import ctypes
import importlib
import math
import numbers
import random
import typing
import warnings
from io import BytesIO
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np
import scipy.ndimage as ndi
import torch
import torchvision.transforms.functional as F
from PIL import Image
from PIL import Image as PILImage
from PIL import ImageFilter, ImageOps
from scipy.ndimage import zoom as scizoom
from scipy.ndimage.interpolation import map_coordinates
from skimage.filters import gaussian
from timm.data import create_transform
from torchvision import transforms
from torchvision.transforms import RandomAffine, RandomResizedCrop
from torchvision.transforms.functional import resize
from torchvision.transforms.transforms import (
    ToPILImage,
    _check_sequence_input,
    _setup_angle,
)

ocrodeg = importlib.util.find_spec("ocrodeg")
ocrodeg_available = ocrodeg is not None
if ocrodeg_available:
    import ocrodeg


class DictTransform(object):
    """
    Applies the transformation on given keys for dictionary outputs

    Args:
        keys (list): List of keys
        transform (callable): Transformation to be applied
    """

    def __init__(self, keys: list, transform: callable):
        super().__init__()

        self.keys = keys
        self.transform = transform

    def __call__(self, sample):
        for key in self.keys:
            if key in sample:
                sample[key] = self.transform(sample[key])

        return sample


class RandomResizedCropCustom(RandomResizedCrop):
    @staticmethod
    def get_params(img, scale, ratio, region_mask):
        """Get parameters for ``crop`` for a random sized crop.
        Args:
            img (PIL Image or Tensor): Input image.
            scale (list): range of scale of the origin size cropped
            ratio (list): range of aspect ratio of the origin aspect ratio cropped
        Returns:
            tuple: params (i, j, h, w) to be passed to ``crop`` for a random
            sized crop.
        """
        width, height = F._get_image_size(img)
        area = height * width

        log_ratio = torch.log(torch.tensor(ratio))
        for _ in range(10):
            target_area = area * torch.empty(1).uniform_(scale[0], scale[1]).item()
            aspect_ratio = torch.exp(
                torch.empty(1).uniform_(log_ratio[0], log_ratio[1])
            ).item()

            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))

            if 0 < w <= width and 0 < h <= height:
                # mask = region_mask[h//2:height-h//2, w//2:width-w//2]
                pixel_list = region_mask.nonzero()
                if len(pixel_list) == 0:
                    i = torch.randint(0, height - h + 1, size=(1,)).item()
                    j = torch.randint(0, width - w + 1, size=(1,)).item()
                else:
                    p_idx = torch.randint(0, len(pixel_list), size=(1,)).item()
                    i = pixel_list[p_idx][0] - h // 2
                    j = pixel_list[p_idx][1] - w // 2
                    i = int(torch.clip(i, min=0, max=height - h - 1))
                    j = int(torch.clip(j, min=0, max=width - w - 1))

                return i, j, h, w

        # Fallback to central crop
        in_ratio = float(width) / float(height)
        if in_ratio < min(ratio):
            w = width
            h = int(round(w / min(ratio)))
        elif in_ratio > max(ratio):
            h = height
            w = int(round(h * max(ratio)))
        else:  # whole image
            w = width
            h = height
        i = (height - h) // 2
        j = (width - w) // 2
        return i, j, h, w

    def forward(self, img, pixel_list):
        """
        Args:
            img (PIL Image or Tensor): Image to be cropped and resized.
        Returns:
            PIL Image or Tensor: Randomly cropped and resized image.
        """
        i, j, h, w = self.get_params(img, self.scale, self.ratio, pixel_list)
        return F.resized_crop(img, i, j, h, w, self.size, self.interpolation)


class RandomResizedMaskedCrop(object):
    def __init__(self, image_size, scale):
        self.t = RandomResizedCropCustom((image_size, image_size), scale=scale)

    def get_black_and_white_regions_mask(self, image_tensor):
        black_and_white_threshold = 0.5
        c, h, w = image_tensor.shape
        ky = 8
        kx = 8
        black_and_white_regions_fast = (
            (
                image_tensor[0].unfold(0, ky, kx).unfold(1, ky, kx)
                < black_and_white_threshold
            )
            .any(dim=2)
            .any(dim=2)
        )
        black_and_white_regions_fast = black_and_white_regions_fast.repeat_interleave(
            ky, dim=0
        ).repeat_interleave(kx, dim=1)
        black_and_white_regions_fast = transforms.functional.resize(
            black_and_white_regions_fast.unsqueeze(0), [h, w]
        ).squeeze()
        return (black_and_white_regions_fast).float()

    def __call__(self, img):
        return self.t(img, self.get_black_and_white_regions_mask(img)) / 255.0


class DeterministicAffine(RandomAffine):
    def __init__(
        self,
        degrees,
        translate=None,
        scale=None,
        shear=None,
        interpolation=F.InterpolationMode.NEAREST,
        fill=0,
        fillcolor=None,
        resample=None,
    ):
        torch.nn.Module.__init__(self)

        if resample is not None:
            warnings.warn(
                "Argument resample is deprecated and will be removed since v0.10.0. Please, use interpolation instead"
            )
            interpolation = F._interpolation_modes_from_int(resample)

        # Backward compatibility with integer value
        if isinstance(interpolation, int):
            warnings.warn(
                "Argument interpolation should be of type InterpolationMode instead of int. "
                "Please, use InterpolationMode enum."
            )
            interpolation = F._interpolation_modes_from_int(interpolation)

        if fillcolor is not None:
            warnings.warn(
                "Argument fillcolor is deprecated and will be removed since v0.10.0. Please, use fill instead"
            )
            fill = fillcolor

        self.degrees = degrees

        if translate is not None:
            _check_sequence_input(translate, "translate", req_sizes=(2,))
            for t in translate:
                if not (0.0 <= t <= 1.0):
                    raise ValueError("translation values should be between 0 and 1")
        self.translate = translate

        if scale is not None:
            if scale <= 0:
                raise ValueError("scale values should be positive")
        self.scale = scale

        if shear is not None:
            self.shear = _setup_angle(shear, name="shear", req_sizes=(2,))
        else:
            self.shear = shear

        self.resample = self.interpolation = interpolation

        if fill is None:
            fill = 0
        elif not isinstance(fill, (Sequence, numbers.Number)):
            raise TypeError("Fill should be either a sequence or a number.")

        self.fillcolor = self.fill = fill

    @staticmethod
    def get_params(
        degrees: List[float],
        translate: Optional[List[float]],
        scale: Optional[List[float]],
        shears: Optional[List[float]],
        image_size: List[int],
    ) -> Tuple[float, Tuple[int, int], float, Tuple[float, float]]:
        """Get parameters for affine transformation

        Returns:
            params to be passed to the affine transformation
        """
        angle = degrees
        if translate is not None:
            tx = float(translate[0] * image_size[0])
            ty = float(translate[1] * image_size[1])
            translations = (tx, ty)
        else:
            translations = (0, 0)

        if scale is None:
            scale = 1.0

        shear_x = shear_y = 0.0
        if shears is not None:
            shear_x = shears[0]
            shear_y = shears[1]

        shear = (shear_x, shear_y)

        return angle, translations, scale, shear


def disk(radius, alias_blur=0.1, dtype=np.float32):
    if radius <= 8:
        L = np.arange(-8, 8 + 1)
        ksize = (3, 3)
    else:
        L = np.arange(-radius, radius + 1)
        ksize = (5, 5)
    X, Y = np.meshgrid(L, L)
    aliased_disk = np.array((X ** 2 + Y ** 2) <= radius ** 2, dtype=dtype)
    aliased_disk /= np.sum(aliased_disk)

    # supersample disk to antialias
    return cv2.GaussianBlur(aliased_disk, ksize=ksize, sigmaX=alias_blur)


class Brightness(object):
    def __init__(self, beta):
        self.beta = beta

    def __call__(self, image):
        return np.clip(image + self.beta, 0, 1)


class Contrast(object):
    def __init__(self, alpha):
        self.alpha = alpha

    def __call__(self, image):
        channel_means = np.mean(image, axis=(0, 1))
        return np.clip((image - channel_means) * self.alpha + channel_means, 0, 1)


class Translation(object):
    def __init__(self, magnitude):
        self.magnitude = magnitude

    def __call__(self, image):
        return ocrodeg.transform_image(image, translation=self.magnitude)


class Scale(object):
    def __init__(self, scale, fill=1):
        self.scale = scale
        self.fill = fill

    def __call__(self, image):
        image = torch.tensor(image).unsqueeze(0)
        scale = np.random.choice(self.scale)
        scale = [scale - 0.025, scale + 0.025]
        t = RandomAffine(degrees=0, scale=scale, fill=self.fill)
        image = t(image).squeeze().numpy()
        return image


class Rotation(object):
    def __init__(self, magnitude):
        self.magnitude = magnitude

    def __call__(self, image):
        return ndi.rotate(image, self.magnitude)


class Affine(object):
    def __init__(self, degrees, translate=[0, 0], shear=[0, 0], fill=1):
        self.degrees = degrees
        self.translate = translate
        self.shear = shear
        self.fill = fill

    def __call__(self, image):
        image = torch.tensor(image).unsqueeze(0)
        translate = np.random.choice(self.translate)
        translate = [translate - 0.01, translate + 0.01]
        degrees = np.random.choice(self.degrees)
        degrees = [degrees - 1, degrees + 1]
        shear = np.random.choice(self.shear)
        shear = [shear - 0.5, shear + 0.05]
        t = RandomAffine(
            degrees=degrees, translate=translate, shear=shear, fill=self.fill
        )
        image = t(image).squeeze().numpy()
        return image


class BinaryBlur(object):
    def __init__(self, sigma):
        self.sigma = sigma

    def __call__(self, image):
        return ocrodeg.binary_blur(image, sigma=self.sigma)


class NoisyBinaryBlur(object):
    def __init__(self, sigma, noise):
        self.sigma = sigma
        self.noise = noise

    def __call__(self, image):
        return ocrodeg.binary_blur(image, sigma=self.sigma, noise=self.noise)


class DefocusBlur(object):
    def __init__(self, radius, alias_blur):
        self.radius = radius
        self.alias_blur = alias_blur

    def __call__(self, image):
        kernel = disk(radius=self.radius, alias_blur=self.alias_blur)
        return np.clip(cv2.filter2D(image, -1, kernel), 0, 1)


class MotionBlur(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, image):
        # generating the kernel
        kernel_motion_blur = np.zeros((self.size, self.size))
        kernel_motion_blur[int((self.size - 1) / 2), :] = np.ones(
            self.size, dtype=np.float32
        )
        kernel_motion_blur = cv2.warpAffine(
            kernel_motion_blur,
            cv2.getRotationMatrix2D(
                (self.size / 2 - 0.5, self.size / 2 - 0.5),
                np.random.uniform(-45, 45),
                1.0,
            ),
            (self.size, self.size),
        )
        kernel_motion_blur = kernel_motion_blur * (1.0 / np.sum(kernel_motion_blur))
        return cv2.filter2D(image, -1, kernel_motion_blur)


class ZoomBlur(object):
    def __init__(self, zoom_factor_start, zoom_factor_end, zoom_factor_step):
        self.zoom_factor_start = zoom_factor_start
        self.zoom_factor_end = zoom_factor_end
        self.zoom_factor_step = zoom_factor_step

    def clipped_zoom(self, image, zoom_factor):
        h = image.shape[0]
        w = image.shape[1]
        # ceil crop height(= crop width)
        ch = int(np.ceil(h / float(zoom_factor)))
        cw = int(np.ceil(w / float(zoom_factor)))
        top = (h - ch) // 2
        left = (w - cw) // 2
        img = scizoom(
            image[top : top + ch, left : left + cw],
            (self.zoom_factor, self.zoom_factor),
            order=1,
        )
        # trim off any extra pixels
        trim_top = (img.shape[0] - h) // 2
        trim_left = (img.shape[1] - w) // 2

        return img[trim_top : trim_top + h, trim_left : trim_left + w]

    def __call__(self, image):
        out = np.zeros_like(image)
        zoom_factor_range = np.arange(
            self.zoom_factor_start, self.zoom_factor_end, self.zoom_factor_step
        )
        for zoom_factor in zoom_factor_range:
            out += self.clipped_zoom(image, zoom_factor)
        return np.clip((image + out) / (len(zoom_factor_range) + 1), 0, 1)


class RandomDistortion(object):
    def __init__(self, sigma, maxdelta):
        self.sigma = sigma
        self.maxdelta = maxdelta

    def __call__(self, image):
        noise = ocrodeg.bounded_gaussian_noise(image.shape, self.sigma, self.maxdelta)
        return ocrodeg.distort_with_noise(image, noise)


class RandomBlotches(object):
    def __init__(self, fgblobs, bgblobs, fgscale, bgscale):
        self.fgblobs = fgblobs
        self.bgblobs = bgblobs
        self.fgscale = fgscale
        self.bgscale = bgscale

    def __call__(self, image):
        return ocrodeg.random_blotches(
            image,
            fgblobs=self.fgblobs,
            bgblobs=self.bgblobs,
            fgscale=self.fgscale,
            bgscale=self.bgscale,
        )


class SurfaceDistortion(object):
    def __init__(self, magnitude):
        self.magnitude = magnitude

    def __call__(self, image):
        noise = ocrodeg.noise_distort1d(image.shape, magnitude=self.magnitude)
        return ocrodeg.distort_with_noise(image, noise)


class Threshold(object):
    def __init__(self, magnitude):
        self.magnitude = magnitude

    def __call__(self, image):
        blurred = ndi.gaussian_filter(image, self.magnitude)
        return 1.0 * (blurred > 0.5)


class GaussianBlur(object):
    def __init__(self, magnitude):
        self.magnitude = magnitude

    def __call__(self, image):
        return ndi.gaussian_filter(image, self.magnitude)


class GaussianNoiseRGB(object):
    def __init__(self, magnitude):
        self.magnitude = magnitude

    def __call__(self, image):
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        return np.clip(
            image + np.random.normal(size=image.shape, scale=self.magnitude), 0, 1
        )


class ShotNoiseRGB(object):
    def __init__(self, magnitude):
        self.magnitude = magnitude

    def __call__(self, image):
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

        return np.clip(
            np.random.poisson(image * self.magnitude) / float(self.magnitude), 0, 1
        )


class FibrousNoise(object):
    def __init__(self, blur, blotches):
        self.blur = blur
        self.blotches = blotches

    def __call__(self, image):
        return ocrodeg.printlike_fibrous(image, blur=self.blur, blotches=self.blotches)


class MultiscaleNoise(object):
    def __init__(self, blur, blotches):
        self.blur = blur
        self.blotches = blotches

    def __call__(self, image):
        return ocrodeg.printlike_multiscale(
            image, blur=self.blur, blotches=self.blotches
        )


class Pixelate(object):
    def __init__(self, magnitude):
        self.magnitude = magnitude

    def __call__(self, image):
        h, w = image.shape
        image = cv2.resize(
            image,
            (int(w * self.magnitude), int(h * self.magnitude)),
            interpolation=cv2.INTER_LINEAR,
        )
        return cv2.resize(image, (w, h), interpolation=cv2.INTER_NEAREST)


class JPEGCompression(object):
    def __init__(self, quality):
        self.quality = quality

    def __call__(self, image):
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), self.quality]
        result, encimg = cv2.imencode(".jpg", image * 255, encode_param)
        decimg = cv2.imdecode(encimg, 0) / 255.0
        return decimg


class Elastic(object):
    def __init__(self, alpha, sigma, alpha_affine, random_state=None):
        self.alpha = alpha
        self.sigma = sigma
        self.alpha_affine = alpha_affine
        self.random_state = random_state

    def __call__(self, image):
        assert len(image.shape) == 2
        shape = image.shape
        shape_size = shape[:2]

        image = np.array(image, dtype=np.float32) / 255.0
        shape = image.shape
        shape_size = shape[:2]

        # random affine
        center_square = np.float32(shape_size) // 2
        square_size = min(shape_size) // 3
        pts1 = np.float32(
            [
                center_square + square_size,
                [center_square[0] + square_size, center_square[1] - square_size],
                center_square - square_size,
            ]
        )
        pts2 = pts1 + np.random.uniform(
            -self.alpha_affine, self.alpha_affine, size=pts1.shape
        ).astype(np.float32)
        M = cv2.getAffineTransform(pts1, pts2)
        image = cv2.warpAffine(
            image, M, shape_size[::-1], borderMode=cv2.BORDER_REFLECT_101
        )

        dx = (
            gaussian(
                np.random.uniform(-1, 1, size=shape[:2]),
                self.sigma,
                mode="reflect",
                truncate=3,
            )
            * self.alpha
        ).astype(np.float32)
        dy = (
            gaussian(
                np.random.uniform(-1, 1, size=shape[:2]),
                self.sigma,
                mode="reflect",
                truncate=3,
            )
            * self.alpha
        ).astype(np.float32)

        x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
        indices = np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1))
        return (
            np.clip(
                map_coordinates(image, indices, order=1, mode="reflect").reshape(shape),
                0,
                1,
            )
            * 255
        )


class GaussianBlurPIL(object):
    def __init__(self, sigma=[0.1, 2.0]):
        self.sigma = sigma

    def __call__(self, image):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        image = image.filter(ImageFilter.GaussianBlur(radius=sigma))
        return image


class Solarization(object):
    def __init__(self, p):
        self.p = p

    def __call__(self, image):
        if random.random() < self.p:
            return ImageOps.solarize(image)
        else:
            return image


class GrayScaleToRGB(object):
    """
    Applies the transformation on an image to convert grayscale to rgb
    """

    def __init__(self):
        super().__init__()

    def __call__(self, image):
        if image.shape[0] == 1:
            return image.repeat(3, 1, 1)
        else:
            return image


class RandomRescale(object):
    """
    Randomly rescales the images based on the max/min dims.

    Args:
        rescale_dims (list): Possible random scale dims for shorter dim
        max_rescale_dim (bool): Maximum rescale dimension for larger dim
        random_sample_max_iters (int): Maximum random sampling iterations
    """

    def __init__(
        self,
        # [320, 416, 512, 608, 704]
        rescale_dims: typing.List[int],
        max_rescale_dim: int,
        random_sample_max_iters: Optional[int] = 100,
    ):
        super().__init__()

        self.rescale_dims = rescale_dims
        self.max_rescale_dim = max_rescale_dim
        self.random_sample_max_iters = random_sample_max_iters

    def __call__(self, image):
        # randomly rescale the image in the batch as done in ViBertGrid
        # shape (C, H, W)
        image_height = image.shape[1]
        image_width = image.shape[2]

        # get larger dim
        larger_dim_idx = 0 if image_height > image_width else 1
        smaller_dim_idx = 0 if image_height < image_width else 1

        rescale_dims = [i for i in self.rescale_dims]

        # find random rescale dim
        rescaled_shape = None
        for iter in range(self.random_sample_max_iters):
            if len(rescale_dims) > 0:
                # get smaller dim out of possible dims
                idx, smaller_dim = random.choice(list(enumerate(rescale_dims)))

                # find the rescale ratio
                rescale_ratio = smaller_dim / image.shape[smaller_dim_idx]

                # rescale larger dim
                larger_dim = rescale_ratio * image.shape[larger_dim_idx]

                # check if larger dim is smaller than max large
                if larger_dim > self.max_rescale_dim:
                    rescale_dims.pop(idx)
                else:
                    rescaled_shape = list(image.shape)
                    rescaled_shape[larger_dim_idx] = int(larger_dim)
                    rescaled_shape[smaller_dim_idx] = int(smaller_dim)
                    break
            else:
                # if no smaller dim is possible rescale image according to
                # larger dim
                larger_dim = self.max_rescale_dim

                # find the rescale ratio
                rescale_ratio = larger_dim / image.shape[larger_dim_idx]

                # rescale smaller dim
                smaller_dim = rescale_ratio * image.shape[smaller_dim_idx]

                rescaled_shape = list(image.shape)
                rescaled_shape[larger_dim_idx] = int(larger_dim)
                rescaled_shape[smaller_dim_idx] = int(smaller_dim)
                break

        if rescaled_shape is not None:
            # resize the image according to the output shape
            return resize(image, rescaled_shape[1:])
        else:
            return image


class Rescale(object):
    """
    Randomly rescales the images based on the max/min dims.

    Args:
        rescale_dim (int): Rescale dimension for smaller dim
        rescale_smaller_dim (bool): Whether to rescale smaller dim, otherwise larger
            dimension is scaled
    """

    def __init__(
        self,
        rescale_dim: int,
        rescale_smaller_dim: bool = True,
        rescale_both_dims: bool = False,
    ):
        super().__init__()

        self.rescale_dim = rescale_dim
        self.rescale_smaller_dim = rescale_smaller_dim
        self.rescale_both_dims = rescale_both_dims

    def __call__(self, image):
        # randomly rescale the image in the batch as done in ViBertGrid
        # shape (C, H, W)
        image_height = image.shape[1]
        image_width = image.shape[2]

        if not self.rescale_both_dims:
            # get smaller dim
            larger_dim_idx = 0 if image_height > image_width else 1
            smaller_dim_idx = 0 if image_height < image_width else 1

            dim_idx = smaller_dim_idx if self.rescale_smaller_dim else larger_dim_idx
            other_dim_idx = (
                larger_dim_idx if self.rescale_smaller_dim else smaller_dim_idx
            )

            # find the rescale ratio
            rescale_ratio = self.rescale_dim / image.shape[dim_idx]

            # rescale the other dim
            other_dim = rescale_ratio * image.shape[other_dim_idx]

            rescaled_shape = list(image.shape)
            rescaled_shape[dim_idx] = int(self.rescale_dim)
            rescaled_shape[other_dim_idx] = int(other_dim)
        else:
            rescaled_shape = list(image.shape)
            if isinstance(self.rescale_dim, list):
                rescaled_shape[1] = self.rescale_dim[0]
                rescaled_shape[2] = self.rescale_dim[1]
            else:
                rescaled_shape[1] = self.rescale_dim
                rescaled_shape[2] = self.rescale_dim

        # resize the image according to the output shape
        return resize(image, rescaled_shape[1:])


class RGBToBGR(object):
    """
    Applies the transformation on an image to convert grayscale to rgb
    """

    def __init__(self):
        super().__init__()

    def __call__(self, image):
        return image.permute(2, 1, 0)


class ImageRescaleStrategyEnum:
    RESCALE_SIMPLE = "rescale_simple"
    RESCALE_RANDOM = "rescale_random"
    RANDOM_RESIZED_CROP = "random_resized_crop"


class BasicImageAugmentation(object):
    def __init__(
        self,
        gray_to_rgb=False,
        rgb_to_bgr=False,
        rescale_strategy: str = ImageRescaleStrategyEnum.RESCALE_SIMPLE,
        rescale_params: Optional[dict] = None,
        normalize=True,
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ):

        t = []
        if gray_to_rgb:
            t.append(GrayScaleToRGB())

        if rgb_to_bgr:
            t.append(RGBToBGR())

        if rescale_strategy == ImageRescaleStrategyEnum.RESCALE_SIMPLE:
            t.append(Rescale(**rescale_params))
        elif rescale_strategy == ImageRescaleStrategyEnum.RESCALE_RANDOM:
            t.append(RandomRescale(**rescale_params))
        elif rescale_strategy == ImageRescaleStrategyEnum.RANDOM_RESIZED_CROP:
            t.append(transforms.RandomResizedCrop(**rescale_params))
        t.append(transforms.ConvertImageDtype(torch.float))
        if normalize:
            t.append(transforms.Normalize(mean, std))
        self.aug = transforms.Compose(t)

    def __call__(self, image):
        return self.aug(image)


class RandAugmentation(object):
    def __init__(
        self,
        image_size=224,
        color_jitter=0.4,
        auto_augment="rand-m9-mstd0.5-inc1",
        interpolation="bicubic",
        re_prob=0.25,
        re_mode="pixel",
        re_count=1,
        n_augs=1,
    ):
        self.n_augs = n_augs

        self.aug = create_transform(
            input_size=image_size,
            is_training=True,
            color_jitter=color_jitter,
            auto_augment=auto_augment,
            interpolation=interpolation,
            re_prob=re_prob,
            re_mode=re_mode,
            re_count=re_count,
        )

    def __call__(self, image):
        if self.n_augs == 1:
            return self.aug(image)
        else:
            augs = []
            for _ in self.n_augs:
                augs.append(self.aug(image))
                augs.append(self.aug(image))
        return augs


class MocoAugmentations(object):
    def __init__(
        self,
        image_size,
        gray_to_rgb=True,
        to_pil=True,
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ):
        base_aug = []
        if gray_to_rgb:
            base_aug.append(GrayScaleToRGB())
        if to_pil:
            base_aug.append(transforms.ToPILImage())

        self.aug = transforms.Compose(
            base_aug
            + [
                transforms.RandomResizedCrop(
                    image_size, scale=(0.2, 1.0), interpolation=PILImage.BICUBIC
                ),
                transforms.RandomApply(
                    [transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8
                ),
                transforms.RandomGrayscale(p=0.5),
                transforms.RandomApply([GaussianBlurPIL([0.1, 2.0])], p=0.5),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ]
        )

    def __call__(self, image):
        crops = []
        crops.append(self.aug(image))
        crops.append(self.aug(image))
        return crops


class BarlowTwinsAugmentations(object):
    def __init__(
        self,
        image_size,
        gray_to_rgb=True,
        to_pil=True,
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ):
        base_aug = []
        if gray_to_rgb:
            base_aug.append(GrayScaleToRGB())
        if to_pil:
            base_aug.append(transforms.ToPILImage())

        self.aug1 = transforms.Compose(
            base_aug
            + [
                transforms.RandomResizedCrop(
                    image_size, interpolation=PILImage.BICUBIC
                ),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomApply(
                    [
                        transforms.ColorJitter(
                            brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1
                        )
                    ],
                    p=0.8,
                ),
                transforms.RandomGrayscale(p=0.2),
                transforms.RandomApply([GaussianBlurPIL([0.1, 2.0])], p=1.0),
                # Solarization(p=0.0),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ]
        )

        self.aug2 = transforms.Compose(
            base_aug
            + [
                transforms.RandomResizedCrop(image_size, interpolation=Image.BICUBIC),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomApply(
                    [
                        transforms.ColorJitter(
                            brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1
                        )
                    ],
                    p=0.8,
                ),
                transforms.RandomGrayscale(p=0.2),
                transforms.RandomApply([GaussianBlurPIL([0.1, 2.0])], p=0.1),
                # Solarization(p=0.2),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    def __call__(self, image):
        crops = []
        crops.append(self.aug1(image))
        crops.append(self.aug2(image))
        return crops


class MultiCropAugmentation(object):
    def __init__(
        self,
        image_size,
        gray_to_rgb=True,
        to_pil=True,
        global_crops_scale=(0.4, 1.0),
        local_crops_scale=(0.05, 0.4),
        local_crops_number=12,
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ):
        flip_and_color_jitter = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomApply(
                    [
                        transforms.ColorJitter(
                            brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1
                        )
                    ],
                    p=0.8,
                ),
                transforms.RandomGrayscale(p=0.2),
            ]
        )
        normalize = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )

        base_aug = []
        if gray_to_rgb:
            base_aug.append(GrayScaleToRGB())
        if to_pil:
            base_aug.append(transforms.ToPILImage())

        # first global crop
        self.global_transfo1 = transforms.Compose(
            base_aug
            + [
                transforms.RandomResizedCrop(
                    image_size, scale=global_crops_scale, interpolation=PILImage.BICUBIC
                ),
                # flip_and_color_jitter,
                # transforms.RandomApply([GaussianBlurPIL([0.1, 2.0])], p=1.0),
                # normalize,
                transforms.RandomApply(
                    [transforms.RandomAffine((-2, 2), fill=255)], p=0.5
                ),
                transforms.RandomApply(
                    [transforms.RandomAffine(0, translate=(0.2, 0.2), fill=255)], p=0.5
                ),
                transforms.RandomApply(
                    [transforms.RandomAffine(0, scale=(0.9, 1.0), fill=255)], p=0.5
                ),
                transforms.RandomApply(
                    [transforms.RandomAffine(0, shear=(-2, 2), fill=255)], p=0.5
                ),
                transforms.ToTensor(),
            ]
        )
        # second global crop
        self.global_transfo2 = transforms.Compose(
            base_aug
            + [
                transforms.RandomResizedCrop(
                    image_size, scale=global_crops_scale, interpolation=PILImage.BICUBIC
                ),
                # flip_and_color_jitter,
                # transforms.RandomApply([GaussianBlurPIL([0.1, 2.0])], p=0.1),
                # Solarization(0.2),
                # normalize,
                transforms.RandomApply(
                    [transforms.RandomAffine((-2, 2), fill=255)], p=0.5
                ),
                transforms.RandomApply(
                    [transforms.RandomAffine(0, translate=(0.2, 0.2), fill=255)], p=0.5
                ),
                transforms.RandomApply(
                    [transforms.RandomAffine(0, scale=(0.9, 1.0), fill=255)], p=0.5
                ),
                transforms.RandomApply(
                    [transforms.RandomAffine(0, shear=(-2, 2), fill=255)], p=0.5
                ),
                transforms.ToTensor(),
            ]
        )
        # transformation for the local small crops
        self.local_crops_number = local_crops_number
        self.local_transfo = transforms.Compose(
            base_aug
            + [
                transforms.RandomResizedCrop(
                    96, scale=local_crops_scale, interpolation=PILImage.BICUBIC
                ),
                # flip_and_color_jitter,
                # utils.GaussianBlur(p=0.5),
                # transforms.RandomApply([GaussianBlurPIL([0.1, 2.0])], p=0.5),
                # normalize,
                transforms.RandomApply(
                    [transforms.RandomAffine((-2, 2), fill=255)], p=0.5
                ),
                transforms.RandomApply(
                    [transforms.RandomAffine(0, translate=(0.2, 0.2), fill=255)], p=0.5
                ),
                transforms.RandomApply(
                    [transforms.RandomAffine(0, scale=(0.9, 1.0), fill=255)], p=0.5
                ),
                transforms.RandomApply(
                    [transforms.RandomAffine(0, shear=(-2, 2), fill=255)], p=0.5
                ),
                transforms.ToTensor(),
            ]
        )

    def __call__(self, image):
        crops = []
        crops.append(self.global_transfo1(image))
        crops.append(self.global_transfo2(image))
        for _ in range(self.local_crops_number):
            crops.append(self.local_transfo(image))
        return crops


class TwinDocsAugmentations(object):
    def __init__(
        self,
        image_size,
        gray_to_rgb=True,
        to_pil=True,
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ):
        base_aug = []
        if gray_to_rgb:
            base_aug.append(GrayScaleToRGB())
        if to_pil:
            base_aug.append(transforms.ToPILImage())
        self.aug1 = transforms.Compose(
            [
                RandomResizedMaskedCrop(image_size, (0.4, 0.8)),
                *base_aug,
                transforms.RandomApply(
                    [transforms.RandomAffine((-2, 2), fill=255)], p=0.5
                ),
                transforms.RandomApply(
                    [transforms.RandomAffine(0, translate=(0.2, 0.2), fill=255)], p=0.5
                ),
                transforms.RandomApply(
                    [transforms.RandomAffine(0, scale=(0.9, 1.0), fill=255)], p=0.5
                ),
                transforms.RandomApply(
                    [transforms.RandomAffine(0, shear=(-2, 2), fill=255)], p=0.5
                ),
                # transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomApply(
                    [
                        transforms.ColorJitter(
                            brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1
                        )
                    ],
                    p=0.8,
                ),
                transforms.RandomGrayscale(p=0.2),
                transforms.RandomApply([GaussianBlurPIL([0.1, 0.5])], p=1.0),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ]
        )
        self.aug2 = transforms.Compose(
            [
                RandomResizedMaskedCrop(image_size, (0.4, 0.8)),
                *base_aug,
                transforms.RandomApply(
                    [transforms.RandomAffine((-5, 5), fill=255)], p=0.5
                ),
                transforms.RandomApply(
                    [transforms.RandomAffine(0, translate=(0.2, 0.2), fill=255)], p=0.5
                ),
                transforms.RandomApply(
                    [transforms.RandomAffine(0, scale=(0.9, 1.0), fill=255)], p=0.5
                ),
                transforms.RandomApply(
                    [transforms.RandomAffine(0, shear=(-5, 5), fill=255)], p=0.5
                ),
                # transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomApply(
                    [
                        transforms.ColorJitter(
                            brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1
                        )
                    ],
                    p=0.8,
                ),
                transforms.RandomGrayscale(p=0.2),
                transforms.RandomApply([GaussianBlurPIL([0.1, 0.5])], p=1.0),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ]
        )

    def __call__(self, image):
        crops = []
        crops.append(self.aug1(image))
        crops.append(self.aug2(image))
        return crops
