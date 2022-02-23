import dataclasses
from enum import Enum
from typing import List

import numpy as np
import yaml
from das.data.augmentations.augmentations import *
from matplotlib import pyplot as plt


class AugmentationsEnum(str, Enum):
    # basic
    BRIGHTNESS = "brightness"  # 0
    CONTRAST = "contrast"  # 1

    # transforms
    TRANSLATION = "translation"  # 2
    SCALE = "scale"  # 3
    ROTATION = "rotation"  # 4
    AFFINE = "affine"  # 5

    # blurs
    BINARY_BLUR = "binary_blur"  # 6
    NOISY_BINARY_BLUR = "noisy_binary_blur"  # 7
    DEFOCUS_BLUR = "defocus_blur"  # 8
    MOTION_BLUR = "motion_blur"  # 9
    ZOOM_BLUR = "zoom_blur"  # 10
    GAUSSIAN_BLUR = "gaussian_blur"  # 18

    # distortions
    RANDOM_DISTORTION = "random_distortion"  # 11
    RANDOM_BLOTCHES = "random_blotches"  # 12
    SURFACE_DISTORTION = "surface_distortion"  # 13
    THRESHOLD = "threshold"  # 14
    PIXELATE = "pixelate"  # 15
    JPEG_COMPRESSION = "jpeg_compression"  # 16
    ELASTIC = "elastic"  # 17

    # noise
    GAUSSIAN_NOISE_RGB = "gaussian_noise_rgb"  # 19
    SHOT_NOISE_RGB = "shot_noise_rgb"  # 20
    FIBROUS_NOISE = "fibrous_noise"  # 21
    MULTISCALE_NOISE = "multiscale_noise"  # 22

    # advanced
    BASIC_IMAGE_AUG = "basic_image"
    RAND_AUG = "rand"
    MOCO = "moco"
    BARLOW = "barlow"
    MULTI_CROP = "multicrop"
    TWIN_DOCS = "twin_docs"


AUG_MAP = {
    AugmentationsEnum.BRIGHTNESS: Brightness,
    AugmentationsEnum.CONTRAST: Contrast,
    AugmentationsEnum.TRANSLATION: Translation,
    AugmentationsEnum.SCALE: Scale,
    AugmentationsEnum.ROTATION: Rotation,
    AugmentationsEnum.AFFINE: Affine,
    AugmentationsEnum.BINARY_BLUR: BinaryBlur,
    AugmentationsEnum.NOISY_BINARY_BLUR: NoisyBinaryBlur,
    AugmentationsEnum.DEFOCUS_BLUR: DefocusBlur,
    AugmentationsEnum.MOTION_BLUR: MotionBlur,
    AugmentationsEnum.ZOOM_BLUR: ZoomBlur,
    AugmentationsEnum.GAUSSIAN_BLUR: GaussianBlur,
    AugmentationsEnum.RANDOM_BLOTCHES: RandomBlotches,
    AugmentationsEnum.RANDOM_DISTORTION: RandomDistortion,
    AugmentationsEnum.SURFACE_DISTORTION: SurfaceDistortion,
    AugmentationsEnum.THRESHOLD: Threshold,
    AugmentationsEnum.PIXELATE: Pixelate,
    AugmentationsEnum.JPEG_COMPRESSION: JPEGCompression,
    AugmentationsEnum.ELASTIC: Elastic,
    AugmentationsEnum.GAUSSIAN_NOISE_RGB: GaussianNoiseRGB,
    AugmentationsEnum.SHOT_NOISE_RGB: ShotNoiseRGB,
    AugmentationsEnum.FIBROUS_NOISE: FibrousNoise,
    AugmentationsEnum.MULTISCALE_NOISE: MultiscaleNoise,
    # advanced augmentations for training
    AugmentationsEnum.BASIC_IMAGE_AUG: BasicImageAugmentation,
    AugmentationsEnum.RAND_AUG: RandAugmentation,
    AugmentationsEnum.MOCO: MocoAugmentations,
    AugmentationsEnum.BARLOW: BarlowTwinsAugmentations,
    AugmentationsEnum.MULTI_CROP: MultiCropAugmentation,
    AugmentationsEnum.TWIN_DOCS: TwinDocsAugmentations,
}


@dataclasses.dataclass
class DataAugmentationArguments:
    name: str
    cls_name = "aug_args"
    keys: Optional[List[str]] = None
    params: List[dict] = dataclasses.field(default_factory=lambda: [{}])

    def create(self, severity=None):
        self.name = AugmentationsEnum(self.name)
        aug_class = AUG_MAP.get(self.name, None)
        if aug_class is None:
            raise ValueError(f"Augmentation [{self.name}] is not supported.")
        if severity is None:
            if self.keys is None:
                return aug_class(**self.params[0])
            else:
                return DictTransform(self.keys, aug_class(**self.params[0]))
        else:
            if severity > len(self.params):
                return None
            return aug_class(input, **self.params[severity - 1])


@dataclasses.dataclass
class AugmentatorArguments:
    output_aug_dir: str
    cls_name = "aug_args"
    n_parallel_jobs: int = 4
    debug: bool = False
    datasets: List[str] = dataclasses.field(default_factory=lambda: ["test"])
    augmentations: List[DataAugmentationArguments] = dataclasses.field(
        default_factory=lambda: [{}, {}, {}, {}, {}]
    )
