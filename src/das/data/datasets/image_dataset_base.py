"""
Defines the base class for Image related Datasets.
"""

import sys

import cv2
import numpy as np
import pandas as pd
import torch
import torchvision
from das.data.datasets.datasets_base import DatasetsBase
from das.data.datasets.utils import DataKeysEnum
from das.utils.basic_utils import create_logger
from PIL import Image
from torchvision.io.image import decode_image, read_file, read_image
from torchvision.transforms.transforms import Normalize
from tqdm import tqdm

logger = create_logger(__name__)


class ImageDatasetsBase(DatasetsBase):
    mean_std = None

    def _save_to_cache(self):
        file_path = super()._save_to_cache()
        if (
            self.data_args.use_dataset_normalization_params
            and self.mean_std is not None
        ):
            mean_std_file = (
                file_path.parents[1]
                / f"mean_std_{self.data_args.data_caching_args.cached_data_name}.df"
            )
            self.mean_std.to_pickle(mean_std_file)

        return file_path

    def _load_from_cache(self):
        data, file_path = super()._load_from_cache()
        if self.data_args.use_dataset_normalization_params:
            mean_std_file = (
                file_path.parents[1] / f"mean_std_{self.data_args.cached_data_name}.df"
            )
            if mean_std_file.exists():
                self.mean_std = pd.read_pickle(mean_std_file)
            else:
                return None, file_path
        return data, file_path

    def compute_dataset_mean_std(self):
        nimages = 0
        mean = 0.0
        var = 0.0
        logger.info("Computing dataset mean/std. This may take a while...")
        for index, _ in tqdm(self.data.iterrows()):
            if self.data_args.data_caching_args.cache_resized_images:
                sample = self.get_sample(index)
                if DataKeysEnum.IMAGE in sample:
                    sample[
                        DataKeysEnum.IMAGE
                    ] = torchvision.transforms.functional.resize(
                        sample[DataKeysEnum.IMAGE],
                        self.data_args.data_caching_args.cache_image_size,
                    )
            else:
                sample = self.get_sample(index)

            # Compute mean and std here
            image = sample[DataKeysEnum.IMAGE] / 255.0
            image = image.view(image.shape[0], -1)
            mean += image.mean(1)
            var += image.var(1)
            nimages += 1

        mean /= nimages
        var /= nimages
        std = torch.sqrt(var)

        self.mean_std = pd.DataFrame([mean, std])

    def _after_load_dataset(self):
        if (
            self.data_args.use_dataset_normalization_params
            and self.mean_std is None
            and self.save_to_cache
            and self.split == "train"
        ):
            self.compute_dataset_mean_std()

    def _after_load_from_cache(self):
        if self.data_args.use_dataset_normalization_params:
            if self.mean_std is None:
                logger.error("No dataset mean_std found for normalization.")
                sys.exit(1)

            mean_tensor = self.mean_std[0][0]
            std_tensor = self.mean_std[0][1]
            fixed_mean = []
            fixed_std = []
            placeholder_mean = self.data_args.data_transforms_args.dataset_mean[
                DataKeysEnum.IMAGE
            ]
            if len(mean_tensor.shape) == 0:
                for idx in range(len(placeholder_mean)):
                    fixed_mean.append(mean_tensor.item())
                    fixed_std.append(std_tensor.item())
            elif len(mean_tensor.shape) != len(placeholder_mean):
                for idx in range(placeholder_mean):
                    fixed_mean.append(mean_tensor[idx])
                    fixed_std.append(std_tensor[idx])

            for module in self.transforms.modules():
                if isinstance(module, Normalize):
                    module.mean = fixed_mean
                    module.std = fixed_std

    def get_sample(self, idx, load_image=True):
        if load_image:
            # get image file path
            image_file_path = self.data.iloc[idx][DataKeysEnum.IMAGE_FILE_PATH]

            try:
                if image_file_path.endswith(("png")):
                    # load image into torch tensor
                    image = read_image(image_file_path)

                    # add a channel to image if not present
                    if len(image.shape) == 2:
                        image = torch.unsqueeze(image, 0)
                else:
                    image = cv2.imread(image_file_path)

                    # convert image to RGB as opencv reads in in BGR
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    image = torch.tensor(image)

                    # permute channels to mimic torch tensors
                    if len(image.shape) == 3:
                        image = image.permute(2, 0, 1)

                    # add a channel to image if not present
                    if len(image.shape) == 2:
                        image = torch.unsqueeze(image, 0)
            except Exception as e:
                logger.exception(
                    f"Exception raised while opening image file: {image_file_path}: ", e
                )
        else:
            image = None

        # get the data
        sample_data = self.data.iloc[idx].to_dict()
        if image is not None:
            sample = {DataKeysEnum.IMAGE: image, **sample_data}
        else:
            sample = {**sample_data}
        return sample
