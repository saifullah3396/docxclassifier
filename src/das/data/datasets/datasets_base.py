"""
Defines the base class for Datasets.
"""

import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from das.data.data_args import DataArguments
from das.data.datasets.data_cacher import DataCacher, DatadingsDataCacher
from das.data.datasets.utils import DataKeysEnum
from das.utils.basic_utils import create_logger
from pandas.core.frame import DataFrame
from torch.utils.data import Dataset

logger = create_logger(__name__)
pd.set_option("display.max_columns", 10)


class DatasetsBase(Dataset):
    """
    Defines the base class for handling different datasets.

    Args:
        data_args: Data related arguments.
        split: Dataset split.
        transforms: Data transforms required.
        download: Whether to download the dataset.
        use_cached: Whether to use cached data.
        save_to_cache: Whether to save data to cache.
        prepare_only:
            Whether to only check if cached data is present otherwise create it
        train_val_indices: The subset indices to be used for training/validation dataset
            splits if they don't exist in the dataset already.
    """

    is_downloadable = False
    supported_splits = ["train", "test"]

    def __init__(
        self,
        data_args: DataArguments,
        split: str,
        transforms=None,
        download: bool = False,
        use_cached: bool = False,
        save_to_cache: bool = True,
        prepare_only: bool = False,
        train_val_indices: list = [],
    ):
        """
        Args:
            root_dir (string): Directory with all the data images and annotations.
            use_cached (bool): Whether to use cached data or prepare it again
        """
        # initialize the class variables
        self.data_args = data_args
        self.split = split
        self.transforms = transforms
        self.download = download
        self.use_cached = use_cached
        self.save_to_cache = save_to_cache
        self.prepare_only = prepare_only
        self.train_val_indices = train_val_indices

        # initialize the datacacher
        if self.data_args.data_caching_args.use_datadings:
            self.data_cache_handler = DatadingsDataCacher(data_args, split)
        else:
            self.data_cache_handler = DataCacher(data_args, split)

    def load(self):
        """
        Loads the dataset, from cache if present, otherwise loads it from directory or
        downloads it. This function also performs other post-processing operations
        required on the dataset such as tokenization and finally saves the data to
        cache.
        """

        try:
            # initialize the main data holder object
            self.data = None

            # see if the cached data already exists and return if true
            if self.prepare_only:
                if self.data_cache_handler.validate_cache():
                    return

            # if use_cache is True load data from cache if available
            if self.use_cached:
                # get the dataset directory
                self.root_dir = Path(self.data_args.data_caching_args.dataset_cache_dir)

                # check if dataset directory exists
                if not self.root_dir.exists():
                    self.root_dir.mkdir(parents=True)

                self.data, _ = self._load_from_cache()

                # hook for after loading data from cache
                if self.data is not None:
                    self._after_load_from_cache()

                if self.transforms is not None:
                    logger.info(f"Defining data transformations [{self.split}]:")
                    for x in self.transforms:
                        print("\t", x.transform)

            if self.data is None:
                if self.is_downloadable and self.download:
                    logger.info(
                        f"Downloading the dataset [{self.data_args.dataset_name}-{self.split}]..."
                    )
                else:
                    if self.save_to_cache:
                        logger.info(
                            f"Initializing the dataset [{self.data_args.dataset_name}-{self.split}] "
                            f"from directory: {self.data_args.dataset_dir}."
                        )

                    # get the dataset directory
                    if self.data_args.dataset_dir is not None:
                        self.root_dir = Path(self.data_args.dataset_dir)

                        # check if dataset directory exists
                        if not self.root_dir.exists():
                            raise ValueError(
                                f"Could not find the dataset directory: {self.root_dir}"
                            )

                if self.split not in self.supported_splits:
                    if len(self.train_val_indices) == 0:
                        raise ValueError(
                            f"Split argument '{self.split}' not supported."
                        )

                # load the dataset
                self.data = self._load_dataset()

                # tokenizing dataset
                if (
                    self.save_to_cache
                    and self.data_args.data_tokenization_args.tokenize_dataset
                    and not self.data_args.data_tokenization_args.tokenize_per_sample
                ):
                    logger.info(f"Tokenizing the dataset...")
                    self.data = self._tokenize(self.data)

                # hook after loading dataset
                self._after_load_dataset()

                if self.save_to_cache:
                    self._save_to_cache()

            if self.data is not None and isinstance(self.data, DataFrame):
                logger.debug(f"Dataset:\n{self.data.head(5)}")
        except Exception as exc:
            logger.exception(
                f"Exception raised while loading dataset "
                f"[{self.data_args.dataset_name}]: {exc}"
            )
            sys.exit(1)

    def _load_dataset(self):
        """
        This must be defined in the child dataset class to actually load the dataset.
        """
        raise NotImplementedError()

    def _after_load_dataset(self):
        """
        Hook for performing operations after loading the dataset.
        """
        pass

    def _tokenize(self, data):
        """
        This must be defined in the child class to tokenize the dataset if required.
        """
        raise NotImplementedError()

    def _tokenize_sample(self, sample):
        """
        This must be defined in the child class to tokenize the dataset per sample
        if required.
        """
        raise NotImplementedError()

    def __len__(self):
        """
        Returns the total size of the dataset.
        """

        # if data is not defined just return an error
        if self.data is None:
            raise ValueError(
                f"No data loaded in the dataset: {self.data_args.dataset_name}"
            )

        # set train_val_split if indices are available
        if self.split in ["train", "val"]:
            if len(self.train_val_indices) > 0:
                return len(self.train_val_indices)

        if isinstance(self.data, DataFrame):
            return self.data.shape[0]
        else:
            return len(self.data)

    def get_sample(self, idx, caching=False):
        """
        This must be defined in the child class to actually return one sample at
        given index.

        Args:
            idx: Sample index.
        """
        raise NotImplementedError()

    def __getitem__(self, idx):
        """
        Called by torch dataloaders to get the sample.
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # if train/val indices are defined then first correct the index
        if len(self.train_val_indices) > 0:
            idx = self.train_val_indices[idx]

        # get the sample from the cached data if needed
        sample = self.data_cache_handler.get_sample(self, idx)

        if (
            self.data_args.data_caching_args.use_datadings
            and DataKeysEnum.IMAGE in sample
        ):
            if self.data_args.data_caching_args.cache_encoded_images:
                image = cv2.imdecode(
                    np.fromstring(sample[DataKeysEnum.IMAGE], dtype="uint8"),
                    cv2.IMREAD_COLOR,
                )
            else:
                image = sample[DataKeysEnum.IMAGE]
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = torch.tensor(image)

            # permute channels to mimic torch tensors
            if len(image.shape) == 3:
                image = image.permute(2, 0, 1)

            # add a channel to image if not present
            if len(image.shape) == 2:
                image = torch.unsqueeze(image, 0)
            sample[DataKeysEnum.IMAGE] = image

        if (
            self.data_args.data_tokenization_args.tokenize_dataset
            and self.data_args.data_tokenization_args.tokenize_per_sample
        ):
            sample = self._tokenize_sample(sample)

        # perform transformations on data if required
        if self.transforms is not None:
            for t in self.transforms:
                sample = t(sample)

        return {**sample, DataKeysEnum.INDEX: idx}

    def _save_to_cache(self):
        """
        Saves the data to cache according to required caching functionality
        """
        return self.data_cache_handler.save_to_cache(self)

    def _load_from_cache(self):
        """
        Loads the data from cache according to required caching functionality
        """
        return self.data_cache_handler.load_from_cache()

    def _after_load_from_cache(self):
        """
        Hook for performing operations after loading data from cache
        """
        pass
