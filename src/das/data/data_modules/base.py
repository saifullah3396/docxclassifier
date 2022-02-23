"""
Defines the abstract base class for handling the functionality for Lightning
DataModules.
"""

import copy
import sys
from abc import ABC, abstractmethod
from typing import Optional

import pytorch_lightning as pl
import torch
from das.data.data_args import DataArguments
from das.data.samplers import TrainValSamplerFactory
from das.utils.basic_args import BasicArguments
from das.utils.basic_utils import create_logger
from torch.utils.data import (
    BatchSampler,
    DataLoader,
    DistributedSampler,
    RandomSampler,
    Subset,
)
from torch.utils.data.sampler import SequentialSampler

from ..utils.group_batch_sampler import GroupedBatchSampler, create_aspect_ratio_groups

# setup logging
logger = create_logger(__name__)


class BaseDataModule(pl.LightningDataModule, ABC):
    """
    The base data module class for loading data, adding transforms, and creating
    dataset splits.

    Args:
        basic_args: The global basic arguments.
        data_args: The data related arguments
        collate_fns: Dictionary of data collation functions for train/val/test sets.
        transforms: Dictionary of data transformation functions for train/val/test sets.
    """

    @property
    def data_transforms(self):
        transforms = {}
        default_transforms = self.default_transforms()
        for k in ["train", "test", "val"]:
            transforms[k] = (
                self.transforms[k] if k in self.transforms else default_transforms[k]
            )
        return transforms

    def __init__(
        self,
        basic_args: BasicArguments,
        data_args: DataArguments,
        collate_fns: dict = {
            "train": None,
            "val": None,
            "test": None,
        },
        transforms: dict = {},
    ):

        super().__init__()

        # initialize the arguments
        self.data_args = data_args
        self.basic_args = basic_args

        # correct the collate_fns if not correctly passed by the user
        if collate_fns is None:
            self.collate_fns: dict = {
                "train": None,
                "val": None,
                "test": None,
            }
        else:
            self.collate_fns = collate_fns

        # initialize the data transforms
        self.transforms = transforms

        # if training is required, initialize the sampler for training/validation sets
        if self.basic_args.do_train:
            # initialize the sampler according to the configuration
            self._train_val_sampler = TrainValSamplerFactory.create_sampler(
                data_args, basic_args
            )

            # if training/validation split sampler is not defined and the dataset class
            # has no predefined validation split, remove the validation dataloader from
            # the datamodule
            if (
                self._train_val_sampler is None
                and "val" not in self.dataset_class.supported_splits
            ):
                delattr(BaseDataModule, "val_dataloader")

            if "test" not in self.dataset_class.supported_splits:
                delattr(BaseDataModule, "test_dataloader")

    @property
    def data_transforms(self):
        """
        The data transforms for each split.
        """

        transforms = {}
        if self.data_args.train_aug_args is None:
            transforms["train"] = None
        else:
            transforms["train"] = [t.create() for t in self.data_args.train_aug_args]

        if self.data_args.eval_aug_args is None:
            transforms["test"] = None
            transforms["val"] = None
        else:
            transforms["val"] = [t.create() for t in self.data_args.eval_aug_args]
            transforms["test"] = [t.create() for t in self.data_args.eval_aug_args]

        return transforms

    @property
    @abstractmethod
    def dataset_class(self):
        """The underlying dataset class"""

    def load_dataset(
        self,
        data_args: DataArguments,
        split: str = "train",
        transforms=None,
        download: bool = False,
        use_cached: bool = False,
        save_to_cache: bool = True,
        prepare_only: bool = False,
        train_val_indices: list = [],
    ):
        """
        Loads the dataset based on its name and performs validation of some arguments.

        Args:
            data_args: The data related arguments
            split: Train, val or test split to load
            transforms: Data transformations to be used on the dataset
            download: Whether to download the data if posisble
            use_cached: Whether to load data from cached directory
            save_to_cache: Whether to save data to cache directory
        """
        try:
            # check if dataset_class property is correctly defined
            if self.dataset_class is None:
                raise ValueError(
                    "Please set the class variable [dataset_class] in the respective "
                    "child datamodule."
                )

            # check if dataset directory is provided
            if data_args.dataset_dir is None:
                # if dataset is not downloadable, raise an error
                if not self.dataset_class.is_downloadable:
                    raise ValueError(
                        "Please provide dataset directory with the --dataset_dir "
                        "argument as this dataset is not downloadable."
                    )

            # initialize the underlying dataset class
            dataset = self.dataset_class(
                data_args,
                split,
                transforms=transforms,
                download=download,
                use_cached=use_cached,
                save_to_cache=save_to_cache,
                prepare_only=prepare_only,
                train_val_indices=train_val_indices,
            )

            # load the dataset
            dataset.load()

            return dataset
        except Exception as exc:
            logger.exception(
                f"Exception raised while loading the dataset "
                f"[{data_args.dataset_name}]: {exc}"
            )
            sys.exit(1)

    def prepare_data(self):
        """
        Runs only on a single GPU/TPU to prepare data and save it in cache dir.
        """

        logger.info("Preparing / preprocesing dataset and saving to cache...")

        if self.basic_args.do_train:
            self.load_train_val_datasets(prepare_only=True)

        if self.basic_args.do_test:
            self.load_test_dataset(prepare_only=True)

    def setup(self, stage: Optional[str] = None):
        """
        Runs on every GPU/TPU in a distributed system. Loads data from the cached dir.
        """

        logger.info(f"Training stage == {stage}")

        # Assign train/val datasets for use in dataloaders using the train/val sampler
        if stage == "fit" or stage is None:
            if not self.basic_args.do_train:
                raise ValueError(
                    "Cannot run training stage with basic_args.do_train=False."
                )

            self.train_dataset, self.val_dataset = self.load_train_val_datasets()

            # if max_train_samples is set get the given number of examples
            # from the dataset
            if self.data_args.data_loader_args.max_train_samples is not None:
                self.train_dataset = Subset(
                    self.train_dataset,
                    range(0, self.data_args.data_loader_args.max_train_samples),
                )

            # if max_val_samples is set get the given number of examples
            # from the dataset
            if self.data_args.data_loader_args.max_val_samples is not None:
                self.val_dataset = Subset(
                    self.val_dataset,
                    range(0, self.data_args.data_loader_args.max_val_samples),
                )

            logger.info(f"Training set size = {len(self.train_dataset)}")
            if self.val_dataset is not None:
                logger.info(f"Validation set size = {len(self.val_dataset)}")

        # Assign test dataset for use in dataloader(s)
        if stage == "test" or stage is None:
            self.test_dataset = self.load_test_dataset()

            if self.test_dataset is not None:
                # if max_test_samples is set get the given number of examples
                # from the dataset
                if self.data_args.data_loader_args.max_test_samples is not None:
                    self.test_dataset = Subset(
                        self.test_dataset,
                        range(0, self.data_args.data_loader_args.max_test_samples),
                    )
                if self.test_dataset is not None:
                    logger.info(f"Test set size = {len(self.test_dataset)}")

    def load_train_val_datasets(self, prepare_only=False):
        """
        Loads or generates the train/validation datasets either from original split
        or from the train-val sampling strategy provided.

        Args:
            prepare_only: Only checks if data is cached if true
        """

        if prepare_only:
            self.load_dataset(
                self.data_args,
                split="train",
                download=True,
                use_cached=True,
                prepare_only=True,
                transforms=self.data_transforms["train"],
            )
            if "val" in self.dataset_class.supported_splits:
                self.load_dataset(
                    self.data_args,
                    split="val",
                    download=True,
                    use_cached=True,
                    prepare_only=True,
                    transforms=self.data_transforms["val"],
                )
        else:
            # sample the dataset according to training/validation sampling strategy
            logger.info("Setting up train/validation dataset...")
            train_dataset = self.load_dataset(
                self.data_args,
                split="train",
                download=True,
                use_cached=True,
                transforms=self.data_transforms["train"],
            )

            if "val" in self.dataset_class.supported_splits:
                val_dataset = self.load_dataset(
                    self.data_args,
                    split="val",
                    download=True,
                    use_cached=True,
                    transforms=self.data_transforms["val"],
                )
                return train_dataset, val_dataset
            elif self._train_val_sampler is not None:
                train_subset, val_subset = next(self._train_val_sampler(train_dataset))

                # generate new train dataset based on indices
                train_dataset.train_val_indices = train_subset.indices

                # generate val dataset from train dataset
                val_dataset = copy.deepcopy(train_dataset)
                val_dataset.train_val_indices = val_subset.indices
                val_dataset.transforms = self.data_transforms["val"]
                return train_dataset, val_dataset
            else:
                return train_dataset, None

    def load_test_dataset(self, prepare_only=False):
        """
        Loads the test dataset.

        Args:
            prepare_only: Only checks if data is cached if true.
        """

        if "test" in self.dataset_class.supported_splits:
            data_transforms = self.data_transforms
            return self.load_dataset(
                self.data_args,
                split="test",
                download=True,
                use_cached=True,
                prepare_only=prepare_only,
                transforms=data_transforms["test"],
            )
        else:
            return None

    def train_dataloader(self):
        """
        Defines the torch dataloader for train dataset.
        """

        if self.basic_args.distributed_accelerator is not None:
            sampler = DistributedSampler(
                self.train_dataset, shuffle=self.data_args.data_loader_args.shuffle_data
            )
        else:
            sampler = RandomSampler(self.train_dataset)

        if self.data_args.data_loader_args.aspect_ratio_grouping_factor >= 0:
            group_ids = create_aspect_ratio_groups(
                self.train_dataset,
                k=self.data_args.data_loader_args.aspect_ratio_grouping_factor,
            )
            batch_sampler = GroupedBatchSampler(
                sampler,
                group_ids,
                self.data_args.data_loader_args.per_device_train_batch_size,
            )
        else:
            batch_sampler = BatchSampler(
                sampler,
                self.data_args.data_loader_args.per_device_train_batch_size,
                drop_last=self.data_args.data_loader_args.dataloader_drop_last,
            )

        return DataLoader(
            self.train_dataset,
            batch_sampler=batch_sampler,
            collate_fn=self.collate_fns["train"],
            num_workers=self.data_args.data_loader_args.dataloader_num_workers,
            pin_memory=self.data_args.data_loader_args.pin_memory,
        )

    def val_dataloader(self):
        """
        Defines the torch dataloader for validation dataset.
        """
        if (
            self.basic_args.distributed_accelerator is not None
            and self.basic_args.n_gpu > 1
            and self.data_args.data_loader_args.dist_eval
        ):
            num_devices = torch.distributed.get_world_size()
            if len(self.val_dataset) % num_devices != 0:
                logger.warning(
                    "Enabling distributed evaluation with an eval dataset not divisible by process number. "
                    "This will slightly alter validation results as extra duplicate entries are added to achieve "
                    "equal num of samples per-process."
                )
            sampler = DistributedSampler(self.val_dataset, shuffle=False)
        else:
            sampler = SequentialSampler(self.val_dataset)

        if self.data_args.data_loader_args.aspect_ratio_grouping_factor >= 0:
            group_ids = create_aspect_ratio_groups(
                self.val_dataset,
                k=self.data_args.data_loader_args.aspect_ratio_grouping_factor,
            )
            batch_sampler = GroupedBatchSampler(
                sampler,
                group_ids,
                self.data_args.data_loader_args.per_device_eval_batch_size,
            )
        else:
            batch_sampler = BatchSampler(
                sampler,
                self.data_args.data_loader_args.per_device_eval_batch_size,
                drop_last=self.data_args.data_loader_args.dataloader_drop_last,
            )

        return DataLoader(
            self.val_dataset,
            batch_sampler=batch_sampler,
            collate_fn=self.collate_fns["val"],
            num_workers=self.data_args.data_loader_args.dataloader_num_workers,
            pin_memory=self.data_args.data_loader_args.pin_memory,
        )

    def test_dataloader(self):
        """
        Defines the torch dataloader for test dataset.
        """
        if (
            self.basic_args.distributed_accelerator is not None
            and self.basic_args.n_gpu > 1
            and self.data_args.data_loader_args.dist_eval
        ):
            num_devices = torch.distributed.get_world_size()
            if len(self.test_dataset) % num_devices != 0:
                logger.warning(
                    "Enabling distributed evaluation with an eval dataset not divisible by process number. "
                    "This will slightly alter validation results as extra duplicate entries are added to achieve "
                    "equal num of samples per-process."
                )
            sampler = DistributedSampler(self.test_dataset, shuffle=False)
        else:
            sampler = SequentialSampler(self.test_dataset)

        if self.data_args.data_loader_args.aspect_ratio_grouping_factor >= 0:
            group_ids = create_aspect_ratio_groups(
                self.test_dataset,
                k=self.data_args.data_loader_args.aspect_ratio_grouping_factor,
            )
            batch_sampler = GroupedBatchSampler(
                sampler,
                group_ids,
                self.data_args.data_loader_args.per_device_eval_batch_size,
            )
        else:
            batch_sampler = BatchSampler(
                sampler,
                self.data_args.data_loader_args.per_device_eval_batch_size,
                drop_last=self.data_args.data_loader_args.dataloader_drop_last,
            )

        return DataLoader(
            self.test_dataset,
            batch_sampler=batch_sampler,
            collate_fn=self.collate_fns["test"],
            num_workers=self.data_args.data_loader_args.dataloader_num_workers,
            pin_memory=self.data_args.data_loader_args.pin_memory,
        )
