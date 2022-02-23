"""
Defines the DataHandler class that manages the loading/preprocessing of all the
supported datasets.
"""

import typing
from abc import ABC, abstractmethod

import torch
from das.data.data_args import DataArguments
from das.utils.basic_args import BasicArguments
from das.utils.basic_utils import create_logger
from sklearn.model_selection import KFold
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import random_split

from .utils.common import TrainValSamplingStrategy


class TrainValSampler(ABC):
    """
    Base class for loading different types of training/validation sampling strategies.

    Args:
        data_args: Dataset arguments with validation strategy related parameters set
            to valid values.
    """

    _logger = create_logger("TrainValSampler")

    def __init__(self, data_args, basic_args) -> None:
        self._load_params(data_args, basic_args)

    @abstractmethod
    def _load_params(self, data_args, basic_args):
        pass

    @abstractmethod
    def __call__(self, dataset) -> typing.Tuple[torch.Tensor, torch.Tensor]:
        pass


class RandomSplitSampler(TrainValSampler):
    """
    The splitting strategy for random split based on size ratio.

    Args:
        data_args: Dataset arguments with:
            - 'random_split_ratio' set to valid value from 0.0 to 1.0.
    """

    def __init__(self, data_args, basic_args) -> None:
        super().__init__(data_args, basic_args)

    def _load_params(self, data_args, basic_args):
        self._seed = basic_args.seed
        self._random_split_ratio = data_args.data_splitting_args.random_split_ratio

    def __call__(self, train_dataset) -> typing.Tuple[DataLoader, DataLoader]:
        """
        Takes the training set as input and returns split train / validation
        sets
        """
        train_dataset_size = len(train_dataset)
        val_dataset_size = \
            int(train_dataset_size * round(1. - self._random_split_ratio, 2))
        train_set, val_set = random_split(
            train_dataset,
            [train_dataset_size - val_dataset_size, val_dataset_size],
            generator=torch.Generator().manual_seed(self._seed))
        yield train_set, val_set


class KFoldCrossValSampler(TrainValSampler):
    """
    The splitting strategy for based on k-fold cross validation technique.

    Args:
        data_args: Dataset arguments with:
            - 'k_folds' set to valid value.
    """

    def __init__(self, data_args, basic_args) -> None:
        super().__init__(data_args, basic_args)

    def _load_params(self, data_args, basic_args):
        self._k_fold_sampler = KFold(
            n_splits=data_args.data_splitting_args.k_folds, shuffle=True)

    def __call__(self, train_dataset) -> typing.Tuple[torch.Tensor, torch.Tensor]:
        """
        Takes the training set as input and returns split train / validation
        sets
        """
        for k_fold, (train_ids, val_ids) in enumerate(
            self._k_fold_sampler.split(train_dataset)
        ):
            yield train_dataset[train_ids], train_dataset[val_ids]


SUPPORTED_SAMPLERS = {
    TrainValSamplingStrategy.RANDOM_SPLIT: RandomSplitSampler,
    TrainValSamplingStrategy.K_FOLD_CROSS_VAL: KFoldCrossValSampler,
}


class TrainValSamplerFactory:
    @staticmethod
    def create_sampler(
            data_args: DataArguments, basic_args: BasicArguments):
        if data_args.data_splitting_args is not None:
            sampler = SUPPORTED_SAMPLERS.get(
                data_args.data_splitting_args.train_val_sampling_strategy, None)
            if sampler is not None:
                return sampler(data_args, basic_args)
        return None
