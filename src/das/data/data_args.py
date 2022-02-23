"""
Defines the dataclass for holding data related arguments.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Union

from das.data.augmentations.factory import DataAugmentationArguments

from .utils.common import TrainValSamplingStrategy


@dataclass
class DataLoaderArguments:
    per_device_train_batch_size: int = field(
        default=8, metadata={"help": "Batch size per GPU/TPU core/CPU for training."}
    )
    per_device_eval_batch_size: int = field(
        default=8, metadata={"help": "Batch size per GPU/TPU core/CPU for evaluation."}
    )
    dataloader_drop_last: bool = field(
        default=False,
        metadata={
            "help": (
                "Drop the last incomplete batch if it is not divisible by the "
                "batch size."
            )
        },
    )
    shuffle_data: bool = field(
        default=True, metadata={"help": ("Whether to shuffle the data on load or not.")}
    )
    pin_memory: bool = field(
        default=False, metadata={"help": ("Whether to load data into pinned memory")}
    )
    dataloader_num_workers: int = field(
        default=0,
        metadata={
            "help": (
                "Number of subprocesses to use for data loading (PyTorch only). 0 "
                "means that the data will be loaded in the main process."
            )
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number "
            "of training examples to this value if set."
        },
    )
    max_val_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number "
            "of validation examples to this value if set."
        },
    )
    max_test_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number "
            "of test examples to this value if set."
        },
    )
    aspect_ratio_grouping_factor: int = field(
        default=-1,
        metadata={"help": ("The size of the groups for aspect ratio grouping.")},
    )
    dist_eval: bool = field(
        default=True,
        metadata={"help": "Whether to perform evaluation with distributed sampler."},
    )


@dataclass
class DataTokenizationArguments:
    tokenize_dataset: Optional[bool] = field(
        default=False,
        metadata={
            "help": "Whether to use tokenized version of the dataset for NLP datasets."
        },
    )
    tokenizer_name: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "The name of the dataset tokenizer to use. The tokenizer is named "
                "according to the tokenizers available in huggingface library."
            )
        },
    )
    tokenizer_lib: Optional[str] = field(
        default="huggingface",
        metadata={
            "help": ("The name of the tokenizer library, torchtext or huggingface")
        },
    )
    pad_to_max_length: bool = field(
        default=True,
        metadata={
            "help": "Whether to pad all samples to model maximum sentence length. "
            "If False, will pad the samples dynamically when batching to the maximum "
            "length in the batch. More efficient on GPU but very bad for TPU."
        },
    )
    seq_max_length: int = field(default=512, metadata={"help": "Max sequence length."})
    max_seqs_per_sample: int = field(
        default=5, metadata={"help": "Max number of seqeunces per sample."}
    )
    compute_word_to_toke_maps: bool = field(
        default=False, metadata={"help": "Whether to compute word to token mapping."}
    )
    overflow_samples_combined: bool = field(
        default=False,
        metadata={
            "help": "Whether to combine overflowing tokens into one sample or to make "
            "multiple separate samples for those."
        },
    )
    tokenize_per_sample: Optional[bool] = field(
        default=False,
        metadata={
            "help": "Whether to tokenize the sample on the go while loading the samples."
        },
    )
    fetch_local_files: Optional[bool] = field(
        default=False,
        metadata={
            "help": "Bypasses calling request from huggingface for downloading stuff."
        },
    )


@dataclass
class DataCachingArguments:
    use_datadings: Optional[bool] = field(
        default=False,
        metadata={"help": ("Whether to use datadings to store the data into cache.")},
    )
    cache_resized_images: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to cache images after resizing them to small size."},
    )
    cache_grayscale_images: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to cache images in grayscale."},
    )
    cache_encoded_images: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to cache encoded images."},
    )
    cache_image_size: Optional[Union[list]] = field(
        default_factory=lambda: [224, 224],
        metadata={"help": ("Image size to rescale to if caching is wanted.")},
    )
    load_data_to_ram: Optional[bool] = field(
        default=False,
        metadata={
            "help": ("Whether to load data into RAM or read directly from datadings.")
        },
    )
    cached_data_name: Optional[str] = field(
        default=None, metadata={"help": ("The name of the cached_data file.")}
    )
    dataset_cache_dir: Optional[str] = field(
        default=None, metadata={"help": "Where do you want to store the dataset files."}
    )
    workers: Optional[int] = field(
        default=4, metadata={"help": "Number of workers for caching."}
    )


@dataclass
class DataSplittingArguments:
    train_val_sampling_strategy: Optional[str] = field(
        default=None,
        metadata={
            "help": "The data splitting strategy to use.",
            "choices": [e.value for e in TrainValSamplingStrategy],
        },
    )
    random_split_ratio: float = field(
        default=0.8, metadata={"help": "The train/validation dataset split ratio."}
    )
    k_folds: int = field(
        default=5,
        metadata={
            "help": (
                "The number of K-folds to use if using kfold cross validation "
                "data sampling strategy"
            )
        },
    )


# @dataclass
# class DataAugmentationArguments:
#     train_image_rescale_strategy: Optional[ImageRescaleStrategy] = field(
#         default=None,
#         metadata={
#             "help": ("The image rescaling splitting strategy to use for train set.")
#         },
#     )
#     eval_image_rescale_strategy: Optional[ImageRescaleStrategy] = field(
#         default=None,
#         metadata={
#             "help": ("The image rescaling splitting strategy to use for train set.")
#         },
#     )
#     normalize_dataset: Optional[bool] = field(
#         default=False,
#         metadata={
#             "help": "Whether to normalize the dataset with given dataset mean and std."
#         },
#     )
#     convert_grayscale_to_rgb: Optional[bool] = field(
#         default=False,
#         metadata={
#             "help": "Whether to convert gray scale images to RGB 3-channel images"
#         },
#     )
#     convert_rgb_to_bgr: Optional[bool] = field(
#         default=False, metadata={"help": "Whether to convert RGB image to BGR image."}
#     )
#     dataset_mean: Optional[Union[list, dict]] = field(
#         default_factory=lambda: [0.485, 0.456, 0.406],
#         metadata={"help": ("Dataset mean")},
#     )
#     dataset_std: Optional[Union[list, dict]] = field(
#         default_factory=lambda: [0.229, 0.224, 0.225],
#         metadata={"help": ("Possible andom scale dims for shorter dims.")},
#     )


@dataclass
class DataArguments:
    """
    Arguments pertaining to what data we are going to input our model for training
    and eval.
    """

    cls_name = "data_args"

    dataset_name: str = field(
        default=None,
        metadata={"help": "The name of the dataset to use (via the datasets library)."},
    )
    dataset_dir: Optional[str] = field(
        default=None, metadata={"help": "The path to the directory of the dataset."}
    )
    data_caching_args: DataCachingArguments = field(
        default=DataCachingArguments(),
        metadata={"help": ("Arguments related to data caching.")},
    )
    data_splitting_args: Optional[DataSplittingArguments] = field(
        default=None,
        metadata={"help": ("Arguments related to data test/val splitting")},
    )
    train_aug_args: Optional[List[DataAugmentationArguments]] = field(
        default=None,
        metadata={
            "help": (
                "Arguments related to defining default data augmentations for training data."
            )
        },
    )
    eval_aug_args: Optional[List[DataAugmentationArguments]] = field(
        default=None,
        metadata={
            "help": (
                "Arguments related to defining default data augmentations for validation data."
            )
        },
    )
    data_loader_args: DataLoaderArguments = field(
        default=DataLoaderArguments(),
        metadata={
            "help": (
                "Arguments related to data loading or specifically torch dataloaders."
            )
        },
    )
    data_tokenization_args: DataTokenizationArguments = field(
        default=DataTokenizationArguments(),
        metadata={"help": ("Arguments related to data tokenization.")},
    )
    use_dataset_normalization_params: Optional[bool] = field(
        default=False,
        metadata={
            "help": "Whether to compute normalization params from the dataset while "
            "loading such as image mean / std."
        },
    )
    extras: Optional[dict] = field(
        default=None,
        metadata={
            "help": ("Any additional argument required specifically for the dataset.")
        },
    )

    def __post_init__(self):
        if (
            self.data_tokenization_args.tokenize_dataset is True
            and self.data_tokenization_args.tokenizer_name is None
        ):
            raise ValueError(
                "Please provide tokenizer name if dataset is to be tokenized."
            )
        if (
            self.data_splitting_args is not None
            and self.data_splitting_args.train_val_sampling_strategy is not None
        ):
            self.data_splitting_args.train_val_sampling_strategy = (
                TrainValSamplingStrategy(
                    self.data_splitting_args.train_val_sampling_strategy
                )
            )
