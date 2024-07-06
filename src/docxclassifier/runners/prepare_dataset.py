from __future__ import annotations

import logging
from typing import Type

import hydra
import ignite.distributed as idist
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from torchfusion.core.args.args import FusionArguments
from torchfusion.core.constants import DataKeys
from torchfusion.core.data.data_augmentations.advanced import ImagePreprocess
from torchfusion.core.data.data_augmentations.general import DictTransform
from torchfusion.core.data.factory.batch_sampler import BatchSamplerFactory
from torchfusion.core.data.utilities.containers import TransformsDict
from torchfusion.core.data.utilities.loaders import load_datamodule_from_args
from torchfusion.core.training.utilities.constants import TrainingStage
from torchfusion.core.training.utilities.general import (
    initialize_torch,
    print_tf_from_loader,
    setup_logging,
)
from torchfusion.core.utilities.dataclasses.dacite_wrapper import from_dict
from torchfusion.core.utilities.logging import get_logger
from torchvision.transforms import Compose

from docxclassifier import *  # noqa

logger = get_logger()


def setup_custom_preprocess_transforms():
    # create custom preprocessing transforms here if needed. These are applied during data caching for datasets
    # huggingface-based that are based on FusionDataset.
    # for torch-based datasets, these are ignored as no caching is applied in that case.
    image_preprocess_transform = Compose(
        [
            ImagePreprocess(
                square_pad=False,
                rescale_size=(256, 256),
                encode_image=True,
                encode_format="JPEG",
            )
        ]
    )

    transforms = TransformsDict()
    transforms.train = Compose(
        [
            DictTransform(
                key=DataKeys.IMAGE,  # apply this transform only to the image key from the output of the dataset
                transform=image_preprocess_transform,
            )
        ]
    )
    transforms.validation = Compose(
        [
            DictTransform(
                key=DataKeys.IMAGE,  # apply this transform only to the image key from the output of the dataset
                transform=image_preprocess_transform,
            )
        ]
    )
    transforms.test = Compose(
        [
            DictTransform(
                key=DataKeys.IMAGE,  # apply this transform only to the image key from the output of the dataset
                transform=image_preprocess_transform,
            )
        ]
    )

    return transforms


def setup_custom_realtime_transforms():
    # create custom transforms here if needed. If not provided, transforms defined in the configuration file will be used
    from torchvision.transforms import (
        Compose,
        RandomCrop,
        RandomHorizontalFlip,
        ToTensor,
    )

    transforms = TransformsDict()
    transforms.train = Compose(
        [
            DictTransform(
                key=DataKeys.IMAGE,
                transform=Compose(
                    [
                        RandomHorizontalFlip(),
                        RandomCrop(224, padding=4),
                        ToTensor(),
                    ]
                ),
            ),
        ]
    )
    transforms.validation = Compose(
        [
            DictTransform(
                key=DataKeys.IMAGE,
                transform=Compose([ToTensor()]),
            ),
        ]
    )
    transforms.test = Compose(
        [
            DictTransform(
                key=DataKeys.IMAGE,
                transform=Compose([ToTensor()]),
            ),
        ]
    )

    return transforms


def prepare_datasets(
    args: FusionArguments, hydra_config: OmegaConf, runtime_config: DictConfig
):
    # initialize training
    initialize_torch(
        args,
        seed=args.general_args.seed,
        deterministic=args.general_args.deterministic,
    )

    # initialize logging directory and tensorboard logger
    _, _ = setup_logging(
        output_dir=hydra_config.runtime.output_dir,
        setup_tb_logger=False,
    )

    # for custom preprocess, we also change the cache file name otherwise default cache file will be used which
    # will not have the custom preprocess applied
    # args.data_args.cache_file_name = "custom_preprocess_cache_file_256x256"
    # custom_preprocess_transforms = setup_custom_preprocess_transforms()
    # custom_realtime_transforms = setup_custom_realtime_transforms()

    # setup datamodule
    datamodule = load_datamodule_from_args(
        args,
        stage=None,
        # preprocess_transforms=custom_preprocess_transforms,
        # realtime_transforms=custom_realtime_transforms,
    )

    # setup batch sampler if needed
    batch_sampler_wrapper = BatchSamplerFactory.create(
        args.data_loader_args.train_batch_sampler.name,
        **args.data_loader_args.train_batch_sampler.kwargs,
    )

    # setup custom data collators if required
    # collator = BatchToTensorDataCollator(
    #     # for example
    #     data_key_type_map={DataKeys.IMAGE: torch.float32, DataKeys.LABEL: torch.long}
    # )
    # datamodule._collate_fns = CollateFnDict(
    #     train=collator, validation=collator, test=collator
    # )

    # setup dataloaders
    train_dataloader = datamodule.train_dataloader(
        args.data_loader_args.per_device_train_batch_size,
        dataloader_num_workers=args.data_loader_args.dataloader_num_workers,
        pin_memory=args.data_loader_args.pin_memory,
        shuffle_data=args.data_loader_args.shuffle_data,
        dataloader_drop_last=args.data_loader_args.dataloader_drop_last,
        batch_sampler_wrapper=batch_sampler_wrapper,
    )
    val_dataloader = datamodule.val_dataloader(
        args.data_loader_args.per_device_eval_batch_size,
        dataloader_num_workers=args.data_loader_args.dataloader_num_workers,
        pin_memory=args.data_loader_args.pin_memory,
    )
    test_dataloader = datamodule.test_dataloader(
        args.data_loader_args.per_device_eval_batch_size,
        dataloader_num_workers=args.data_loader_args.dataloader_num_workers,
        pin_memory=args.data_loader_args.pin_memory,
    )

    # print transforms before training run just for sanity check
    logging.debug("Final sanity check about transforms...")
    print_tf_from_loader(
        train_dataloader, stage=TrainingStage.train, log_level=logging.DEBUG
    )
    print_tf_from_loader(
        val_dataloader, stage=TrainingStage.validation, log_level=logging.DEBUG
    )
    print_tf_from_loader(
        test_dataloader, stage=TrainingStage.test, log_level=logging.DEBUG
    )

    if runtime_config.visualize:
        # show training batch
        logger.info("Showing training batch...")
        for batch in train_dataloader:
            datamodule.show_batch(batch)
            break

        # show validation batch
        logger.info("Showing validation batch...")
        for batch in val_dataloader:
            datamodule.show_batch(batch)
            break

        # show testing batch
        logger.info("Showing testing batch...")
        for batch in test_dataloader:
            datamodule.show_batch(batch)
            break


def main(
    cfg: DictConfig,
    hydra_config: DictConfig,
    data_class: Type[FusionArguments] = FusionArguments,
):
    # initialize general configuration for script, any additional arguments may be direclty passed by hydra
    # and they will be available in the cfg object
    runtime_cfg = cfg
    cfg = OmegaConf.to_object(cfg)
    args = from_dict(data_class=data_class, data=cfg["args"])

    # setup logging
    logger = get_logger(hydra_config=hydra_config)
    logger.info("Starting torchfusion testing script with arguments:")
    logger.info(args)

    try:
        return prepare_datasets(args, hydra_config, runtime_cfg)
    except Exception as e:
        logging.exception(e)
    finally:
        return None


@hydra.main(version_base=None, config_name="hydra")
def app(cfg: DictConfig) -> None:
    # get hydra config
    hydra_config = HydraConfig.get()

    # train and evaluate the model
    main(cfg, hydra_config)

    # wait for all processes to complete before exiting
    idist.barrier()


if __name__ == "__main__":
    app()
