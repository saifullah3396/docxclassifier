"""
The main script that serves as the entry-point for all kinds of training experiments.
"""

import argparse
import dataclasses
import gc
import logging
import os
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms.functional as F
import tqdm
from das.data.augmentations.factory import AugmentatorArguments
from das.data.data_args import DataArguments
from das.data.data_modules.factory import DataModuleFactory
from das.utils.arg_parser import DASArgumentParser
from das.utils.basic_args import BasicArguments
from das.utils.basic_utils import configure_logger, create_logger
from joblib import Parallel, delayed
from PIL import Image

# setup logging
logger = create_logger(__name__)

# define dataclasses to parse arguments from
ARG_DATA_CLASSES = [BasicArguments, DataArguments, AugmentatorArguments]

# torch hub bug fix https://github.com/pytorch/vision/issues/4156
torch.hub._validate_not_a_forked_repo = lambda a, b, c: True


def parse_args():
    """
    Parses script arguments.
    """

    arg_parser_main = argparse.ArgumentParser()
    arg_parser_main.add_argument("--cfg", required=False)

    # get config file path
    args, unknown = arg_parser_main.parse_known_args()

    # initialize the argument parsers
    arg_parser = DASArgumentParser(ARG_DATA_CLASSES)

    # parse arguments either based on a json file or directly
    if args.cfg is not None:
        if args.cfg.endswith(".json"):
            return args.cfg, arg_parser.parse_json_file(os.path.abspath(args.cfg))
        elif args.cfg.endswith(".yaml"):
            return args.cfg, arg_parser.parse_yaml_file(
                os.path.abspath(args.cfg), unknown
            )
    else:
        return args.cfg, arg_parser.parse_args_into_dataclasses()


def print_args(title, args):
    """
    Pretty prints the arguments.
    """
    args_message = f"\n{title}:\n"
    for (k, v) in dataclasses.asdict(args).items():
        args_message += f"\t{k}: {v}\n"
    print(args_message)


def empty_cache():
    torch.cuda.empty_cache()
    gc.collect()


def augment_sample(idx, dataset, aug_args, output_aug_dir):
    sample = dataset[idx]
    output_image_path = (
        output_aug_dir / "images" / sample["image_file_path"].split("images/")[1]
    )
    if not output_image_path.parent.exists():
        output_image_path.parent.mkdir(parents=True)
    image_output = str(output_image_path)[:-4] + "/" + str(output_image_path.name)
    image_orig = sample["image"]
    image_orig = (image_orig.squeeze() / 255.0).numpy()
    if len(image_orig.shape) == 2:
        image_orig = cv2.cvtColor(image_orig, cv2.COLOR_GRAY2RGB)
    if not Path(image_output).parent.exists():
        Path(image_output).parent.mkdir(parents=True)

    image_output_path = str(image_output)[:-4] + ".jpg"
    save_image = False
    if Path(image_output_path).exists():
        try:
            Image.open(image_output_path)
        except Exception as e:
            logger.exception(
                "Exception raised while loading image {image_output_path}."
            )
            save_image = True
    else:
        save_image = True

    if save_image:
        plt.imsave(image_output_path, image_orig, vmin=0, vmax=1.0)

    image = resize_image_to_1000(sample["image"])
    image = (image.squeeze() / 255.0).numpy()

    for aug in aug_args.augmentations:
        for severity in [1, 2, 3, 4, 5]:
            output_augmented_image_dir = Path(str(output_image_path)[:-4]) / str(
                aug.name.value
            )
            if not output_augmented_image_dir.exists():
                output_augmented_image_dir.mkdir(parents=True)

            output_augmented_image_path = Path(
                str(output_augmented_image_dir / str(severity)) + ".jpg"
            )

            if output_augmented_image_path.exists():
                try:
                    Image.open(output_augmented_image_path)
                    continue
                except Exception as e:
                    logger.exception(
                        f"Exception raised while loading image {image_output_path}."
                    )
            augmented_image = aug(image, severity=severity)
            if augmented_image is not None:
                if len(augmented_image.shape) == 2:
                    augmented_image = cv2.cvtColor(
                        np.clip(augmented_image, 0, 1).astype("float32"),
                        cv2.COLOR_GRAY2RGB,
                    )
                plt.imsave(output_augmented_image_path, augmented_image, vmin=0, vmax=1)


def resize_image_to_1000(image):
    larger_dim = 1 if image.shape[1] > image.shape[2] else 2
    smaller_dim = 1 if image.shape[1] < image.shape[2] else 2
    new_shape = list(image.shape)
    new_shape[larger_dim] = 1000
    new_shape[smaller_dim] = int(
        image.shape[smaller_dim] / image.shape[larger_dim] * 1000
    )
    return F.resize(image, new_shape[1:])


def main():
    """
    Initializes the training of a model given dataset, and their configurations.
    """

    # empty cuda cache
    empty_cache()

    # parse arguments
    cfg_file, (basic_args, data_args, aug_args) = parse_args()

    # print arguments for verbosity
    logger.info("Initializing augmentation with the following arguments:")
    print_args("Basic arguments", basic_args)
    print_args("Dataset arguments", data_args)
    print_args("Augmentation arguments", aug_args)

    # configure pytorch-lightning logger
    pl_logger = logging.getLogger("pytorch_lightning")
    configure_logger(pl_logger)

    # intialize torch random seed
    torch.manual_seed(basic_args.seed)

    datamodule = DataModuleFactory.create_datamodule(basic_args, data_args)

    # prepare data for usage later on model
    datamodule.prepare_data()
    datamodule.setup()

    # load the data
    output_aug_dir = Path(aug_args.output_aug_dir)
    if not output_aug_dir.exists():
        output_aug_dir.mkdir()

    datasets = []
    for dataset_split in aug_args.datasets:
        if dataset_split == "train":
            datasets.append(datamodule.train_dataset)
        if dataset_split == "test":
            datasets.append(datamodule.test_dataset)
    for dataset in datasets:
        if not aug_args.debug:
            Parallel(n_jobs=aug_args.n_parallel_jobs)(
                delayed(augment_sample)(idx, dataset, aug_args, output_aug_dir)
                for idx in tqdm.tqdm(range(len(dataset)))
            )
        else:
            for idx in tqdm.tqdm(range(len(dataset))):
                sample = dataset[idx]
                image = resize_image_to_1000(sample["image"])
                image = (image.squeeze() / 255.0).numpy()
                for aug in aug_args.augmentations:
                    # print("Augmentation: ", aug.name.value)
                    aug_images = []
                    for severity in [1, 2, 3, 4, 5]:
                        aug_images.append(aug(image, severity=severity))

                    if len(image.shape) == 2:
                        show_image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

                    for idx, aug_image in enumerate(aug_images):
                        if len(aug_image.shape) == 2:
                            aug_images[idx] = cv2.cvtColor(
                                aug_image.astype("float32"), cv2.COLOR_GRAY2RGB
                            )

                    fig, axs = plt.subplots(2, 4)
                    axs[0, 0].imshow(show_image)
                    if aug_images[0] is not None:
                        axs[0, 1].imshow(aug_images[0], vmin=0, vmax=1)
                    if aug_images[1] is not None:
                        axs[0, 2].imshow(aug_images[1], vmin=0, vmax=1)
                    if aug_images[2] is not None:
                        axs[0, 3].imshow(aug_images[2], vmin=0, vmax=1)
                    if aug_images[3] is not None:
                        axs[1, 0].imshow(aug_images[3], vmin=0, vmax=1)
                    if aug_images[4] is not None:
                        axs[1, 1].imshow(aug_images[4], vmin=0, vmax=1)
                    plt.show()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.exception(f"Exception found while training/testing the model: {e}")
