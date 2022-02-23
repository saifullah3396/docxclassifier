"""
The main script that serves as the entry-point for all kinds of training experiments.
"""

import argparse
import dataclasses
import gc
import logging
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import pytorch_lightning as pl
import torch
from das.data.data_args import DataArguments
from das.data.data_modules.factory import DataModuleFactory
from das.models.model_args import ModelArguments, ModelFactory
from das.utils.arg_parser import DASArgumentParser
from das.utils.basic_args import BasicArguments
from das.utils.basic_utils import configure_logger, create_logger
from matplotlib import pyplot as plt
from pytorch_lightning.callbacks import TQDMProgressBar

# from pytorch_lightning.profiler.advanced import AdvancedProfiler

# setup logging
logger = create_logger(__name__)

# define dataclasses to parse arguments from
ARG_DATA_CLASSES = [BasicArguments, DataArguments, ModelArguments]

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
        print(args.cfg)
        if args.cfg.endswith(".json"):
            return args.cfg, arg_parser.parse_json_file(os.path.abspath(args.cfg))
        elif args.cfg.endswith(".yaml"):
            return args.cfg, arg_parser.parse_yaml_file(
                os.path.abspath(args.cfg), unknown
            )
    else:
        return args.cfg, arg_parser.parse_args_into_dataclasses()


def empty_cache():
    torch.cuda.empty_cache()
    gc.collect()


def main():
    """
    Initializes the training of a model given dataset, and their configurations.
    """

    # empty cuda cache
    empty_cache()

    # parse arguments
    cfg_file, (basic_args, data_args, model_args) = parse_args()

    # configure pytorch-lightning logger
    pl_logger = logging.getLogger("pytorch_lightning")
    configure_logger(pl_logger)

    # intialize torch random seed
    torch.manual_seed(basic_args.seed)

    # get model class
    model_class = ModelFactory.get_model_class(
        model_args.model_name, model_args.model_task
    )

    # # get data collator required for the model
    collate_fns = model_class.get_data_collators(data_args, None)

    # initialize data-handling module
    datamodule = DataModuleFactory.create_datamodule(
        basic_args, data_args, collate_fns=collate_fns
    )

    # prepare data for usage later on model
    datamodule.prepare_data()

    # if model checkpoint is present, use it to load the weights
    if model_args.model_checkpoint_file is None:
        # intialize the model for training
        model = model_class(
            basic_args,
            model_args,
            training_args=None,
            data_args=data_args,
            datamodule=datamodule,
        )
    else:
        if not model_args.model_checkpoint_file.startswith("http"):
            model_checkpoint = Path(model_args.model_checkpoint_file)
            if not model_checkpoint.exists():
                logger.error(
                    f"Checkpoint not found, cannot load weights from {model_checkpoint}."
                )
                sys.exit(1)
        else:
            model_checkpoint = model_args.model_checkpoint_file
        logger.info(f"Loading model from model checkpoint: {model_checkpoint}")

        # load model weights from checkpoint
        model = model_class.load_from_checkpoint(
            model_checkpoint,
            strict=True,
            basic_args=basic_args,
            model_args=model_args,
            training_args=None,
            data_args=data_args,
            datamodule=datamodule,
        )

    # setup data
    datamodule.setup(stage="test")
    test_dataset = datamodule.test_dataset

    # set model to cuda
    model.cuda()

    # generate attention maps for 10 first samples
    for i in range(10):
        image = test_dataset[i]["image"]
        output = model(image=image.unsqueeze(0).to("cuda"))
        all_attentions = [output.attn]
        _attentions = [att.detach().cpu().numpy() for att in all_attentions]
        attentions_mat = np.asarray(_attentions)[:, 0]

        # normalize attention maps
        res_att_mat = attentions_mat.sum(axis=1) / attentions_mat.shape[1]
        res_att_mat = res_att_mat + np.eye(res_att_mat.shape[1])[None, ...]
        res_att_mat = res_att_mat / res_att_mat.sum(axis=-1)[..., None]

        # get attention map of last layer
        v = res_att_mat[-1]
        grid_size = int(np.sqrt(res_att_mat.shape[-1]))
        mask = v[0, 1:].reshape(grid_size, grid_size)
        im_np = image.cpu().numpy()
        mask = cv2.resize(mask / mask.max(), im_np.shape[1:], cv2.INTER_NEAREST)[
            ..., np.newaxis
        ]

        imm = cv2.imread(test_dataset[i]["image_file_path"])
        imm = cv2.resize(imm, (384, 384))
        fig, ax = plt.subplots()
        ax.imshow(imm, alpha=1, cmap="gray")
        im = ax.imshow(mask, alpha=0.5, cmap="jet")
        plt.axis("off")

        label = test_dataset[i]
        if not os.path.exists(f"./attns/"):
            os.mkdir(f"./attns/")
        plt.savefig(f"./attns/{i}.png")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.exception(f"Exception found while training/testing the model: {e}")
