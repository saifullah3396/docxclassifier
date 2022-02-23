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

import pytorch_lightning as pl
import torch
from das.data.data_args import DataArguments
from das.data.data_modules.factory import DataModuleFactory
from das.models.model_args import ModelArguments, ModelFactory
from das.utils.arg_parser import DASArgumentParser
from das.utils.basic_args import BasicArguments
from das.utils.basic_utils import configure_logger, create_logger
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


def debug_data(datamodule):
    datamodule.prepare_data()
    datamodule.setup()

    dl = datamodule.train_dataloader()
    logger.info(f"Loading train data of size: {len(dl)}")
    for data in datamodule.train_dataloader():
        print(f"\t\t\tData:\n{data}")
        for k, v in data.items():
            print(k, v, v.shape)
        break

    dl = datamodule.val_dataloader()
    logger.info(f"Loading val data of size: {len(dl)}")
    for data in datamodule.val_dataloader():
        print(f"\t\t\tData:\n{data}")
        break

    dl = datamodule.test_dataloader()
    logger.info(f"Loading test data of size: {len(dl)}")
    for data in datamodule.test_dataloader():
        print(f"\t\t\tData:\n{data}")
        break
    exit(1)


class MeterlessProgressBar(TQDMProgressBar):
    def init_train_tqdm(self):
        bar = super().init_train_tqdm()
        bar.dynamic_ncols = False
        bar.ncols = 0
        bar.leave = True
        return bar

    def init_validation_tqdm(self):
        bar = super().init_validation_tqdm()
        bar.dynamic_ncols = False
        bar.ncols = 0
        bar.leave = True
        return bar


def main():
    """
    Initializes the training of a model given dataset, and their configurations.
    """

    # empty cuda cache
    empty_cache()

    # parse arguments
    cfg_file, (basic_args, data_args, model_args) = parse_args()

    # print arguments for verbosity
    logger.info("Initializing the training script with the following arguments:")
    print_args("Basic arguments", basic_args)
    print_args("Dataset arguments", data_args)
    print_args("Model arguments", model_args)

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

    # test dataset for debugging purposes...
    if basic_args.debug_data:
        debug_data(datamodule)

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

    # generate a tensorboard logger
    output_dir = basic_args.output_dir + model_args.model_task + "/"
    if hasattr(model_args, "model_type"):
        version = (
            model_args.model_name
            + model_args.model_type
            + "/"
            + model_args.model_version
        )
    else:
        version = model_args.model_name + "/" + model_args.model_version

    trainer = pl.Trainer(
        gpus=basic_args.n_gpu,
        num_nodes=basic_args.n_nodes,
        default_root_dir=output_dir,
    )
    trainer.test(model, datamodule=datamodule)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.exception(f"Exception found while training/testing the model: {e}")
