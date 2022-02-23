"""
Defines the dataclass for holding model related configuration.
"""

import os
import sys
from dataclasses import dataclass, field
from importlib import import_module
from typing import List, Optional, Union

from das.utils.basic_utils import create_logger
from numpy.core.fromnumeric import std

logger = create_logger(__name__)


MODELS_PATH = os.path.dirname(os.path.realpath(__file__))
SUPPORTED_MODEL_ARGUMENTS = {}
SUPPORTED_MODEL_TASKS = {}
for dir in os.listdir(MODELS_PATH):
    abs_path = MODELS_PATH + "/" + dir
    if os.path.isdir(abs_path) and dir not in ["__pycache__", "common"]:
        model = f"das.models.{dir}.model"
        model_args = f"das.models.{dir}.model_args"
        SUPPORTED_MODEL_ARGUMENTS[dir] = lambda model_args=model_args: import_module(
            model_args
        ).ChildModelArguments
        SUPPORTED_MODEL_TASKS[dir] = lambda model=model: import_module(
            model
        ).SUPPORTED_TASKS


@dataclass
class ModelArguments:
    """
    Configuration arguments pertaining to which model/config/tokenizer we are going to
    train.
    """

    cls_name = "model_args"

    model_name: str = field(
        default=None,
        metadata={"help": "Which model do you want to run the training script on."},
    )
    model_type: str = field(
        default="",
        metadata={"help": "Model subtype"},
    )
    model_task: str = field(
        default=None,
        metadata={"help": "The task of the model."},
    )
    model_version: str = field(
        default=None,
        metadata={"help": "Model training version."},
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models."},
    )
    model_checkpoint_file: Optional[str] = field(
        default=None,
        metadata={"help": "Checkpoint file name to load the model weights from."},
    )
    finetune_checkpoint_file: Optional[str] = field(
        default=None,
        metadata={
            "help": "Checkpoint file name to load the model weights from for finetuning."
        },
    )
    grad_batch_size: int = field(
        default=8,
        metadata={
            "help": "Batch size to use for computing gradients. This is required for shap."
        },
    )

    def __post_init__(self):
        try:
            if self.model_name is None:
                raise ValueError("The child arguments must initialize the model name!")
        except ValueError as exc:
            logger.error(exc)
            sys.exit(1)

        if self.model_version is None:
            self.model_version = "v0.1"


class ModelArgumentsFactory:
    @staticmethod
    def create_model_arguments(model_name: str):
        """
        Returns the model arguments class if present

        Args:
            model_name: The model name for which the configuration arguments are to
                be returned.
        """
        try:
            model_args_class = SUPPORTED_MODEL_ARGUMENTS.get(model_name, None)
            if model_args_class is None:
                raise ValueError(f"Model {model_name} is not supported!")
            else:
                model_args_class = model_args_class()
            return model_args_class
        except Exception as exc:
            logger.exception(
                f"Exception raised while loading model arguments "
                f"[{model_name}]: {exc}"
            )
            sys.exit(1)


class TasksFactory:
    @staticmethod
    def get_model_class(supported_tasks: str, model_task: str):
        """
        Returns the model class if present

        Args:
            model_name: The model name
            model_task: The model task
        """
        try:
            model_class = supported_tasks.get(model_task, None)
            if model_class is None:
                raise ValueError(f"Model task {model_task} is not supported!")
            return model_class
        except Exception as exc:
            logger.exception(
                f"Exception raised while loading model arguments "
                f"[{model_task}]: {exc}"
            )
            sys.exit(1)


class ModelFactory:
    @staticmethod
    def get_model_class(model_name: str, model_task: str):
        """
        Returns the model class if present

        Args:
            model_name: The model name
            model_task: The model task
        """
        try:
            supported_tasks = SUPPORTED_MODEL_TASKS.get(model_name, None)
            if supported_tasks is None:
                raise ValueError(f"Model {model_name} is not supported!")
            else:
                supported_tasks = supported_tasks()  # apply lambda
                model_class = TasksFactory.get_model_class(supported_tasks, model_task)
            return model_class
        except Exception as exc:
            logger.exception(
                f"Exception raised while loading model arguments "
                f"[{model_name}-{model_task}]: {exc}"
            )
            sys.exit(1)
