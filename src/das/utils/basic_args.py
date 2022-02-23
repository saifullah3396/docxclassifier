"""
Defines the dataclass for holding training related arguments.
"""

import os
from dataclasses import dataclass, field
from typing import Optional

from das.utils.basic_utils import create_logger

logger = create_logger(__name__)


@dataclass
class BasicArguments:
    """
    General arguments
    """
    cls_name = 'basic_args'

    output_dir: str = field(
        metadata={
            "help": (
                "The output directory where the model predictions and checkpoints "
                "will be written.")})
    do_train: bool = field(
        default=False,
        metadata={
            "help": "Whether to run training."})
    do_eval: bool = field(
        default=False,
        metadata={
            "help": "Whether to run eval on the dev set."})
    do_test: bool = field(
        default=False,
        metadata={
            "help": "Whether to run predictions on the test set."})
    debug_data: bool = field(
        default=False,
        metadata={
            "help": "Whether to just debug data."})
    overwrite_output_dir: bool = field(
        default=False,
        metadata={
            "help": (
                "Overwrite the content of the output directory. Use this to "
                "continue training if output_dir points to a checkpoint directory.")
        })
    seed: int = field(
        default=42,
        metadata={
            "help": "Random seed that will be set at the beginning of training."})
    distributed_accelerator: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "The accelerator to use when performing distributed training.")
        })
    n_gpu: int = field(
        default=0,
        metadata={
            "help": (
                "The number of gpus per node required for training.")
        })
    n_nodes: int = field(
        default=1,
        metadata={
            "help": (
                "The number of nodes required for training.")
        })

    def __post_init__(self):
        # make directories if not already available
        if self.output_dir is not None:
            self.output_dir = os.path.expanduser(self.output_dir)
