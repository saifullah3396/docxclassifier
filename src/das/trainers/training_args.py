"""
Defines the dataclass for holding training related arguments.
"""

import json
import logging
import math
import os
from dataclasses import asdict, dataclass, field
from datetime import timedelta
from enum import Enum
from pathlib import Path
from tokenize import group
from typing import Any, Dict, List, Mapping, Optional, TypedDict, Union

import torch
from das.models.common.optimizer import LARS
from das.utils.basic_utils import ExplicitEnum, create_logger
from das.utils.lr_schedulers import CosineScheduler, PolynomialLRDecay
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.lr_monitor import LearningRateMonitor
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from transformers.file_utils import cached_property

logger = create_logger(__name__)

log_levels = {
    "debug": logging.DEBUG,
    "info": logging.INFO,
    "warning": logging.WARNING,
    "error": logging.ERROR,
    "critical": logging.CRITICAL,
}
trainer_log_levels = dict(**log_levels, passive=-1)


def default_logdir() -> str:
    """
    Same default as PyTorch
    """
    import socket
    from datetime import datetime

    current_time = datetime.now().strftime("%b%d_%H-%M-%S")
    return os.path.join("runs", current_time + "_" + socket.gethostname())


@dataclass
class ModelCheckpointConfig:
    dirpath: Optional[Union[str, Path]] = None
    filename: Optional[str] = None
    monitor: Optional[str] = None
    verbose: bool = False
    save_last: Optional[bool] = None
    save_top_k: int = 1
    save_weights_only: bool = False
    mode: str = "min"
    auto_insert_metric_name: bool = True
    every_n_train_steps: Optional[int] = None
    train_time_interval: Optional[timedelta] = None
    every_n_epochs: Optional[int] = None
    save_on_train_epoch_end: Optional[bool] = None

    def create(self):
        return ModelCheckpoint(
            dirpath=self.dirpath,
            filename=self.filename,
            monitor=self.monitor,
            verbose=self.verbose,
            save_last=self.save_last,
            save_top_k=self.save_top_k,
            save_weights_only=self.save_weights_only,
            mode=self.mode,
            auto_insert_metric_name=self.auto_insert_metric_name,
            every_n_train_steps=self.every_n_train_steps,
            train_time_interval=self.train_time_interval,
            every_n_epochs=self.every_n_epochs,
            save_on_train_epoch_end=self.save_on_train_epoch_end,
        )


class CustomLearningRateMonitor(LearningRateMonitor):
    @staticmethod
    def _should_log(trainer) -> bool:
        return (trainer.global_step + 1) % 5 == 0 or trainer.should_stop


@dataclass
class LearningRateMonitorConfig:
    logging_interval: Optional[str] = None
    log_momentum: bool = False

    def create(self):
        return CustomLearningRateMonitor(
            logging_interval=self.logging_interval, log_momentum=self.log_momentum
        )


@dataclass
class EarlyStoppingConfig:
    monitor: Optional[str] = None
    min_delta: float = 0.0
    patience: int = 3
    verbose: bool = False
    mode: str = "min"
    strict: bool = True
    check_finite: bool = True
    stopping_threshold: Optional[float] = None
    divergence_threshold: Optional[float] = None
    check_on_train_epoch_end: bool = True

    def create(self):
        return EarlyStopping(
            monitor=self.monitor,
            min_delta=self.min_delta,
            patience=self.patience,
            verbose=self.verbose,
            mode=self.mode,
            strict=self.strict,
            check_finite=self.check_finite,
            stopping_threshold=self.stopping_threshold,
            divergence_threshold=self.divergence_threshold,
            check_on_train_epoch_end=self.check_on_train_epoch_end,
        )


class OptimizerEnum(str, Enum):
    ADAM_W = "adam_w"
    SGD = "sgd"
    LARS = "lars"


class LRSchedulerEnum(str, Enum):
    LAMBDA_LR = "lambda_lr"
    STEP_LR = "step_lr"
    REDUCE_LR_ON_PLATEAU = "reduce_lr_on_plateau"
    COSINE_ANNEALING_LR = "cosine_annealing_lr"
    CYCLIC_LR = "cyclic_lr"
    POLYNOMIAL_DECAY_LR = "poly_decay_lr"


class WDSchedulerEnum(str, Enum):
    COSINE = "cosine"


@dataclass
class Optimizer:
    name: str
    params: List[dict]

    def __post_init__(self):
        self.name = OptimizerEnum(self.name)

    def create(self, model_param_groups, lr_scale=1):
        for params_list in self.params:
            params_list["lr"] = params_list["lr"] * lr_scale

        model_parameters = []
        for params_list in self.params:
            group_name = params_list["group"]
            if group_name in model_param_groups.keys():
                model_parameters.append({})
                model_parameters[-1]["params"] = model_param_groups[group_name]
                for k, v in params_list.items():
                    model_parameters[-1][k] = v

        base_params = self.params[0]
        base_params.pop("group")

        opt = None
        if self.name == OptimizerEnum.ADAM_W:
            opt = torch.optim.AdamW(model_parameters, **base_params)
        elif self.name == OptimizerEnum.SGD:
            opt = torch.optim.SGD(model_parameters, **base_params)
        elif self.name == OptimizerEnum.LARS:
            if "lars_exclude" in base_params:
                base_params.pop("lars_exclude")
            opt = LARS(model_parameters, **base_params)
        if opt is None:
            raise ValueError(f"Optimizer {self.name} is not supported!")
        return opt


@dataclass
class LRScheduler:
    name: str
    params: dict
    restarts: bool = False
    interval: str = "step"
    frequency: int = 1

    def __post_init__(self):
        self.name = LRSchedulerEnum(self.name)

    def create(
        self, optimizer, num_training_steps=None, num_warmup_steps=None, num_epochs=None
    ):
        res = {}
        res["interval"] = self.interval
        res["frequency"] = self.frequency

        if self.name == LRSchedulerEnum.STEP_LR:
            res["scheduler"] = torch.optim.lr_scheduler.StepLR(optimizer, **self.params)
        elif self.name == LRSchedulerEnum.REDUCE_LR_ON_PLATEAU:
            res["scheduler"] = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, **self.params
            )
        elif self.name == LRSchedulerEnum.LAMBDA_LR:
            type = self.params["type"]
            lambda_fn = None
            if type == "linear":

                def lr_lambda(current_step: int):
                    if current_step < num_warmup_steps:
                        return float(current_step) / float(max(1, num_warmup_steps))
                    return max(
                        0.0,
                        float(num_training_steps - current_step)
                        / float(max(1, num_training_steps - num_warmup_steps)),
                    )

                lambda_fn = lr_lambda

            res["scheduler"] = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda_fn)
        elif self.name == LRSchedulerEnum.COSINE_ANNEALING_LR:
            if not self.restarts:
                res["scheduler"] = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, num_training_steps, **self.params
                )
            else:
                res["scheduler"] = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, num_training_steps // num_epochs, **self.params
                )
        elif self.name == LRSchedulerEnum.CYCLIC_LR:
            res["scheduler"] = torch.optim.lr_scheduler.CyclicLR(
                optimizer, **self.params
            )
        elif self.name == LRSchedulerEnum.POLYNOMIAL_DECAY_LR:
            max_decay_steps = self.params.pop("max_decay_steps")
            max_decay_steps = (
                num_training_steps if max_decay_steps == -1 else max_decay_steps
            )
            res["scheduler"] = PolynomialLRDecay(
                optimizer, max_decay_steps=max_decay_steps, **self.params
            )
        return res


@dataclass
class WDScheduler:
    name: str
    params: List[dict]

    def __post_init__(self):
        self.name = WDSchedulerEnum(self.name)

    def create(self, optimizer, num_epochs=None, steps_per_epoch=None):
        group_keys = []
        initial_wds = []
        for group in optimizer.param_groups:
            group_keys.append(group["group"])
            initial_wds.append(group["weight_decay"])

        wd_schedulers = [None] * len(group_keys)
        for params_list in self.params:
            group_name = params_list["group"]
            if group_name in group_keys:
                idx = group_keys.index(group_name)
                if self.name == WDSchedulerEnum.COSINE:
                    wd_schedulers[idx] = CosineScheduler(
                        initial_wds[idx],
                        params_list["wd_end"],
                        num_epochs,
                        steps_per_epoch,
                        warmup_epochs=params_list["warmup_epochs"],
                    )
        return wd_schedulers


@dataclass
class TrainingArguments:
    """
    Arguments related to the training loop.
    """

    cls_name = "training_args"

    num_sanity_val_steps: int = field(
        default=0, metadata={"help": "Number of validation sanity check steps"}
    )
    gradient_accumulation_steps: int = field(
        default=1,
        metadata={
            "help": (
                "Number of updates steps to accumulate before performing a "
                "backward/update pass."
            )
        },
    )
    enable_grad_clipping: bool = field(
        default=False, metadata={"help": "Whether to use gradient clipping or not."}
    )
    max_grad_norm: float = field(default=1.0, metadata={"help": "Max gradient norm."})
    min_steps: Optional[int] = field(
        default=None, metadata={"help": "Min number of training steps to perform."}
    )
    max_steps: Optional[int] = field(
        default=None, metadata={"help": "Total number of training steps to perform."}
    )
    min_epochs: Optional[float] = field(
        default=None, metadata={"help": "Min number of training epochs to perform."}
    )
    max_epochs: Optional[float] = field(
        default=None, metadata={"help": "Max number of training epochs to perform."}
    )
    warmup_ratio: float = field(
        default=0.0,
        metadata={"help": "Linear warmup over warmup_ratio fraction of total steps."},
    )
    warmup_steps: int = field(
        default=0, metadata={"help": "Linear warmup over warmup_steps."}
    )
    logging_steps: int = field(
        default=500, metadata={"help": "Log every X updates steps."}
    )
    precision: int = field(
        default=32, metadata={"help": "The precision to use 16, 32, 64"}
    )
    eval_every_n_epochs: Optional[int] = field(
        default=1, metadata={"help": "Run an evaluation every X steps."}
    )
    eval_training: Optional[bool] = field(
        default=False, metadata={"help": "Evaluate metrics for training data."}
    )
    model_checkpoint_config: Optional[ModelCheckpointConfig] = field(
        default=ModelCheckpointConfig(),
        metadata={
            "help": "ModelCheckpoint configuration that defines the saving strategy "
            "of model checkpoints."
        },
    )
    resume_from_checkpoint: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to resume from latest checkpoint"},
    )
    resume_checkpoint_file: Optional[str] = field(
        default=None,
        metadata={
            "help": "Checkpoint file name to load the model weights for resuming training."
        },
    )
    test_checkpoint_file: Optional[str] = field(
        default=None,
        metadata={"help": "Checkpoint file name to load the model weights from."},
    )
    lr_monitor_config: Optional[LearningRateMonitorConfig] = field(
        default=LearningRateMonitorConfig(),
        metadata={
            "help": "LearningRateMonitorconfiguration that defines the saving strategy "
            "of learning rate schedulers."
        },
    )
    early_stopping_config: Optional[EarlyStoppingConfig] = field(
        default=None,
        metadata={
            "help": "EarlyStoppingConfig that defines the strategy to stop the training"
            "early if the specified metric is not improving anymore"
        },
    )
    sync_batchnorm: bool = field(
        default=True,
        metadata={"help": "Whether to synchronize batches accross multiple GPUs."},
    )
    optimizers: Mapping[str, Optimizer] = field(
        default_factory=lambda: {
            "default": Optimizer(
                OptimizerEnum.ADAM_W,
                params=[
                    {
                        "group": "default",
                        "lr": 2e-5,
                        "betas": (0.9, 0.999),
                        "eps": 1e-8,
                        "weight_decay": 1e-2,
                    }
                ],
            )
        },
        metadata={"help": "Training optimizer to use."},
    )
    lr_schedulers: Optional[Mapping[str, LRScheduler]] = field(
        default_factory=lambda: {
            "default": LRScheduler(
                name=LRSchedulerEnum.LAMBDA_LR,
                params=[{"type": "linear"}],
            ),
        },
        metadata={"help": "Learning rate scheduler."},
    )
    wd_schedulers: Optional[Mapping[str, WDScheduler]] = field(
        default=None,
        metadata={"help": "Weight decay scheduler."},
    )
    base_lr_batch_size: int = field(
        default=32,
        metadata={"help": "The base batch size for learning rate scaling."},
    )
    two_step_finetuning_strategy: bool = field(
        default=False, metadata={"help": "Whether to use two step finetuning strategy"}
    )
    finetune_switch_epoch: int = 5

    def __post_init__(self):
        if self.warmup_ratio < 0 or self.warmup_ratio > 1:
            raise ValueError("warmup_ratio must lie in range [0,1]")
        elif self.warmup_ratio > 0 and self.warmup_steps > 0:
            logger.info(
                "Both warmup_ratio and warmup_steps given, warmup_steps will override"
                " any effect of warmup_ratio during training"
            )

    def __str__(self):
        self_as_dict = asdict(self)

        attrs_as_str = [f"{k}={v},\n" for k, v in sorted(self_as_dict.items())]
        return f"{self.__class__.__name__}(\n{''.join(attrs_as_str)})"

    __repr__ = __str__

    @property
    def train_batch_size(self) -> int:
        """
        The actual batch size for training (may differ from
        per_device_train_batch_size in distributed training).
        """
        per_device_batch_size = self.per_device_train_batch_size
        train_batch_size = per_device_batch_size * max(1, self.n_gpu)
        return train_batch_size

    @property
    def eval_batch_size(self) -> int:
        """
        The actual batch size for evaluation (may differ from
        per_device_eval_batch_size in distributed training).
        """
        per_device_batch_size = self.per_device_eval_batch_size
        eval_batch_size = per_device_batch_size * max(1, self.n_gpu)
        return eval_batch_size

    def get_warmup_steps(self, num_training_steps: int):
        """
        Get number of steps used for a linear warmup.
        """
        warmup_steps = (
            self.warmup_steps
            if self.warmup_steps > 0
            else math.ceil(num_training_steps * self.warmup_ratio)
        )
        return warmup_steps

    def to_dict(self):
        """
        Serializes this instance while replace `Enum` by their values (for JSON
        serialization support).
        """
        d = asdict(self)
        for k, v in d.items():
            if isinstance(v, Enum):
                d[k] = v.value
            if isinstance(v, list) and len(v) > 0 and isinstance(v[0], Enum):
                d[k] = [x.value for x in v]
        return d

    def to_json_string(self):
        """
        Serializes this instance to a JSON string.
        """
        return json.dumps(self.to_dict(), indent=2)

    def to_sanitized_dict(self) -> Dict[str, Any]:
        """
        Sanitized serialization to use with TensorBoard’s hparams
        """
        d = self.to_dict()
        d = {
            **d,
            **{
                "train_batch_size": self.train_batch_size,
                "eval_batch_size": self.eval_batch_size,
            },
        }

        valid_types = [bool, int, float, str]
        valid_types.append(torch.Tensor)

        return {k: v if type(v) in valid_types else str(v) for k, v in d.items()}
