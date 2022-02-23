""" PyTorch lightning module that defines the base model for training/testing etc. """


import sys
from abc import ABC, abstractmethod
from pathlib import Path
from typing import IO, Any, Callable, Dict, Optional, Union

import pytorch_lightning as pl
import timm
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import torchmetrics
from das.data.data_args import DataArguments
from das.data.datasets.utils import DataKeysEnum
from das.models.common.data_collators import BatchToTensorDataCollator
from das.models.common.model_outputs import ClassificationModelOutput
from das.models.common.utils import load_model
from das.models.model_args import ModelArguments
from das.trainers.training_args import TrainingArguments
from das.utils.basic_args import BasicArguments
from das.utils.basic_utils import create_logger
from das.utils.lr_schedulers import PolynomialLRDecay
from das.utils.metrics import TrueLabelConfidence
from pytorch_lightning.core.datamodule import LightningDataModule
from pytorch_lightning.utilities.cloud_io import load as pl_load
from pytorch_lightning.utilities.warnings import rank_zero_warn
from seqeval.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
from torch import nn

logger = create_logger(__name__)


class BaseModel(pl.LightningModule):
    @classmethod
    def load_from_checkpoint(
        cls,
        checkpoint_path: Union[str, IO],
        map_location: Optional[
            Union[Dict[str, str], str, torch.device, int, Callable]
        ] = None,
        hparams_file: Optional[str] = None,
        strict: bool = True,
        **kwargs,
    ):
        if map_location is not None:
            checkpoint = pl_load(checkpoint_path, map_location=map_location)
        else:
            checkpoint = pl_load(
                checkpoint_path, map_location=lambda storage, loc: storage
            )

        model = cls._load_model_state(checkpoint, strict=strict, **kwargs)
        return model

    @classmethod
    def _load_model_state(
        cls, checkpoint: Dict[str, Any], strict: bool = True, **cls_kwargs
    ):

        model = cls(**cls_kwargs)

        # give model a chance to load something
        model.on_load_checkpoint(checkpoint)

        # load the state_dict on the model automatically
        keys = model.load_checkpoint(checkpoint, strict=strict)

        if not strict:
            if keys.missing_keys:
                rank_zero_warn(
                    f"Found keys that are in the model state dict but not in the checkpoint: {keys.missing_keys}"
                )
            if keys.unexpected_keys:
                rank_zero_warn(
                    f"Found keys that are not in the model state dict but in the checkpoint: {keys.unexpected_keys}"
                )

        return model

    @property
    def model_name(self):
        if hasattr(self.model_args, "model_type"):
            return f"{self.model_args.model_name}{self.model_args.model_type}"
        else:
            return self.model_args.model_name

    @property
    def steps_per_epoch(self) -> int:
        """Total training steps inferred from datamodule and devices."""
        if self.trainer.max_steps > 0:
            return self.trainer.max_steps

        limit_batches = self.trainer.limit_train_batches
        batches = len(self.datamodule.train_dataloader())
        batches = (
            min(batches, limit_batches)
            if isinstance(limit_batches, int)
            else int(limit_batches * batches)
        )
        effective_accum = self.trainer.accumulate_grad_batches
        return batches // effective_accum

    @property
    def num_training_steps(self) -> int:
        """Total training steps inferred from datamodule and devices."""
        return self.steps_per_epoch * self.trainer.max_epochs

    def __init__(
        self,
        basic_args: BasicArguments,
        model_args: ModelArguments,
        training_args: Optional[TrainingArguments] = None,
        data_args: Optional[DataArguments] = None,
        datamodule: Optional[LightningDataModule] = None,
    ):
        super().__init__()

        # initialize model arguments
        self.basic_args = basic_args
        self.model_args = model_args
        self.training_args = training_args
        self.data_args = data_args
        self.datamodule = datamodule

        # initialize metrics
        self.train_metrics = None
        self.val_metrics = None
        self.test_metrics = None

        # define model state dict default key
        self.checkpoint_state_dict_key = "state_dict"

        # build model
        self.build_model()
        self.train_metrics, self.val_metrics, self.test_metrics = self.init_metrics()

    def init_metrics(self):
        return None, None, None

    def setup(self, stage: Optional[str]) -> None:
        super().setup(stage=stage)

        if stage == "fit":
            if self.training_args is None:
                raise ValueError(
                    "Training arguments must be passed to model if training is required!"
                )

            # initialize manual optimization
            self.automatic_optimization = False
            self.accumulated_loss = 0

            # gradient accumulation related parameters
            self.grad_accum_in_process = False
            self.grad_accum_steps = self.training_args.gradient_accumulation_steps

            # get the total number of batches
            self.total_batches = len(self.datamodule.train_dataloader())

            # setup warmup steps
            self.warmup_steps = self.training_args.get_warmup_steps(
                self.num_training_steps
            )

    @abstractmethod
    def build_model(self):
        pass

    def load_checkpoint(self, checkpoint, strict):
        return self.load_state_dict(
            checkpoint[self.checkpoint_state_dict_key], strict=strict
        )

    def finetune_freeze(self):
        pass

    def finetune_unfreeze(self):
        pass

    def evaluate(
        self,
        output,
        batch,
        stage=None,
        metrics=None,
        on_step=True,
        on_epoch=False,
        prog_bar=True,
    ):
        return {}

    def evaluate_epoch(
        self,
        outputs,
        stage=None,
        metrics=None,
        on_step=False,
        on_epoch=True,
        prog_bar=True,
    ):
        pass

    def training_step(self, batch, batch_idx):
        if self.training_args.two_step_finetuning_strategy:
            if self.current_epoch > self.training_args.finetune_switch_epoch:
                if not self.finetune_unfroze:
                    self.finetune_unfreeze()
                    self.restart_schedulers(force=True)
                    self.finetune_unfroze = True

        optimizers = self.optimizers()
        if not isinstance(optimizers, list):
            optimizers = [optimizers]

        # get the model output
        output = self(**batch, stage="fit")

        # update the gradients for the same loss
        if not self.grad_accum_in_process:
            # if this is the last accum batch we check if our batch size is less then
            # the accumulation steps
            if self.total_batches - batch_idx < self.grad_accum_steps:
                self.grad_accum_steps = self.total_batches - batch_idx

        # current loss
        if isinstance(output.loss, dict):
            loss = output.loss["final"] / self.grad_accum_steps
        else:
            loss = output.loss / self.grad_accum_steps

        # store this loss for logging
        self.accumulated_loss += loss.item()

        # propagate the loss backwards
        self.manual_backward(loss)

        self.grad_accum_in_process = True

        # log the training metrics
        eval_results = None
        if self.training_args.eval_training:
            eval_results = self.evaluate(
                output,
                batch=batch,
                stage="fit",
                metrics=self.train_metrics,
                on_step=True,
                on_epoch=True,
            )

        # if gradient_accumulation_steps
        if (
            (batch_idx + 1) % self.training_args.gradient_accumulation_steps == 0
            or batch_idx + 1 == self.total_batches
        ):

            # perform gradient clipping
            if self.training_args.enable_grad_clipping:
                torch.nn.utils.clip_grad_norm_(
                    self.parameters(), self.training_args.max_grad_norm
                )

            # set warmup initially
            if self.trainer.global_step == 0 and self.warmup_steps > 0:
                lr_scale = (
                    min(1.0, float(self.trainer.global_step + 1) / self.warmup_steps)
                    if self.warmup_steps > 0
                    else 1
                )
                for idx, opt in enumerate(optimizers):
                    for pg_idx, pg in enumerate(opt.param_groups):
                        pg["lr"] = lr_scale * self.base_optimizer_lrs[idx][pg_idx]

            # handle weight decay
            if len(self.wd_schedulers) > 0:
                for idx, opt in enumerate(optimizers):
                    opt_wd_schs = self.wd_schedulers[idx]
                    for pg_idx, pg in enumerate(opt.param_groups):
                        if opt_wd_schs[pg_idx] is not None:
                            pg["weight_decay"] = opt_wd_schs[pg_idx].step()

            for opt in optimizers:
                opt.step()
                opt.zero_grad()

            lr_schedulers = self.lr_schedulers()
            if not isinstance(lr_schedulers, list):
                lr_schedulers = [lr_schedulers]

            for sch in lr_schedulers:
                if isinstance(sch, torch.optim.lr_scheduler.LambdaLR):
                    sch.step()
                elif self.trainer.global_step + 1 > self.warmup_steps:
                    if isinstance(sch, torch.optim.lr_scheduler.CosineAnnealingLR):
                        sch.step()
                    elif isinstance(sch, torch.optim.lr_scheduler.CyclicLR):
                        sch.step()
                    if isinstance(sch, PolynomialLRDecay):
                        sch.step()

            if self.trainer.global_step + 1 <= self.warmup_steps:
                lr_scale = (
                    min(1.0, float(self.trainer.global_step + 1) / self.warmup_steps)
                    if self.warmup_steps > 0
                    else 1
                )
                for idx, opt in enumerate(optimizers):
                    for pg_idx, pg in enumerate(opt.param_groups):
                        pg["lr"] = lr_scale * self.base_optimizer_lrs[idx][pg_idx]

            # log the training step loss
            self.log(
                f"fl",
                self.accumulated_loss,
                on_step=True,
                on_epoch=True,
                prog_bar=True,
                sync_dist=True,
                reduce_fx="mean",
            )

            # zero out logged loss
            self.grad_accum_in_process = False
            self.accumulated_loss = 0.0

        if eval_results is None:
            return {"loss": loss}
        else:
            return {"loss": loss, **eval_results}

    def training_epoch_end(self, outputs):
        lr_schedulers = self.lr_schedulers()
        if not isinstance(lr_schedulers, list):
            lr_schedulers = [lr_schedulers]

        # If the selected scheduler is a ReduceLROnPlateau scheduler.
        if self.trainer.global_step + 1 > self.warmup_steps:
            for sch in lr_schedulers:
                if isinstance(sch, torch.optim.lr_scheduler.StepLR):
                    sch.step()

        # restart schedulers
        self.restart_schedulers()

        # reset gradient accum steps
        self.grad_accum_steps = self.training_args.gradient_accumulation_steps

        if self.training_args.eval_training:
            # gather all outputs
            if self.trainer.accelerator is not None:
                outputs = self.all_gather(outputs)

            # get metrics on epoch
            print("In training epoch end: evaluate_epoch")
            self.evaluate_epoch(outputs, stage="fit", metrics=self.train_metrics)

    def validation_step(self, batch, batch_idx):
        # get the model output
        output = self(**batch, stage="val")

        # log the training step loss
        if output.loss is not None:
            self.log(
                f"vl",
                output.loss,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                sync_dist=True,
                reduce_fx="mean",
            )

        # log the validation metrics
        eval_results = self.evaluate(
            output,
            batch,
            stage="val",
            metrics=self.val_metrics,
            on_step=False,
            on_epoch=True,
        )

        return {"loss": output.loss, **eval_results}

    def validation_epoch_end(self, outputs):
        if self.trainer.global_step + 1 > self.warmup_steps:
            lr_schedulers = self.lr_schedulers()
            if not isinstance(lr_schedulers, list):
                lr_schedulers = [lr_schedulers]

            # If the selected scheduler is a ReduceLROnPlateau scheduler.
            for sch in lr_schedulers:
                # If the selected scheduler is a ReduceLROnPlateau scheduler.
                if isinstance(sch, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    # call step on epoch val loss
                    sch.step(self.trainer.callback_metrics["val_loss"])

        # gather all outputs
        if (
            self.trainer.accelerator is not None
            and self.data_args.data_loader_args.dist_eval
        ):
            outputs = self.all_gather(outputs)

        self.evaluate_epoch(outputs, stage="val", metrics=self.val_metrics)

    def test_step(self, batch, batch_idx):
        # get the model output
        output = self(**batch, stage="test")

        # log the validation metrics
        eval_results = self.evaluate(
            output,
            batch,
            stage="test",
            metrics=self.test_metrics,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )

        return {**eval_results}

    def test_epoch_end(self, outputs):
        # gather all outputs
        if (
            self.trainer.accelerator is not None
            and self.data_args.data_loader_args.dist_eval
        ):
            outputs = self.all_gather(outputs)

        self.evaluate_epoch(outputs, stage="test", metrics=self.test_metrics)

    def predict_step(self, batch, batch_idx):
        # get the model output
        return self(**batch, stage="predict")

    def restart_schedulers(self, force=False):
        optimizers = self.optimizers()
        if not isinstance(optimizers, list):
            optimizers = [optimizers]
        current_lr_schedulers = self.lr_schedulers()
        if not isinstance(current_lr_schedulers, list):
            current_lr_schedulers = [current_lr_schedulers]

        if self.trainer.global_step + 1 > self.warmup_steps:
            for idx, curr_sch in enumerate(current_lr_schedulers):
                if (
                    self.training_args.lr_schedulers[self.opt_keys[idx]].restarts
                    or force
                ):
                    self.trainer.lr_schedulers[idx][
                        "scheduler"
                    ] = self.training_args.lr_schedulers[idx].create(
                        curr_sch.optimizer,
                        self.num_training_steps,
                        self.warmup_steps,
                        self.trainer.max_epochs,
                    )[
                        "scheduler"
                    ]

    def get_model_param_groups(self):
        return {
            "default": list(self.parameters()),
        }

    def get_optimizers_schedulers(self):
        # configure optimizer for textual backbone
        model_param_groups = self.get_model_param_groups()

        # if its multi-gpu training scale the learning rate
        batch_size = self.data_args.data_loader_args.per_device_train_batch_size
        # if self.basic_args.distributed_accelerator is not None:
        #    lr_scale = (
        #        batch_size
        #        * torch.distributed.get_world_size()
        #        / self.training_args.base_lr_batch_size
        #    )

        optimizers_dict = {}
        for k, opt_config in self.training_args.optimizers.items():
            optimizers_dict[k] = opt_config.create(
                model_param_groups,
                lr_scale=1,
            )

        # configure schedulers
        lr_schedulers_dict = {}
        if self.training_args.lr_schedulers is not None:
            for k, sch in self.training_args.lr_schedulers.items():
                lr_schedulers_dict[k] = sch.create(
                    optimizers_dict[k],
                    self.num_training_steps,
                    self.warmup_steps,
                    self.trainer.max_epochs,
                )

        # configure schedulers
        wd_schedulers_dict = {}
        if self.training_args.wd_schedulers is not None:
            for k, sch in self.training_args.wd_schedulers.items():
                wd_schedulers_dict[k] = sch.create(
                    optimizers_dict[k],
                    self.trainer.max_epochs,
                    self.steps_per_epoch,
                )

        self.opt_keys = []
        optimizers = []
        lr_schedulers = []
        wd_schedulers = []
        base_optimizer_lrs = []
        for k in optimizers_dict.keys():
            self.opt_keys.append(k)
            optimizers.append(optimizers_dict[k])
            lr_schedulers.append(lr_schedulers_dict[k])
            if k in wd_schedulers_dict:
                wd_schedulers.append(wd_schedulers_dict[k])
            base_optimizer_lrs.append(
                [pg["lr"] for pg in optimizers_dict[k].param_groups]
            )

        return optimizers, lr_schedulers, wd_schedulers, base_optimizer_lrs

    def configure_optimizers(self):
        (
            optimizers,
            lr_schedulers,
            self.wd_schedulers,
            self.base_optimizer_lrs,
        ) = self.get_optimizers_schedulers()
        return optimizers, lr_schedulers

    @staticmethod
    def get_data_collators(
        data_args: Optional[DataArguments] = None,
        training_args: Optional[TrainingArguments] = None,
    ):
        return {
            "train": None,
            "val": None,
            "test": None,
        }


class RepresentationLearningBase(BaseModel):
    eval_training = False

    def evaluate(
        self,
        output,
        batch,
        stage=None,
        metrics=None,
        on_step=True,
        on_epoch=False,
        prog_bar=True,
    ):
        return {}

    def get_pred_target_labels(self, output, batch):
        pass


class SeqevalBaseModel(BaseModel):
    eval_training = True

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def init_metrics(self):
        if self.model_args.use_seqeval:
            train_metrics = {
                "f1_macro": {
                    "fn": lambda true, pred: f1_score(true, pred, average="macro"),
                    "prog_bar": True,
                },
                "f1_micro": {
                    "fn": lambda true, pred: f1_score(true, pred, average="micro"),
                    "prog_bar": True,
                },
            }

            val_metrics = {
                "acc": {
                    "fn": lambda true, pred: accuracy_score(true, pred),
                    "prog_bar": False,
                },
                "prec_macro": {
                    "fn": lambda true, pred: precision_score(
                        true, pred, average="macro"
                    ),
                    "prog_bar": False,
                },
                "prec_micro": {
                    "fn": lambda true, pred: precision_score(
                        true, pred, average="micro"
                    ),
                    "prog_bar": False,
                },
                "recall_macro": {
                    "fn": lambda true, pred: recall_score(true, pred, average="macro"),
                    "prog_bar": False,
                },
                "recall_micro": {
                    "fn": lambda true, pred: recall_score(true, pred, average="micro"),
                    "prog_bar": False,
                },
                "f1_macro": {
                    "fn": lambda true, pred: f1_score(true, pred, average="macro"),
                    "prog_bar": True,
                },
                "f1_micro": {
                    "fn": lambda true, pred: f1_score(true, pred, average="micro"),
                    "prog_bar": True,
                },
            }

            test_metrics = {
                "acc": {
                    "fn": lambda true, pred: accuracy_score(true, pred),
                    "prog_bar": False,
                },
                "prec_macro": {
                    "fn": lambda true, pred: precision_score(
                        true, pred, average="macro"
                    ),
                    "prog_bar": False,
                },
                "prec_micro": {
                    "fn": lambda true, pred: precision_score(
                        true, pred, average="micro"
                    ),
                    "prog_bar": False,
                },
                "recall_macro": {
                    "fn": lambda true, pred: recall_score(true, pred, average="macro"),
                    "prog_bar": False,
                },
                "recall_micro": {
                    "fn": lambda true, pred: recall_score(true, pred, average="micro"),
                    "prog_bar": False,
                },
                "f1_macro": {
                    "fn": lambda true, pred: f1_score(true, pred, average="macro"),
                    "prog_bar": False,
                },
                "f1_micro": {
                    "fn": lambda true, pred: f1_score(true, pred, average="micro"),
                    "prog_bar": False,
                },
            }
        else:
            train_metrics = nn.ModuleDict(
                {
                    "f1": torchmetrics.F1(num_classes=self.num_labels, average="macro"),
                }
            )

            val_metrics = nn.ModuleDict(
                {
                    "f1": torchmetrics.F1(num_classes=self.num_labels, average="macro"),
                }
            )

            test_metrics = nn.ModuleDict(
                {
                    "acc": torchmetrics.Accuracy(),
                    "precision": torchmetrics.Precision(
                        num_classes=self.num_labels, average="macro"
                    ),
                    "recall": torchmetrics.Recall(
                        num_classes=self.num_labels, average="macro"
                    ),
                    "f1": torchmetrics.F1(num_classes=self.num_labels, average="macro"),
                }
            )
        return train_metrics, val_metrics, test_metrics

    def get_pred_target_labels(self, output, batch):
        if self.model_args.use_seqeval:
            # first we check the output predictions based on binary classifier
            # probability
            labels = self.get_labels(batch)
            preds = self.get_preds(output)

            target_labels = [[]]
            pred_labels = [[]]
            pad_token_label_id = -100
            for j in range(preds.shape[0]):
                if labels[j].item() != pad_token_label_id:
                    pred_labels[-1].append(self.labels_map[preds[j].item()])
                    target_labels[-1].append(self.labels_map[labels[j].item()])
            return pred_labels, target_labels
        else:
            labels = self.get_labels(batch)
            preds = self.get_preds(output)
            return preds, labels

    @abstractmethod
    def get_labels(self, batch):
        pass

    @abstractmethod
    def get_preds(self, output):
        pass

    def evaluate(
        self,
        output,
        batch,
        stage=None,
        metrics=None,
        on_step=True,
        on_epoch=False,
        prog_bar=True,
    ):

        pred_labels, target_labels = self.get_pred_target_labels(output, batch)

        if metrics is not None and not self.model_args.use_seqeval:
            for (name, met) in metrics.items():
                if isinstance(met, torchmetrics.ConfusionMatrix):
                    met(pred_labels, target_labels)
                    continue

                self.log(
                    f"{stage[0]}{name}",
                    met(pred_labels, target_labels),
                    on_step=on_step,
                    on_epoch=on_epoch,
                    prog_bar=prog_bar,
                    # sync_dist=True,
                )

        return {"pred_labels": pred_labels, "target_labels": target_labels}

    def evaluate_epoch(
        self,
        outputs,
        stage=None,
        metrics=None,
        on_step=False,
        on_epoch=True,
        prog_bar=True,
    ):
        if self.model_args.use_seqeval:
            total_size = len(outputs)
            target_labels = []
            pred_labels = []
            for i in range(total_size):
                for j in range(len(outputs[i]["pred_labels"])):
                    pred_labels.append(outputs[i]["pred_labels"][j])
                    target_labels.append(outputs[i]["target_labels"][j])

            if metrics is not None:
                for (name, met) in metrics.items():
                    if isinstance(met, torchmetrics.ConfusionMatrix):
                        met(pred_labels, target_labels)
                        continue

                    self.log(
                        f"{stage[0]}{name}_ep",
                        met["fn"](target_labels, pred_labels),
                        prog_bar=met["prog_bar"],
                        on_epoch=on_epoch,
                        on_step=on_step,
                        rank_zero_only=True,
                    )
                    self.print(classification_report(target_labels, pred_labels))


class TorchHubBaseModelForImageClassification(BaseModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.feature_extraction = False

    def init_metrics(self):
        train_metrics = nn.ModuleDict(
            {
                "acc": torchmetrics.Accuracy(num_classes=self.num_labels),
            }
        )

        val_metrics = nn.ModuleDict(
            {
                "acc": torchmetrics.Accuracy(num_classes=self.num_labels),
            }
        )

        test_metrics = nn.ModuleDict(
            {"acc": torchmetrics.Accuracy(num_classes=self.num_labels)}
        )

        return train_metrics, val_metrics, test_metrics

    def build_model(self) -> None:
        self.labels = self.datamodule.labels
        self.labels_map = {i: label for i, label in enumerate(self.labels)}
        self.num_labels = len(self.labels)
        self.model = self.load_model(
            model_name=self.model_name,
            num_labels=len(self.datamodule.labels),
            pretrained=self.model_args.pretrained,
        )

        if self.training_args and self.training_args.two_step_finetuning_strategy:
            self.finetune_freeze()
            self.finetune_unfroze = False

        self.loss_fn = nn.CrossEntropyLoss()

    def setup_for_features(self):
        self.feature_extraction = True

    def forward(
        self, index=None, image=None, label=None, return_dict=None, stage=None, **kwargs
    ):
        if not self.feature_extraction:
            return_dict = (
                return_dict
                if return_dict is not None
                else self.model_args.use_return_dict
            )
            logits = self.model(image)

            loss = None
            if label is not None:
                loss = self.loss_fn(logits, label)

            if return_dict:
                return ClassificationModelOutput(loss=loss, logits=logits)
            else:
                return (
                    loss,
                    logits,
                )
        else:
            return self.model(image)

    def evaluate(
        self,
        output,
        batch,
        stage=None,
        metrics=None,
        on_step=True,
        on_epoch=False,
        prog_bar=True,
    ):
        pred_labels, target_labels = self.get_pred_target_labels(output, batch)

        if metrics is not None:
            for (name, met) in metrics.items():
                if isinstance(met, torchmetrics.ConfusionMatrix):
                    met(pred_labels, target_labels)
                    continue

                if isinstance(met, TrueLabelConfidence):
                    scores = F.softmax(output.logits, dim=1)
                    met(scores, target_labels)
                    continue

                self.log(
                    f"{stage[0]}{name}",
                    met(pred_labels, target_labels),
                    on_step=on_step,
                    on_epoch=on_epoch,
                    prog_bar=prog_bar,
                    # sync_dist=True,
                )

        return {"pred_labels": pred_labels, "target_labels": target_labels}

    def get_pred_target_labels(self, output, batch):
        labels = batch["label"].view(-1)
        preds = output.logits.view(-1, self.num_labels).argmax(dim=1)
        return preds, labels

    @classmethod
    def load_model(
        cls, model_name, num_labels=None, use_timm=True, pretrained=True, **kwargs
    ):
        return load_model(
            model_name=model_name,
            num_labels=num_labels,
            use_timm=use_timm,
            pretrained=pretrained,
            **kwargs,
        )

    @staticmethod
    def get_data_collators(
        data_args: Optional[DataArguments] = None,
        training_args: Optional[TrainingArguments] = None,
        data_key_type_map: Optional[dict] = None,
    ):
        if data_key_type_map is None:
            data_key_type_map = {
                DataKeysEnum.INDEX: torch.long,
                DataKeysEnum.IMAGE: torch.float,
                DataKeysEnum.LABEL: torch.long,
            }
        else:
            data_key_type_map[DataKeysEnum.INDEX] = torch.long
            data_key_type_map[DataKeysEnum.IMAGE] = torch.float
            data_key_type_map[DataKeysEnum.LABEL] = torch.long

        collate_fn = BatchToTensorDataCollator(
            data_key_type_map=data_key_type_map,
        )

        # initialize the data collators for bert grid based word classification
        return {
            "train": collate_fn,
            "val": collate_fn,
            "test": collate_fn,
        }
