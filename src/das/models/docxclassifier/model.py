""" PyTorch lightning module for the visual backbone of the AlexNetv2 model. """


import torch
from das.models.common.model_outputs import ClassificationModelWithAttentionOutput
from das.utils.basic_utils import create_logger
from torch import nn

from ..model import TorchHubBaseModelForImageClassification
from .model_defs import docxclassifier_base, docxclassifier_large, docxclassifier_xlarge

logger = create_logger(__name__)


class DocXClassifierForImageClassification(TorchHubBaseModelForImageClassification):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @classmethod
    def load_model(cls, model_name, num_labels, use_timm=False, pretrained=True):
        if model_name == "docxclassifier_base":
            model = docxclassifier_base()
        elif model_name == "docxclassifier_large":
            model = docxclassifier_large()
        elif model_name == "docxclassifier_xlarge":
            model = docxclassifier_xlarge()
        pytorch_total_params = sum(p.numel() for p in model.parameters())
        print("pytorch_total_params", pytorch_total_params)
        return cls.update_classifier_for_labels(model, num_labels=num_labels)

    @classmethod
    def update_classifier_for_labels(cls, model, num_labels):
        num_ftrs = model.head2.in_features
        model.head2 = nn.Linear(num_ftrs, num_labels)
        return model

    def load_checkpoint(self, checkpoint, strict=True):
        return self.model.load_state_dict(
            checkpoint[self.checkpoint_state_dict_key], strict=strict
        )

    def forward(
        self, index=None, image=None, label=None, return_dict=None, stage=None, **kwargs
    ):
        if not self.feature_extraction:
            return_dict = (
                return_dict
                if return_dict is not None
                else self.model_args.use_return_dict
            )
            logits, attn = self.model(image)

            loss = None
            if label is not None:
                loss = self.loss_fn(logits, label)

            if return_dict:
                return ClassificationModelWithAttentionOutput(
                    loss=loss, logits=logits, attn=attn
                )
            else:
                return (loss, logits, attn)
        else:
            return self.model(image)


SUPPORTED_TASKS = {
    "image_classification": DocXClassifierForImageClassification,
}
