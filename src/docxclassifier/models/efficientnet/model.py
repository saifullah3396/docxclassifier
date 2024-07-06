""" Lightning modules for the standard VGG model. """

from dataclasses import dataclass

import numpy as np
from matplotlib import pyplot as plt
from torch import nn

from torchfusion.core.constants import DataKeys, MetricKeys
from torchfusion.core.models.args.fusion_model_config import FusionModelConfig
from torchfusion.core.models.image_classification.fusion_nn_model import (
    FusionNNModelForImageClassification,
)
from efficientnet_pytorch import EfficientNet

class EfficientNetForImageClassification(FusionNNModelForImageClassification):
    @dataclass
    class Config(FusionNNModelForImageClassification.Config):
        dropout_rate: float = 0.5
        model_type: str = ""

    def _build_classification_model(self):
        if self.model_args.pretrained:
            return EfficientNet.from_pretrained(
                self.config.model_type,
                num_classes=self.num_labels,
                dropout_rate=self.config.dropout_rate,
            )
        else:
            return EfficientNet.from_name(
                self.config.model_type,
                num_classes=self.num_labels,
                dropout_rate=self.config.dropout_rate,
            )
