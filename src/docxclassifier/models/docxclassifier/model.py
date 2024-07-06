""" PyTorch lightning module for the visual backbone of the AlexNetv2 model. """

import torch
from torch import nn
from torchfusion.core.constants import DataKeys
from torchfusion.core.models.classification.image import (
    FusionModelForImageClassification,
)
from torchfusion.core.training.utilities.constants import TrainingStage
from torchfusion.core.utilities.logging import get_logger
from torchvision.ops import FeaturePyramidNetwork

from docxclassifier.models.docxclassifier.utililities import (
    FeatureSelector,
    set_requires_grad,
)

logger = get_logger(__name__)


class BackboneFeatures(nn.Module):
    def __init__(self, backbone_model, backbone_name, use_fpn=False):
        super().__init__()
        self.backbone_name = backbone_name
        self.backbone_model = backbone_model
        self.use_fpn = use_fpn

    def forward(self, x):
        if self.backbone_name == "resnet50":
            x = self.backbone_model.conv1(x)
            x = self.backbone_model.bn1(x)
            x = self.backbone_model.relu(x)
            x = self.backbone_model.maxpool(x)

            x1 = self.backbone_model.layer1(x)
            x2 = self.backbone_model.layer2(x1)
            x3 = self.backbone_model.layer3(x2)
            x4 = self.backbone_model.layer4(x3)
            if self.use_fpn:
                return x3, x4
            else:
                return x4
        elif self.backbone_name == "res2net50_26w_8s":
            x = self.backbone_model.conv1(x)
            x = self.backbone_model.bn1(x)
            x = self.backbone_model.act1(x)
            x = self.backbone_model.maxpool(x)

            x1 = self.backbone_model.layer1(x)
            x2 = self.backbone_model.layer2(x1)
            x3 = self.backbone_model.layer3(x2)
            x4 = self.backbone_model.layer4(x3)
            if self.use_fpn:
                return x3, x4
            else:
                return x4
        elif self.backbone_name == "densenet121":
            x = self.backbone_model.features.conv0(x)
            x = self.backbone_model.features.norm0(x)
            x = self.backbone_model.features.pool0(x)
            x = self.backbone_model.features.denseblock1(x)
            x = self.backbone_model.features.transition1(x)
            x = self.backbone_model.features.denseblock2(x)
            x = self.backbone_model.features.transition2(x)
            x = self.backbone_model.features.denseblock3(x)
            x = self.backbone_model.features.transition3.norm(x)
            f1 = self.backbone_model.features.transition3.conv(x)
            x = self.backbone_model.features.transition3.pool(f1)
            x = self.backbone_model.features.denseblock4(x)
            f2 = self.backbone_model.features.norm5(x)
            if self.use_fpn:
                return f1, f2
            else:
                return f2
        elif self.backbone_name == "resnext101_32x8d":
            x = self.backbone_model.conv1(x)
            x = self.backbone_model.bn1(x)
            x = self.backbone_model.act1(x)
            x = self.backbone_model.maxpool(x)

            x1 = self.backbone_model.layer1(x)
            x2 = self.backbone_model.layer2(x1)
            x3 = self.backbone_model.layer3(x2)
            x4 = self.backbone_model.layer4(x3)
            if self.use_fpn:
                return x3, x4
            else:
                return x4
        elif self.backbone_name == "efficientnet-b4":
            endpoints = self.backbone_model.extract_endpoints(x)
            if self.use_fpn:
                return endpoints["reduction_4"], endpoints["reduction_6"]
            else:
                return endpoints["reduction_6"]
        elif "convnext" in self.backbone_name:
            features = []
            for i in range(4):
                x = self.backbone_model.downsample_layers[i](x)
                x = self.backbone_model.stages[i](x)
                if i in [2, 3]:
                    features.append(x)
            if self.use_fpn:
                return features
            else:
                return features[-1]
        else:
            raise NotImplementedError(f"Backbone {self.backbone_name} not supported.")


class DocXClassifier(nn.Module):
    def __init__(
        self,
        backbone_model,
        backbone_model_name,
        num_labels: int,
        freeze_backbone_model: bool = True,
        use_fpn: bool = False,
        cls_embed_dim: int = 768,
        input_size: int = 224,
    ):
        super().__init__()
        self.backbone_model = BackboneFeatures(
            backbone_model, backbone_model_name, use_fpn
        )
        self.num_labels = num_labels
        self.freeze_backbone_model = freeze_backbone_model
        self.use_fpn = use_fpn
        self.cls_embed_dim = cls_embed_dim
        self.input_size = input_size

        # freeze backbone model if needed
        if self.freeze_backbone_model:
            set_requires_grad(self.backbone_model, False)

        # get output shape of backbone model
        with torch.no_grad():
            if self.use_fpn:
                f1, f2 = self.backbone_model(
                    torch.randn(1, 3, self.input_size, self.input_size)
                )
                self.feature_dims = [f1.shape[1], f2.shape[1]]
                logger.info(f"Output shapes: {f1.shape}, {f2.shape}")
            else:
                output_shape = self.backbone_model(
                    torch.randn(1, 3, self.input_size, self.input_size)
                ).shape
                self.feature_dim = output_shape[1]
                logger.info(f"Output shape: {output_shape}")
                logger.info(f"Feature dimension: {self.feature_dim}")

        # build explanation head modules
        self._build_explanation_head()

    def _build_explanation_head(self):
        if self.use_fpn:
            self.fpn = FeaturePyramidNetwork(
                [self.feature_dims[0], self.feature_dims[1]], self.cls_embed_dim
            )
        else:
            self.feature_to_embed = nn.Linear(self.feature_dim, self.cls_embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, int(self.cls_embed_dim)))
        self.feature_selector = FeatureSelector(
            dim=int(self.cls_embed_dim), num_heads=1
        )
        self.cls_norm = nn.LayerNorm(self.cls_embed_dim, eps=1e-6)
        self.cls_head = nn.Linear(self.cls_embed_dim, self.num_labels)

    def forward(self, image):
        B = image.shape[0]
        cls_tokens = self.cls_token.expand(B, -1, -1)
        features = self.backbone_model(image)

        if self.use_fpn:
            feature_maps_dict = {"feat0": features[0], "feat1": features[1]}
            feature_maps_dict = self.fpn(feature_maps_dict)
            features = feature_maps_dict["feat0"].flatten(2).transpose(1, 2)
        else:
            features = self.feature_to_embed(features.flatten(2).transpose(1, 2))
        x_cls, feature_weights = self.feature_selector(features, cls_tokens)
        x_cls = torch.cat((x_cls, features), dim=1)
        x_cls = self.cls_norm(x_cls)
        logits = self.cls_head(x_cls[:, 0])
        return logits, feature_weights


class DocXClassifierForImageClassification(FusionModelForImageClassification):
    def _training_step(self, engine, batch, tb_logger, **kwargs) -> None:
        assert self._LABEL_KEY in batch, "Label must be passed for training"

        # get data
        input = self._prepare_input(engine, batch, tb_logger, **kwargs)
        label = self._prepare_label(engine, batch, tb_logger, **kwargs)
        print("input", input.shape)

        # compute logits
        logits, attention_maps = self._model_forward(input)

        # compute loss
        loss = self.loss_fn_train(logits, label)

        # return outputs
        if self.model_args.return_dict:
            return {
                DataKeys.LOSS: loss,
                DataKeys.LOGITS: logits,
                DataKeys.ATTENTION_MAPS: attention_maps,
                self._LABEL_KEY: label,
            }
        else:
            return (loss, logits, attention_maps, label)

    def _evaluation_step(
        self,
        engine,
        training_engine,
        batch,
        tb_logger,
        stage: TrainingStage = TrainingStage.test,
        **kwargs,
    ) -> None:
        assert self._LABEL_KEY in batch, "Label must be passed for evaluation"

        # get data
        input = self._prepare_input(engine, batch, tb_logger, **kwargs)
        label = self._prepare_label(engine, batch, tb_logger, **kwargs)

        # compute logits
        logits, attention_maps = self._model_forward(input)

        # compute loss
        loss = self.loss_fn_eval(logits, label)

        # return outputs
        if self.model_args.return_dict:
            return {
                DataKeys.LOSS: loss,
                DataKeys.LOGITS: logits,
                DataKeys.ATTENTION_MAPS: attention_maps,
                self._LABEL_KEY: label,
            }
        else:
            return (loss, logits, attention_maps, label)

    def _predict_step(self, engine, batch, tb_logger, **kwargs) -> None:
        # get data
        input = self._prepare_input(engine, batch, tb_logger, **kwargs)

        # compute logits
        logits, attention_maps = self._model_forward(input)

        # return outputs
        if self.model_args.return_dict:
            return {
                DataKeys.LOGITS: logits,
                DataKeys.ATTENTION_MAPS: attention_maps,
            }
        else:
            return (logits, attention_maps)

    def _model_forward(self, input, return_logits=True):
        self.torch_model.eval()
        if isinstance(input, dict):
            # compute logits
            output = self.torch_model(**input)
        else:
            output = self.torch_model(input)
        return output
