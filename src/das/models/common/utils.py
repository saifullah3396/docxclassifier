import timm
import torch


def load_model(model_name, num_labels=None, pretrained=True, use_timm=True, **kwargs):
    if use_timm:
        if num_labels is None:
            model = timm.create_model(model_name, pretrained=pretrained, **kwargs)
        else:
            model = timm.create_model(
                model_name, pretrained=pretrained, num_classes=num_labels, **kwargs
            )
    else:
        model = torch.hub.load(
            "pytorch/vision:v0.10.0", model_name, pretrained=pretrained, **kwargs
        )
    return model
