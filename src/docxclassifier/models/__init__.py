from torchfusion.core.models.tasks import ModelTasks
from torchfusion.core.utilities.module_import import (
    ModuleLazyImporter,
    ModuleRegistryItem,
)

from .convnext import *  # noqa

_import_structure = {
    "docxclassifier.model": [
        ModuleRegistryItem(
            "DocXClassifierForImageClassification",
            "docxclassifier",
            ModelTasks.image_classification,
        ),
    ],
}

ModuleLazyImporter.register_fusion_models(__name__, _import_structure)

_import_structure = {
    "docxclassifier.model": [
        ModuleRegistryItem(
            "DocXClassifier",
            "docxclassifier",
            ModelTasks.image_classification,
        ),
    ],
}

ModuleLazyImporter.register_torch_models(__name__, _import_structure)
