from torchfusion.core.models.tasks import ModelTasks
from torchfusion.utilities.module_import import ModuleLazyImporter, ModuleRegistryItem

_import_structure = {
    "model": [
        ModuleRegistryItem(
            "EfficientNetForImageClassification",
            "efficientnet",
            ModelTasks.image_classification,
        ),
    ],
}

ModuleLazyImporter.register_models(__name__, _import_structure)
