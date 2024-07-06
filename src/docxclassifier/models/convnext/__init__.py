from torchfusion.core.models.tasks import ModelTasks
from torchfusion.core.utilities.module_import import (
    ModuleLazyImporter,
    ModuleRegistryItem,
)

_import_structure = {
    "definition": [
        ModuleRegistryItem(
            "convnext_base",
            "convnext_base",
            ModelTasks.image_classification,
        ),
        ModuleRegistryItem(
            "convnext_large",
            "convnext_large",
            ModelTasks.image_classification,
        ),
        ModuleRegistryItem(
            "convnext_xlarge",
            "convnext_xlarge",
            ModelTasks.image_classification,
        ),
    ],
}

ModuleLazyImporter.register_torch_models(__name__, _import_structure)
