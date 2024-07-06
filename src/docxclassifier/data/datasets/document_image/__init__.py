from torchfusion.core.utilities.module_import import ModuleLazyImporter

_import_structure = {
    "tobacco3482": [
        "Tobacco3482",
    ],
    "rvlcdip": [
        "RvlCdip",
    ],
}
ModuleLazyImporter.register_datasets(__name__, _import_structure)
