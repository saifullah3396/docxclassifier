"""
Defines data modules related utility functions
"""

import os
from importlib import import_module


def create_datamodules_import_dict():
    """
    Creates a dictionary of all the data modules defined in impl so they can be easily
    imported and initialized.
    """

    data_modules_path = os.path.dirname(os.path.realpath(__file__)) + '/impl/'
    import_dict = {}
    for file in os.listdir(data_modules_path):
        abs_path = data_modules_path+file
        if os.path.isfile(abs_path):
            module_name = file[:-3]
            module = f'das.data.data_modules.impl.{module_name}'
            import_dict[module_name] = \
                lambda module=module: import_module(module).DATA_MODULE
    return import_dict
