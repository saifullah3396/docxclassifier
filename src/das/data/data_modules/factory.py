"""
Defines the factory for loading data modules
"""

import sys

from das.data.data_args import DataArguments
from das.data.data_modules.utils import create_datamodules_import_dict
from das.utils.basic_args import BasicArguments
from das.utils.basic_utils import create_logger

# create file logger
logger = create_logger(__name__)


class DataModuleFactory:
    # holds all the supported data modules
    __supported_datamodules = None

    @ staticmethod
    def create_datamodule(
            basic_args: BasicArguments,
            data_args: DataArguments,
            collate_fns: dict = {
                'train': None,
                'val': None,
                'test': None,
            },
            transforms: dict = {}):
        """
        Initializes and returns the child data module class if present

        Args:
            basic_args: Training arguments.
            data_args: Dataset arguments.
            collate_fns: a dictionary of data collation functions for
                train/val/test sets.
            transforms: a dictionary of data transformation functions for
                train/val/test sets.
        """
        try:
            if DataModuleFactory.__supported_datamodules is None:
                DataModuleFactory.__supported_datamodules = \
                    create_datamodules_import_dict()

            # import the module by calling the lambda
            datamodule_class = \
                DataModuleFactory.__supported_datamodules[
                    data_args.dataset_name]()

            # initialize the module
            datamodule_class = datamodule_class(
                basic_args,
                data_args,
                transforms=transforms,
                collate_fns=collate_fns)
            return datamodule_class
        except KeyError:
            logger.error(
                f"Data module {data_args.dataset_name} is not supported!")
            sys.exit(1)
        except Exception as exc:
            logger.exception(
                f"Exception raised while loading data module "
                f"[{data_args.dataset_name}]: {exc}")
            sys.exit(1)
