"""
Defines the DataModule for Tobacco3842 dataset.
"""

from das.data.data_modules.base import BaseDataModule
from das.data.datasets.impl.tobacco3842 import Tobacco3842Dataset


class Tobacco3842DataModule(BaseDataModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # get labels for this task
        self.labels = self.dataset_class.LABELS
        if self.labels is not None:
            self.num_labels = len(self.labels)

    @property
    def dataset_class(self):
        return Tobacco3842Dataset


DATA_MODULE = Tobacco3842DataModule
