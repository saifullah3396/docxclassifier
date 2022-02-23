"""
Defines the DataModule for RVLCDIP dataset.
"""

from das.data.data_modules.base import BaseDataModule
from das.data.datasets.impl.rvlcdip import RVLCDIPDataset


class RVLCDIPDataModule(BaseDataModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # get labels for this task
        self.labels = self.dataset_class.LABELS
        if self.labels is not None:
            self.num_labels = len(self.labels)

    @property
    def dataset_class(self):
        return RVLCDIPDataset


DATA_MODULE = RVLCDIPDataModule
