"""
Defines the RVLCDIP dataset.
"""

import os

import pandas as pd
from das.data.datasets.image_dataset_base import ImageDatasetsBase
from das.data.datasets.utils import DataKeysEnum
from das.utils.basic_utils import create_logger

logger = create_logger(__name__)


class RVLCDIPDataset(ImageDatasetsBase):
    """RVLCDIP dataset from https://www.cs.cmu.edu/~aharley/rvl-cdip/."""

    is_downloadable = False
    supported_splits = ["train", "test", "val"]

    # define dataset labels
    LABELS = [
        "letter",
        "form",
        "email",
        "handwritten",
        "advertisement",
        "scientific report",
        "scientific publication",
        "specification",
        "file folder",
        "news article",
        "budgetv",
        "invoice",
        "presentation",
        "questionnaire",
        "resume",
        "memo",
    ]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _load_dataset(self):
        if self.split not in self.supported_splits:
            raise ValueError(f"Split argument '{self.split}' not supported.")

        # load the annotations
        data_columns = [DataKeysEnum.IMAGE_FILE_PATH, DataKeysEnum.LABEL]
        data = pd.read_csv(
            self.root_dir / f"labels/{self.split}.txt",
            names=data_columns,
            delim_whitespace=True,
        )
        if self.data_args.extras is not None and "version" in self.data_args.extras:
            data[DataKeysEnum.IMAGE_FILE_PATH] = [
                f"{self.root_dir}/images/{x[:-4]}{self.data_args.extras['version']}{x[-4:]}"
                for x in data[DataKeysEnum.IMAGE_FILE_PATH]
            ]
        else:
            data[DataKeysEnum.IMAGE_FILE_PATH] = [
                f"{self.root_dir}/images/{x}"
                for x in data[DataKeysEnum.IMAGE_FILE_PATH]
            ]
        return data
