"""
Defines the Tobacco3842 dataset.
"""

import os
from pathlib import Path

import tqdm
import pandas as pd
from das.data.datasets.image_dataset_base import ImageDatasetsBase
from das.data.datasets.utils import DataKeysEnum
from das.utils.basic_utils import create_logger

logger = create_logger(__name__)


class Tobacco3842Dataset(ImageDatasetsBase):
    """Tobacco3842 dataset from https://www.kaggle.com/patrickaudriaz/tobacco3482jpg."""

    is_downloadable = False
    train_test_split_ratio = 0.8
    supported_splits = ["train", "test"]

    # define dataset labels
    LABELS = [
        "Letter",
        "Resume",
        "Scientific",
        "ADVE",
        "Email",
        "Report",
        "News",
        "Memo",
        "Form",
        "Note",
    ]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _read_data(self):
        # load all the data into a list
        files = []
        with open(f'{self.root_dir}/{self.split}.txt', 'r') as f:
            files = f.readlines()
        files = [f.strip() for f in files]

        data = []
        for file in tqdm.tqdm(files):
            sample = []

            # generate the filepath
            fp = Path(self.root_dir) / Path(file)

            # add image path
            sample.append(str(fp))
        
            # add label
            label_str = str(fp.parent.name)
            label_idx = self.LABELS.index(label_str)
            sample.append(label_idx)

            # add sample to data
            data.append(sample)

        # convert data list to df
        data_columns = [DataKeysEnum.IMAGE_FILE_PATH, DataKeysEnum.LABEL]
        return pd.DataFrame(data, columns=data_columns)

    # def _shuffle_data(self, data):
    #     shuffled_per_label_data = []
    #     for label_idx in range(len(self.LABELS)):
    #         label_samples = data[data[DataKeysEnum.LABEL] == label_idx]

    #         # make sure they are reproduceable
    #         label_samples = label_samples.sample(frac=1, random_state=1).reset_index(
    #             drop=True
    #         )

    #         train_images_per_label = int(
    #             self.train_test_split_ratio * len(label_samples)
    #         )

    #         if self.split in ["train", "val"]:
    #             shuffled_per_label_data.append(label_samples[:train_images_per_label])
    #         elif self.split == "test":
    #             shuffled_per_label_data.append(label_samples[train_images_per_label:])

    #     shuffled_per_label_data = (
    #         pd.concat(shuffled_per_label_data)
    #         .reset_index(drop=True)
    #         .sample(frac=1, random_state=1)
    #         .reset_index(drop=True)
    #     )
    #     return shuffled_per_label_data

    def _load_dataset(self):
        return self._read_data()
        # return self._shuffle_data(data)
