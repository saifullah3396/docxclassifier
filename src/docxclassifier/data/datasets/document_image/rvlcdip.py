# Copyright 2022 The HuggingFace Datasets Authors and the current dataset script contributor.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""RVL-CDIP (Ryerson Vision Lab Complex Document Information Processing) dataset"""


import os
from pathlib import Path

import datasets
import numpy as np
import PIL
from torchfusion.core.constants import DataKeys
from torchfusion.core.data.datasets.fusion_image_dataset import (
    FusionImageDataset,
    FusionImageDatasetConfig,
)

_CITATION = """\
@inproceedings{harley2015icdar,
    title = {Evaluation of Deep Convolutional Nets for Document Image Classification and Retrieval},
    author = {Adam W Harley and Alex Ufkes and Konstantinos G Derpanis},
    booktitle = {International Conference on Document Analysis and Recognition ({ICDAR})}},
    year = {2015}
}
"""


_DESCRIPTION = """\
The RVL-CDIP (Ryerson Vision Lab Complex Document Information Processing) dataset consists of 400,000 grayscale images in 16 classes, with 25,000 images per class. There are 320,000 training images, 40,000 validation images, and 40,000 test images.
"""


_HOMEPAGE = "https://www.cs.cmu.edu/~aharley/rvl-cdip/"


_LICENSE = "https://www.industrydocuments.ucsf.edu/help/copyright/"

_IMAGE_DATA_NAME = "rvl-cdip"
_OCR_DATA_NAME = "rvl-cdip-ocr"

_URLS = {
    _IMAGE_DATA_NAME: f"https://huggingface.co/datasets/rvl_cdip/resolve/main/data/{_IMAGE_DATA_NAME}.tar.gz",
    _OCR_DATA_NAME: f"https://huggingface.co/datasets/sasa3396/rvlcdip/resolve/main/data/{_OCR_DATA_NAME}.tar.gz",
}

_METADATA_URLS = {  # for default let us always have tobacco3482 overlap removed from the dataset
    "default": {
        "train": "https://huggingface.co/datasets/sasa3396/rvlcdip/resolve/main/data/tobacco3482_excluded/train.txt",
        "test": "https://huggingface.co/datasets/sasa3396/rvlcdip/resolve/main/data/tobacco3482_excluded/test.txt",  # original test set has one corrupted file which is removed. Test set here is of size 39999
        "val": "https://huggingface.co/datasets/sasa3396/rvlcdip/resolve/main/data/tobacco3482_excluded/val.txt",
    },
    "tobacco3482_included": {
        "train": "https://huggingface.co/datasets/sasa3396/rvlcdip/resolve/main/data/default/train.txt",
        "test": "https://huggingface.co/datasets/sasa3396/rvlcdip/resolve/main/data/default/test.txt",
        "val": "https://huggingface.co/datasets/sasa3396/rvlcdip/resolve/main/data/default/val.txt",
    },
}

_CLASSES = [
    "letter",  # 0
    "form",  # 1
    "email",  # 2
    "handwritten",  # 3
    "advertisement",  # 4
    "scientific report",  # 5
    "scientific publication",  # 6
    "specification",  # 7
    "file folder",  # 8
    "news article",  # 9
    "budget",  # 10
    "invoice",  # 11
    "presentation",  # 12
    "questionnaire",  # 13
    "resume",  # 14
    "memo",  # 15
]

_IMAGES_DIR = "images/"


def extract_archive(path):
    import tarfile

    root_path = path.parent
    folder_name = path.name.replace(".tar.gz", "")

    def extract_nonexisting(archive):
        for member in archive.members:
            name = member.name
            if not (root_path / folder_name / name).exists():
                archive.extract(name, path=root_path / folder_name)

    # print(f"Extracting {path.name} into {root_path / folder_name}...")
    with tarfile.open(path) as archive:
        extract_nonexisting(archive)


def folder_iterator(folder):
    for subdir, dirs, files in os.walk(folder):
        for file in files:
            yield os.path.join(subdir, file)


class RvlCdipConfig(FusionImageDatasetConfig):
    """BuilderConfig for RvlCdip."""

    def __init__(
        self,
        image_url,
        metadata_urls,
        ocr_url,
        use_auth_token=True,
        load_ocr=False,
        ocr_conf_threshold: float = 95,
        **kwargs,
    ):
        """BuilderConfig for RvlCdip.
        Args:
            image_url: `string`, url to download the zip file from.
            metadata_urls: dictionary with keys 'train' and 'validation' containing the archive metadata URLs
            **kwargs: keyword arguments forwarded to super.
        """
        super(RvlCdipConfig, self).__init__(version=datasets.Version("1.0.0"), **kwargs)
        self.image_url = image_url
        self.ocr_url = ocr_url
        self.metadata_urls = metadata_urls
        self.load_ocr = load_ocr
        self.use_auth_token = use_auth_token
        self.ocr_conf_threshold = ocr_conf_threshold


class RvlCdip(FusionImageDataset):
    """Ryerson Vision Lab Complex Document Information Processing dataset."""

    VERSION = datasets.Version("1.0.0")

    BUILDER_CONFIGS = [
        RvlCdipConfig(
            name="default",
            description=_DESCRIPTION,
            homepage=_HOMEPAGE,
            citation=_CITATION,
            license=_LICENSE,
            labels=_CLASSES,
            image_url=_URLS[_IMAGE_DATA_NAME],
            ocr_url=_URLS[_OCR_DATA_NAME],
            metadata_urls=_METADATA_URLS["default"],
            use_auth_token=True,
        ),
        RvlCdipConfig(
            name="default_test",
            description=_DESCRIPTION,
            homepage=_HOMEPAGE,
            citation=_CITATION,
            license=_LICENSE,
            labels=_CLASSES,
            image_url=_URLS[_IMAGE_DATA_NAME],
            ocr_url=_URLS[_OCR_DATA_NAME],
            metadata_urls=_METADATA_URLS["default"],
            use_auth_token=True,
        ),
        RvlCdipConfig(
            name="default_with_ocr",
            description=_DESCRIPTION
            + " Tobaco3482 overalp is removed from the dataset.",
            homepage=_HOMEPAGE,
            citation=_CITATION,
            license=_LICENSE,
            labels=_CLASSES,
            image_url=_URLS[_IMAGE_DATA_NAME],
            ocr_url=_URLS[_OCR_DATA_NAME],
            metadata_urls=_METADATA_URLS["default"],
            use_auth_token=True,
            load_ocr=True,
        ),
        RvlCdipConfig(
            name="tobacco3482_included",
            description=_DESCRIPTION
            + " Tobaco3482 overalp is removed from the dataset.",
            homepage=_HOMEPAGE,
            citation=_CITATION,
            license=_LICENSE,
            labels=_CLASSES,
            image_url=_URLS[_IMAGE_DATA_NAME],
            ocr_url=_URLS[_OCR_DATA_NAME],
            metadata_urls=_METADATA_URLS["tobacco3482_included"],
            use_auth_token=True,
        ),
    ]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.use_auth_token = self.config.use_auth_token

    def _dataset_features(self):
        dataset_features = super()._dataset_features()
        if self.config.load_ocr:
            dataset_features[DataKeys.WORDS] = datasets.Sequence(
                datasets.Value(dtype="string")
            )
            dataset_features[DataKeys.WORD_BBOXES] = datasets.Sequence(
                datasets.Sequence(datasets.Value(dtype="float"), length=4)
            )
            dataset_features[DataKeys.WORD_ANGLES] = datasets.Sequence(
                datasets.Value(dtype="int32")
            )
            dataset_features[DataKeys.WORD_CONFS] = datasets.Sequence(
                datasets.Value(dtype="int32")
            )
        return dataset_features

    def _split_generators(self, dl_manager):
        if self.config.data_dir is None:
            raise ValueError(
                "You must specify a local data directory using the data_dir keyword argument."
            )

        self.image_data_path = Path(self.config.data_dir) / f"{_IMAGE_DATA_NAME}"
        if not self.image_data_path.exists():
            self.image_data_path = dl_manager.download(
                self.config.image_url,
            )
            self.image_data_path = dl_manager.extract(self.image_data_path)
        self.image_data_path = Path(self.image_data_path)

        if self.config.load_ocr:
            self.ocr_data_path = Path(self.config.data_dir) / f"{_OCR_DATA_NAME}"
            if not self.ocr_data_path.exists():
                self.ocr_data_path = dl_manager.download(self.config.ocr_url)
                self.ocr_data_path = dl_manager.extract(self.ocr_data_path)
            self.ocr_data_path = Path(self.ocr_data_path)

        labels_path = dl_manager.download(self.config.metadata_urls)

        if "test" in self.config.name:
            return [
                datasets.SplitGenerator(
                    name=datasets.Split.TEST,
                    gen_kwargs={
                        "labels_filepath": labels_path["test"],
                    },
                ),
            ]

        else:
            return [
                datasets.SplitGenerator(
                    name=datasets.Split.TRAIN,
                    gen_kwargs={
                        "labels_filepath": labels_path["train"],
                        # "shard": [0, 1, 3, 4, 5, 6, 7, 8],
                        # "num_shards": [2] * 2,
                    },
                ),
                datasets.SplitGenerator(
                    name=datasets.Split.TEST,
                    gen_kwargs={
                        "labels_filepath": labels_path["test"],
                    },
                ),
                datasets.SplitGenerator(
                    name=datasets.Split.VALIDATION,
                    gen_kwargs={
                        "labels_filepath": labels_path["val"],
                    },
                ),
            ]

    def _generate_examples_impl(self, labels_filepath, shard=None, num_shards=None):
        labels_filepath = (
            labels_filepath[0] if isinstance(labels_filepath, list) else labels_filepath
        )
        with open(labels_filepath, encoding="utf-8") as f:
            data = f.read().splitlines()

        if shard is not None and num_shards is not None:
            shard, num_shards = shard[0], num_shards[0]
            shard_length = len(data) // num_shards
            start_idx = shard_length * shard
            end_idx = start_idx + shard_length
            data = data[start_idx:end_idx]

        for item in data:
            try:
                local_image_path, label = item.split(" ")
                local_image_path = Path(_IMAGES_DIR) / local_image_path
                image_file_path = str(self.image_data_path / local_image_path)
                image = np.array(PIL.Image.open(image_file_path))

                output = {
                    DataKeys.IMAGE: image,
                    DataKeys.IMAGE_FILE_PATH: local_image_path,
                    DataKeys.LABEL: label,
                }
                if self.config.load_ocr:
                    from torchfusion.core.utilities.text_utilities import (
                        TesseractOCRReader,
                    )

                    ocr_file_path = (
                        self.ocr_data_path
                        / _IMAGES_DIR
                        / Path(local_image_path).with_suffix(".hocr.lstm")
                    )
                    (
                        output[DataKeys.WORDS],
                        output[DataKeys.WORD_BBOXES],
                        output[DataKeys.WORD_ANGLES],
                        output[DataKeys.WORD_CONFS],
                    ) = TesseractOCRReader(
                        ocr_file_path, conf_threshold=self.config.ocr_conf_threshold
                    ).parse(
                        return_confs=True,
                    )
                yield image_file_path, output
            except Exception as exc:
                self._logger.exception(exc)
                continue
