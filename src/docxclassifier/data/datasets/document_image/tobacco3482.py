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

"""Tobacco3482 dataset"""


import os
from pathlib import Path

import datasets
import PIL
from torchfusion.core.constants import DataKeys
from torchfusion.core.data.datasets.download_manager import FusionDownloadManager
from torchfusion.core.data.datasets.fusion_image_dataset import (
    FusionImageDataset,
    FusionImageDatasetConfig,
)

_CITATION = """\
@article{Kumar2014StructuralSF,
    title={Structural similarity for document image classification and retrieval},
    author={Jayant Kumar and Peng Ye and David S. Doermann},
    journal={Pattern Recognit. Lett.},
    year={2014},
    volume={43},
    pages={119-126}
}
"""


_DESCRIPTION = """\
The Tobacco3482 dataset consists of 3842 grayscale images in 10 classes. In this version, the dataset is plit into 2782 training images, and 700 test images.
"""


_HOMEPAGE = "https://www.kaggle.com/datasets/patrickaudriaz/tobacco3482jpg"


_LICENSE = "https://www.industrydocuments.ucsf.edu/help/copyright/"

_IMAGE_DATA_NAME = "tobacco3482"
_OCR_DATA_NAME = "tobacco3482_ocr"

_URLS = {
    _IMAGE_DATA_NAME: f"https://huggingface.co/datasets/sasa3396/tobacco3482/resolve/main/data/{_IMAGE_DATA_NAME}.tar.gz",
    _OCR_DATA_NAME: f"https://huggingface.co/datasets/sasa3396/tobacco3482/resolve/main/data/{_OCR_DATA_NAME}.tar.gz",
}

_METADATA_URLS = {
    "default": {
        "train": "https://huggingface.co/datasets/sasa3396/tobacco3482/resolve/main/data/train.txt",
        "test": "https://huggingface.co/datasets/sasa3396/tobacco3482/resolve/main/data/test.txt",
    },
}

_CLASSES = [
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


class Tobacco3482Config(FusionImageDatasetConfig):
    """BuilderConfig for Tobacco3482."""

    def __init__(
        self,
        image_url,
        ocr_url,
        metadata_urls,
        load_ocr=False,
        load_images=True,
        ocr_conf_threshold: float = 0.99,
        tokenizer_config: dict = None,
        **kwargs,
    ):
        """BuilderConfig for Tobacco3482.
        Args:
            data_url: `string`, url to download the zip file from.
            metadata_urls: dictionary with keys 'train' and 'validation' containing the archive metadata URLs
          **kwargs: keyword arguments forwarded to super.
        """
        super(Tobacco3482Config, self).__init__(
            version=datasets.Version("1.0.0"), **kwargs
        )
        self.image_url = image_url
        self.ocr_url = ocr_url
        self.metadata_urls = metadata_urls
        self.load_ocr = load_ocr
        self.load_images = load_images
        self.ocr_conf_threshold = ocr_conf_threshold
        self.tokenizer_config = tokenizer_config


class Tobacco3482(FusionImageDataset):
    """Ryerson Vision Lab Complex Document Information Processing dataset."""

    VERSION = datasets.Version("1.0.0")

    BUILDER_CONFIGS = [
        Tobacco3482Config(
            name="default",
            description=_DESCRIPTION,
            homepage=_HOMEPAGE,
            citation=_CITATION,
            license=_LICENSE,
            labels=_CLASSES,
            image_url=_URLS[_IMAGE_DATA_NAME],
            ocr_url=_URLS[_OCR_DATA_NAME],
            metadata_urls=_METADATA_URLS["default"],
            load_ocr=False,
        ),
        Tobacco3482Config(
            name="with_ocr",
            description=_DESCRIPTION
            + " This configuration contains additional OCR information.",
            homepage=_HOMEPAGE,
            citation=_CITATION,
            license=_LICENSE,
            labels=_CLASSES,
            image_url=_URLS[_IMAGE_DATA_NAME],
            ocr_url=_URLS[_OCR_DATA_NAME],
            metadata_urls=_METADATA_URLS["default"],
            load_ocr=True,
        ),
    ]

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
        return dataset_features

    def _split_generators(self, dl_manager: FusionDownloadManager):
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

        if self.config.load_ocr:
            return [
                datasets.SplitGenerator(
                    name=datasets.Split.TRAIN,
                    gen_kwargs={
                        "labels_filepath": labels_path["train"],
                    },
                ),
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
                    },
                ),
                datasets.SplitGenerator(
                    name=datasets.Split.TEST,
                    gen_kwargs={
                        "labels_filepath": labels_path["test"],
                    },
                ),
            ]

    @staticmethod
    def _get_image_to_class_map(data):
        from pathlib import Path

        image_to_class_id = {}
        for item in data:
            # add label
            image_path, class_id = item, _CLASSES.index(Path(item).parent.name)
            image_to_class_id[image_path] = int(class_id)

        return image_to_class_id

    def _generate_examples_impl(self, labels_filepath):
        with open(labels_filepath, encoding="utf-8") as f:
            data = f.read().splitlines()

        for item in data:
            try:
                local_image_path, label = item, int(
                    _CLASSES.index(Path(item).parent.name)
                )
                image_file_path = str(self.image_data_path / local_image_path)
                image = PIL.Image.open(image_file_path)

                output = {
                    DataKeys.IMAGE: image,
                    DataKeys.IMAGE_FILE_PATH: local_image_path,
                    DataKeys.LABEL: label,
                }
                if self.config.load_ocr:
                    from torchfusion.core.utilities.text_utilities import (
                        TesseractOCRReader,
                    )

                    ocr_file_path = self.ocr_data_path / local_image_path.replace(
                        ".jpg", ".hocr"
                    )
                    (
                        output[DataKeys.WORDS],
                        output[DataKeys.WORD_BBOXES],
                        output[DataKeys.WORD_ANGLES],
                        output[DataKeys.WORD_CONFS],
                    ) = TesseractOCRReader(
                        ocr_file_path, conf_threshold=self.config.ocr_conf_threshold
                    ).parse()
                yield local_image_path, output
            except Exception as exc:
                self._logger.exception(exc)
                continue
