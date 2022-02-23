"""
Defines the data cacher classes.
"""

import pickle
import sys
from multiprocessing.pool import ThreadPool
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torchvision
from das.data.data_args import DataArguments
from das.data.datasets.datadings_writer import CustomFileWriter
from das.utils.basic_utils import create_logger
from datadings.reader import MsgpackReader
from tqdm import tqdm

logger = create_logger(__name__)


class DataCacher:
    """
    Handles the dataset caching functionality.

    Args:
        data_args: Data related arguments.
        split: Dataset split.
    """

    def __init__(self, data_args: DataArguments, split: str) -> None:
        self.data_args = data_args
        self.split = split

    @property
    def file_ext(self):
        """
        Extension to for saved dataset file.
        """
        return "df"

    @property
    def data_file_name(self):
        """
        Cached dataset filename.
        """
        file_name = self.data_args.data_caching_args.cached_data_name
        filename = file_name if file_name is not None else "data"
        if (
            self.data_args.data_tokenization_args.tokenize_dataset
            and not self.data_args.data_tokenization_args.tokenize_per_sample
        ):
            filename += f"{self.data_args.data_tokenization_args.tokenizer_name}"
        return f"{filename}.{self.file_ext}"

    @property
    def file_path(self):
        """
        Cached dataset file path.
        """
        file_path = (
            Path(self.data_args.data_caching_args.dataset_cache_dir)
            / self.data_args.dataset_name
            / self.split
        )
        return file_path / self.data_file_name

    def validate_cache(self):
        return self.file_path.exists()

    def save_to_cache(self, dataset):
        """
        Saves the data from dataset to file.

        Args:
            dataset: The dataset class to save the data from.
        """

        if dataset.data is not None:
            logger.info(
                f"Saving dataset [{self.data_args.dataset_name}-{self.split}] "
                "to cache..."
            )
            if not self.file_path.parent.exists():
                self.file_path.parent.mkdir(parents=True)
            dataset.data.to_pickle(self.file_path)
        return self.file_path

    def load_from_cache(self):
        """
        Loads the data from cached file.
        """
        if self.validate_cache():
            logger.info(
                f"Loading dataset [{self.data_args.dataset_name}-{self.split}] "
                f"from cached file: {self.file_path}"
            )
            data = pd.read_pickle(self.file_path)
            return data, self.file_path
        else:
            return None, self.file_path

    def get_sample(self, dataset, idx):
        """
        Returns the sample from the data. This is called from the dataset as in some
        cases, the data cacher might get the sample directly from the file.

        Args:
            dataset: The dataset to load the sample from.
            idx: The sample index.
        """
        return dataset.get_sample(idx)


class DatadingsDataCacher(DataCacher):
    """
    Handles the datadings based caching functionality.
    """

    @property
    def file_ext(self):
        """Datadings cache file extension."""
        return "msgpack"

    def save_to_cache(self, dataset):
        """
        Saves the data from dataset to a datadings file.

        Args:
            dataset: The dataset class to save the data from.
        """

        if dataset.data is not None:
            logger.info(
                f"Saving dataset [{self.data_args.dataset_name}-{self.split}] "
                "to cache..."
            )
            if not self.file_path.parent.exists():
                self.file_path.parent.mkdir(parents=True)

            try:
                # save dataset meta info
                self.save_dataset_meta(dataset)

                cached_data_size = 0
                if self.file_path.exists():
                    data_reader = MsgpackReader(self.file_path)
                    cached_data_size = len(data_reader)

                def get_samples():
                    for idx, _ in dataset.data.iterrows():
                        if idx < cached_data_size:
                            print(idx, cached_data_size)
                            yield idx, None
                        else:
                            yield idx, dataset.get_sample(idx)

                def preprocess_sample(sample):
                    idx, sample = sample
                    if idx < cached_data_size:
                        return None
                    if self.data_args.data_caching_args.cache_resized_images:
                        if "image" in sample:
                            sample["image"] = torchvision.transforms.functional.resize(
                                sample["image"],
                                self.data_args.data_caching_args.cache_image_size,
                            )

                    if self.data_args.data_caching_args.cache_grayscale_images:
                        if "image" in sample:
                            sample["image"] = sample["image"][0]

                    if self.data_args.data_caching_args.cache_encoded_images:
                        sample["image"] = cv2.imencode(".png", sample["image"].numpy())[
                            1
                        ].tobytes()

                    sample = {"key": str(idx), "data": pickle.dumps(sample)}
                    return sample

                gen = get_samples()
                writer = CustomFileWriter(self.file_path, overwrite=False)
                pool = ThreadPool(self.data_args.data_caching_args.workers)
                with writer:
                    logger.info(
                        "Writing all data into a datadings file. "
                        "This might take a while... Please do not press ctrl-C."
                    )
                    for sample in tqdm(pool.imap_unordered(preprocess_sample, gen)):
                        if sample:
                            writer.write({**sample})

                    # progress = tqdm(
                    #     dataset.data.iterrows(), total=dataset.data.shape[0]
                    # )
                    # for index, _ in progress:
                    #     if index < cached_data_size:
                    #         continue
                    #     sample = dataset.get_sample(index)
                    #     # if (
                    #     #     self.data_args.data_tokenization_args.tokenize_dataset
                    #     #     and self.data_args.data_tokenization_args.tokenize_per_sample
                    #     # ):
                    #     #     sample = dataset._tokenize_sample(sample)
                    #     if self.data_args.data_caching_args.cache_resized_images:
                    #         if "image" in sample:
                    #             sample[
                    #                 "image"
                    #             ] = torchvision.transforms.functional.resize(
                    #                 sample["image"],
                    #                 self.data_args.data_caching_args.cache_image_size,
                    #             )

                    #     if self.data_args.data_caching_args.cache_grayscale_images:
                    #         if "image" in sample:
                    #             sample["image"] = sample["image"][0]

                    #     if self.data_args.data_caching_args.cache_encoded_images:
                    #         sample["image"] = cv2.imencode(
                    #             ".png", sample["image"].numpy()
                    #         )[1].tobytes()

                    #     sample = {"data": pickle.dumps(sample)}
                    #     writer.write({"key": str(index), **sample})
            except KeyboardInterrupt as exc:
                logger.error(f"Data caching interrupted. Exiting...")
                sys.exit(1)
            except Exception as exc:
                logger.exception(
                    f"Exception raised while saving dataset "
                    f"[{self.data_args.dataset_name}] into datading: {exc}"
                )
                sys.exit(1)
        return self.file_path

    def save_dataset_meta(self, dataset):
        sample = dataset.get_sample(0)
        # if (
        #     self.data_args.data_tokenization_args.tokenize_dataset
        #     and self.data_args.data_tokenization_args.tokenize_per_sample
        # ):
        #     sample = dataset._tokenize_sample(sample)
        dataset_meta = {"size": len(dataset.data), "keys": list(sample.keys())}
        dataset_meta_fp = (
            self.file_path.parent
            / f"{self.data_args.data_caching_args.cached_data_name}_meta.pickle"
        )
        with open(dataset_meta_fp, "wb") as f:
            pickle.dump(dataset_meta, f)

    def validate_cache(self):
        try:
            dataset_meta = None
            dataset_meta_fp = (
                self.file_path.parent
                / f"{self.data_args.data_caching_args.cached_data_name}_meta.pickle"
            )
            if dataset_meta_fp.exists():
                with open(dataset_meta_fp, "rb") as f:
                    dataset_meta = pickle.load(f)
            if dataset_meta is None:
                return False

            data_reader = None
            if self.file_path.exists():
                data_reader = MsgpackReader(self.file_path)
                size_check = len(data_reader) == dataset_meta["size"]
                if not size_check:
                    return False
            else:
                return False
            return True
        except Exception as exc:
            logger.exception(f"Exception raised while validating cache: {exc}")
            return False

    def load_from_cache(self):
        """
        Loads the data from cached file.
        """
        if self.validate_cache():
            logger.info(
                f"Loading dataset [{self.data_args.dataset_name}-{self.split}] "
                f"from cached file: {self.file_path}"
            )
            data_reader = MsgpackReader(self.file_path)
            if self.data_args.data_caching_args.load_data_to_ram:
                return self.load_data_from_datadings(data_reader), self.file_path
            return data_reader, self.file_path
        else:
            return None, self.file_path

    def load_data_from_datadings(self, data_reader):
        """
        Loads all the data from datadings file into ram.
        """
        logger.info("Loading all data into RAM. This might take a while...")
        data = []
        for idx in tqdm(range(len(data_reader))):
            sample = data_reader.get(idx)
            data.append(sample)
        return pd.DataFrame(data)

    def get_sample(self, dataset, idx):
        """
        Returns the sample from the data. This is called from the dataset class as in
        this case, the sample is read directly from the datadings file.

        Args:
            dataset: The dataset to load the sample from.
            idx: The sample index.
        """

        if self.data_args.data_caching_args.load_data_to_ram:
            sample = dataset.data.iloc[idx].to_dict()
        else:
            sample = dataset.data.get(idx)
        if "data" in sample:
            sample = pickle.loads(sample["data"])
        return sample
