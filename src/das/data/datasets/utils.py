"""
Defines dataset related utility functions
"""


import random
from pathlib import Path

import bs4
import pandas as pd
from das.data.tokenizer import DataTokenizer
from das.utils.basic_utils import create_logger

logger = create_logger(__name__)


class DataKeysEnum:
    INDEX = "index"
    IMAGE = "image"
    IMAGE_FILE_PATH = "image_file_path"
    OCR_FILE_PATH = "ocr_file_path"
    LABEL = "label"
    WORDS = "words"
    NER_TAGS = "ner_tags"
    WORD_BBOXES = "word_bboxes"
    WORD_ANGLES = "word_angles"
    TOKEN_IDS = "input_ids"
    TOKEN_TYPE_IDS = "token_type_ids"
    ATTENTION_MASKS = "attention_mask"
    OVERFLOW_MAPPING = "overflow_to_sample_mapping"
    TOKEN_BBOXES = "token_bboxes"
    TOKEN_ANGLES = "token_angles"
    WORD_TO_TOKEN_MAPS = "word_to_token_maps"
    AUGMENTATION = "augmentation"
    SEVERITY = "severity"


def normalize_bbox(bbox, size):
    return [
        int(1000 * bbox[0] / size[0]),
        int(1000 * bbox[1] / size[1]),
        int(1000 * bbox[2] / size[0]),
        int(1000 * bbox[3] / size[1]),
    ]


def read_ocr_data(ocr_file_path):
    # get hocr of file
    words = []
    word_bboxes = []
    word_angles = []
    try:
        ocr_file = Path(ocr_file_path)
        if ocr_file.exists() and ocr_file.stat().st_size > 0:
            with open(ocr_file, "r", encoding="utf-8") as f:
                xml_input = eval(f.read())
            soup = bs4.BeautifulSoup(xml_input, "lxml")
            ocr_page = soup.findAll("div", {"class": "ocr_page"})
            image_size_str = ocr_page[0]["title"].split("; bbox")[1]
            w, h = map(int, image_size_str[4 : image_size_str.find(";")].split())
            # ocr_lines = soup.findAll("span", {"class": "ocr_line"})

            # for line in ocr_lines:
            #     title = line["title"]

            #     # get text angle from line title
            #     textangle = 0
            #     if "textangle" in title:
            #         textangle = int(title.split("textangle")[1][1:3])

            #     ocr_words = line.findAll("span", {"class": "ocrx_word"})
            #     for word in ocr_words:
            #         title = word["title"]
            #         conf = int(title[title.find(";") + 10 :])
            #         if conf < 80 or word.text.strip() == "":
            #             continue
            #         x1, y1, x2, y2 = map(int, title[5 : title.find(";")].split())
            #         words.append(word.text.strip())
            #         word_bboxes.append([x1 / w, y1 / h, x2 / w, y2 / h])
            #         word_angles.append(textangle)
            ocr_words = soup.findAll("span", {"class": "ocrx_word"})
            for word in ocr_words:
                title = word["title"]
                conf = int(title[title.find(";") + 10 :])
                if word.text.strip() == "" or conf < 50:
                    continue

                # get text angle from line title
                textangle = 0
                parent_title = word.parent["title"]
                if "textangle" in parent_title:
                    textangle = int(parent_title.split("textangle")[1][1:3])

                x1, y1, x2, y2 = map(int, title[5 : title.find(";")].split())
                words.append(word.text.strip())
                word_bboxes.append([x1 / w, y1 / h, x2 / w, y2 / h])
                word_angles.append(textangle)
        else:
            logger.warning(f"Cannot read file: {ocr_file}.")
    except Exception as e:
        logger.exception(
            f"Exception raised while reading ocr data from file {ocr_file}: {e}"
        )
    return words, word_bboxes, word_angles


def tokenize_sample(sample, data_args):
    tokenized_data = DataTokenizer.tokenize_textual_data(
        sample[DataKeysEnum.WORDS], data_args
    )

    if data_args.data_tokenization_args.overflow_samples_combined:
        tokenized_size = len(tokenized_data[DataKeysEnum.TOKEN_IDS])
        if data_args.data_tokenization_args.compute_word_to_toke_maps:
            word_to_token_maps = []
        token_bboxes_list = []
        token_angles_list = []
        for batch_index in range(tokenized_size):
            word_ids = tokenized_data.word_ids(batch_index=batch_index)
            previous_word_idx = None
            if data_args.data_tokenization_args.compute_word_to_toke_maps:
                word_to_token_map = []
            token_bboxes = []
            token_angles = []
            seq_length = len(word_ids)

            for (idx, word_idx) in enumerate(word_ids):
                # Special tokens have a word id that is None. We set the labels only
                # for our known words since our classifier is word basd
                # We set the label and bounding box for the first token of each word.
                if word_idx is not None:
                    if data_args.data_tokenization_args.compute_word_to_toke_maps:
                        if word_idx != previous_word_idx:
                            # word_to_token_map.append(idx)
                            word_to_token_map.append([0] * seq_length)
                        word_to_token_map[-1][idx] = 1

                    token_bboxes.append(sample[DataKeysEnum.WORD_BBOXES][word_idx])
                    token_angles.append(sample[DataKeysEnum.WORD_ANGLES][word_idx])
                else:
                    token_bboxes.append([0, 0, 0, 0])
                    token_angles.append(0)

                previous_word_idx = word_idx
            if data_args.data_tokenization_args.compute_word_to_toke_maps:
                word_to_token_maps.append(word_to_token_map)
            token_bboxes_list.append(token_bboxes)
            token_angles_list.append(token_angles)

        for k in [
            DataKeysEnum.TOKEN_IDS,
            DataKeysEnum.ATTENTION_MASKS,
            DataKeysEnum.TOKEN_TYPE_IDS,
        ]:
            if k in tokenized_data:
                sample[k] = tokenized_data[k]
        if data_args.data_tokenization_args.compute_word_to_toke_maps:
            sample[DataKeysEnum.WORD_TO_TOKEN_MAP] = word_to_token_maps
        sample[DataKeysEnum.TOKEN_BBOXES] = token_bboxes_list
        sample[DataKeysEnum.TOKEN_ANGLES] = token_angles_list

        if (
            len(sample[DataKeysEnum.TOKEN_IDS])
            > data_args.data_tokenization_args.max_seqs_per_sample
        ):
            indices = list(range(len(sample[DataKeysEnum.TOKEN_IDS])))
            # random.shuffle(indices)
            indices = indices[: data_args.data_tokenization_args.max_seqs_per_sample]
            for k in [
                DataKeysEnum.TOKEN_IDS,
                DataKeysEnum.ATTENTION_MASKS,
                DataKeysEnum.TOKEN_TYPE_IDS,
                DataKeysEnum.TOKEN_BBOXES,
                DataKeysEnum.TOKEN_ANGLES,
                DataKeysEnum.WORD_TO_TOKEN_MAPS,
            ]:
                if k in sample:
                    sample[k] = [sample[k][i] for i in indices]
        return sample
    else:
        raise ValueError(
            "overflow_samples_combined cannot be set to False for per sample "
            "tokenization."
        )


def tokenize_dataset(data, data_args):
    tokenized_data = DataTokenizer.tokenize_textual_data(
        data[DataKeysEnum.WORDS].to_list(), data_args
    )

    if data_args.data_tokenization_args.overflow_samples_combined:
        input_ids_combined = [[] for _ in range(len(data))]
        attention_mask_combined = [[] for _ in range(len(data))]
        token_type_ids_combined = [[] for _ in range(len(data))]
        if data_args.data_tokenization_args.compute_word_to_toke_maps:
            word_to_token_map_combined = [[] for _ in range(len(data))]
        token_bboxes_combined = [[] for _ in range(len(data))]
        token_angles_combined = [[] for _ in range(len(data))]
        for batch_index in range(len(tokenized_data[DataKeysEnum.TOKEN_IDS])):
            word_ids = tokenized_data.word_ids(batch_index=batch_index)
            org_batch_index = tokenized_data[DataKeysEnum.OVERFLOW_MAPPING][batch_index]
            input_ids_combined[org_batch_index].append(
                tokenized_data[DataKeysEnum.TOKEN_IDS][batch_index]
            )
            attention_mask_combined[org_batch_index].append(
                tokenized_data[DataKeysEnum.ATTENTION_MASKS][batch_index]
            )
            token_type_ids_combined[org_batch_index].append(
                tokenized_data[DataKeysEnum.TOKEN_TYPE_IDS][batch_index]
            )

            previous_word_idx = None
            if data_args.data_tokenization_args.compute_word_to_toke_maps:
                word_to_token_map = []
            token_bboxes = []
            token_angles = []
            seq_length = len(word_ids)

            for (idx, word_idx) in enumerate(word_ids):
                # Special tokens have a word id that is None. We set the labels only
                # for our known words since our classifier is word basd
                # We set the label and bounding box for the first token of each word.
                if word_idx is not None:
                    if data_args.data_tokenization_args.compute_word_to_toke_maps:
                        if word_idx != previous_word_idx:
                            word_to_token_map.append([0] * seq_length)
                        word_to_token_map[-1][idx] = 1

                    token_bboxes.append(
                        data[DataKeysEnum.WORD_BBOXES][org_batch_index][word_idx]
                    )
                    token_angles.append(
                        data[DataKeysEnum.WORD_ANGLES][org_batch_index][word_idx]
                    )
                else:
                    token_bboxes.append([0, 0, 0, 0])
                    token_angles.append(0)

                previous_word_idx = word_idx
            if data_args.data_tokenization_args.compute_word_to_toke_maps:
                word_to_token_map_combined[org_batch_index].append(word_to_token_map)
            token_bboxes_combined[org_batch_index].append(token_bboxes)
            token_angles_combined[org_batch_index].append(token_angles)

        data[DataKeysEnum.TOKEN_IDS] = input_ids_combined
        data[DataKeysEnum.ATTENTION_MASKS] = attention_mask_combined
        data[DataKeysEnum.TOKEN_TYPE_IDS] = token_type_ids_combined
        if data_args.data_tokenization_args.compute_word_to_toke_maps:
            data[DataKeysEnum.WORD_TO_TOKEN_MAPS] = word_to_token_map_combined
        data[DataKeysEnum.TOKEN_BBOXES] = token_bboxes_combined
        data[DataKeysEnum.TOKEN_ANGLES] = token_angles_combined
        return data
    else:
        new_data = pd.DataFrame()
        new_data[DataKeysEnum.TOKEN_IDS] = tokenized_data[DataKeysEnum.TOKEN_IDS]
        new_data[DataKeysEnum.ATTENTION_MASKS] = tokenized_data[
            DataKeysEnum.ATTENTION_MASKS
        ]
        new_data[DataKeysEnum.TOKEN_TYPE_IDS] = tokenized_data[
            DataKeysEnum.TOKEN_TYPE_IDS
        ]

        image_file_paths = []
        labels = []
        words_list = []
        token_bboxes_list = []
        token_angles_list = []
        if data_args.data_tokenization_args.compute_word_to_toke_maps:
            word_to_token_maps = []
        for batch_index in range(len(tokenized_data[DataKeysEnum.TOKEN_IDS])):
            word_ids = tokenized_data.word_ids(batch_index=batch_index)
            org_batch_index = tokenized_data[DataKeysEnum.OVERFLOW_MAPPING][batch_index]

            previous_word_idx = None
            if data_args.data_tokenization_args.compute_word_to_toke_maps:
                word_to_token_map = []
            token_bboxes = []
            token_angles = []
            seq_length = len(word_ids)

            for (idx, word_idx) in enumerate(word_ids):
                # Special tokens have a word id that is None. We set the labels only
                # for our known words since our classifier is word basd
                # We set the label and bounding box for the first token of each word.
                if word_idx is not None:
                    if data_args.data_tokenization_args.compute_word_to_toke_maps:
                        if word_idx != previous_word_idx:
                            word_to_token_map.append([0] * seq_length)
                        word_to_token_map[-1][idx] = 1

                    token_bboxes.append(
                        data[DataKeysEnum.WORD_BBOXES][org_batch_index][word_idx]
                    )
                    token_angles.append(
                        data[DataKeysEnum.WORD_ANGLES][org_batch_index][word_idx]
                    )
                else:
                    token_bboxes.append([0, 0, 0, 0])
                    token_angles.append(0)

                previous_word_idx = word_idx

            image_file_paths.append(data[DataKeysEnum.IMAGE_FILE_PATH][org_batch_index])
            labels.append(data[DataKeysEnum.LABEL][org_batch_index])
            words_list.append(data[DataKeysEnum.WORDS][org_batch_index])
            if data_args.data_tokenization_args.compute_word_to_toke_maps:
                word_to_token_maps.append(word_to_token_map)
            token_angles_list.append(token_angles)
            token_bboxes_list.append(token_bboxes)

        new_data[DataKeysEnum.IMAGE_FILE_PATH] = image_file_paths
        new_data[DataKeysEnum.LABEL] = labels
        new_data[DataKeysEnum.WORDS] = words_list
        new_data[DataKeysEnum.TOKEN_BBOXES] = token_bboxes_list
        new_data[DataKeysEnum.TOKEN_ANGLES] = token_angles_list
        if data_args.data_tokenization_args.compute_word_to_toke_maps:
            new_data[DataKeysEnum.WORD_TO_TOKEN_MAPS] = word_to_token_maps
        return new_data
