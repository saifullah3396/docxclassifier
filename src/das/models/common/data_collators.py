from dataclasses import dataclass, field
from typing import List

import torch
from das.data.data_args import DataArguments
from das.data.datasets.utils import DataKeysEnum
from das.data.tokenizer import DataTokenizer
from das.trainers.training_args import TrainingArguments
from das.utils.basic_utils import create_logger
from numpy import expand_dims

logger = create_logger(__name__)


def pad_sequences(sequences, padding_side, max_length, padding_elem):
    if padding_side == "right":
        if isinstance(sequences[0][0], list):
            padded_sequences = []
            for seq_list in sequences:
                padded_sequences.append([])
                for seq in seq_list:
                    padded_sequences[-1].append(
                        seq + [padding_elem] * (max_length - len(seq))
                    )
            return padded_sequences
        else:
            return [seq + [padding_elem] * (max_length - len(seq)) for seq in sequences]
    else:
        if isinstance(sequences[0][0], list):
            padded_sequences = []
            for seq_list in sequences:
                padded_sequences.append([])
                for seq in seq_list:
                    padded_sequences[-1].append(
                        [padding_elem] * (max_length - len(seq)) + seq
                    )
            return padded_sequences
        else:
            return [[padding_elem] * (max_length - len(seq)) + seq for seq in sequences]


@dataclass
class BatchToTensorDataCollator:
    """
    Data collator for converting data in the batch to a dictionary of pytorch tensors.
    """

    data_key_type_map: dict = field(default_factory=lambda: {})

    def __call__(self, features):
        batch = {}
        for k, dtype in self.data_key_type_map.items():
            if isinstance(features[0][k], torch.Tensor):
                batch[k] = torch.stack([sample[k] for sample in features]).type(dtype)
            elif isinstance(features[0][k], list):
                batch[k] = torch.tensor([sample[k] for sample in features], dtype=dtype)
            else:
                batch[k] = torch.tensor([sample[k] for sample in features], dtype=dtype)
        return batch


@dataclass
class SequenceDataCollator:
    data_args: DataArguments
    training_args: TrainingArguments
    data_key_type_map: dict = field(default_factory=lambda: {})
    data_padding_dict: dict = field(default_factory=lambda: {})
    expand_batch: bool = True

    def __post_init__(self) -> None:
        # get the tokenizer
        self.tokenizer = DataTokenizer.get_tokenizer(self.data_args)

        # get token padding configuration
        self.padding = (
            "max_length"
            if self.data_args.data_tokenization_args.pad_to_max_length
            else False
        )

        # initialize padding skips
        if self.training_args is not None:
            self.pad_to_multiple_of = 8 if self.training_args.precision == 16 else None
        else:
            self.pad_to_multiple_of = None

        # sequence keys dict
        if DataKeysEnum.TOKEN_IDS not in self.data_padding_dict:
            self.data_padding_dict[DataKeysEnum.TOKEN_IDS] = self.tokenizer.pad_token_id
        if DataKeysEnum.TOKEN_TYPE_IDS not in self.data_padding_dict:
            self.data_padding_dict[
                DataKeysEnum.TOKEN_TYPE_IDS
            ] = self.tokenizer.pad_token_type_id
        if DataKeysEnum.ATTENTION_MASKS not in self.data_padding_dict:
            self.data_padding_dict[DataKeysEnum.ATTENTION_MASKS] = 0
        if DataKeysEnum.TOKEN_BBOXES not in self.data_padding_dict:
            self.data_padding_dict[DataKeysEnum.TOKEN_BBOXES] = [0, 0, 0, 0]
        if DataKeysEnum.TOKEN_ANGLES not in self.data_padding_dict:
            self.data_padding_dict[DataKeysEnum.TOKEN_ANGLES] = 0

    def __call__(self, features):
        batch = {}
        for k in features[0].keys():
            if k not in self.data_key_type_map.keys():
                continue
            if k not in batch:
                batch[k] = []
            for f in features:
                batch[k].append(f[k])

        # pad sequences
        for k, padding_elem in self.data_padding_dict.items():
            if k in batch:
                batch[k] = pad_sequences(
                    batch[k],
                    self.tokenizer.padding_side,
                    self.data_args.data_tokenization_args.seq_max_length,
                    padding_elem,
                )

        # convert all objects in batch to torch tensors
        for (k, v) in batch.items():
            if isinstance(v, list):
                if isinstance(v[0], torch.Tensor):
                    batch[k] = torch.stack(v).type(self.data_key_type_map[k])
                elif isinstance(v[0], list):
                    batch[k] = [
                        torch.tensor(vv, dtype=self.data_key_type_map[k]) for vv in v
                    ]
                else:
                    batch[k] = torch.tensor(v, dtype=self.data_key_type_map[k])

        if self.data_args.data_tokenization_args.overflow_samples_combined:
            # generate overflow sample ids
            batch[DataKeysEnum.OVERFLOW_MAPPING] = []
            for idx, token_ids in enumerate(batch[DataKeysEnum.TOKEN_IDS]):
                for _ in range(len(token_ids)):
                    batch[DataKeysEnum.OVERFLOW_MAPPING].append(idx)
            batch[DataKeysEnum.OVERFLOW_MAPPING] = torch.tensor(
                batch[DataKeysEnum.OVERFLOW_MAPPING]
            )

            # generate overflow token mapping
            overflow_to_sample_matrix = torch.zeros(
                len(batch["overflow_to_sample_mapping"]),
                batch["overflow_to_sample_mapping"].max() + 1,
            ).scatter_(1, batch["overflow_to_sample_mapping"].unsqueeze(1), 1.0)
            overflow_to_sample_matrix = torch.nn.functional.normalize(
                overflow_to_sample_matrix.T, p=1, dim=1
            )
            batch["overflow_to_sample_matrix"] = overflow_to_sample_matrix

        return batch
