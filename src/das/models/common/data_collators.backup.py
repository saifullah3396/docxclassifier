from dataclasses import dataclass, field
from typing import List

import torch
from das.data.data_args import DataArguments
from das.data.datasets.utils import DataKeysEnum
from das.data.tokenizer import DataTokenizer
from das.trainers.training_args import TrainingArguments
from das.utils.basic_utils import create_logger
import copy 

logger = create_logger(__name__)


def pad_sequence(seq, padding_side, max_length, padding_elem):
    if padding_side == "right":
        return [x + [padding_elem] * (max_length - len(x)) for x in seq]
    else:
        return [[padding_elem] * (max_length - len(x)) + x for x in seq]


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

        # sequence keys
        self.seq_keys = [
            DataKeysEnum.TOKEN_IDS, 
            DataKeysEnum.ATTENTION_MASKS,
            DataKeysEnum.TOKEN_BBOXES, 
            DataKeysEnum.TOKEN_TYPE_IDS, 
            DataKeysEnum.TOKEN_ANGLES]

    def __call__(self, features):
        for f in features:
            for key in list(f.keys()):
                if key not in list(self.data_key_type_map.keys()):
                    f.pop(key)
        
        if self.data_args.data_tokenization_args.overflow_samples_combined:
            # expand sequences per image to repeated images with repeated sequences
            features_expanded = []
            for f_idx, f in enumerate(features):
                if DataKeysEnum.TOKEN_IDS in f and isinstance(f[DataKeysEnum.TOKEN_IDS], list):
                    for i in range(len(f[DataKeysEnum.TOKEN_IDS])):
                        features_expanded.append(copy.copy(f))         
                        for k in self.seq_keys: # convert list of lists to single list
                            if k in features_expanded[-1]:
                                features_expanded[-1][k] = features_expanded[-1][k][i]

                        # generate overflow sample ids
                        features_expanded[-1][DataKeysEnum.OVERFLOW_MAPPING] = f_idx
            
            self.data_key_type_map[DataKeysEnum.OVERFLOW_MAPPING] = torch.long

        batch = self.tokenizer.pad(
            features_expanded,
            padding=self.padding,
            max_length=self.data_args.data_tokenization_args.seq_max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
        )

        # pad sequences
        for k, padding_elem in self.data_padding_dict.items():
            if k in batch:
                batch[k] = pad_sequence(
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
                else:
                    batch[k] = torch.tensor(v, dtype=self.data_key_type_map[k])

        # generate overflow token mapping
        overflow_to_sample_matrix = \
            torch.zeros(
                len(batch['overflow_to_sample_mapping']), 
                batch['overflow_to_sample_mapping'].max()+1).scatter_(
                    1, batch['overflow_to_sample_mapping'].unsqueeze(1), 1.)
        overflow_to_sample_matrix = \
            torch.nn.functional.normalize(overflow_to_sample_matrix.T, p=1, dim=1)
        batch['overflow_to_sample_matrix'] = overflow_to_sample_matrix

        return batch
