"""
Defines dataset related utility functions
"""

import typing
from typing import List

import torchtext
from das.data.custom_tokenizers import LayoutLMv2TokenizerFastCustom
from das.data.data_args import DataArguments
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import Vocab, build_vocab_from_iterator
from transformers.models.auto.tokenization_auto import AutoTokenizer
from transformers.tokenization_utils_base import PreTrainedTokenizerBase


class DataTokenizer:
    _tokenizers = {}
    _vocab = {}

    @staticmethod
    def get_tokenizer(data_args: DataArguments) -> PreTrainedTokenizerBase:
        """
        Initializes a new tokenizer or returns the previously initialized one from dict

        Args:
            data_args: The dataset arguments with
                - 'tokenizer_name' set to a valid value for a tokenizer based on
                    huggingface library.
                - 'dataset_cache_dir' set to a valid value for storing tokenizer data
        """
        tokenizer_name = data_args.data_tokenization_args.tokenizer_name
        tokenizer_lib = data_args.data_tokenization_args.tokenizer_lib

        if tokenizer_name not in DataTokenizer._tokenizers:
            if tokenizer_lib == "torchtext":
                DataTokenizer._tokenizers[
                    tokenizer_name
                ] = torchtext.data.utils.get_tokenizer(tokenizer_name)
            elif tokenizer_lib == "huggingface":
                # layoutlmv2 tokenizer has some issues so we use our custom version
                if "layoutlmv2" in tokenizer_name:
                    DataTokenizer._tokenizers[
                        tokenizer_name
                    ] = LayoutLMv2TokenizerFastCustom.from_pretrained(tokenizer_name)
                elif "roberta" in tokenizer_name:
                    # add space before each token
                    DataTokenizer._tokenizers[
                        tokenizer_name
                    ] = AutoTokenizer.from_pretrained(
                        tokenizer_name,
                        cache_dir=data_args.data_caching_args.dataset_cache_dir,
                        local_files_only=data_args.data_tokenization_args.fetch_local_files,
                        add_prefix_space=True,
                        do_lower_case=True,
                    )
                elif "longformer" in tokenizer_name:
                    # add space before each token
                    DataTokenizer._tokenizers[
                        tokenizer_name
                    ] = AutoTokenizer.from_pretrained(
                        tokenizer_name,
                        cache_dir=data_args.data_caching_args.dataset_cache_dir,
                        local_files_only=data_args.data_tokenization_args.fetch_local_files,
                        add_prefix_space=True,
                    )
                else:
                    DataTokenizer._tokenizers[
                        tokenizer_name
                    ] = AutoTokenizer.from_pretrained(
                        tokenizer_name,
                        cache_dir=data_args.data_caching_args.dataset_cache_dir,
                        local_files_only=data_args.data_tokenization_args.fetch_local_files,
                    )

        return DataTokenizer._tokenizers[tokenizer_name]

    @staticmethod
    def get_vocabulary(data_args: DataArguments) -> typing.List:
        tokenizer_name = data_args.data_tokenization_args.tokenizer_name
        if tokenizer_name not in DataTokenizer._vocab:
            return None
        return DataTokenizer._vocab[tokenizer_name]

    @staticmethod
    def set_vocabulary(data_args: DataArguments, vocab: Vocab) -> typing.List:
        tokenizer_name = data_args.data_tokenization_args.tokenizer_name
        DataTokenizer._vocab[tokenizer_name] = vocab

    @staticmethod
    def tokenize_textual_data(
        data: typing.Union[typing.Iterable, typing.List, typing.Dict],
        data_args: DataArguments,
    ) -> typing.List:
        """
        Tokenizes the data based on the tokenizer_name given in data_args.

        Args:
            data: List of token words
            data_args: The dataset arguments with
                - 'tokenizer_name' set to a valid value for a tokenizer based on
                    huggingface library.
                - 'dataset_cache_dir' set to a valid value for storing tokenizer data
                - 'pad_to_max_length' set to True or False
        """
        # initialize the tokenizer based on the tokenizer name
        tokenizer = DataTokenizer.get_tokenizer(data_args)
        tokenizer_name = data_args.data_tokenization_args.tokenizer_name
        tokenizer_lib = data_args.data_tokenization_args.tokenizer_lib

        if tokenizer_lib == "torchtext":
            if tokenizer_name not in DataTokenizer._vocab:

                def yield_tokens(data_iter):
                    for text in data_iter:
                        yield tokenizer(text)

                DataTokenizer._vocab[tokenizer_name] = build_vocab_from_iterator(
                    yield_tokens(data), specials=["<unk>"]
                )
                DataTokenizer._vocab[tokenizer_name].set_default_index(
                    DataTokenizer._vocab[tokenizer_name]["<unk>"]
                )
            return [
                DataTokenizer._vocab[tokenizer_name](tokenizer(sample))
                for sample in data
            ]

        elif tokenizer_lib == "huggingface":
            # get token padding configuration
            padding = (
                "max_length"
                if data_args.data_tokenization_args.pad_to_max_length
                else False
            )

            kwargs = {
                padding: padding,
                "max_length": data_args.data_tokenization_args.seq_max_length,
                "truncation": True,
                "return_overflowing_tokens": True,
                "is_split_into_words": True,
            }
            if "layoutlmv2" in tokenizer_name:
                kwargs.pop("is_split_into_words")

            # tokenize the words
            if isinstance(data, dict):
                tokenized_words = tokenizer(**data, **kwargs)
            else:
                tokenized_words = tokenizer(data, **kwargs)

            return tokenized_words
