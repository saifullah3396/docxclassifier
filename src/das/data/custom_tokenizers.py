"""
Extends the LayoutLMV2 Tokenizer in huggingface as it seems to contain some issues for
our use.
"""

from typing import List, Optional, Union

from transformers.file_utils import PaddingStrategy, TensorType
from transformers.models.layoutlmv2.tokenization_layoutlmv2_fast import (
    LayoutLMv2TokenizerFast,
)
from transformers.tokenization_utils_base import (
    BatchEncoding,
    PreTokenizedInput,
    TextInput,
    TextInputPair,
    TruncationStrategy,
)


class LayoutLMv2TokenizerFastCustom(LayoutLMv2TokenizerFast):
    def __call__(
        self,
        text: Union[
            TextInput, PreTokenizedInput, List[TextInput], List[PreTokenizedInput]
        ],
        text_pair: Optional[Union[PreTokenizedInput, List[PreTokenizedInput]]] = None,
        boxes: Union[List[List[int]], List[List[List[int]]]] = None,
        word_labels: Optional[Union[List[int], List[List[int]]]] = None,
        add_special_tokens: bool = True,
        padding: Union[bool, str, PaddingStrategy] = False,
        truncation: Union[bool, str, TruncationStrategy] = False,
        max_length: Optional[int] = None,
        stride: int = 0,
        pad_to_multiple_of: Optional[int] = None,
        return_tensors: Optional[Union[str, TensorType]] = None,
        return_token_type_ids: Optional[bool] = None,
        return_attention_mask: Optional[bool] = None,
        return_overflowing_tokens: bool = False,
        return_special_tokens_mask: bool = False,
        return_offsets_mapping: bool = False,
        return_length: bool = False,
        verbose: bool = True,
        **kwargs,
    ) -> BatchEncoding:
        # Input type checking for clearer error
        def _is_valid_text_input(t):
            if isinstance(t, str):
                # Strings are fine
                return True
            elif isinstance(t, (list, tuple)):
                # List are fine as long as they are...
                if len(t) == 0:
                    # ... empty
                    return True
                elif isinstance(t[0], str):
                    # ... list of strings
                    return True
                elif isinstance(t[0], (list, tuple)):
                    # ... list with an empty list or with a list of strings
                    return len(t[0]) == 0 or isinstance(t[0][0], str)
                else:
                    return False
            else:
                return False

        if text_pair is not None:
            # in case text + text_pair are provided, text = questions, text_pair = words
            if not _is_valid_text_input(text):
                raise ValueError(
                    "text input must of type `str` (single example) or `List[str]` (batch of examples). "
                )
            if not isinstance(text_pair, (list, tuple)):
                raise ValueError(
                    "words must of type `List[str]` (single pretokenized example),"
                    "or `List[List[str]]` (batch of pretokenized examples)."
                )
        else:
            # in case only text is provided => must be words
            if not isinstance(text, (list, tuple)):
                raise ValueError(
                    "Words must of type `List[str]` (single pretokenized example), "
                    "or `List[List[str]]` (batch of pretokenized examples)."
                )

        if text_pair is not None:
            is_batched = isinstance(text, (list, tuple))
        else:
            is_batched = (
                isinstance(text, (list, tuple))
                and text
                and isinstance(text[0], (list, tuple))
            )

        words = text if text_pair is None else text_pair
        # Comment out the boxes requirement.
        #
        # assert boxes is not None, "You must provide corresponding bounding boxes"
        # if is_batched:
        #     assert len(words) == len(boxes), "You must provide words and boxes for an equal amount of examples"
        #     for words_example, boxes_example in zip(words, boxes):
        #         assert len(words_example) == len(
        #             boxes_example
        #         ), "You must provide as many words as there are bounding boxes"
        # else:
        #     assert len(words) == len(boxes), "You must provide as many words as there are bounding boxes"

        if is_batched:
            if text_pair is not None and len(text) != len(text_pair):
                raise ValueError(
                    f"batch length of `text`: {len(text)} does not match batch length of `text_pair`: {len(text_pair)}."
                )
            batch_text_or_text_pairs = (
                list(zip(text, text_pair)) if text_pair is not None else text
            )
            is_pair = bool(text_pair is not None)
            return self.batch_encode_plus(
                batch_text_or_text_pairs=batch_text_or_text_pairs,
                is_pair=is_pair,
                boxes=boxes,
                word_labels=word_labels,
                add_special_tokens=add_special_tokens,
                padding=padding,
                truncation=truncation,
                max_length=max_length,
                stride=stride,
                pad_to_multiple_of=pad_to_multiple_of,
                return_tensors=return_tensors,
                return_token_type_ids=return_token_type_ids,
                return_attention_mask=return_attention_mask,
                return_overflowing_tokens=return_overflowing_tokens,
                return_special_tokens_mask=return_special_tokens_mask,
                return_offsets_mapping=return_offsets_mapping,
                return_length=return_length,
                verbose=verbose,
                **kwargs,
            )
        else:
            return self.encode_plus(
                text=text,
                text_pair=text_pair,
                boxes=boxes,
                word_labels=word_labels,
                add_special_tokens=add_special_tokens,
                padding=padding,
                truncation=truncation,
                max_length=max_length,
                stride=stride,
                pad_to_multiple_of=pad_to_multiple_of,
                return_tensors=return_tensors,
                return_token_type_ids=return_token_type_ids,
                return_attention_mask=return_attention_mask,
                return_overflowing_tokens=return_overflowing_tokens,
                return_special_tokens_mask=return_special_tokens_mask,
                return_offsets_mapping=return_offsets_mapping,
                return_length=return_length,
                verbose=verbose,
                **kwargs,
            )

    def _batch_encode_plus(
        self,
        batch_text_or_text_pairs: Union[
            List[TextInput],
            List[TextInputPair],
            List[PreTokenizedInput],
        ],
        is_pair: bool = None,
        boxes: Optional[List[List[List[int]]]] = None,
        word_labels: Optional[List[List[int]]] = None,
        add_special_tokens: bool = True,
        padding_strategy: PaddingStrategy = PaddingStrategy.DO_NOT_PAD,
        truncation_strategy: TruncationStrategy = TruncationStrategy.DO_NOT_TRUNCATE,
        max_length: Optional[int] = None,
        stride: int = 0,
        pad_to_multiple_of: Optional[int] = None,
        return_tensors: Optional[str] = None,
        return_token_type_ids: Optional[bool] = None,
        return_attention_mask: Optional[bool] = None,
        return_overflowing_tokens: bool = False,
        return_special_tokens_mask: bool = False,
        return_offsets_mapping: bool = False,
        return_length: bool = False,
        verbose: bool = True,
    ) -> BatchEncoding:

        if not isinstance(batch_text_or_text_pairs, list):
            raise TypeError(
                f"batch_text_or_text_pairs has to be a list (got {type(batch_text_or_text_pairs)})"
            )

        # Set the truncation and padding strategy and restore the initial configuration
        self.set_truncation_and_padding(
            padding_strategy=padding_strategy,
            truncation_strategy=truncation_strategy,
            max_length=max_length,
            stride=stride,
            pad_to_multiple_of=pad_to_multiple_of,
        )

        if is_pair:
            batch_text_or_text_pairs = [
                (text.split(), text_pair)
                for text, text_pair in batch_text_or_text_pairs
            ]

        encodings = self._tokenizer.encode_batch(
            batch_text_or_text_pairs,
            add_special_tokens=add_special_tokens,
            is_pretokenized=True,  # we set this to True as LayoutLMv2 always expects pretokenized inputs
        )

        # Convert encoding to dict
        # `Tokens` has type: Tuple[
        #                       List[Dict[str, List[List[int]]]] or List[Dict[str, 2D-Tensor]],
        #                       List[EncodingFast]
        #                    ]
        # with nested dimensions corresponding to batch, overflows, sequence length
        tokens_and_encodings = [
            self._convert_encoding(
                encoding=encoding,
                return_token_type_ids=return_token_type_ids,
                return_attention_mask=return_attention_mask,
                return_overflowing_tokens=return_overflowing_tokens,
                return_special_tokens_mask=return_special_tokens_mask,
                return_offsets_mapping=True
                if word_labels is not None
                else return_offsets_mapping,  # we use offsets to create the labels
                return_length=return_length,
                verbose=verbose,
            )
            for encoding in encodings
        ]

        # Convert the output to have dict[list] from list[dict] and remove the additional overflows dimension
        # From (variable) shape (batch, overflows, sequence length) to ~ (batch * overflows, sequence length)
        # (we say ~ because the number of overflow varies with the example in the batch)
        #
        # To match each overflowing sample with the original sample in the batch
        # we add an overflow_to_sample_mapping array (see below)
        sanitized_tokens = {}
        for key in tokens_and_encodings[0][0].keys():
            stack = [e for item, _ in tokens_and_encodings for e in item[key]]
            sanitized_tokens[key] = stack
        sanitized_encodings = [e for _, item in tokens_and_encodings for e in item]

        # If returning overflowing tokens, we need to return a mapping
        # from the batch idx to the original sample
        if return_overflowing_tokens:
            overflow_to_sample_mapping = []
            for i, (toks, _) in enumerate(tokens_and_encodings):
                overflow_to_sample_mapping += [i] * len(toks["input_ids"])
            sanitized_tokens["overflow_to_sample_mapping"] = overflow_to_sample_mapping

        for input_ids in sanitized_tokens["input_ids"]:
            self._eventual_warn_about_too_long_sequence(input_ids, max_length, verbose)

        # create the token boxes
        if boxes is not None and boxes != [None]:
            token_boxes = []
            for batch_index in range(len(sanitized_tokens["input_ids"])):
                if return_overflowing_tokens:
                    original_index = sanitized_tokens["overflow_to_sample_mapping"][
                        batch_index
                    ]
                else:
                    original_index = batch_index
                token_boxes_example = []
                for id, sequence_id, word_id in zip(
                    sanitized_tokens["input_ids"][batch_index],
                    sanitized_encodings[batch_index].sequence_ids,
                    sanitized_encodings[batch_index].word_ids,
                ):
                    if word_id is not None:
                        if is_pair and sequence_id == 0:
                            token_boxes_example.append(self.pad_token_box)
                        else:
                            token_boxes_example.append(boxes[original_index][word_id])
                    else:
                        if id == self.cls_token_id:
                            token_boxes_example.append(self.cls_token_box)
                        elif id == self.sep_token_id:
                            token_boxes_example.append(self.sep_token_box)
                        elif id == self.pad_token_id:
                            token_boxes_example.append(self.pad_token_box)
                        else:
                            raise ValueError("Id not recognized")
                token_boxes.append(token_boxes_example)

            sanitized_tokens["bbox"] = token_boxes

        # optionally, create the labels
        if word_labels is not None:
            labels = []
            for batch_index in range(len(sanitized_tokens["input_ids"])):
                if return_overflowing_tokens:
                    original_index = sanitized_tokens["overflow_to_sample_mapping"][
                        batch_index
                    ]
                else:
                    original_index = batch_index
                labels_example = []
                for id, offset, word_id in zip(
                    sanitized_tokens["input_ids"][batch_index],
                    sanitized_tokens["offset_mapping"][batch_index],
                    sanitized_encodings[batch_index].word_ids,
                ):
                    if word_id is not None:
                        if self.only_label_first_subword:
                            if offset[0] == 0:
                                # Use the real label id for the first token of the word, and padding ids for the remaining tokens
                                labels_example.append(
                                    word_labels[original_index][word_id]
                                )
                            else:
                                labels_example.append(self.pad_token_label)
                        else:
                            labels_example.append(word_labels[original_index][word_id])
                    else:
                        labels_example.append(self.pad_token_label)
                labels.append(labels_example)

            sanitized_tokens["labels"] = labels
            # finally, remove offsets if the user didn't want them
            if not return_offsets_mapping:
                del sanitized_tokens["offset_mapping"]

        return BatchEncoding(
            sanitized_tokens, sanitized_encodings, tensor_type=return_tensors
        )
