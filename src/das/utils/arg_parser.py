"""
Defines a customized argument parser that parses arguments based on input dataclasses.
"""

import copy
import dataclasses
import os
import re
from argparse import Namespace
from pathlib import Path
from typing import Iterable, Optional, Tuple, Type, Union, get_type_hints

import yaml
from dacite import from_dict
from dacite.config import Config
from dacite.core import T, _build_value
from dacite.data import Data
from dacite.dataclasses import (
    DefaultValueNotFoundError,
    get_default_value_for_field,
    get_fields,
)
from dacite.exceptions import ForwardReferenceError, MissingValueError, WrongTypeError
from dacite.types import is_instance, transform_value
from das.models.model_args import ModelArguments, ModelArgumentsFactory
from transformers.hf_argparser import (
    DataClass,
    DataClassType,
    HfArgumentParser,
    string_to_bool,
)

path_matcher = re.compile(r"\$\{([^}^{]+)\}")


def path_constructor(loader, node):
    """Extract the matched value, expand env variable, and replace the match"""
    value = node.value
    match = path_matcher.match(value)
    env_var = match.group()[2:-1]
    return os.environ.get(env_var) + value[match.end() :]


yaml.add_implicit_resolver("!path", path_matcher)
yaml.add_constructor("!path", path_constructor)


class DASArgumentParser(HfArgumentParser):
    """
    A custom arguments parser based on huggingface arguments parser.

    Args:
        dataclass_types:
            Dataclass type, or list of dataclass types for which we will "fill"
            instances with the parsed args.
        kwargs:
            (Optional) Passed to `argparse.ArgumentParser()` in the regular way.
    """

    def __init__(
        self, dataclass_types: Union[DataClassType, Iterable[DataClassType]], **kwargs
    ):
        super().__init__(dataclass_types, **kwargs)

    def update_dataclass(
        self, dataclass: Type[T], data: Data, config: Optional[Config] = None
    ):
        config = config or Config()
        fields = get_fields(dataclass)
        try:
            dataclass_hints = get_type_hints(dataclass)
        except NameError as error:
            raise ForwardReferenceError(str(error))

        for field in fields:
            if field.name not in data:
                continue

            field = copy.copy(field)
            field_data = data[field.name]
            if field.name not in dataclass_hints:
                field.type = type(field_data)
            else:
                field.type = dataclass_hints[field.name]
            try:
                if field.type == bool:
                    field_data = string_to_bool(field_data)

                value = _build_value(type_=field.type, data=field_data, config=config)
                if config.check_types and not is_instance(value, field.type):
                    raise WrongTypeError(
                        field_path=field.name, field_type=field.type, value=value
                    )
                setattr(dataclass, field.name, value)
            except KeyError:
                try:
                    value = get_default_value_for_field(field)
                except DefaultValueNotFoundError:
                    if not field.init:
                        continue
                    raise MissingValueError(field.name)

            dataclass.__post_init__()

    def parse_yaml_file(
        self, file: str, extra_args: Optional[Namespace] = None
    ) -> Tuple[DataClass, ...]:
        """
        Alternative helper method that does not use `argparse` at all, instead loading
        a yaml file and populating the dataclass types.
        """
        with open(file) as f:
            # use safe_load instead load
            data = yaml.load(f, Loader=yaml.FullLoader)

            outputs = {}
            for dtype in self.dataclass_types:
                if dtype == ModelArguments:
                    dtype = ModelArgumentsFactory.create_model_arguments(
                        data[dtype.cls_name]["model_name"]
                    )
                obj = from_dict(data_class=dtype, data=data[dtype.cls_name])
                outputs[dtype.cls_name] = obj

            if extra_args is not None:
                extra_args_list = list(extra_args)
                for k, dataclass in outputs.items():
                    current_args = {}
                    for idx, arg in enumerate(extra_args_list):
                        if k in arg:  # here we assume every argument as some value
                            arg_name = arg.replace(f"{k}.", "")
                            arg_value = extra_args_list[idx + 1]
                            current_args[arg_name] = arg_value
                    self.update_dataclass(
                        dataclass, current_args, Config(cast=list(current_args.keys()))
                    )

            return outputs.values()
