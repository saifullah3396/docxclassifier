""" Model configuration """


from dataclasses import dataclass

from das.models.model_args import *

from ..model_args import ModelArguments


@dataclass
class ChildModelArguments(ModelArguments):
    """
    This is the arguments class to store the configuration arguments of this model.
    """

    model_name: str = "docxclassifier"
    model_type: str = "_base"
    pretrained: bool = True
    use_return_dict: bool = True
