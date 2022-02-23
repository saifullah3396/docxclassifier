"""
Defines the ouputs of the different model/sub-models as dataclasses.
"""

from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class SimpleLossOutput:
    """
    Data class for simple loss output

    Args:
        loss: Classification loss.
    """

    loss: Optional[torch.FloatTensor] = None


@dataclass
class ClassificationModelOutput:
    """
    Data class for outputs of any classifier

    Args:
        loss: Classification loss.
        logits: Classifier output logits.
    """

    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None


@dataclass
class ClassificationModelWithAttentionOutput:
    """
    Data class for outputs of any classifier

    Args:
        loss: Classification loss.
        logits: Classifier output logits.
        attn: Attention map output
    """

    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    attn: torch.FloatTensor = None
