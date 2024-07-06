from __future__ import annotations

from typing import TYPE_CHECKING

import hydra
from hydra.core.hydra_config import HydraConfig
from torchfusion.core.training.fusion_trainer import FusionTrainer  # noqa

if TYPE_CHECKING:
    from omegaconf import DictConfig

from docxclassifier import *  # noqa


@hydra.main(version_base=None, config_name="hydra")
def app(cfg: DictConfig) -> None:
    # get hydra config
    hydra_config = HydraConfig.get()

    # evaluate the model
    _ = FusionTrainer.run_test(cfg, hydra_config)


if __name__ == "__main__":
    app()
