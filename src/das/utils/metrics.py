import torch
from torchmetrics import Metric


class TrueLabelConfidence(Metric):
    def __init__(self, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.add_state("confidence", default=torch.tensor(0.),
                       dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0.), dist_reduce_fx="sum")

    def update(self, scores: torch.Tensor, target: torch.Tensor):
        assert scores.shape[0] == target.shape[0]

        self.confidence += torch.sum(
            scores[torch.arange(0, scores.shape[0]), target])
        self.total += target.numel()

    def compute(self):
        return self.confidence.float() / self.total
