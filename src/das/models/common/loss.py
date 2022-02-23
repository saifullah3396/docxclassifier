import random

import torch
import torch.distributed as dist
import torch.nn as nn


class SyncFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, tensor):
        ctx.batch_size = tensor.shape[0]

        gathered_tensor = [
            torch.zeros_like(tensor) for _ in range(torch.distributed.get_world_size())
        ]

        torch.distributed.all_gather(gathered_tensor, tensor)
        gathered_tensor = torch.cat(gathered_tensor, 0)

        return gathered_tensor

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        torch.distributed.all_reduce(
            grad_input, op=torch.distributed.ReduceOp.SUM, async_op=False
        )

        idx_from = torch.distributed.get_rank() * ctx.batch_size
        idx_to = (torch.distributed.get_rank() + 1) * ctx.batch_size
        return grad_input[idx_from:idx_to]


class EntLoss(nn.Module):
    def __init__(self, lam1, lam2, tau=1.0, eps=1e-5):
        super(EntLoss, self).__init__()
        self.lam1 = lam1
        self.lam2 = lam2
        self.tau = tau
        self.eps = eps

    def forward(self, features_1, features_2, use_queue=False):
        # gather representations in case of distributed training
        # features_1_dist: [batch_size * world_size, dim]
        # features_2_dist: [batch_size * world_size, dim]
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            features_1_dist = SyncFunction.apply(features_1)
            features_2_dist = SyncFunction.apply(features_2)
        else:
            features_1_dist = features_1
            features_2_dist = features_2

        probs1 = torch.nn.functional.softmax(features_1_dist, dim=-1)
        probs2 = torch.nn.functional.softmax(features_2_dist, dim=-1)
        loss = dict()
        loss["kl"] = 0.5 * (KL(probs1, probs2, self.eps) + KL(probs2, probs1, self.eps))
        sharpened_probs1 = torch.nn.functional.softmax(
            features_1_dist / self.tau, dim=-1
        )
        sharpened_probs2 = torch.nn.functional.softmax(
            features_2_dist / self.tau, dim=-1
        )
        loss["eh"] = 0.5 * (
            EH(sharpened_probs1, self.eps) + EH(sharpened_probs2, self.eps)
        )

        # whether use historical data
        loss["he"] = 0.5 * (
            HE(sharpened_probs1, self.eps) + HE(sharpened_probs2, self.eps)
        )

        loss["final"] = loss["kl"] + (
            (1 + self.lam1) * loss["eh"] - self.lam2 * loss["he"]
        )
        return loss


def KL(probs1, probs2, eps):
    kl = (probs1 * (probs1 + eps).log() - probs1 * (probs2 + eps).log()).sum(dim=1)
    kl = kl.mean()
    return kl


def CE(probs1, probs2, eps):
    ce = -(probs1 * (probs2 + eps).log()).sum(dim=1)
    ce = ce.mean()
    return ce


def HE(probs, eps):
    mean = probs.mean(dim=0)
    ent = -(mean * (mean + torch.distributed.get_world_size() * eps).log()).sum()
    return ent


def EH(probs, eps):
    ent = -(probs * (probs + eps).log()).sum(dim=1)
    mean = ent.mean()
    return mean
