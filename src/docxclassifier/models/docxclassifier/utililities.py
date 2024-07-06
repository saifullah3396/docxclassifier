""" PyTorch lightning module for the visual backbone of the AlexNetv2 model. """

import numpy as np
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from timm.models.layers import DropPath


def set_requires_grad(layer, requires_grad: bool):
    for param in layer.parameters():
        param.requires_grad = requires_grad


class Mlp(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        drop=0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class AttentionFeatureSelector(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=1,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim**-0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.k = nn.Linear(dim, dim, bias=qkv_bias)
        self.v = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):

        B, N, C = x.shape
        q = (
            self.q(x[:, 0])
            .unsqueeze(1)
            .reshape(B, 1, self.num_heads, C // self.num_heads)
            .permute(0, 2, 1, 3)
        )
        k = (
            self.k(x)
            .reshape(B, N, self.num_heads, C // self.num_heads)
            .permute(0, 2, 1, 3)
        )

        q = q * self.scale
        v = (
            self.v(x)
            .reshape(B, N, self.num_heads, C // self.num_heads)
            .permute(0, 2, 1, 3)
        )

        attn = q @ k.transpose(-2, -1)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x_cls = (attn @ v).transpose(1, 2).reshape(B, 1, C)
        x_cls = self.proj(x_cls)
        x_cls = self.proj_drop(x_cls)

        return x_cls, attn


class FeatureSelector(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        init_values=1e-4,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = AttentionFeatureSelector(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )
        self.gamma_1 = nn.Parameter(init_values * torch.ones((dim)), requires_grad=True)
        self.gamma_2 = nn.Parameter(init_values * torch.ones((dim)), requires_grad=True)

    def forward(self, x, x_cls):
        u = torch.cat((x_cls, x), dim=1)
        u_selected, feature_weights = self.attn(self.norm1(u))
        x_cls = x_cls + self.drop_path(self.gamma_1 * u_selected)
        x_cls = x_cls + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x_cls)))
        return x_cls, feature_weights


def plot_attn_map(image, attn_map):
    attn_map = normalize_attribution(attn_map, size=(*image.shape[2:],)).cpu()
    fig = plt.figure(figsize=(10, 10))
    ax = fig.gca()
    plt.axis("off")

    for i, a in zip(image, attn_map):
        ax.imshow(i.cpu().permute(1, 2, 0), alpha=1, cmap="gray")
        ax.imshow(a.permute(1, 2, 0), cmap="jet", alpha=0.5)
        plt.show()
        break


def normalize_attribution(attn_map, size):
    from torch.nn.functional import upsample

    bs = attn_map.shape[0]
    attn_map = attn_map[:, :, :, 1:]
    grid_size = int(np.sqrt(attn_map.shape[-1]))
    attn_map = attn_map.view(bs, grid_size, grid_size)
    attn_map = upsample(
        attn_map.unsqueeze(1),
        size=size,
        mode="nearest-exact",
    )
    attn_map = attn_map / attn_map.sum()
    return attn_map
