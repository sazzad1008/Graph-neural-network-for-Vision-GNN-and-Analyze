"""Graph neural network for vision training script."""

from __future__ import annotations

import argparse
import math
import os
from typing import Any, Tuple
import xml.etree.ElementTree as ElementTree

import numpy as np
import torch
from torch import nn
from torch.nn import Sequential as Seq, Conv2d
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models.layers import DropPath
from timm.models.registry import register_model


def position_1d(dimension: int, pos: np.ndarray) -> np.ndarray:
    if dimension % 2 != 0:
        raise ValueError("dimension must be even")
    omega = np.arange(dimension // 2, dtype=float)
    omega /= dimension / 2.0
    omega = 1.0 / 10000**omega
    pos = pos.reshape(-1)
    out = np.einsum("m,d->md", pos, omega)
    sin_out = np.sin(out)
    cos_out = np.cos(out)
    return np.concatenate([sin_out, cos_out], axis=1)


def position_2d(dimension: int, grid: np.ndarray) -> np.ndarray:
    embed_h = position_1d(dimension // 2, grid[0])
    embed_w = position_1d(dimension // 2, grid[1])
    return np.concatenate([embed_h, embed_w], axis=1)


def get_2d_position_from_start(embed_dim: int, grid_size: int) -> np.ndarray:
    grid_height = np.arange(grid_size, dtype=np.float32)
    grid_width = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_width, grid_height)
    grid = np.stack(grid, axis=0)
    return position_2d(embed_dim, grid)


def relative_position(embed_dim: int, grid_size: int) -> np.ndarray:
    pos_embedding = get_2d_position_from_start(embed_dim, grid_size)
    return 2 * np.matmul(pos_embedding, pos_embedding.transpose()) / pos_embedding.shape[1]


def pairwise_distance(x: torch.Tensor) -> torch.Tensor:
    """Compute pairwise distance of a point cloud."""
    with torch.no_grad():
        x_inner = -2 * torch.matmul(x, x.transpose(2, 1))
        x_square = torch.sum(torch.mul(x, x), dim=-1, keepdim=True)
        return x_square + x_inner + x_square.transpose(2, 1)


def part_pairwise_distance(x: torch.Tensor, start_idx: int = 0, end_idx: int = 1) -> torch.Tensor:
    """Compute pairwise distance of a point cloud for a slice."""
    with torch.no_grad():
        x_part = x[:, start_idx:end_idx]
        x_square_part = torch.sum(torch.mul(x_part, x_part), dim=-1, keepdim=True)
        x_inner = -2 * torch.matmul(x_part, x.transpose(2, 1))
        x_square = torch.sum(torch.mul(x, x), dim=-1, keepdim=True)
        return x_square_part + x_inner + x_square.transpose(2, 1)


def xy_pairwise_distance(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Compute pairwise distance between x and y point clouds."""
    with torch.no_grad():
        xy_inner = -2 * torch.matmul(x, y.transpose(2, 1))
        x_square = torch.sum(torch.mul(x, x), dim=-1, keepdim=True)
        y_square = torch.sum(torch.mul(y, y), dim=-1, keepdim=True)
        return x_square + xy_inner + y_square.transpose(2, 1)


def dense_knn_matrix(
    x: torch.Tensor, k: int = 16, relative_pos: torch.Tensor | None = None
) -> torch.Tensor:
    with torch.no_grad():
        x = x.transpose(2, 1).squeeze(-1)
        batch_size, n_points, _ = x.shape
        n_part = 10000

        if n_points > n_part:
            nn_idx_list: list[torch.Tensor] = []
            groups = math.ceil(n_points / n_part)
            for i in range(groups):
                start_idx = n_part * i
                end_idx = min(n_points, n_part * (i + 1))
                distance = part_pairwise_distance(x.detach(), start_idx, end_idx)
                if relative_pos is not None:
                    distance += relative_pos[:, start_idx:end_idx]
                _, nn_idx_part = torch.topk(-distance, k=k)
                nn_idx_list.append(nn_idx_part)
            nn_idx = torch.cat(nn_idx_list, dim=1)
        else:
            distance = pairwise_distance(x.detach())
            if relative_pos is not None:
                distance += relative_pos
            _, nn_idx = torch.topk(-distance, k=k)
        center_idx = (
            torch.arange(0, n_points, device=x.device).repeat(batch_size, k, 1).transpose(2, 1)
        )
        return torch.stack((nn_idx, center_idx), dim=0)


def xy_dense_knn_matrix(
    x: torch.Tensor, y: torch.Tensor, k: int = 16, relative_pos: torch.Tensor | None = None
) -> torch.Tensor:
    with torch.no_grad():
        x = x.transpose(2, 1).squeeze(-1)
        y = y.transpose(2, 1).squeeze(-1)
        batch_size, n_points, _ = x.shape
        dist = xy_pairwise_distance(x.detach(), y.detach())
        if relative_pos is not None:
            dist += relative_pos
        _, nn_idx = torch.topk(-dist, k=k)
        center_idx = (
            torch.arange(0, n_points, device=x.device).repeat(batch_size, k, 1).transpose(2, 1)
        )
    return torch.stack((nn_idx, center_idx), dim=0)


class DenseDilated(nn.Module):
    def __init__(self, k: int = 9, dilation: int = 9, stochastic: bool = False, epsilon: float = 0.0):
        super().__init__()
        self.dilation = dilation
        self.stochastic = stochastic
        self.epsilon = epsilon
        self.k = k

    def forward(self, edge_index: torch.Tensor) -> torch.Tensor:
        if self.stochastic:
            if torch.rand(1) < self.epsilon and self.training:
                num = self.k * self.dilation
                randnum = torch.randperm(num)[: self.k]
                edge_index = edge_index[:, :, :, randnum]
            else:
                edge_index = edge_index[:, :, :, :: self.dilation]
        else:
            edge_index = edge_index[:, :, :, :: self.dilation]
        return edge_index


class DenseDilatedKnnGraph(nn.Module):
    """Find the neighbors' indices based on dilated knn."""

    def __init__(self, k: int = 9, dilation: int = 1, stochastic: bool = False, epsilon: float = 0.0):
        super().__init__()
        self.dilation = dilation
        self.stochastic = stochastic
        self.epsilon = epsilon
        self.k = k
        self._dilated = DenseDilated(k, dilation, stochastic, epsilon)

    def forward(
        self,
        x: torch.Tensor,
        y: torch.Tensor | None = None,
        relative_pos: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if y is not None:
            x = F.normalize(x, p=2.0, dim=1)
            y = F.normalize(y, p=2.0, dim=1)
            edge_index = xy_dense_knn_matrix(x, y, self.k * self.dilation, relative_pos)
        else:
            x = F.normalize(x, p=2.0, dim=1)
            edge_index = dense_knn_matrix(x, self.k * self.dilation, relative_pos)
        return self._dilated(edge_index)


def act_layer(act: str, inplace: bool = False, neg_slope: float = 0.2, n_prelu: int = 1) -> nn.Module:
    act = act.lower()
    if act == "relu":
        layer = nn.ReLU(inplace)
    elif act == "leakyrelu":
        layer = nn.LeakyReLU(neg_slope, inplace)
    elif act == "prelu":
        layer = nn.PReLU(num_parameters=n_prelu, init=neg_slope)
    elif act == "gelu":
        layer = nn.GELU()
    elif act == "hswish":
        layer = nn.Hardswish(inplace)
    else:
        raise NotImplementedError(f"activation layer [{act}] is not found")
    return layer


def norm_layer(norm: str, nc: int) -> nn.Module:
    norm = norm.lower()
    if norm == "batch":
        layer = nn.BatchNorm2d(nc, affine=True)
    elif norm == "instance":
        layer = nn.InstanceNorm2d(nc, affine=False)
    else:
        raise NotImplementedError(f"normalization layer [{norm}] is not found")
    return layer


class BasicConv(Seq):
    def __init__(self, channels: list[int], act: str = "relu", norm: str | None = None,
                 bias: bool = True, drop: float = 0.0):
        m = []
        for i in range(1, len(channels)):
            m.append(Conv2d(channels[i - 1], channels[i], 1, bias=bias, groups=4))
            if norm is not None and norm.lower() != "none":
                m.append(norm_layer(norm, channels[i]))
            if act is not None and act.lower() != "none":
                m.append(act_layer(act))
            if drop > 0:
                m.append(nn.Dropout2d(drop))

        super().__init__(*m)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d)):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


def batched_index_select(x: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """Fetches neighbor features from a given neighbor idx."""
    batch_size, num_dims, num_vertices_reduced = x.shape[:3]
    _, num_vertices, k = idx.shape
    idx_base = torch.arange(0, batch_size, device=idx.device).view(-1, 1, 1) * num_vertices_reduced
    idx = idx + idx_base
    idx = idx.contiguous().view(-1)

    x = x.transpose(2, 1)
    feature = x.contiguous().view(batch_size * num_vertices_reduced, -1)[idx, :]
    feature = feature.view(batch_size, num_vertices, k, num_dims).permute(0, 3, 1, 2).contiguous()
    return feature


class MaxRelative(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, act: str = "relu",
                 norm: str | None = None, bias: bool = True):
        super().__init__()
        self.conv1 = BasicConv([in_channels * 2, out_channels], act, norm, bias)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, y: torch.Tensor | None = None) -> torch.Tensor:
        x_i = batched_index_select(x, edge_index[1])
        if y is not None:
            x_j = batched_index_select(y, edge_index[0])
        else:
            x_j = batched_index_select(x, edge_index[0])
        x_j, _ = torch.max(x_j - x_i, -1, keepdim=True)
        b, c, n, _ = x.shape
        x = torch.cat([x.unsqueeze(2), x_j.unsqueeze(2)], dim=2).reshape(b, 2 * c, n, _)
        return self.conv1(x)


class EdgeConv2d(nn.Module):
    """Edge convolution layer (with activation, batch normalization) for dense data type."""

    def __init__(self, in_channels: int, out_channels: int, act: str = "relu",
                 norm: str | None = None, bias: bool = True):
        super().__init__()
        self.nn = BasicConv([in_channels * 2, out_channels], act, norm, bias)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, y: torch.Tensor | None = None) -> torch.Tensor:
        x_i = batched_index_select(x, edge_index[1])
        if y is not None:
            x_j = batched_index_select(y, edge_index[0])
        else:
            x_j = batched_index_select(x, edge_index[0])
        max_value, _ = torch.max(self.nn(torch.cat([x_i, x_j - x_i], dim=1)), -1, keepdim=True)
        return max_value


class GraphSAGE(nn.Module):
    """GraphSAGE Graph Convolution for dense data type."""

    def __init__(self, in_channels: int, out_channels: int, act: str = "relu",
                 norm: str | None = None, bias: bool = True):
        super().__init__()
        self.nn1 = BasicConv([in_channels, in_channels], act, norm, bias)
        self.nn2 = BasicConv([in_channels * 2, out_channels], act, norm, bias)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, y: torch.Tensor | None = None) -> torch.Tensor:
        if y is not None:
            x_j = batched_index_select(y, edge_index[0])
        else:
            x_j = batched_index_select(x, edge_index[0])
        x_j, _ = torch.max(self.nn1(x_j), -1, keepdim=True)
        return self.nn2(torch.cat([x, x_j], dim=1))


class GINConv2d(nn.Module):
    """GIN Graph Convolution for dense data type."""

    def __init__(self, in_channels: int, out_channels: int, act: str = "relu",
                 norm: str | None = None, bias: bool = True):
        super().__init__()
        self.nn = BasicConv([in_channels, out_channels], act, norm, bias)
        eps_init = 0.0
        self.eps = nn.Parameter(torch.Tensor([eps_init]))

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, y: torch.Tensor | None = None) -> torch.Tensor:
        if y is not None:
            x_j = batched_index_select(y, edge_index[0])
        else:
            x_j = batched_index_select(x, edge_index[0])
        x_j = torch.sum(x_j, -1, keepdim=True)
        return self.nn((1 + self.eps) * x + x_j)


class GraphConv2d(nn.Module):
    """Static graph convolution layer."""

    def __init__(self, in_channels: int, out_channels: int, conv: str = "edge", act: str = "relu",
                 norm: str | None = None, bias: bool = True):
        super().__init__()
        if conv == "edge":
            self.gconv = EdgeConv2d(in_channels, out_channels, act, norm, bias)
        elif conv == "mr":
            self.gconv = MaxRelative(in_channels, out_channels, act, norm, bias)
        elif conv == "sage":
            self.gconv = GraphSAGE(in_channels, out_channels, act, norm, bias)
        elif conv == "gin":
            self.gconv = GINConv2d(in_channels, out_channels, act, norm, bias)
        else:
            raise NotImplementedError(f"conv:{conv} is not supported")

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, y: torch.Tensor | None = None) -> torch.Tensor:
        return self.gconv(x, edge_index, y)


class DynamicGraph(GraphConv2d):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 9, dilation: int = 1,
                 conv: str = "edge", act: str = "relu", norm: str | None = None, bias: bool = True,
                 stochastic: bool = False, epsilon: float = 0.0, r: int = 1):
        super().__init__(in_channels, out_channels, conv, act, norm, bias)
        self.k = kernel_size
        self.d = dilation
        self.r = r
        self.knn_graph = DenseDilatedKnnGraph(kernel_size, dilation, stochastic, epsilon)

    def forward(self, x: torch.Tensor, relative_pos: torch.Tensor | None = None) -> torch.Tensor:
        b, c, h, w = x.shape
        y = None
        if self.r > 1:
            y = F.avg_pool2d(x, self.r, self.r)
            y = y.reshape(b, c, -1, 1).contiguous()
        x = x.reshape(b, c, -1, 1).contiguous()
        edge_index = self.knn_graph(x, y, relative_pos)
        x = super().forward(x, edge_index, y)
        return x.reshape(b, -1, h, w).contiguous()


class Grapher(nn.Module):
    def __init__(
        self,
        in_channels: int,
        kernel_size: int = 9,
        dilation: int = 1,
        conv: str = "edge",
        act: str = "relu",
        norm: str | None = None,
        bias: bool = True,
        stochastic: bool = False,
        epsilon: float = 0.0,
        r: int = 1,
        n: int = 196,
        drop_path: float = 0.0,
        relative_pos: bool = False,
    ):
        super().__init__()
        self.channels = in_channels
        self.n = n
        self.r = r
        self.fc1 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 1, stride=1, padding=0),
            nn.BatchNorm2d(in_channels),
        )
        self.graph_conv = DynamicGraph(
            in_channels,
            in_channels * 2,
            kernel_size,
            dilation,
            conv,
            act,
            norm,
            bias,
            stochastic,
            epsilon,
            r,
        )
        self.fc2 = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels, 1, stride=1, padding=0),
            nn.BatchNorm2d(in_channels),
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.relative_pos: torch.Tensor | None = None
        if relative_pos:
            relative_pos_tensor = torch.from_numpy(
                np.float32(relative_position(in_channels, int(n**0.5)))
            ).unsqueeze(0).unsqueeze(1)
            relative_pos_tensor = F.interpolate(
                relative_pos_tensor, size=(n, n // (r * r)), mode="bicubic", align_corners=False
            )
            self.relative_pos = nn.Parameter(-relative_pos_tensor.squeeze(1), requires_grad=False)

    def _get_relative_pos(self, relative_pos: torch.Tensor | None, h: int, w: int) -> torch.Tensor | None:
        if relative_pos is None or h * w == self.n:
            return relative_pos
        n = h * w
        n_reduced = n // (self.r * self.r)
        return F.interpolate(relative_pos.unsqueeze(0), size=(n, n_reduced), mode="bicubic").squeeze(0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x
        x = self.fc1(x)
        b, c, h, w = x.shape
        relative_pos = self._get_relative_pos(self.relative_pos, h, w)
        x = self.graph_conv(x, relative_pos)
        x = self.fc2(x)
        x = self.drop_path(x) + shortcut
        return x


class FFN(nn.Module):
    def __init__(
        self, in_features: int, hidden_features: int | None = None, out_features: int | None = None,
        act: str = "relu", drop_path: float = 0.0
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Sequential(
            nn.Conv2d(in_features, hidden_features, 1, stride=1, padding=0),
            nn.BatchNorm2d(hidden_features),
        )
        self.act = act_layer(act)
        self.fc2 = nn.Sequential(
            nn.Conv2d(hidden_features, out_features, 1, stride=1, padding=0),
            nn.BatchNorm2d(out_features),
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.drop_path(x) + shortcut
        return x


class Stem(nn.Module):
    def __init__(self, in_dim: int = 3, out_dim: int = 768, act: str = "relu"):
        super().__init__()
        self.convs = nn.Sequential(
            nn.Conv2d(in_dim, out_dim // 2, 3, stride=2, padding=1),
            nn.BatchNorm2d(out_dim // 2),
            act_layer(act),
            nn.Conv2d(out_dim // 2, out_dim, 3, stride=2, padding=1),
            nn.BatchNorm2d(out_dim),
            act_layer(act),
            nn.Conv2d(out_dim, out_dim, 3, stride=1, padding=1),
            nn.BatchNorm2d(out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.convs(x)


class Downsample(nn.Module):
    def __init__(self, in_dim: int = 3, out_dim: int = 768):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_dim, out_dim, 3, stride=2, padding=1),
            nn.BatchNorm2d(out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


def _cfg(url: str = "", **kwargs: Any) -> dict:
    return {
        "url": url,
        "num_classes": 6,
        "input_size": (3, 224, 224),
        "pool_size": None,
        "crop_pct": 0.9,
        "interpolation": "bicubic",
        "mean": IMAGENET_DEFAULT_MEAN,
        "std": IMAGENET_DEFAULT_STD,
        "first_conv": "patch_embed.proj",
        "classifier": "head",
        **kwargs,
    }


default_cfgs = {
    "vig_224_gelu": _cfg(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    "vig_b_224_gelu": _cfg(crop_pct=0.95, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
}


class DeepGCN(nn.Module):
    def __init__(self, opt: "OptInit"):
        super().__init__()
        k = opt.k
        act = opt.act
        norm = opt.norm
        bias = opt.bias
        epsilon = opt.epsilon
        stochastic = opt.use_stochastic
        conv = opt.conv
        drop_path = opt.drop_path

        blocks = opt.blocks
        self.n_blocks = sum(blocks)
        channels = opt.channels
        reduce_ratios = [4, 2, 1, 1]
        drop_path_rate = [x.item() for x in torch.linspace(0, drop_path, self.n_blocks)]
        num_knn = [int(x.item()) for x in torch.linspace(k, k, self.n_blocks)]
        max_dilation = 49 // max(num_knn)
        self.stem = Stem(out_dim=channels[0], act=act)
        self.pos_embed = nn.Parameter(torch.zeros(1, channels[0], 224 // 4, 224 // 4))

        hw = 224 // 4 * 224 // 4
        self.backbone = nn.ModuleList([])
        idx = 0
        for i in range(len(blocks)):
            if i > 0:
                self.backbone.append(Downsample(channels[i - 1], channels[i]))
                hw = hw // 4
            for _ in range(blocks[i]):
                self.backbone.append(
                    Seq(
                        Grapher(
                            channels[i],
                            num_knn[idx],
                            min(idx // 4 + 1, max_dilation),
                            conv,
                            act,
                            norm,
                            bias,
                            stochastic,
                            epsilon,
                            reduce_ratios[i],
                            n=hw,
                            drop_path=drop_path_rate[idx],
                            relative_pos=True,
                        ),
                        FFN(channels[i], channels[i] * 4, act=act, drop_path=drop_path_rate[idx]),
                    )
                )
                idx += 1

        self.backbone = Seq(*self.backbone)

        self.prediction = Seq(
            nn.Conv2d(channels[-1], 1024, 1, bias=True),
            nn.BatchNorm2d(1024),
            act_layer(act),
            nn.Dropout(opt.dropout),
            nn.Conv2d(1024, opt.n_classes, 1, bias=True),
        )
        self.model_init()

    def model_init(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                m.weight.requires_grad = True
                if m.bias is not None:
                    m.bias.data.zero_()
                    m.bias.requires_grad = True

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = self.stem(inputs) + self.pos_embed
        for layer in self.backbone:
            x = layer(x)

        x = F.adaptive_avg_pool2d(x, 1)
        return self.prediction(x).squeeze(-1).squeeze(-1)


@register_model
def pvig_ti_224_gelu(pretrained: bool = False, **kwargs: Any) -> DeepGCN:
    class OptInit:
        def __init__(self, num_classes: int = 6, drop_path_rate: float = 0.3, **kwargs: Any):
            self.k = 9
            self.conv = "mr"
            self.act = "gelu"
            self.norm = "batch"
            self.bias = True
            self.dropout = 0.3
            self.use_dilation = True
            self.epsilon = 0.2
            self.use_stochastic = False
            self.drop_path = drop_path_rate
            self.blocks = [2, 2, 6, 2]
            self.channels = [80, 160, 400, 640]
            self.n_classes = num_classes
            self.emb_dims = 1024

    opt = OptInit(**kwargs)
    model = DeepGCN(opt)
    model.default_cfg = default_cfgs["vig_224_gelu"]
    return model


class Split(datasets.ImageFolder):
    def __init__(self, root: str, xml_list: list[str], transform=None, target_transform=None,
                 loader=datasets.folder.default_loader):
        super().__init__(root, transform, target_transform, loader)
        self.file_list: dict[str, np.ndarray] = {}
        self.num_classes = 6
        for file_name in xml_list:
            last_dot_idx = file_name.rfind(".")
            f_name_idx = file_name.rfind("/")
            root_path = file_name[f_name_idx + 1: last_dot_idx]
            tree = ElementTree.parse(file_name)
            root_node = tree.getroot()
            for defect in root_node:
                crop_name = list(defect.attrib.values())[0]
                target = self.multi_target(defect)
                self.file_list[os.path.join(root_path, crop_name)] = target

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        image_batch = super().__getitem__(index)[0]
        image_name = self.imgs[index][0]
        f_name_idx = image_name.rfind("/")
        f_dir_idx = image_name[: f_name_idx].rfind("/")
        de_lim = image_name.rfind("_-_")
        file_type = image_name.rfind(".")
        if de_lim != -1:
            name = image_name[f_dir_idx + 1: de_lim] + image_name[file_type:]
        else:
            name = image_name[f_dir_idx + 1:]
        return image_batch, torch.from_numpy(self.file_list[name])

    def get_class_names(self, xml_list: list[str]) -> list[str]:
        class_names = []
        for file_name in xml_list:
            tree = ElementTree.parse(file_name)
            root_node = tree.getroot()
            for defect in root_node:
                class_name = list(defect.attrib.keys())[0]
                class_names.append(class_name)
        return class_names

    def multi_target(self, defect: ElementTree.Element) -> np.ndarray:
        out = np.zeros(self.num_classes, dtype=np.float32)
        for i in range(self.num_classes):
            if defect[i].text == "1":
                out[i] = 1.0
        return out


class CODEBRIM:
    """CODEBRIM dataset wrapper with train/val/test loaders."""

    def __init__(self, is_gpu: bool, args: argparse.Namespace):
        self.num_classes = 6
        self.dataset_path = args.dataset_path
        self.dataset_xml_list = [
            os.path.join(args.dataset_path, "metadata/defects.xml"),
            os.path.join(args.dataset_path, "metadata/background.xml"),
        ]
        self.train_set, self.val_set, self.test_set = self.get_dataset(args.patch_size)
        self.train_loader, self.val_loader, self.test_loader = self.get_dataset_loader(
            args.batch_size, args.workers, is_gpu
        )

    def get_dataset(self, patch_size: int):
        train_set = Split(
            os.path.join(self.dataset_path, "train"),
            self.dataset_xml_list,
            transform=transforms.Compose(
                [
                    transforms.Resize(patch_size),
                    transforms.RandomCrop(patch_size),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                ]
            ),
        )
        val_set = Split(
            os.path.join(self.dataset_path, "val"),
            self.dataset_xml_list,
            transform=transforms.Compose(
                [
                    transforms.Resize(patch_size),
                    transforms.CenterCrop(patch_size),
                    transforms.ToTensor(),
                ]
            ),
        )
        test_set = Split(
            os.path.join(self.dataset_path, "test"),
            self.dataset_xml_list,
            transform=transforms.Compose(
                [
                    transforms.Resize(patch_size),
                    transforms.CenterCrop(patch_size),
                    transforms.ToTensor(),
                ]
            ),
        )
        return train_set, val_set, test_set

    def get_dataset_loader(self, batch_size: int, workers: int, is_gpu: bool):
        train_loader = torch.utils.data.DataLoader(
            self.train_set, num_workers=workers, batch_size=batch_size, shuffle=True, pin_memory=is_gpu
        )
        val_loader = torch.utils.data.DataLoader(
            self.val_set, num_workers=workers, batch_size=batch_size, shuffle=False, pin_memory=is_gpu
        )
        test_loader = torch.utils.data.DataLoader(
            self.test_set, num_workers=workers, batch_size=batch_size, shuffle=False, pin_memory=is_gpu
        )

        return train_loader, val_loader, test_loader


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the GNN vision model")
    parser.add_argument("--dataset-path", required=True, help="Path to CODEBRIM dataset root")
    parser.add_argument("--patch-size", type=int, default=224, help="Image patch size")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--workers", type=int, default=4, help="Data loader workers")
    parser.add_argument("--epochs", type=int, default=400, help="Number of epochs")
    parser.add_argument("--num-classes", type=int, default=6, help="Number of classes")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-3, help="Weight decay")
    parser.add_argument("--drop-path-rate", type=float, default=0.3, help="Drop path rate")
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run on (cuda or cpu)",
    )
    parser.add_argument(
        "--checkpoint-path",
        default="model_checkpoint.pth",
        help="Path to save the trained checkpoint",
    )
    return parser.parse_args()


def evaluate(model: nn.Module, data_loader: torch.utils.data.DataLoader, device: torch.device,
             criterion: nn.Module) -> tuple[float, float]:
    model.eval()
    correct_predictions = 0
    total_predictions = 0
    running_loss = 0.0
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            predicted = (torch.sigmoid(outputs) > 0.5).byte()
            correct_predictions += (predicted == labels.byte()).all(dim=1).sum().item()
            total_predictions += labels.size(0)
    avg_loss = running_loss / max(len(data_loader), 1)
    accuracy = 100 * correct_predictions / max(total_predictions, 1)
    return avg_loss, accuracy


def train(model: nn.Module, dataset: CODEBRIM, args: argparse.Namespace, device: torch.device) -> None:
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        factor=0.5,
        patience=5,
        threshold=0.001,
        verbose=True,
        min_lr=1e-5,
        threshold_mode="abs",
    )

    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        correct_predictions = 0
        total_predictions = 0

        for inputs, labels in dataset.train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            predicted = (torch.sigmoid(outputs) > 0.5).byte()
            correct_predictions += (predicted == labels.byte()).all(dim=1).sum().item()
            total_predictions += labels.size(0)

        epoch_loss = running_loss / max(len(dataset.train_loader), 1)
        epoch_accuracy = 100 * correct_predictions / max(total_predictions, 1)
        val_loss, val_accuracy = evaluate(model, dataset.val_loader, device, criterion)
        scheduler.step(val_loss)

        print(
            f"Epoch {epoch + 1}/{args.epochs} "
            f"- Loss: {epoch_loss:.4f} "
            f"- Train Acc: {epoch_accuracy:.2f}% "
            f"- Val Loss: {val_loss:.4f} "
            f"- Val Acc: {val_accuracy:.2f}%"
        )

    model_to_save = model.module if isinstance(model, nn.DataParallel) else model
    torch.save(
        {
            "model_state_dict": model_to_save.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        },
        args.checkpoint_path,
    )
    print(f"Checkpoint saved to {args.checkpoint_path}")


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    is_gpu = device.type == "cuda"
    dataset = CODEBRIM(is_gpu, args)
    model = pvig_ti_224_gelu(num_classes=args.num_classes, drop_path_rate=args.drop_path_rate).to(device)
    if torch.cuda.device_count() > 1 and device.type == "cuda":
        model = nn.DataParallel(model)
    train(model, dataset, args, device)


if __name__ == "__main__":
    main()
