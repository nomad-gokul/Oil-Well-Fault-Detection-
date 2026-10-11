"""PyTorch sequence classifiers over windows of shape [batch, time, channels]."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn


class ResidualBlock(nn.Module):
    """Two same-padded conv layers with a skip connection, then stride-2 downsampling."""

    def __init__(self, c_in: int, c_out: int, kernel_size: int, dropout: float) -> None:
        super().__init__()
        pad = kernel_size // 2
        self.body = nn.Sequential(
            nn.Conv1d(c_in, c_out, kernel_size, padding=pad, bias=False),
            nn.BatchNorm1d(c_out),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(c_out, c_out, kernel_size, padding=pad, bias=False),
            nn.BatchNorm1d(c_out),
        )
        self.skip = nn.Conv1d(c_in, c_out, 1, bias=False) if c_in != c_out else nn.Identity()
        self.out = nn.Sequential(nn.GELU(), nn.MaxPool1d(2, ceil_mode=True))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.out(self.body(x) + self.skip(x))


class CNN1D(nn.Module):
    """Residual 1D-CNN with global average + max pooling."""

    def __init__(
        self, n_channels: int, n_classes: int, widths: list[int], kernel_size: int, dropout: float
    ) -> None:
        super().__init__()
        stages, c = [], n_channels
        for w in widths:
            stages.append(ResidualBlock(c, w, kernel_size, dropout))
            c = w
        self.features = nn.Sequential(*stages)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(2 * c, n_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.features(x.transpose(1, 2))  # [B, C, T]
        return self.head(torch.cat([h.mean(dim=2), h.amax(dim=2)], dim=1))


class GRUNet(nn.Module):
    """Average-pool the time axis, then a GRU; classify from the last hidden state."""

    def __init__(
        self, n_channels: int, n_classes: int, hidden: int, layers: int, pool: int, dropout: float
    ) -> None:
        super().__init__()
        self.pool = nn.AvgPool1d(pool, ceil_mode=True)
        self.gru = nn.GRU(n_channels, hidden, layers, batch_first=True,
                          dropout=dropout if layers > 1 else 0.0)  # fmt: skip
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden, n_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.pool(x.transpose(1, 2)).transpose(1, 2)  # [B, T/pool, C]
        _, last = self.gru(h)
        return self.head(last[-1])


def build_model(config: dict[str, Any], n_channels: int, n_classes: int) -> nn.Module:
    """Instantiate the architecture named by ``config['arch']``."""
    arch = config["arch"]
    if arch == "cnn":
        return CNN1D(n_channels, n_classes, **config["cnn"])
    if arch == "gru":
        return GRUNet(n_channels, n_classes, **config["gru"])
    raise ValueError(f"Unknown architecture: {arch}")
