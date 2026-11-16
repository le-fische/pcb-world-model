"""Multi-layer spatial ResNet-style encoder for PCB BoardState tensor S_t [B, 7, H, W]."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualConvBlock(nn.Module):
    """Residual 3x3 convolutional block with GroupNorm and SiLU activation."""

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        groups_1 = min(8, out_ch)
        while out_ch % groups_1 != 0 and groups_1 > 1:
            groups_1 -= 1
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False)
        self.norm1 = nn.GroupNorm(groups_1, out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False)
        self.norm2 = nn.GroupNorm(groups_1, out_ch)
        self.skip = (
            nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False)
            if in_ch != out_ch
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.skip(x)
        h = F.silu(self.norm1(self.conv1(x)))
        h = self.norm2(self.conv2(h))
        return F.silu(h + residual)


class StateEncoder(nn.Module):
    """Hierarchical spatial encoder extracting multi-scale features from S_t [B, 7, H, W]."""

    def __init__(self, in_channels: int = 7, base_channels: int = 16) -> None:
        super().__init__()
        c1 = base_channels
        c2 = base_channels * 2
        c3 = base_channels * 4
        c4 = base_channels * 8

        self.enc1 = ResidualConvBlock(in_channels, c1)
        self.down1 = nn.Conv2d(c1, c1, kernel_size=3, stride=2, padding=1)

        self.enc2 = ResidualConvBlock(c1, c2)
        self.down2 = nn.Conv2d(c2, c2, kernel_size=3, stride=2, padding=1)

        self.enc3 = ResidualConvBlock(c2, c3)
        self.down3 = nn.Conv2d(c3, c3, kernel_size=3, stride=2, padding=1)

        self.bottleneck = ResidualConvBlock(c3, c4)

    def forward(
        self, s_t: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Encode `[B, 7, H, W]` into `(f1, f2, f3, bottleneck)` multi-scale feature maps."""
        f1 = self.enc1(s_t)
        f2 = self.enc2(self.down1(f1))
        f3 = self.enc3(self.down2(f2))
        fb = self.bottleneck(self.down3(f3))
        return (f1, f2, f3, fb)
