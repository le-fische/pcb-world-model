"""Latent World Model U-Net with Bottleneck Self-Attention and Dual Critic/Congestion Heads."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.world.state_encoder import ResidualConvBlock, StateEncoder


class BottleneckSelfAttention(nn.Module):
    """2D spatial Multi-Head Self-Attention block over downsampled bottleneck feature maps."""

    def __init__(self, channels: int, num_heads: int = 4) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(channels)
        self.attn = nn.MultiheadAttention(
            embed_dim=channels, num_heads=num_heads, batch_first=True
        )
        self.ffn = nn.Sequential(
            nn.LayerNorm(channels),
            nn.Linear(channels, channels * 2),
            nn.SiLU(),
            nn.Linear(channels * 2, channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply global spatial self-attention across `[B, C, H, W]` tokens."""
        b, c, h, w = x.shape
        # Lightweight 2D coordinate positional signal
        ys = torch.linspace(-1.0, 1.0, h, device=x.device, dtype=x.dtype)
        xs = torch.linspace(-1.0, 1.0, w, device=x.device, dtype=x.dtype)
        gy, gx = torch.meshgrid(ys, xs, indexing="ij")
        pos = torch.zeros_like(x)
        if c >= 2:
            pos[:, 0] = gx
            pos[:, 1] = gy

        tokens = (x + 0.1 * pos).flatten(2).transpose(1, 2)  # [B, H*W, C]
        normed = self.norm(tokens)
        attn_out, _ = self.attn(normed, normed, normed, need_weights=False)
        tokens = tokens + attn_out
        tokens = tokens + self.ffn(tokens)
        return tokens.transpose(1, 2).reshape(b, c, h, w)


class WorldModelUNet(nn.Module):
    """Spatial World Model predicting future congestion M_{t+1} and Routability V(S_t).

    Heads:
        1. `congestion_head`: Outputs `[B, 2, H, W]` in `[0, 1]` for Layer 1 & Layer 2.
        2. `value_head`: Outputs `[B, 1]` in `[0, 1]` predicting DRC-clean routability.
    """

    def __init__(self, in_channels: int = 7, base_channels: int = 16) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.base_channels = base_channels
        c1 = base_channels
        c2 = base_channels * 2
        c3 = base_channels * 4
        c4 = base_channels * 8

        self.encoder = StateEncoder(in_channels=in_channels, base_channels=base_channels)
        self.attn = BottleneckSelfAttention(c4, num_heads=4)

        self.dec3 = ResidualConvBlock(c4 + c3, c3)
        self.dec2 = ResidualConvBlock(c3 + c2, c2)
        self.dec1 = ResidualConvBlock(c2 + c1, c1)

        # Dual Head 1: Spatial Congestion Heatmap M_{t+1} in [0, 1]^(2 x H x W)
        self.congestion_head = nn.Sequential(
            nn.Conv2d(c1, c1, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(c1, 2, kernel_size=1),
            nn.Sigmoid(),
        )

        # Dual Head 2: Scalar Routability Value V(S_t) in [0, 1]
        self.value_head = nn.Sequential(
            nn.Linear(c4 * 2 + 4, 64),
            nn.SiLU(),
            nn.Linear(64, 32),
            nn.SiLU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

    def forward(self, s_t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass returning `(congestion_map [B, 2, H, W], routability_value [B, 1])`."""
        if s_t.ndim == 3:
            s_t = s_t.unsqueeze(0)

        f1, f2, f3, fb = self.encoder(s_t)
        fb = self.attn(fb)

        # Decoder with exact spatial skip alignment
        u3 = F.interpolate(fb, size=f3.shape[-2:], mode="bilinear", align_corners=False)
        d3 = self.dec3(torch.cat([u3, f3], dim=1))

        u2 = F.interpolate(d3, size=f2.shape[-2:], mode="bilinear", align_corners=False)
        d2 = self.dec2(torch.cat([u2, f2], dim=1))

        u1 = F.interpolate(d2, size=f1.shape[-2:], mode="bilinear", align_corners=False)
        d1 = self.dec1(torch.cat([u1, f1], dim=1))

        congestion = self.congestion_head(d1)

        # Summary physical descriptors appended to bottleneck pooling for calibrated V(S_t)
        avg_pool = fb.mean(dim=(2, 3))
        max_pool = fb.amax(dim=(2, 3))
        courtyard_density = s_t[:, 1].mean(dim=(1, 2)).unsqueeze(1)
        copper_density = 0.5 * (
            s_t[:, 2].mean(dim=(1, 2)).unsqueeze(1)
            + s_t[:, 3].mean(dim=(1, 2)).unsqueeze(1)
        )
        demand_mean = s_t[:, 6].mean(dim=(1, 2)).unsqueeze(1)
        demand_peak = s_t[:, 6].amax(dim=(1, 2)).unsqueeze(1)
        val_feat = torch.cat(
            [avg_pool, max_pool, courtyard_density, copper_density, demand_mean, demand_peak],
            dim=1,
        )
        value = self.value_head(val_feat)
        return congestion, value

    @torch.inference_mode()
    def predict(self, state_tensor: torch.Tensor) -> tuple[np.ndarray, float]:
        """Single forward pass returning `(congestion_map [2, H, W], value)`."""
        self.eval()
        device = next(self.parameters()).device
        x = state_tensor.to(device)
        if x.ndim == 3:
            x = x.unsqueeze(0)
        congestion, value = self.forward(x)
        return (
            congestion.squeeze(0).cpu().numpy().astype(np.float32),
            float(value.squeeze().item()),
        )

    @torch.inference_mode()
    def predict_batch(self, state_tensors: list[torch.Tensor]) -> tuple[np.ndarray, list[float]]:
        """Evaluate a batch of `[7, H, W]` state tensors via optimal device path."""
        if not state_tensors:
            return (np.zeros((0, 2, 0, 0), dtype=np.float32), [])
        self.eval()
        device = next(self.parameters()).device
        if device.type == "mps":
            congs: list[np.ndarray] = []
            vals: list[float] = []
            for s_t in state_tensors:
                c, v = self.predict(s_t)
                congs.append(c)
                vals.append(v)
            return (np.stack(congs, axis=0), vals)
        batch = torch.stack(state_tensors, dim=0).to(device)
        congestion, values = self.forward(batch)
        return (
            congestion.cpu().numpy().astype(np.float32),
            [float(v) for v in values.squeeze(-1).cpu().tolist()],
        )

    @torch.inference_mode()
    def predict_value(self, state_tensor: torch.Tensor) -> float:
        """Predict scalar routability probability V(S_t) in [0, 1]."""
        self.eval()
        device = next(self.parameters()).device
        x = state_tensor.to(device)
        if x.ndim == 3:
            x = x.unsqueeze(0)
        _, value = self.forward(x)
        return float(value.squeeze().item())

    @torch.inference_mode()
    def predict_congestion(self, state_tensor: torch.Tensor) -> np.ndarray:
        """Predict 2-layer congestion heatmap M_{t+1} of shape [2, H, W] in [0, 1]."""
        self.eval()
        device = next(self.parameters()).device
        x = state_tensor.to(device)
        if x.ndim == 3:
            x = x.unsqueeze(0)
        congestion, _ = self.forward(x)
        return congestion.squeeze(0).cpu().numpy().astype(np.float32)
