"""Macro-Placement GNN pose head, differentiable physics loss, and candidate sampling."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch
import torch.nn as nn

from models.placer.gnn_encoder import GraphTensorBatch, PlacerGATv2Encoder, netlist_to_tensors

if TYPE_CHECKING:
    from env.config import BoardConfig
    from env.sexpr_parser import NetlistGraph


def _nearest_cardinal_theta(sin_val: float, cos_val: float) -> int:
    """Snap continuous (sin, cos) angle vector to {0, 90, 180, 270} degrees."""
    deg = math.degrees(math.atan2(sin_val, cos_val)) % 360.0
    return int(round(deg / 90.0) * 90) % 360


class MacroPlacerGNN(nn.Module):
    """End-to-end GATv2 Macro-Placement composer predicting (x, y, sin_theta, cos_theta)."""

    def __init__(
        self,
        node_dim: int = 7,
        edge_dim: int = 3,
        hidden_dim: int = 64,
        num_layers: int = 4,
        heads: int = 4,
    ) -> None:
        super().__init__()
        self.encoder = PlacerGATv2Encoder(
            node_dim=node_dim,
            edge_dim=edge_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            heads=heads,
        )
        self.pose_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 4),  # [dx_logit, dy_logit, sin_theta, cos_theta]
        )

    def forward(
        self,
        batch: GraphTensorBatch,
        board_config: BoardConfig,
        noise_std: float = 0.0,
        generator: torch.Generator | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Predict continuous component centers `[N, 2]` in mm and `[N, 2]` `(sin, cos)`."""
        h = self.encoder(batch.node_feats, batch.edge_index, batch.edge_attr)
        if noise_std > 0.0:
            eps = torch.randn(h.shape, device=h.device, dtype=h.dtype, generator=generator)
            h = h + noise_std * eps

        raw = self.pose_head(h)
        margin = board_config.edge_margin_mm
        bw = board_config.board_width_mm
        bh = board_config.board_height_mm

        # Map sigmoid logits into safe usable board interior [min_c, max_c]
        half_w = batch.sizes_mm[:, 0] * 0.5
        half_h = batch.sizes_mm[:, 1] * 0.5
        min_x = margin + half_w
        max_x = torch.clamp(bw - margin - half_w, min=min_x + 0.5)
        min_y = margin + half_h
        max_y = torch.clamp(bh - margin - half_h, min=min_y + 0.5)

        u = torch.sigmoid(raw[:, 0])
        v = torch.sigmoid(raw[:, 1])
        pred_x = min_x + u * (max_x - min_x)
        pred_y = min_y + v * (max_y - min_y)
        pred_xy = torch.stack([pred_x, pred_y], dim=-1)

        # Anchor locked connectors to their exact fixed coordinates
        pred_xy = torch.where(batch.locked_mask.unsqueeze(-1), batch.locked_xy, pred_xy)

        sincos = raw[:, 2:4]
        norm = torch.norm(sincos, dim=-1, keepdim=True).clamp(min=1e-6)
        sincos = sincos / norm
        return pred_xy, sincos

    @torch.no_grad()
    def sample_placements(
        self,
        netlist_graph: NetlistGraph,
        board_config: BoardConfig,
        num_samples: int = 8,
        noise_scale: float = 0.35,
        seed: int = 42,
    ) -> list[dict[str, tuple[float, float, int]]]:
        """Sample `num_samples` diverse candidate macro-placements from the GNN policy."""
        self.eval()
        device = next(self.parameters()).device
        batch = netlist_to_tensors(netlist_graph, board_config, device=device)
        gen = torch.Generator(device=device)
        gen.manual_seed(seed)

        candidates: list[dict[str, tuple[float, float, int]]] = []
        for k in range(num_samples):
            # Sample k=0 deterministically, k>0 with progressively stronger exploration
            std = 0.0 if k == 0 else noise_scale * (0.6 + 0.25 * k)
            pred_xy, sincos = self.forward(batch, board_config, noise_std=std, generator=gen)

            # Inject targeted spatial diversity across candidates so World Model Critic
            # sees distinct topological configurations
            if k > 0:
                jitter = torch.randn(
                    pred_xy.shape, device=device, dtype=pred_xy.dtype, generator=gen
                ) * (2.5 + 1.1 * k)
                pred_xy = torch.where(
                    batch.locked_mask.unsqueeze(-1), batch.locked_xy, pred_xy + jitter
                )

            placement: dict[str, tuple[float, float, int]] = {}
            for idx, cid in enumerate(batch.comp_ids):
                comp = netlist_graph.components[cid]
                if comp.is_locked:
                    placement[cid] = (float(comp.x), float(comp.y), int(comp.theta) % 360)
                else:
                    x_mm = float(pred_xy[idx, 0].item())
                    y_mm = float(pred_xy[idx, 1].item())
                    theta = _nearest_cardinal_theta(
                        float(sincos[idx, 0].item()), float(sincos[idx, 1].item())
                    )
                    if k % 2 == 1 and comp.comp_type in ("cap", "res", "diode"):
                        theta = (theta + 90 * ((k // 2) % 4)) % 360
                    placement[cid] = (x_mm, y_mm, theta)
            candidates.append(placement)
        return candidates


def compute_placement_loss(
    pred_xy: torch.Tensor,
    pred_sincos: torch.Tensor,
    batch: GraphTensorBatch,
    netlist_graph: NetlistGraph,
    board_config: BoardConfig,
    lambda_hpwl: float = 1.0,
    lambda_overlap: float = 10.0,
    lambda_bound: float = 20.0,
    routing_channel_pad_mm: float = 1.4,
    smooth_temp: float = 1.5,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Compute differentiable placement physics loss:

    L_place = lambda_hpwl * SmoothHPWL + lambda_overlap * OverlapArea + lambda_bound * OutOfBounds.
    """
    comp_idx = {cid: i for i, cid in enumerate(batch.comp_ids)}
    device = pred_xy.device

    # 1. SmoothHPWL using log-sum-exp over rotated pad positions
    sin_t = pred_sincos[:, 0]
    cos_t = pred_sincos[:, 1]
    hpwl_loss = torch.tensor(0.0, device=device)

    for nid in netlist_graph.routable_net_ids:
        net = netlist_graph.nets[nid]
        px_list: list[torch.Tensor] = []
        py_list: list[torch.Tensor] = []
        for comp_id, pad_id in net.pins:
            idx = comp_idx.get(comp_id)
            if idx is None:
                continue
            comp = netlist_graph.components[comp_id]
            pad = next((p for p in comp.pads if p.pad_id == pad_id), None)
            if pad is None:
                continue
            if comp.is_locked:
                wx, wy = comp.pad_world_pos(pad)
                px_list.append(torch.tensor(wx, device=device, dtype=pred_xy.dtype))
                py_list.append(torch.tensor(wy, device=device, dtype=pred_xy.dtype))
            else:
                rx = torch.tensor(pad.rel_x, device=device, dtype=pred_xy.dtype)
                ry = torch.tensor(pad.rel_y, device=device, dtype=pred_xy.dtype)
                wx = pred_xy[idx, 0] + rx * cos_t[idx] - ry * sin_t[idx]
                wy = pred_xy[idx, 1] + rx * sin_t[idx] + ry * cos_t[idx]
                px_list.append(wx)
                py_list.append(wy)

        if len(px_list) >= 2:
            xs = torch.stack(px_list)
            ys = torch.stack(py_list)
            max_x = smooth_temp * torch.logsumexp(xs / smooth_temp, dim=0)
            min_x = -smooth_temp * torch.logsumexp(-xs / smooth_temp, dim=0)
            max_y = smooth_temp * torch.logsumexp(ys / smooth_temp, dim=0)
            min_y = -smooth_temp * torch.logsumexp(-ys / smooth_temp, dim=0)
            hpwl_loss = hpwl_loss + net.weight * ((max_x - min_x) + (max_y - min_y))

    # 2. Pairwise Courtyard OverlapArea (including routing channel padding)
    # Approximate rotated half-extents smoothly using |cos| and |sin|
    abs_cos = torch.abs(cos_t)
    abs_sin = torch.abs(sin_t)
    eff_w = batch.sizes_mm[:, 0] * abs_cos + batch.sizes_mm[:, 1] * abs_sin + routing_channel_pad_mm
    eff_h = batch.sizes_mm[:, 0] * abs_sin + batch.sizes_mm[:, 1] * abs_cos + routing_channel_pad_mm

    x0 = pred_xy[:, 0] - eff_w * 0.5
    x1 = pred_xy[:, 0] + eff_w * 0.5
    y0 = pred_xy[:, 1] - eff_h * 0.5
    y1 = pred_xy[:, 1] + eff_h * 0.5

    overlap_x = torch.relu(torch.minimum(x1.unsqueeze(1), x1.unsqueeze(0)) - torch.maximum(x0.unsqueeze(1), x0.unsqueeze(0)))
    overlap_y = torch.relu(torch.minimum(y1.unsqueeze(1), y1.unsqueeze(0)) - torch.maximum(y0.unsqueeze(1), y0.unsqueeze(0)))
    n_comp = pred_xy.size(0)
    upper_mask = torch.triu(torch.ones((n_comp, n_comp), device=device, dtype=torch.bool), diagonal=1)
    overlap_area = (overlap_x * overlap_y)[upper_mask].sum()

    # 3. Out-Of-Bounds penalty
    margin = board_config.edge_margin_mm
    bw = board_config.board_width_mm
    bh = board_config.board_height_mm
    oob = (
        torch.relu(margin - x0)
        + torch.relu(x1 - (bw - margin))
        + torch.relu(margin - y0)
        + torch.relu(y1 - (bh - margin))
    ).sum()

    total = lambda_hpwl * hpwl_loss + lambda_overlap * overlap_area + lambda_bound * oob
    stats = {
        "total_loss": float(total.item()),
        "smooth_hpwl": float(hpwl_loss.item()),
        "overlap_area": float(overlap_area.item()),
        "oob_penalty": float(oob.item()),
    }
    return total, stats
