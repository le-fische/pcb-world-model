"""Pure-PyTorch 4-layer GATv2 Graph Attention Encoder for PCB Netlist Hypergraphs."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
import torch.nn.functional as F

if TYPE_CHECKING:
    from env.config import BoardConfig
    from env.sexpr_parser import NetlistGraph


@dataclass
class GraphTensorBatch:
    """Tensorized Netlist Hypergraph inputs for MacroPlacerGNN."""

    comp_ids: list[str]
    node_feats: torch.Tensor  # [N, 7]
    edge_index: torch.Tensor  # [2, E]
    edge_attr: torch.Tensor   # [E, 3]
    sizes_mm: torch.Tensor    # [N, 2]
    locked_mask: torch.Tensor # [N] bool
    locked_xy: torch.Tensor   # [N, 2]
    locked_theta: torch.Tensor  # [N] int64


def netlist_to_tensors(
    netlist_graph: NetlistGraph,
    board_config: BoardConfig,
    device: torch.device | str = "cpu",
) -> GraphTensorBatch:
    """Convert a NetlistGraph into component node features and hyperedge-expanded graph tensors.

    Node features [7]:
        `[width_norm, height_norm, area_norm, num_pins_norm, is_ic, is_decoupling_cap, is_connector]`
    Edge features [3]:
        `[net_fanout_norm, is_power_ground, weight]`
    """
    comp_ids = sorted(netlist_graph.components.keys())
    comp_idx = {cid: i for i, cid in enumerate(comp_ids)}
    bw = max(1.0, netlist_graph.width_mm)
    bh = max(1.0, netlist_graph.height_mm)

    node_rows: list[list[float]] = []
    sizes: list[list[float]] = []
    locked_mask: list[bool] = []
    locked_xy: list[list[float]] = []
    locked_theta: list[int] = []

    for cid in comp_ids:
        c = netlist_graph.components[cid]
        w_norm = c.width / bw
        h_norm = c.height / bh
        area_norm = (c.width * c.height) / (bw * bh)
        pins_norm = len(c.pads) / 16.0
        is_ic = 1.0 if c.comp_type == "ic" else 0.0
        is_cap = 1.0 if c.comp_type == "cap" else 0.0
        is_conn = 1.0 if c.comp_type == "connector" else 0.0
        node_rows.append([w_norm, h_norm, area_norm, pins_norm, is_ic, is_cap, is_conn])
        sizes.append([c.width, c.height])
        locked_mask.append(bool(c.is_locked))
        locked_xy.append([c.x, c.y])
        locked_theta.append(int(c.theta) % 360)

    src_list: list[int] = []
    dst_list: list[int] = []
    edge_rows: list[list[float]] = []

    # Expand each net hyperedge into clique connections with edge attributes
    for nid in netlist_graph.routable_net_ids:
        net = netlist_graph.nets[nid]
        connected_comps = sorted({comp_id for comp_id, _ in net.pins if comp_id in comp_idx})
        if len(connected_comps) < 2:
            continue
        fanout_norm = min(1.0, len(net.pins) / 8.0)
        is_pwr = 1.0 if net.is_power_ground else 0.0
        weight = float(net.weight)
        for i in range(len(connected_comps)):
            for j in range(len(connected_comps)):
                if i == j:
                    continue
                u = comp_idx[connected_comps[i]]
                v = comp_idx[connected_comps[j]]
                src_list.append(u)
                dst_list.append(v)
                edge_rows.append([fanout_norm, is_pwr, weight])

    # Add self-loops so every node always receives attention
    for i in range(len(comp_ids)):
        src_list.append(i)
        dst_list.append(i)
        edge_rows.append([0.0, 0.0, 1.0])

    return GraphTensorBatch(
        comp_ids=comp_ids,
        node_feats=torch.tensor(node_rows, dtype=torch.float32, device=device),
        edge_index=torch.tensor([src_list, dst_list], dtype=torch.long, device=device),
        edge_attr=torch.tensor(edge_rows, dtype=torch.float32, device=device),
        sizes_mm=torch.tensor(sizes, dtype=torch.float32, device=device),
        locked_mask=torch.tensor(locked_mask, dtype=torch.bool, device=device),
        locked_xy=torch.tensor(locked_xy, dtype=torch.float32, device=device),
        locked_theta=torch.tensor(locked_theta, dtype=torch.long, device=device),
    )


class GATv2Layer(nn.Module):
    """Dynamic Graph Attention v2 layer supporting multi-head edge-conditioned attention."""

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        edge_dim: int = 3,
        heads: int = 4,
        negative_slope: float = 0.2,
    ) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.heads = heads
        self.head_dim = max(1, out_dim // heads)
        proj_dim = self.heads * self.head_dim

        self.w_left = nn.Linear(in_dim, proj_dim, bias=False)
        self.w_right = nn.Linear(in_dim, proj_dim, bias=False)
        self.w_edge = nn.Linear(edge_dim, proj_dim, bias=False)
        self.attn = nn.Parameter(torch.empty(1, heads, self.head_dim))
        self.bias = nn.Parameter(torch.zeros(proj_dim))
        self.out_proj = (
            nn.Linear(proj_dim, out_dim) if proj_dim != out_dim else nn.Identity()
        )
        self.negative_slope = negative_slope
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.w_left.weight)
        nn.init.xavier_uniform_(self.w_right.weight)
        nn.init.xavier_uniform_(self.w_edge.weight)
        nn.init.xavier_uniform_(self.attn)
        nn.init.zeros_(self.bias)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
    ) -> torch.Tensor:
        """Compute GATv2 message passing over `[N, in_dim]` node embeddings."""
        num_nodes = x.size(0)
        src, dst = edge_index[0], edge_index[1]

        h_l = self.w_left(x).view(num_nodes, self.heads, self.head_dim)
        h_r = self.w_right(x).view(num_nodes, self.heads, self.head_dim)
        h_e = self.w_edge(edge_attr).view(-1, self.heads, self.head_dim)

        # GATv2 dynamic attention: a^T LeakyReLU(W_l h_dst + W_r h_src + W_e e_ij)
        pair_feat = F.leaky_relu(
            h_l[dst] + h_r[src] + h_e, negative_slope=self.negative_slope
        )
        logits = (pair_feat * self.attn).sum(dim=-1)  # [E, heads]

        # Numerically stable segment softmax over dst nodes
        max_logits = torch.full(
            (num_nodes, self.heads), -1e9, device=x.device, dtype=x.dtype
        )
        max_logits.scatter_reduce_(
            0,
            dst.unsqueeze(-1).expand_as(logits),
            logits,
            reduce="amax",
            include_self=True,
        )
        exp_logits = torch.exp(logits - max_logits[dst])
        sum_exp = torch.zeros(
            (num_nodes, self.heads), device=x.device, dtype=x.dtype
        )
        sum_exp.scatter_add_(0, dst.unsqueeze(-1).expand_as(exp_logits), exp_logits)
        alpha = exp_logits / (sum_exp[dst] + 1e-9)  # [E, heads]

        messages = h_r[src] * alpha.unsqueeze(-1)  # [E, heads, head_dim]
        out = torch.zeros(
            (num_nodes, self.heads, self.head_dim), device=x.device, dtype=x.dtype
        )
        out.scatter_add_(
            0,
            dst.view(-1, 1, 1).expand_as(messages),
            messages,
        )
        out_flat = out.reshape(num_nodes, self.heads * self.head_dim) + self.bias
        return self.out_proj(out_flat)


class PlacerGATv2Encoder(nn.Module):
    """4-layer GATv2Conv encoder with residual connections and LayerNorm."""

    def __init__(
        self,
        node_dim: int = 7,
        edge_dim: int = 3,
        hidden_dim: int = 64,
        num_layers: int = 4,
        heads: int = 4,
    ) -> None:
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(node_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
        )
        self.layers = nn.ModuleList(
            [
                GATv2Layer(hidden_dim, hidden_dim, edge_dim=edge_dim, heads=heads)
                for _ in range(num_layers)
            ]
        )
        self.norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(num_layers)])

    def forward(
        self,
        node_feats: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
    ) -> torch.Tensor:
        """Encode component node features into contextual embeddings `[N, hidden_dim]`."""
        h = self.input_proj(node_feats)
        for gat, norm in zip(self.layers, self.norms):
            delta = gat(h, edge_index, edge_attr)
            h = norm(h + F.silu(delta))
        return h
