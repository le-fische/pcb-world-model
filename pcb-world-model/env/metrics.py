"""Routing and placement metrics: HPWL, MST wirelength, congestion, and net ranking."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import networkx as nx
import numpy as np

if TYPE_CHECKING:
    from env.board_state import BoardState
    from env.sexpr_parser import NetlistGraph


def compute_net_mst_pairs(
    netlist_graph: NetlistGraph,
    net_id: int,
    placement: dict[str, tuple[float, float, int]] | None = None,
) -> list[tuple[tuple[float, float, int], tuple[float, float, int]]]:
    """Decompose a multi-pin net into 2-pin Minimum Spanning Tree segments.

    Each pin endpoint is `(x_mm, y_mm, layer_mask)`.
    """
    net = netlist_graph.nets.get(net_id)
    if net is None or len(net.pins) < 2:
        return []

    pin_coords: list[tuple[float, float, int]] = []
    for comp_id, pad_id in net.pins:
        comp = netlist_graph.components.get(comp_id)
        if comp is None:
            continue
        pad = next((p for p in comp.pads if p.pad_id == pad_id), None)
        if pad is None:
            continue
        if placement is not None and comp_id in placement:
            cx, cy, ctheta = placement[comp_id]
            rad = math.radians(ctheta % 360)
            wx = cx + (pad.rel_x * math.cos(rad) - pad.rel_y * math.sin(rad))
            wy = cy + (pad.rel_x * math.sin(rad) + pad.rel_y * math.cos(rad))
        else:
            wx, wy = comp.pad_world_pos(pad)
        pin_coords.append((wx, wy, pad.layer_mask))

    if len(pin_coords) < 2:
        return []
    if len(pin_coords) == 2:
        return [(pin_coords[0], pin_coords[1])]

    g = nx.Graph()
    for i in range(len(pin_coords)):
        g.add_node(i)
    for i in range(len(pin_coords)):
        x1, y1, _ = pin_coords[i]
        for j in range(i + 1, len(pin_coords)):
            x2, y2, _ = pin_coords[j]
            dist = math.hypot(x2 - x1, y2 - y1)
            g.add_edge(i, j, weight=dist)

    mst = nx.minimum_spanning_tree(g, weight="weight")
    pairs: list[tuple[tuple[float, float, int], tuple[float, float, int]]] = []
    for u, v in mst.edges():
        pairs.append((pin_coords[u], pin_coords[v]))
    return pairs


def compute_hpwl(
    netlist_graph: NetlistGraph,
    placement: dict[str, tuple[float, float, int]] | None = None,
) -> float:
    """Calculate total weighted Half-Perimeter Wirelength (HPWL) in mm."""
    total_hpwl = 0.0
    for net_id in netlist_graph.routable_net_ids:
        net = netlist_graph.nets[net_id]
        xs: list[float] = []
        ys: list[float] = []
        for comp_id, pad_id in net.pins:
            comp = netlist_graph.components.get(comp_id)
            if comp is None:
                continue
            pad = next((p for p in comp.pads if p.pad_id == pad_id), None)
            if pad is None:
                continue
            if placement is not None and comp_id in placement:
                cx, cy, ctheta = placement[comp_id]
                rad = math.radians(ctheta % 360)
                wx = cx + (pad.rel_x * math.cos(rad) - pad.rel_y * math.sin(rad))
                wy = cy + (pad.rel_x * math.sin(rad) + pad.rel_y * math.cos(rad))
            else:
                wx, wy = comp.pad_world_pos(pad)
            xs.append(wx)
            ys.append(wy)
        if len(xs) >= 2:
            total_hpwl += net.weight * ((max(xs) - min(xs)) + (max(ys) - min(ys)))
    return float(total_hpwl)


def compute_mst_wirelength(
    netlist_graph: NetlistGraph,
    placement: dict[str, tuple[float, float, int]] | None = None,
) -> float:
    """Calculate total Minimum Spanning Tree wirelength across all routable nets in mm."""
    total = 0.0
    for net_id in netlist_graph.routable_net_ids:
        for (x1, y1, _), (x2, y2, _) in compute_net_mst_pairs(
            netlist_graph, net_id, placement
        ):
            total += math.hypot(x2 - x1, y2 - y1)
    return float(total)


def compute_routed_wirelength(board_state: BoardState) -> float:
    """Compute actual routed copper segment wirelength in mm."""
    total = 0.0
    for segs in board_state.routed_traces.values():
        for seg in segs:
            total += math.hypot(seg.x2_mm - seg.x1_mm, seg.y2_mm - seg.y1_mm)
    return float(total)


def rank_nets_by_criticality(netlist_graph: NetlistGraph) -> list[int]:
    """Order nets for sequential routing: short signals & low-fanout critical nets first, high-fanout rails balanced."""
    scored: list[tuple[tuple[float, float, int], int]] = []
    for net_id in netlist_graph.routable_net_ids:
        net = netlist_graph.nets[net_id]
        pairs = compute_net_mst_pairs(netlist_graph, net_id)
        mst_len = sum(math.hypot(b[0] - a[0], b[1] - a[1]) for a, b in pairs)
        fanout = len(net.pins)
        # Route short, tight 2-3 pin signal nets first so high-fanout power/GND buses can weave around via L2
        is_high_fanout_rail = 1 if (net.is_power_ground and fanout >= 4) else 0
        priority_score = (is_high_fanout_rail, mst_len + 1.5 * fanout, net_id)
        scored.append((priority_score, net_id))
    scored.sort(key=lambda item: item[0])
    return [nid for _, nid in scored]


def compute_rudy_congestion_target(board_state: BoardState) -> np.ndarray:
    """Compute Rectangular Uniform Wire Density (RUDY) + actual trace density map [2, H, W].

    Used both as ground-truth supervision for training WorldModelUNet and as a calibration metric.
    """
    H, W = board_state.H, board_state.W
    res = board_state.config.grid_resolution_mm
    rudy = np.zeros((2, H, W), dtype=np.float32)

    for net_id in board_state.netlist.routable_net_ids:
        pairs = compute_net_mst_pairs(board_state.netlist, net_id)
        for (x1, y1, _), (x2, y2, _) in pairs:
            gx1 = max(0, min(W - 1, int(round(x1 / res))))
            gy1 = max(0, min(H - 1, int(round(y1 / res))))
            gx2 = max(0, min(W - 1, int(round(x2 / res))))
            gy2 = max(0, min(H - 1, int(round(y2 / res))))
            x_lo, x_hi = min(gx1, gx2), max(gx1, gx2)
            y_lo, y_hi = min(gy1, gy2), max(gy1, gy2)
            pad_cells = 3
            x_lo = max(0, x_lo - pad_cells)
            x_hi = min(W - 1, x_hi + pad_cells)
            y_lo = max(0, y_lo - pad_cells)
            y_hi = min(H - 1, y_hi + pad_cells)
            area_w = max(1, x_hi - x_lo + 1)
            area_h = max(1, y_hi - y_lo + 1)
            density = min(0.5, (area_w + area_h) / float(area_w * area_h + 4.0))
            rudy[0, y_lo : y_hi + 1, x_lo : x_hi + 1] += density
            rudy[1, y_lo : y_hi + 1, x_lo : x_hi + 1] += density * 0.65

    # Incorporate actual copper occupancy & pad crowding
    l1_occ = (board_state.occ_L1 > 0).astype(np.float32)
    l2_occ = (board_state.occ_L2 > 0).astype(np.float32)
    rudy[0] += 0.45 * l1_occ
    rudy[1] += 0.45 * l2_occ

    max_val = max(1.0, float(rudy.max()))
    if max_val > 1.0:
        rudy = np.clip(rudy / max_val, 0.0, 1.0)
    return rudy
