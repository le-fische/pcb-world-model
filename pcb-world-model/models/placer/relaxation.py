"""Differentiable overlap repulsion relaxation, orientation optimization, and grid snapping."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

from env.metrics import compute_hpwl

if TYPE_CHECKING:
    from env.config import BoardConfig
    from env.sexpr_parser import NetlistGraph


def _rects_overlap(
    ax: float, ay: float, aw: float, ah: float,
    bx: float, by: float, bw: float, bh: float,
    pad: float = 0.0,
) -> bool:
    """Check whether two axis-aligned boxes overlap with optional clearance padding."""
    return (
        abs(ax - bx) < 0.5 * (aw + bw) + pad - 1e-4
        and abs(ay - by) < 0.5 * (ah + bh) + pad - 1e-4
    )


def _snap(val: float, grid: float) -> float:
    return round(round(val / grid) * grid, 4)


def relax_and_snap_to_grid(
    placement: dict[str, tuple[float, float, int]],
    netlist_graph: NetlistGraph,
    board_config: BoardConfig,
    iterations: int = 50,
    lr: float = 0.45,
    routing_buffer_mm: float = 1.6,
) -> dict[str, tuple[float, float, int]]:
    """Run 50 iterations of differentiable overlap repulsion and snap to 0.5mm manufacturing grid."""
    comp_ids = sorted(netlist_graph.components.keys())
    comp_idx = {cid: i for i, cid in enumerate(comp_ids)}
    n = len(comp_ids)
    if n == 0:
        return {}

    margin = board_config.edge_margin_mm + 0.5
    bw = board_config.board_width_mm
    bh = board_config.board_height_mm

    # Prepare initial tensors
    init_xy: list[list[float]] = []
    sizes: list[list[float]] = []
    locked: list[bool] = []
    thetas: list[int] = []

    for cid in comp_ids:
        comp = netlist_graph.components[cid]
        if comp.is_locked:
            x, y, th = comp.x, comp.y, int(comp.theta) % 360
        else:
            x, y, th = placement.get(cid, (comp.x, comp.y, comp.theta))
            th = int(round((th % 360) / 90.0) * 90) % 360
        ew, eh = (comp.height, comp.width) if th in (90, 270) else (comp.width, comp.height)
        x = max(margin + ew * 0.5, min(bw - margin - ew * 0.5, float(x)))
        y = max(margin + eh * 0.5, min(bh - margin - eh * 0.5, float(y)))
        init_xy.append([x, y])
        sizes.append([ew, eh])
        locked.append(bool(comp.is_locked))
        thetas.append(th)

    xy_var = torch.tensor(init_xy, dtype=torch.float32, requires_grad=True)
    fixed_xy = torch.tensor(init_xy, dtype=torch.float32)
    size_t = torch.tensor(sizes, dtype=torch.float32)
    locked_t = torch.tensor(locked, dtype=torch.bool)

    # Build component-component net spring pairs for gentle wirelength cohesion
    spring_pairs: list[tuple[int, int, float]] = []
    for nid in netlist_graph.routable_net_ids:
        net = netlist_graph.nets[nid]
        cset = sorted({c for c, _ in net.pins if c in comp_idx})
        for i in range(len(cset)):
            for j in range(i + 1, len(cset)):
                spring_pairs.append((comp_idx[cset[i]], comp_idx[cset[j]], net.weight))

    optimizer = torch.optim.Adam([xy_var], lr=lr)
    upper_mask = torch.triu(torch.ones((n, n), dtype=torch.bool), diagonal=1)

    for step in range(iterations):
        optimizer.zero_grad()
        xy = torch.where(locked_t.unsqueeze(-1), fixed_xy, xy_var)

        eff_w = size_t[:, 0] + routing_buffer_mm
        eff_h = size_t[:, 1] + routing_buffer_mm
        x0 = xy[:, 0] - eff_w * 0.5
        x1 = xy[:, 0] + eff_w * 0.5
        y0 = xy[:, 1] - eff_h * 0.5
        y1 = xy[:, 1] + eff_h * 0.5

        ox = torch.relu(torch.minimum(x1.unsqueeze(1), x1.unsqueeze(0)) - torch.maximum(x0.unsqueeze(1), x0.unsqueeze(0)))
        oy = torch.relu(torch.minimum(y1.unsqueeze(1), y1.unsqueeze(0)) - torch.maximum(y0.unsqueeze(1), y0.unsqueeze(0)))
        overlap_loss = ((ox * oy)[upper_mask]).sum()

        # Repel components gently from exact identical centers in case of exact tie
        dists_sq = ((xy.unsqueeze(1) - xy.unsqueeze(0)) ** 2).sum(dim=-1)
        coincidence_repel = torch.relu(9.0 - dists_sq)[upper_mask].sum() * 0.15

        oob_loss = (
            torch.relu(margin - (xy[:, 0] - size_t[:, 0] * 0.5))
            + torch.relu((xy[:, 0] + size_t[:, 0] * 0.5) - (bw - margin))
            + torch.relu(margin - (xy[:, 1] - size_t[:, 1] * 0.5))
            + torch.relu((xy[:, 1] + size_t[:, 1] * 0.5) - (bh - margin))
        ).sum()

        hpwl_pull = torch.tensor(0.0)
        for u, v, wgt in spring_pairs:
            hpwl_pull = hpwl_pull + wgt * torch.sqrt(((xy[u] - xy[v]) ** 2).sum() + 1e-4)

        # Prioritize overlap legalization heavily over wirelength pull
        loss = 20.0 * overlap_loss + 30.0 * oob_loss + coincidence_repel + 0.08 * hpwl_pull
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        final_xy = torch.where(locked_t.unsqueeze(-1), fixed_xy, xy_var).numpy()

    grid = board_config.snap_grid_mm
    legal: dict[str, tuple[float, float, int]] = {}
    for idx, cid in enumerate(comp_ids):
        comp = netlist_graph.components[cid]
        if comp.is_locked:
            legal[cid] = (float(comp.x), float(comp.y), int(comp.theta) % 360)
            continue
        ew, eh = sizes[idx]
        cx = _snap(float(final_xy[idx, 0]), grid)
        cy = _snap(float(final_xy[idx, 1]), grid)
        cx = max(_snap(margin + ew * 0.5, grid), min(_snap(bw - margin - ew * 0.5, grid), cx))
        cy = max(_snap(margin + eh * 0.5, grid), min(_snap(bh - margin - eh * 0.5, grid), cy))
        legal[cid] = (cx, cy, thetas[idx])

    # Discrete orientation refinement (0, 90, 180, 270) when overlap-free
    for cid in comp_ids:
        comp = netlist_graph.components[cid]
        if comp.is_locked:
            continue
        cur_x, cur_y, best_th = legal[cid]
        best_hpwl = compute_hpwl(netlist_graph, legal)
        for cand_th in (0, 90, 180, 270):
            if cand_th == best_th:
                continue
            ew, eh = (comp.height, comp.width) if cand_th in (90, 270) else (comp.width, comp.height)
            if (
                cur_x - ew * 0.5 < margin - 0.1
                or cur_x + ew * 0.5 > bw - margin + 0.1
                or cur_y - eh * 0.5 < margin - 0.1
                or cur_y + eh * 0.5 > bh - margin + 0.1
            ):
                continue
            collides = False
            for other_id, (ox_c, oy_c, o_th) in legal.items():
                if other_id == cid:
                    continue
                oc = netlist_graph.components[other_id]
                ow, oh = (oc.height, oc.width) if o_th in (90, 270) else (oc.width, oc.height)
                if _rects_overlap(cur_x, cur_y, ew, eh, ox_c, oy_c, ow, oh, pad=1.0):
                    collides = True
                    break
            if collides:
                continue
            trial = dict(legal)
            trial[cid] = (cur_x, cur_y, cand_th)
            trial_hpwl = compute_hpwl(netlist_graph, trial)
            if trial_hpwl + 0.1 < best_hpwl:
                best_hpwl = trial_hpwl
                best_th = cand_th
        legal[cid] = (cur_x, cur_y, best_th)

    # Final deterministic spiral grid search if any pair still overlaps after snapping
    placed_ids = [cid for cid in comp_ids if netlist_graph.components[cid].is_locked]
    unlocked_ids = [cid for cid in comp_ids if not netlist_graph.components[cid].is_locked]
    for cid in unlocked_ids:
        comp = netlist_graph.components[cid]
        cx, cy, th = legal[cid]
        ew, eh = (comp.height, comp.width) if th in (90, 270) else (comp.width, comp.height)

        def is_valid(tx: float, ty: float) -> bool:
            if (
                tx - ew * 0.5 < margin - 0.01
                or tx + ew * 0.5 > bw - margin + 0.01
                or ty - eh * 0.5 < margin - 0.01
                or ty + eh * 0.5 > bh - margin + 0.01
            ):
                return False
            for pid in placed_ids:
                pc = netlist_graph.components[pid]
                px, py, pth = legal[pid]
                pw, ph = (pc.height, pc.width) if pth in (90, 270) else (pc.width, pc.height)
                if _rects_overlap(tx, ty, ew, eh, px, py, pw, ph, pad=1.2):
                    return False
            return True

        if not is_valid(cx, cy):
            found = False
            for r in range(1, 50):
                step = r * grid
                for ang_idx in range(16):
                    ang = 2.0 * math.pi * ang_idx / 16.0
                    tx = _snap(cx + step * math.cos(ang), grid)
                    ty = _snap(cy + step * math.sin(ang), grid)
                    if is_valid(tx, ty):
                        cx, cy = tx, ty
                        found = True
                        break
                if found:
                    break
        legal[cid] = (cx, cy, th)
        placed_ids.append(cid)

    return legal
