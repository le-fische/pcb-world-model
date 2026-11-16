"""Neural-Guided 45-degree Octilinear 2-layer A* Router with DRC & Congestion Fields."""

from __future__ import annotations

import heapq
import math
from typing import TYPE_CHECKING

import numpy as np
import torch
import torch.nn.functional as F

from env.board_state import BoardState, TraceSegment, Via
from env.metrics import compute_net_mst_pairs

if TYPE_CHECKING:
    pass


# 8 octilinear planar steps: (dx, dy, base_step_cost)
_SQRT2 = math.sqrt(2.0)
_OCTILINEAR_DIRS: tuple[tuple[int, int, float], ...] = (
    (1, 0, 1.0),
    (-1, 0, 1.0),
    (0, 1, 1.0),
    (0, -1, 1.0),
    (1, 1, _SQRT2),
    (1, -1, _SQRT2),
    (-1, 1, _SQRT2),
    (-1, -1, _SQRT2),
)


def _dilate_mask(mask: np.ndarray, radius_cells: int) -> np.ndarray:
    """Fast C++-backed morphological binary dilation using PyTorch max_pool2d."""
    if radius_cells <= 0:
        return mask.astype(bool)
    t = torch.from_numpy(mask.astype(np.float32)).unsqueeze(0).unsqueeze(0)
    k = 2 * radius_cells + 1
    dilated = F.max_pool2d(t, kernel_size=k, stride=1, padding=radius_cells)
    return dilated.squeeze(0).squeeze(0).numpy() > 0.5


def _compress_collinear_segments(
    path_cells: list[tuple[int, int, int]],
    res: float,
    net_id: int,
    trace_width_mm: float,
    via_diameter_mm: float,
    via_drill_mm: float,
) -> tuple[list[TraceSegment], list[Via]]:
    """Convert discrete `(layer, gx, gy)` cell path into clean TraceSegments and Vias."""
    if len(path_cells) < 2:
        return ([], [])

    segments: list[TraceSegment] = []
    vias: list[Via] = []

    run_start_idx = 0
    prev_dir: tuple[int, int] | None = None

    for i in range(1, len(path_cells)):
        l_prev, x_prev, y_prev = path_cells[i - 1]
        l_cur, x_cur, y_cur = path_cells[i]

        if l_cur != l_prev:
            # Flush planar segment up to i - 1
            if i - 1 > run_start_idx:
                _, sx, sy = path_cells[run_start_idx]
                segments.append(
                    TraceSegment(
                        net_id=net_id,
                        layer=l_prev,
                        x1_mm=round(sx * res, 4),
                        y1_mm=round(sy * res, 4),
                        x2_mm=round(x_prev * res, 4),
                        y2_mm=round(y_prev * res, 4),
                        width_mm=trace_width_mm,
                    )
                )
            vias.append(
                Via(
                    net_id=net_id,
                    x_mm=round(x_cur * res, 4),
                    y_mm=round(y_cur * res, 4),
                    diameter_mm=via_diameter_mm,
                    drill_mm=via_drill_mm,
                )
            )
            run_start_idx = i
            prev_dir = None
            continue

        cur_dir = (x_cur - x_prev, y_cur - y_prev)
        if prev_dir is not None and cur_dir != prev_dir:
            _, sx, sy = path_cells[run_start_idx]
            segments.append(
                TraceSegment(
                    net_id=net_id,
                    layer=l_cur,
                    x1_mm=round(sx * res, 4),
                    y1_mm=round(sy * res, 4),
                    x2_mm=round(x_prev * res, 4),
                    y2_mm=round(y_prev * res, 4),
                    width_mm=trace_width_mm,
                )
            )
            run_start_idx = i - 1
        prev_dir = cur_dir

    # Flush trailing run
    last_l, last_x, last_y = path_cells[-1]
    start_l, sx, sy = path_cells[run_start_idx]
    if last_l == start_l and (last_x != sx or last_y != sy):
        segments.append(
            TraceSegment(
                net_id=net_id,
                layer=last_l,
                x1_mm=round(sx * res, 4),
                y1_mm=round(sy * res, 4),
                x2_mm=round(last_x * res, 4),
                y2_mm=round(last_y * res, 4),
                width_mm=trace_width_mm,
            )
        )

    return (segments, vias)


def _octilinear_heuristic(gx: int, gy: int, targets: list[tuple[int, int]]) -> float:
    """Admissible octilinear Chebyshev-Euclidean lower bound to nearest target."""
    best = float("inf")
    for tx, ty in targets:
        dx = abs(tx - gx)
        dy = abs(ty - gy)
        d = max(dx, dy) + (_SQRT2 - 1.0) * min(dx, dy)
        if d < best:
            best = d
    return best


def _build_pad_masks_for_net(
    state: BoardState, net_id: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute foreign copper masks and own-pad masks for `net_id`."""
    H, W = state.H, state.W
    res = state.res
    foreign_pads_L1 = np.zeros((H, W), dtype=bool)
    foreign_pads_L2 = np.zeros((H, W), dtype=bool)
    own_pads_mask = np.zeros((2, H, W), dtype=bool)

    for comp in state.netlist.components.values():
        for pad in comp.pads:
            wx, wy = comp.pad_world_pos(pad)
            pw, ph = comp.pad_effective_size(pad)
            x0 = max(0, int(math.floor((wx - pw / 2.0) / res)))
            x1 = min(W - 1, int(math.ceil((wx + pw / 2.0) / res)))
            y0 = max(0, int(math.floor((wy - ph / 2.0) / res)))
            y1 = min(H - 1, int(math.ceil((wy + ph / 2.0) / res)))
            if pad.net_id == net_id:
                if pad.layer_mask & 1:
                    own_pads_mask[0, y0 : y1 + 1, x0 : x1 + 1] = True
                if pad.layer_mask & 2:
                    own_pads_mask[1, y0 : y1 + 1, x0 : x1 + 1] = True
            else:
                if pad.layer_mask & 1:
                    foreign_pads_L1[y0 : y1 + 1, x0 : x1 + 1] = True
                if pad.layer_mask & 2:
                    foreign_pads_L2[y0 : y1 + 1, x0 : x1 + 1] = True

    return foreign_pads_L1, foreign_pads_L2, own_pads_mask


def _astar_2pin(
    H: int,
    W: int,
    start_nodes: list[tuple[int, int, int]],
    target_nodes: list[tuple[int, int, int]],
    blocked: np.ndarray,      # [2, H, W] bool
    via_blocked: np.ndarray,  # [H, W] bool
    soft_penalty: np.ndarray, # [2, H, W] float32
    cost_field: np.ndarray,   # [2, H, W] float32
    occ_layers: np.ndarray,   # [2, H, W] int32
    net_id: int,
    gamma_via: float,
    max_expansions: int = 120_000,
) -> list[tuple[int, int, int]] | None:
    """Run fast 2-layer octilinear A* from start_nodes to any node in target_nodes."""
    target_set = set(target_nodes)
    target_xy = [(tx, ty) for _, tx, ty in target_nodes]
    hw = H * W
    total_nodes = 2 * hw

    g_score = np.full(total_nodes, np.inf, dtype=np.float32)
    parent = np.full(total_nodes, -1, dtype=np.int32)
    closed = np.zeros(total_nodes, dtype=bool)

    heap: list[tuple[float, float, int]] = []
    for l, sx, sy in start_nodes:
        if not (0 <= sx < W and 0 <= sy < H):
            continue
        idx = (l * H + sy) * W + sx
        g_score[idx] = 0.0
        h0 = _octilinear_heuristic(sx, sy, target_xy)
        heapq.heappush(heap, (h0, 0.0, idx))

    expansions = 0
    while heap and expansions < max_expansions:
        _, cur_g, idx = heapq.heappop(heap)
        if closed[idx]:
            continue
        closed[idx] = True
        expansions += 1

        l = idx // hw
        rem = idx % hw
        y = rem // W
        x = rem % W

        if (l, x, y) in target_set:
            # Reconstruct path
            path: list[tuple[int, int, int]] = []
            cur = idx
            while cur != -1:
                cl = cur // hw
                crem = cur % hw
                cy = crem // W
                cx = crem % W
                path.append((cl, cx, cy))
                cur = int(parent[cur])
            path.reverse()
            return path

        # 8 planar octilinear neighbors
        for dx, dy, step_d in _OCTILINEAR_DIRS:
            nx = x + dx
            ny = y + dy
            if not (0 <= nx < W and 0 <= ny < H):
                continue
            n_idx = (l * H + ny) * W + nx
            if closed[n_idx]:
                continue
            is_target = (l, nx, ny) in target_set
            if not is_target and blocked[l, ny, nx]:
                continue

            # Prevent diagonal X-crossing through a foreign trace corner
            if dx != 0 and dy != 0 and not is_target:
                occ_a = int(occ_layers[l, y, nx])
                occ_b = int(occ_layers[l, ny, x])
                if occ_a > 0 and occ_a != net_id and occ_b > 0 and occ_b != net_id:
                    continue

            cell_cost = float(cost_field[l, ny, nx]) + float(soft_penalty[l, ny, nx])
            tentative_g = cur_g + step_d * (1.0 + cell_cost)
            if tentative_g < g_score[n_idx]:
                g_score[n_idx] = tentative_g
                parent[n_idx] = idx
                h = _octilinear_heuristic(nx, ny, target_xy)
                heapq.heappush(heap, (tentative_g + h, tentative_g, n_idx))

        # Layer transition via (l <-> 1 - l)
        other_l = 1 - l
        if not via_blocked[y, x] or (other_l, x, y) in target_set:
            n_idx = (other_l * H + y) * W + x
            if not closed[n_idx]:
                tentative_g = cur_g + gamma_via + float(soft_penalty[other_l, y, x])
                if tentative_g < g_score[n_idx]:
                    g_score[n_idx] = tentative_g
                    parent[n_idx] = idx
                    h = _octilinear_heuristic(x, y, target_xy)
                    heapq.heappush(heap, (tentative_g + h, tentative_g, n_idx))

    return None


def octilinear_astar_route(
    state: BoardState,
    net_id: int,
    cost_field: np.ndarray,
    alpha: float | None = None,
    beta: float = 1.0,
    gamma: float | None = None,
    allow_soft_collision: bool = True,
) -> tuple[list[TraceSegment], list[Via], bool]:
    """Route all pins of `net_id` via 45-degree octilinear 2-layer Steiner-MST A*.

    Args:
        state: Current BoardState (other nets already committed).
        net_id: Target net ID to route.
        cost_field: Combined `[2, H, W]` spatial cost field (`congestion + penalty_field`).
        alpha: Scaling factor for `cost_field` (defaults to `state.config.alpha_congestion`).
        beta: Scaling factor for board-edge / courtyard proximity cost.
        gamma: Via transition cost (defaults to `state.config.gamma_via_cost`).
        allow_soft_collision: If True and strict obstacle avoidance fails, fall back
            to high-penalty soft collision routing so DRC verifier can localize errors.

    Returns:
        `(segments, vias, fully_connected)`
    """
    cfg = state.config
    res = state.res
    H, W = state.H, state.W
    alpha_val = cfg.alpha_congestion if alpha is None else alpha
    gamma_val = cfg.gamma_via_cost if gamma is None else gamma

    pairs = compute_net_mst_pairs(state.netlist, net_id)
    if not pairs:
        return ([], [], True)

    # Clearance dilation radii in grid cells
    trace_clear_cells = max(
        2, int(math.ceil((cfg.clearance_mm + cfg.trace_width_mm) / res))
    )
    pad_clear_cells = max(
        2, int(math.ceil((cfg.clearance_mm + cfg.trace_width_mm * 0.5) / res))
    )
    via_clear_cells = max(
        3, int(math.ceil((cfg.clearance_mm + cfg.via_diameter_mm * 0.5 + cfg.trace_width_mm * 0.5) / res))
    )

    foreign_pads_L1, foreign_pads_L2, own_pads_mask = _build_pad_masks_for_net(state, net_id)

    # Foreign copper occupancy (pads + committed traces + vias of OTHER nets)
    foreign_L1 = (state.occ_L1 > 0) & (state.occ_L1 != net_id)
    foreign_L2 = (state.occ_L2 > 0) & (state.occ_L2 != net_id)
    unconn_L1 = state.occ_L1 < 0
    unconn_L2 = state.occ_L2 < 0

    raw_foreign_L1 = foreign_L1 | unconn_L1 | foreign_pads_L1
    raw_foreign_L2 = foreign_L2 | unconn_L2 | foreign_pads_L2

    # Trace clearance zone around foreign copper
    hard_block_L1 = _dilate_mask(raw_foreign_L1, trace_clear_cells)
    hard_block_L2 = _dilate_mask(raw_foreign_L2, trace_clear_cells)

    # Never block own net's pad footprints unless directly occupied by foreign copper
    hard_block_L1 = (hard_block_L1 & (~own_pads_mask[0])) | foreign_pads_L1
    hard_block_L2 = (hard_block_L2 & (~own_pads_mask[1])) | foreign_pads_L2

    # Keep inside usable board outline
    out_of_board = state.outline_grid < 0.5
    hard_block_L1 |= out_of_board
    hard_block_L2 |= out_of_board

    blocked = np.stack([hard_block_L1, hard_block_L2], axis=0)

    # Vias require clearance on BOTH layers and cannot sit on SMD pads of other nets
    via_block_L1 = _dilate_mask(raw_foreign_L1, via_clear_cells)
    via_block_L2 = _dilate_mask(raw_foreign_L2, via_clear_cells)
    pad_avoid_via = _dilate_mask(own_pads_mask[0] | own_pads_mask[1], 1)
    via_blocked = via_block_L1 | via_block_L2 | out_of_board | pad_avoid_via

    scaled_cost = (alpha_val * np.clip(cost_field, 0.0, 100.0)).astype(np.float32)
    # Add mild courtyard escape preference so traces don't hug unrelated pads unnecessarily
    courtyard_pref = (beta * 0.15 * state.courtyard_grid).astype(np.float32)
    scaled_cost[0] += courtyard_pref
    scaled_cost[1] += courtyard_pref * 0.5

    occ_layers = np.stack([state.occ_L1, state.occ_L2], axis=0)

    # Prepare soft-collision fallback masks (pads remain strongly discouraged; traces get +80 cost)
    pad_block_L1 = _dilate_mask(foreign_pads_L1 | unconn_L1, pad_clear_cells) | out_of_board
    pad_block_L2 = _dilate_mask(foreign_pads_L2 | unconn_L2, pad_clear_cells) | out_of_board
    soft_blocked = np.stack(
        [pad_block_L1 & (~own_pads_mask[0]), pad_block_L2 & (~own_pads_mask[1])],
        axis=0,
    )
    soft_penalty_zero = np.zeros((2, H, W), dtype=np.float32)
    soft_penalty_high = np.zeros((2, H, W), dtype=np.float32)
    soft_penalty_high[0] = np.where(hard_block_L1 & (~soft_blocked[0]), 85.0, 0.0)
    soft_penalty_high[1] = np.where(hard_block_L2 & (~soft_blocked[1]), 85.0, 0.0)

    all_segments: list[TraceSegment] = []
    all_vias: list[Via] = []
    connected_cells: list[tuple[int, int, int]] = []
    all_connected = True

    def endpoint_nodes(pt: tuple[float, float, int]) -> list[tuple[int, int, int]]:
        px, py, lmask = pt
        gx, gy = state.mm_to_grid(px, py)
        nodes: list[tuple[int, int, int]] = []
        if lmask & 1:
            nodes.append((0, gx, gy))
        if lmask & 2:
            nodes.append((1, gx, gy))
        if not nodes:
            nodes.append((0, gx, gy))
        return nodes

    # Route MST pairs sequentially, reusing already-routed Steiner copper of `net_id`
    for idx_pair, (src_pt, dst_pt) in enumerate(pairs):
        src_nodes = endpoint_nodes(src_pt)
        dst_nodes = endpoint_nodes(dst_pt)

        # For subsequent pairs on multi-pin nets, target can attach to any existing copper cell of this net
        start_cands = src_nodes if idx_pair == 0 else dst_nodes
        target_cands = dst_nodes if idx_pair == 0 else list(set(connected_cells + src_nodes))

        # Keep endpoints unblocked
        for l, gx, gy in start_cands + target_cands:
            blocked[l, gy, gx] = False
            soft_blocked[l, gy, gx] = False

        path = _astar_2pin(
            H=H,
            W=W,
            start_nodes=start_cands,
            target_nodes=target_cands,
            blocked=blocked,
            via_blocked=via_blocked,
            soft_penalty=soft_penalty_zero,
            cost_field=scaled_cost,
            occ_layers=occ_layers,
            net_id=net_id,
            gamma_via=gamma_val,
        )

        if path is None and allow_soft_collision:
            path = _astar_2pin(
                H=H,
                W=W,
                start_nodes=start_cands,
                target_nodes=target_cands,
                blocked=soft_blocked,
                via_blocked=out_of_board,
                soft_penalty=soft_penalty_high,
                cost_field=scaled_cost,
                occ_layers=occ_layers,
                net_id=net_id,
                gamma_via=gamma_val,
            )

        if path is None:
            all_connected = False
            continue

        segs, vias = _compress_collinear_segments(
            path_cells=path,
            res=res,
            net_id=net_id,
            trace_width_mm=cfg.trace_width_mm,
            via_diameter_mm=cfg.via_diameter_mm,
            via_drill_mm=cfg.via_drill_mm,
        )
        all_segments.extend(segs)
        all_vias.extend(vias)

        # Sample points along `path` as valid Steiner attachment points for remaining pins
        step = max(1, len(path) // 16)
        connected_cells.extend(path[::step])
        if path[-1] not in connected_cells:
            connected_cells.append(path[-1])

        # Prevent future via placements of this net right on top of newly placed vias
        for v in vias:
            vgx, vgy = state.mm_to_grid(v.x_mm, v.y_mm)
            x0, x1 = max(0, vgx - 2), min(W - 1, vgx + 2)
            y0, y1 = max(0, vgy - 2), min(H - 1, vgy + 2)
            via_blocked[y0 : y1 + 1, x0 : x1 + 1] = True

    return (all_segments, all_vias, all_connected)
