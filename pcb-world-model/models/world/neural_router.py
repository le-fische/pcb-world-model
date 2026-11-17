"""Neural-Guided 45-degree Octilinear 2-layer A* Router with DRC & Congestion Fields."""

from __future__ import annotations

import heapq
import math
from typing import TYPE_CHECKING

import numpy as np
import torch
import torch.nn.functional as F

from env.board_state import BoardState, TraceSegment, Via, get_legal_via_spans
from env.metrics import compute_net_mst_pairs
from models.world.native_kernel import native_astar_available, run_native_astar

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
    num_layers: int = 2,
    microvia_diameter_mm: float = 0.40,
    microvia_drill_mm: float = 0.20,
    buried_via_diameter_mm: float = 0.45,
    buried_via_drill_mm: float = 0.20,
    plane_layers: set[int] | None = None,
) -> tuple[list[TraceSegment], list[Via]]:
    """Convert discrete `(layer, gx, gy)` cell path into clean TraceSegments and Vias."""
    if len(path_cells) < 2:
        return ([], [])

    segments: list[TraceSegment] = []
    vias: list[Via] = []

    run_start_idx = 0
    prev_dir: tuple[int, int] | None = None
    bottom_layer = max(1, num_layers - 1)

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
            lo_l = min(l_prev, l_cur)
            hi_l = max(l_prev, l_cur)
            if num_layers >= 8:
                if lo_l == 0 and hi_l == bottom_layer:
                    vtype, v_diam, v_drill = "through", via_diameter_mm, via_drill_mm
                elif hi_l - lo_l == 1:
                    vtype, v_diam, v_drill = "micro", microvia_diameter_mm, microvia_drill_mm
                else:
                    vtype, v_diam, v_drill = "buried", buried_via_diameter_mm, buried_via_drill_mm
            else:
                is_micro = num_layers >= 4 and (
                    (0 < l_prev < bottom_layer) or (0 < l_cur < bottom_layer)
                )
                vtype = "micro" if is_micro else "through"
                v_diam = microvia_diameter_mm if is_micro else via_diameter_mm
                v_drill = microvia_drill_mm if is_micro else via_drill_mm
            vias.append(
                Via(
                    net_id=net_id,
                    x_mm=round(x_cur * res, 4),
                    y_mm=round(y_cur * res, 4),
                    diameter_mm=v_diam,
                    drill_mm=v_drill,
                    start_layer=lo_l,
                    end_layer=hi_l,
                    via_type=vtype,
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

    # Flush trailing run (skip virtual segments inside actual internal power/ground planes)
    last_l, last_x, last_y = path_cells[-1]
    start_l, sx, sy = path_cells[run_start_idx]
    if plane_layers is not None:
        is_internal_plane_layer = last_l in plane_layers
    elif num_layers == 4:
        is_internal_plane_layer = last_l in (1, 2)
    elif num_layers == 6:
        is_internal_plane_layer = last_l in (1, 4)
    else:
        is_internal_plane_layer = False
    if last_l == start_l and not is_internal_plane_layer and (last_x != sx or last_y != sy):
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


def _mark_microvia_stagger_zone(
    stagger_blocked: np.ndarray,
    vgx: int,
    vgy: int,
    lo_l: int,
    hi_l: int,
    min_stagger_cells: float,
    num_layers: int,
) -> None:
    """Block adjacent buildup interfaces within `min_stagger_cells` of a 1-step microvia."""
    if min_stagger_cells <= 0.0 or (hi_l - lo_l) != 1:
        return
    _, H, W = stagger_blocked.shape
    r_int = int(math.ceil(min_stagger_cells))
    x0 = max(0, vgx - r_int)
    x1 = min(W - 1, vgx + r_int)
    y0 = max(0, vgy - r_int)
    y1 = min(H - 1, vgy + r_int)
    if x0 > x1 or y0 > y1:
        return
    yy, xx = np.ogrid[y0 : y1 + 1, x0 : x1 + 1]
    circle = ((xx - vgx) ** 2 + (yy - vgy) ** 2) < (min_stagger_cells ** 2 - 1e-4)
    for adj_iface in (lo_l - 1, lo_l + 1):
        if 0 <= adj_iface < num_layers - 1:
            stagger_blocked[adj_iface, y0 : y1 + 1, x0 : x1 + 1] |= circle.astype(np.uint8)


def _build_stagger_blocked_mask(
    state: BoardState, min_stagger_cells: float
) -> np.ndarray | None:
    """Build initial [L, H, W] uint8 stagger exclusion mask from already-routed microvias."""
    num_layers = getattr(state, "num_layers", 2)
    if min_stagger_cells <= 0.0 or num_layers < 6:
        return None
    stagger_blocked = np.zeros((num_layers, state.H, state.W), dtype=np.uint8)
    bottom = max(1, num_layers - 1)
    for vias in state.routed_vias.values():
        for v in vias:
            lo = max(0, min(int(getattr(v, "start_layer", 0)), int(getattr(v, "end_layer", bottom))))
            hi = min(bottom, max(int(getattr(v, "start_layer", 0)), int(getattr(v, "end_layer", bottom))))
            if hi - lo == 1:
                vgx, vgy = state.mm_to_grid(v.x_mm, v.y_mm)
                _mark_microvia_stagger_zone(
                    stagger_blocked, vgx, vgy, lo, hi, min_stagger_cells, num_layers
                )
    return stagger_blocked


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


def _is_hv_isolation_net(name: str) -> bool:
    """Return True if net carries high/switched primary voltage requiring creepage isolation."""
    up = name.upper()
    return any(tok in up for tok in ("HV_", "AC_", "MAINS", "RELAY_COM", "RELAY_NO", "RELAY_NC"))


def _find_diff_pair_mate_net(state: BoardState, net_id: int) -> int | None:
    """Locate the complementary differential-pair net ID if one exists (e.g. USB_DP <-> USB_DM)."""
    net = state.netlist.nets.get(net_id)
    if net is None:
        return None
    up = net.name.upper()
    pair_tokens = (
        ("_DP", "_DM"),
        ("_DM", "_DP"),
        ("D+", "D-"),
        ("D-", "D+"),
        ("CANH", "CANL"),
        ("CANL", "CANH"),
        ("TX+", "TX-"),
        ("TX-", "TX+"),
        ("RX+", "RX-"),
        ("RX-", "RX+"),
    )
    for src_tok, dst_tok in pair_tokens:
        if src_tok in up:
            target_up = up.replace(src_tok, dst_tok, 1)
            for other_id, other_net in state.netlist.nets.items():
                if other_id != net_id and other_net.name.upper() == target_up:
                    return other_id
    if up.endswith("_P") or up.endswith("_N"):
        target_up = (up[:-2] + "_N") if up.endswith("_P") else (up[:-2] + "_P")
        for other_id, other_net in state.netlist.nets.items():
            if other_id != net_id and other_net.name.upper() == target_up:
                return other_id
    return None


def _generate_thermal_ep_vias(state: BoardState, net_id: int) -> list[Via]:
    """Synthesize 2x2 or 3x3 thermal GND via array inside exposed thermal pads (thermal_via_ep)."""
    if not getattr(state.config, "enable_special_layout_rules", True):
        return []
    bottom_layer = getattr(state, "bottom_layer", 1)
    is_compact = max(state.netlist.width_mm, state.netlist.height_mm) <= 35.0
    v_diam = 0.40 if is_compact else 0.50
    v_drill = 0.20 if is_compact else 0.25
    plane_layer: int | None = None
    if getattr(state, "num_layers", 2) >= 4 and getattr(state, "plane_assignments", None):
        for l_idx, (pnid, _) in state.plane_assignments.items():
            if pnid == net_id:
                plane_layer = l_idx
                break

    thermal_vias: list[Via] = []
    for comp in state.netlist.components.values():
        if "thermal_via_ep" not in getattr(comp, "special_layout_rules", []):
            continue
        for pad in comp.pads:
            if (
                pad.net_id == net_id
                and abs(pad.rel_x) < 0.25
                and abs(pad.rel_y) < 0.25
                and min(pad.width, pad.height) >= 1.2
            ):
                wx, wy = comp.pad_world_pos(pad)
                pw, ph = comp.pad_effective_size(pad)
                n_grid = 3 if min(pw, ph) >= 2.2 else 2
                span_x = pw * 0.48
                span_y = ph * 0.48
                xs = np.linspace(-span_x * 0.5, span_x * 0.5, n_grid)
                ys = np.linspace(-span_y * 0.5, span_y * 0.5, n_grid)
                for dy in ys:
                    for dx in xs:
                        vx = round(wx + float(dx), 4)
                        vy = round(wy + float(dy), 4)
                        if plane_layer is not None:
                            thermal_vias.append(
                                Via(
                                    net_id=net_id,
                                    x_mm=vx,
                                    y_mm=vy,
                                    diameter_mm=v_diam,
                                    drill_mm=v_drill,
                                    start_layer=0,
                                    end_layer=plane_layer,
                                    via_type="micro",
                                )
                            )
                        else:
                            thermal_vias.append(
                                Via(
                                    net_id=net_id,
                                    x_mm=vx,
                                    y_mm=vy,
                                    diameter_mm=v_diam,
                                    drill_mm=v_drill,
                                    start_layer=0,
                                    end_layer=bottom_layer,
                                    via_type="through",
                                )
                            )
    return thermal_vias


def _build_multilayer_clearance_masks(
    state: BoardState,
    net_id: int,
    trace_width_mm: float,
    via_diameter_mm: float | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute sub-grid Euclidean clearance masks across all stackup layers `[L, H, W]`."""
    L = getattr(state, "num_layers", 2)
    bottom_layer = max(1, L - 1)
    H, W = state.H, state.W
    res = state.res
    c_min = state.config.clearance_mm
    d_via = state.config.via_diameter_mm if via_diameter_mm is None else via_diameter_mm
    d_drill = getattr(state.config, "microvia_drill_mm", 0.20) if d_via < 0.5 else state.config.via_drill_mm

    r_trace_edge = c_min + trace_width_mm * 0.5 + 0.36 * res
    r_via_edge = c_min + d_via * 0.5 + 0.025
    r_soft_pad = c_min + trace_width_mm * 0.5

    trace_blocked = np.zeros((L, H, W), dtype=bool)
    via_blocked_layers = np.zeros((L, H, W), dtype=bool)
    soft_blocked = np.zeros((L, H, W), dtype=bool)
    own_pads_mask = np.zeros((L, H, W), dtype=bool)

    pad_margin = max(r_trace_edge, r_via_edge, r_soft_pad) + res
    xs_all = np.arange(W, dtype=np.float32) * res
    ys_all = np.arange(H, dtype=np.float32) * res

    # 1. Exact Euclidean clearance around all component pads (cached per BoardState geometry)
    cached_entry = getattr(state, "_pad_dist_cache", None)
    if cached_entry is not None and cached_entry[0] >= pad_margin - 1e-6:
        pad_entries = cached_entry[1]
    else:
        max_tw = max(trace_width_mm, getattr(state.config, "power_trace_width_mm", 0.40), 0.50)
        max_vd = max(
            d_via,
            state.config.via_diameter_mm,
            getattr(state.config, "microvia_diameter_mm", 0.40),
        )
        cache_margin = max(
            pad_margin,
            c_min + max_tw * 0.5 + 0.36 * res + res,
            c_min + max_vd * 0.5 + 0.025 + res,
        )
        pad_entries: list[tuple[int, int, int, int, int, int, np.ndarray, bool]] = []
        for comp in state.netlist.components.values():
            has_ep_rule = "thermal_via_ep" in getattr(comp, "special_layout_rules", [])
            for pad in comp.pads:
                wx, wy = comp.pad_world_pos(pad)
                pw, ph = comp.pad_effective_size(pad)
                x0, x1 = wx - pw * 0.5, wx + pw * 0.5
                y0, y1 = wy - ph * 0.5, wy + ph * 0.5
                is_ep_pad = (
                    has_ep_rule
                    and abs(pad.rel_x) < 0.25
                    and abs(pad.rel_y) < 0.25
                    and min(pad.width, pad.height) >= 1.2
                )

                gx_lo = max(0, int(math.floor((x0 - cache_margin) / res)))
                gx_hi = min(W - 1, int(math.ceil((x1 + cache_margin) / res)))
                gy_lo = max(0, int(math.floor((y0 - cache_margin) / res)))
                gy_hi = min(H - 1, int(math.ceil((y1 + cache_margin) / res)))
                if gx_lo > gx_hi or gy_lo > gy_hi:
                    continue

                xx_mm = xs_all[gx_lo : gx_hi + 1][None, :]
                yy_mm = ys_all[gy_lo : gy_hi + 1][:, None]
                dx = np.maximum(0.0, np.maximum(x0 - xx_mm, xx_mm - x1))
                dy = np.maximum(0.0, np.maximum(y0 - yy_mm, yy_mm - y1))
                d_pad = np.hypot(dx, dy)
                pad_entries.append(
                    (pad.net_id, pad.layer_mask, gy_lo, gy_hi + 1, gx_lo, gx_hi + 1, d_pad, is_ep_pad)
                )
        state._pad_dist_cache = (cache_margin, pad_entries)

    for p_net_id, p_lmask, gy0, gy1, gx0, gx1, d_pad, is_ep_pad in pad_entries:
        if p_net_id == net_id:
            own_hit = d_pad <= 0.5 * res
            if p_lmask == 3:
                own_pads_mask[:, gy0:gy1, gx0:gx1] |= own_hit[None, :, :]
            else:
                if p_lmask & 1:
                    own_pads_mask[0, gy0:gy1, gx0:gx1] |= own_hit
                if p_lmask & 2:
                    own_pads_mask[bottom_layer, gy0:gy1, gx0:gx1] |= own_hit
                # Explicitly block via-in-pad on SMD pads on the pad's side (exempt exposed thermal pads)
                if not is_ep_pad:
                    smd_via_hit = d_pad < 0.12
                    if p_lmask & 1:
                        via_blocked_layers[0, gy0:gy1, gx0:gx1] |= smd_via_hit
                    if p_lmask & 2:
                        via_blocked_layers[bottom_layer, gy0:gy1, gx0:gx1] |= smd_via_hit
        else:
            trace_hit = d_pad < r_trace_edge
            soft_hit = d_pad < r_soft_pad
            via_hit = d_pad < r_via_edge
            if p_lmask == 3:
                trace_blocked[:, gy0:gy1, gx0:gx1] |= trace_hit[None, :, :]
                soft_blocked[:, gy0:gy1, gx0:gx1] |= soft_hit[None, :, :]
                via_blocked_layers[:, gy0:gy1, gx0:gx1] |= via_hit[None, :, :]
            else:
                if p_lmask & 1:
                    trace_blocked[0, gy0:gy1, gx0:gx1] |= trace_hit
                    soft_blocked[0, gy0:gy1, gx0:gx1] |= soft_hit
                    via_blocked_layers[0, gy0:gy1, gx0:gx1] |= via_hit
                if p_lmask & 2:
                    trace_blocked[bottom_layer, gy0:gy1, gx0:gx1] |= trace_hit
                    soft_blocked[bottom_layer, gy0:gy1, gx0:gx1] |= soft_hit
                    via_blocked_layers[bottom_layer, gy0:gy1, gx0:gx1] |= via_hit

    has_iso = (
        getattr(state.config, "enable_special_layout_rules", True)
        and any(
            "isolation_creepage" in getattr(c, "special_layout_rules", [])
            for c in state.netlist.components.values()
        )
    )
    curr_net = state.netlist.nets.get(net_id)
    curr_hv = _is_hv_isolation_net(curr_net.name) if curr_net else False
    creep_mm = float(getattr(state.config, "creepage_mm", 2.5))
    r_iso_edge = max(r_trace_edge, creep_mm + trace_width_mm * 0.5)
    r_iso_via = max(r_via_edge, creep_mm + d_via * 0.5)

    # 2. Exact point-to-segment clearance around foreign routed traces (+ symmetric HV/LV creepage)
    for other_net_id, segs in state.routed_traces.items():
        if other_net_id == net_id:
            continue
        other_net = state.netlist.nets.get(other_net_id)
        opp_iso = has_iso and other_net is not None and (_is_hv_isolation_net(other_net.name) != curr_hv)
        req_trace = r_iso_edge if opp_iso else r_trace_edge
        req_via = r_iso_via if opp_iso else r_via_edge
        for seg in segs:
            half_w = seg.width_mm * 0.5
            seg_margin = max(req_trace, req_via) + half_w + res
            bx0 = min(seg.x1_mm, seg.x2_mm) - seg_margin
            bx1 = max(seg.x1_mm, seg.x2_mm) + seg_margin
            by0 = min(seg.y1_mm, seg.y2_mm) - seg_margin
            by1 = max(seg.y1_mm, seg.y2_mm) + seg_margin

            gx_lo = max(0, int(math.floor(bx0 / res)))
            gx_hi = min(W - 1, int(math.ceil(bx1 / res)))
            gy_lo = max(0, int(math.floor(by0 / res)))
            gy_hi = min(H - 1, int(math.ceil(by1 / res)))
            if gx_lo > gx_hi or gy_lo > gy_hi:
                continue

            xx_mm = xs_all[gx_lo : gx_hi + 1][None, :]
            yy_mm = ys_all[gy_lo : gy_hi + 1][:, None]
            vx = seg.x2_mm - seg.x1_mm
            vy = seg.y2_mm - seg.y1_mm
            len_sq = vx * vx + vy * vy
            if len_sq <= 1e-12:
                d_seg = np.hypot(xx_mm - seg.x1_mm, yy_mm - seg.y1_mm)
            else:
                t = np.clip(
                    ((xx_mm - seg.x1_mm) * vx + (yy_mm - seg.y1_mm) * vy) / len_sq,
                    0.0,
                    1.0,
                )
                cx = seg.x1_mm + t * vx
                cy = seg.y1_mm + t * vy
                d_seg = np.hypot(xx_mm - cx, yy_mm - cy)

            edge_dist = d_seg - half_w
            layer_idx = max(0, min(bottom_layer, int(seg.layer)))
            trace_blocked[layer_idx, gy_lo : gy_hi + 1, gx_lo : gx_hi + 1] |= (
                edge_dist < req_trace
            )
            via_blocked_layers[layer_idx, gy_lo : gy_hi + 1, gx_lo : gx_hi + 1] |= (
                edge_dist < req_via
            )

    # 3. Exact Euclidean clearance around foreign routed vias (layer-span aware)
    for other_net_id, vias in state.routed_vias.items():
        if other_net_id == net_id:
            continue
        other_net = state.netlist.nets.get(other_net_id)
        opp_iso = has_iso and other_net is not None and (_is_hv_isolation_net(other_net.name) != curr_hv)
        req_trace = r_iso_edge if opp_iso else r_trace_edge
        req_via = r_iso_via if opp_iso else r_via_edge
        for via in vias:
            half_d = via.diameter_mm * 0.5
            r_drill_edge = c_min + 0.5 * (d_drill + via.drill_mm) + 0.025
            via_margin = max(req_trace, req_via) + half_d + res
            gx_lo = max(0, int(math.floor((via.x_mm - via_margin) / res)))
            gx_hi = min(W - 1, int(math.ceil((via.x_mm + via_margin) / res)))
            gy_lo = max(0, int(math.floor((via.y_mm - via_margin) / res)))
            gy_hi = min(H - 1, int(math.ceil((via.y_mm + via_margin) / res)))
            if gx_lo > gx_hi or gy_lo > gy_hi:
                continue

            xx_mm = xs_all[gx_lo : gx_hi + 1][None, :]
            yy_mm = ys_all[gy_lo : gy_hi + 1][:, None]
            center_dist = np.hypot(xx_mm - via.x_mm, yy_mm - via.y_mm)
            edge_dist = center_dist - half_d
            trace_hit = edge_dist < req_trace
            via_hit = edge_dist < req_via
            drill_hit = center_dist < r_drill_edge

            v_lo = max(0, min(getattr(via, "start_layer", 0), getattr(via, "end_layer", bottom_layer)))
            v_hi = min(bottom_layer, max(getattr(via, "start_layer", 0), getattr(via, "end_layer", bottom_layer)))
            trace_blocked[v_lo : v_hi + 1, gy_lo : gy_hi + 1, gx_lo : gx_hi + 1] |= trace_hit[None, :, :]
            via_blocked_layers[v_lo : v_hi + 1, gy_lo : gy_hi + 1, gx_lo : gx_hi + 1] |= via_hit[None, :, :]
            if v_lo > 0:
                via_blocked_layers[:v_lo, gy_lo : gy_hi + 1, gx_lo : gx_hi + 1] |= drill_hit[None, :, :]
            if v_hi + 1 < L:
                via_blocked_layers[v_hi + 1 :, gy_lo : gy_hi + 1, gx_lo : gx_hi + 1] |= drill_hit[None, :, :]

    # 4. Keep inside usable board outline + enforce RF antenna / isolation keepout zones
    out_of_board = state.outline_grid < 0.5
    trace_blocked |= out_of_board[None, :, :]
    soft_blocked |= out_of_board[None, :, :]
    via_blocked_layers |= out_of_board[None, :, :]

    keepout_mask = getattr(state, "keepout_layers", None)
    if keepout_mask is not None and keepout_mask.shape == trace_blocked.shape and keepout_mask.any():
        trace_blocked |= keepout_mask
        soft_blocked |= keepout_mask
        via_blocked_layers |= keepout_mask

    # Rev 14: Block vias inside DynamicFlexZone bend areas (0 vias permitted inside flex)
    if getattr(state, "has_rigid_flex", False):
        for fz in getattr(state.netlist, "substrate_zones", []) or []:
            if getattr(fz, "zone_type", "rigid") == "dynamic_flex" and not getattr(fz, "allow_vias", False):
                gx0 = max(0, int(math.floor((float(fz.x0_mm) - 0.1) / res)))
                gx1 = min(W - 1, int(math.ceil((float(fz.x1_mm) + 0.1) / res)))
                gy0 = max(0, int(math.floor((float(fz.y0_mm) - 0.1) / res)))
                gy1 = min(H - 1, int(math.ceil((float(fz.y1_mm) + 0.1) / res)))
                if gx0 <= gx1 and gy0 <= gy1:
                    via_blocked_layers[:, gy0 : gy1 + 1, gx0 : gx1 + 1] = True

    # 5. Isolation creepage enforcement against opposite-class pads (HV primary vs LV secondary)
    if has_iso:
        iso_margin = r_iso_edge + res
        for comp in state.netlist.components.values():
            for pad in comp.pads:
                if pad.net_id <= 0 or pad.net_id == net_id:
                    continue
                other_net = state.netlist.nets.get(pad.net_id)
                if not other_net or _is_hv_isolation_net(other_net.name) == curr_hv:
                    continue
                wx, wy = comp.pad_world_pos(pad)
                pw, ph = comp.pad_effective_size(pad)
                x0, x1 = wx - pw * 0.5, wx + pw * 0.5
                y0, y1 = wy - ph * 0.5, wy + ph * 0.5
                gx_lo = max(0, int(math.floor((x0 - iso_margin) / res)))
                gx_hi = min(W - 1, int(math.ceil((x1 + iso_margin) / res)))
                gy_lo = max(0, int(math.floor((y0 - iso_margin) / res)))
                gy_hi = min(H - 1, int(math.ceil((y1 + iso_margin) / res)))
                if gx_lo > gx_hi or gy_lo > gy_hi:
                    continue
                xx_mm = xs_all[gx_lo : gx_hi + 1][None, :]
                yy_mm = ys_all[gy_lo : gy_hi + 1][:, None]
                dx = np.maximum(0.0, np.maximum(x0 - xx_mm, xx_mm - x1))
                dy = np.maximum(0.0, np.maximum(y0 - yy_mm, yy_mm - y1))
                iso_hit = np.hypot(dx, dy) < r_iso_edge
                if pad.layer_mask == 3:
                    trace_blocked[:, gy_lo : gy_hi + 1, gx_lo : gx_hi + 1] |= iso_hit[None, :, :]
                    via_blocked_layers[:, gy_lo : gy_hi + 1, gx_lo : gx_hi + 1] |= iso_hit[None, :, :]
                else:
                    if pad.layer_mask & 1:
                        trace_blocked[0, gy_lo : gy_hi + 1, gx_lo : gx_hi + 1] |= iso_hit
                        via_blocked_layers[0, gy_lo : gy_hi + 1, gx_lo : gx_hi + 1] |= iso_hit
                    if pad.layer_mask & 2:
                        trace_blocked[bottom_layer, gy_lo : gy_hi + 1, gx_lo : gx_hi + 1] |= iso_hit
                        via_blocked_layers[bottom_layer, gy_lo : gy_hi + 1, gx_lo : gx_hi + 1] |= iso_hit

    return trace_blocked, via_blocked_layers, soft_blocked, own_pads_mask


def _build_exact_clearance_masks(
    state: BoardState,
    net_id: int,
    trace_width_mm: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute sub-grid Euclidean clearance masks for traces, vias, soft fallback, and own pads."""
    trace_blocked, via_blocked_layers, soft_blocked, own_pads_mask = _build_multilayer_clearance_masks(
        state=state,
        net_id=net_id,
        trace_width_mm=trace_width_mm,
    )
    via_blocked = via_blocked_layers.any(axis=0)
    return trace_blocked, via_blocked, soft_blocked, own_pads_mask


def _astar_2pin_python(
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
    """Pure-Python fallback 2-layer octilinear A* from start_nodes to any node in target_nodes."""
    if not target_nodes:
        return None
    target_set = set(target_nodes)
    unique_xy = list({(tx, ty) for _, tx, ty in target_nodes})

    xs_grid = np.arange(W, dtype=np.float32)[None, :]
    ys_grid = np.arange(H, dtype=np.float32)[:, None]
    h_map = np.full((H, W), np.inf, dtype=np.float32)
    diag_delta = np.float32(_SQRT2 - 1.0)
    for tx, ty in unique_xy:
        dx = np.abs(xs_grid - float(tx))
        dy = np.abs(ys_grid - float(ty))
        d = np.maximum(dx, dy) + diag_delta * np.minimum(dx, dy)
        np.minimum(h_map, d, out=h_map)
    h_flat: list[float] = h_map.ravel().tolist()

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
        h0 = h_flat[sy * W + sx]
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

        p_idx = int(parent[idx])
        if p_idx != -1:
            pl = p_idx // hw
            prem = p_idx % hw
            py = prem // W
            px = prem % W
            if pl == l:
                prev_dx = x - px
                prev_dy = y - py
            else:
                prev_dx, prev_dy = 0, 0
        else:
            pl = -1
            prev_dx, prev_dy = 0, 0

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

            if dx != 0 and dy != 0 and not is_target:
                if blocked[l, y, nx] and blocked[l, ny, x]:
                    continue
                occ_a = int(occ_layers[l, y, nx])
                occ_b = int(occ_layers[l, ny, x])
                if occ_a > 0 and occ_a != net_id and occ_b > 0 and occ_b != net_id:
                    continue

            bend_pen = (
                0.06
                if (p_idx != -1 and pl == l and (dx != prev_dx or dy != prev_dy))
                else 0.0
            )
            cell_cost = float(cost_field[l, ny, nx]) + float(soft_penalty[l, ny, nx])
            tentative_g = cur_g + step_d * (1.0 + cell_cost) + bend_pen
            if tentative_g < g_score[n_idx]:
                g_score[n_idx] = tentative_g
                parent[n_idx] = idx
                h = h_flat[ny * W + nx]
                heapq.heappush(heap, (tentative_g + h, tentative_g, n_idx))

        other_l = 1 - l
        if not via_blocked[y, x]:
            n_idx = (other_l * H + y) * W + x
            if not closed[n_idx]:
                tentative_g = cur_g + gamma_via + float(soft_penalty[other_l, y, x])
                if tentative_g < g_score[n_idx]:
                    g_score[n_idx] = tentative_g
                    parent[n_idx] = idx
                    h = h_flat[y * W + x]
                    heapq.heappush(heap, (tentative_g + h, tentative_g, n_idx))

    return None


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
    """Run compiled C++20 2-layer octilinear A* (with pure-Python fallback)."""
    if not target_nodes or not start_nodes:
        return None
    if native_astar_available():
        via_2l = np.broadcast_to(via_blocked[None, :, :], (2, H, W))
        allowed = np.array([1, 1], dtype=np.uint8)
        trans = np.array([[0.0, float(gamma_via)], [float(gamma_via), 0.0]], dtype=np.float32)
        return run_native_astar(
            L=2,
            H=H,
            W=W,
            start_nodes=start_nodes,
            target_nodes=target_nodes,
            blocked=blocked,
            via_blocked=via_2l,
            soft_penalty=soft_penalty,
            cost_field=cost_field,
            occ_layers=occ_layers,
            allowed_layers=allowed,
            transition_costs=trans,
            net_id=net_id,
            max_expansions=max_expansions,
        )
    return _astar_2pin_python(
        H=H,
        W=W,
        start_nodes=start_nodes,
        target_nodes=target_nodes,
        blocked=blocked,
        via_blocked=via_blocked,
        soft_penalty=soft_penalty,
        cost_field=cost_field,
        occ_layers=occ_layers,
        net_id=net_id,
        gamma_via=gamma_via,
        max_expansions=max_expansions,
    )


def _astar_multilayer_python(
    L: int,
    H: int,
    W: int,
    start_nodes: list[tuple[int, int, int]],
    target_nodes: list[tuple[int, int, int]],
    blocked: np.ndarray,
    via_blocked_layers: np.ndarray,
    soft_penalty: np.ndarray,
    cost_field: np.ndarray,
    occ_layers: np.ndarray,
    allowed_layers: np.ndarray,
    transition_costs: np.ndarray,
    net_id: int,
    max_expansions: int = 150_000,
    stagger_blocked: np.ndarray | None = None,
    min_stagger_cells: float = 0.0,
) -> list[tuple[int, int, int]] | None:
    """Pure-Python multi-layer fallback A* when C++20 native kernel is unavailable."""
    if not target_nodes or not start_nodes:
        return None
    target_set = set(target_nodes)
    unique_xy = list({(tx, ty) for _, tx, ty in target_nodes})

    xs_grid = np.arange(W, dtype=np.float32)[None, :]
    ys_grid = np.arange(H, dtype=np.float32)[:, None]
    h_map = np.full((H, W), np.inf, dtype=np.float32)
    diag_delta = np.float32(_SQRT2 - 1.0)
    for tx, ty in unique_xy:
        dx = np.abs(xs_grid - float(tx))
        dy = np.abs(ys_grid - float(ty))
        d = np.maximum(dx, dy) + diag_delta * np.minimum(dx, dy)
        np.minimum(h_map, d, out=h_map)
    h_flat: list[float] = h_map.ravel().tolist()

    hw = H * W
    total_nodes = L * hw
    g_score = np.full(total_nodes, np.inf, dtype=np.float32)
    parent = np.full(total_nodes, -1, dtype=np.int32)
    closed = np.zeros(total_nodes, dtype=bool)
    heap: list[tuple[float, float, int]] = []

    for l, sx, sy in start_nodes:
        if 0 <= l < L and 0 <= sx < W and 0 <= sy < H:
            idx = (l * H + sy) * W + sx
            g_score[idx] = 0.0
            heapq.heappush(heap, (h_flat[sy * W + sx], 0.0, idx))

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
            path: list[tuple[int, int, int]] = []
            cur = idx
            while cur != -1:
                cl = cur // hw
                crem = cur % hw
                cx, cy = crem % W, crem // W
                path.append((cl, cx, cy))
                p = int(parent[cur])
                if p != -1:
                    pl_p = p // hw
                    if abs(cl - pl_p) == 2 and not allowed_layers[(cl + pl_p) // 2]:
                        mid_l = (cl + pl_p) // 2
                        prem = p % hw
                        px_p, py_p = prem % W, prem // W
                        dx_b, dy_b = px_p - cx, py_p - cy
                        steps_b = max(abs(dx_b), abs(dy_b))
                        ux_b = 1 if dx_b > 0 else (-1 if dx_b < 0 else 0)
                        uy_b = 1 if dy_b > 0 else (-1 if dy_b < 0 else 0)
                        for s_b in range(steps_b + 1):
                            path.append((mid_l, cx + s_b * ux_b, cy + s_b * uy_b))
                cur = p
            path.reverse()
            return path

        p_idx = int(parent[idx])
        pl = p_idx // hw if p_idx != -1 else -1
        prev_dx = (x - ((p_idx % hw) % W)) if (p_idx != -1 and pl == l) else 0
        prev_dy = (y - ((p_idx % hw) // W)) if (p_idx != -1 and pl == l) else 0

        if allowed_layers[l]:
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
                if dx != 0 and dy != 0 and not is_target:
                    if blocked[l, y, nx] and blocked[l, ny, x]:
                        continue
                    occ_a = int(occ_layers[l, y, nx])
                    occ_b = int(occ_layers[l, ny, x])
                    if occ_a > 0 and occ_a != net_id and occ_b > 0 and occ_b != net_id:
                        continue
                bend_pen = 0.06 if (p_idx != -1 and pl == l and (dx != prev_dx or dy != prev_dy)) else 0.0
                cell_cost = float(cost_field[l, ny, nx]) + float(soft_penalty[l, ny, nx])
                tentative_g = cur_g + step_d * (1.0 + cell_cost) + bend_pen
                if tentative_g < g_score[n_idx]:
                    g_score[n_idx] = tentative_g
                    parent[n_idx] = idx
                    heapq.heappush(heap, (tentative_g + h_flat[ny * W + nx], tentative_g, n_idx))

        for next_l in range(L):
            if next_l == l:
                continue
            t_cost = float(transition_costs[l, next_l])
            if t_cost < 0.0 or not math.isfinite(t_cost):
                continue
            lo_l, hi_l = min(l, next_l), max(l, next_l)

            if min_stagger_cells > 0.0 and (hi_l - lo_l) == 1:
                if stagger_blocked is not None and stagger_blocked[lo_l, y, x] != 0:
                    continue
                min_stag_sq = min_stagger_cells * min_stagger_cells - 1e-4
                cur = idx
                hops = 0
                stagger_violated = False
                while cur != -1 and hops < 64:
                    p = int(parent[cur])
                    if p == -1:
                        break
                    p_l = p // hw
                    if p_l != l:
                        if abs(p_l - l) == 1:
                            crem = cur % hw
                            cy = crem // W
                            cx = crem % W
                            if ((x - cx) ** 2 + (y - cy) ** 2) < min_stag_sq:
                                stagger_violated = True
                        break
                    cur = p
                    hops += 1
                if stagger_violated:
                    continue

            if via_blocked_layers[lo_l : hi_l + 1, y, x].any():
                continue
            n_idx = (next_l * H + y) * W + x
            if closed[n_idx]:
                continue
            tentative_g = cur_g + t_cost + float(soft_penalty[next_l, y, x])
            if tentative_g < g_score[n_idx]:
                g_score[n_idx] = tentative_g
                parent[n_idx] = idx
                heapq.heappush(heap, (tentative_g + h_flat[y * W + x], tentative_g, n_idx))

        if min_stagger_cells > 0.0 and allowed_layers[l]:
            s_ortho = max(1, int(math.ceil(min_stagger_cells)))
            s_diag = max(1, int(math.ceil(min_stagger_cells / _SQRT2)))
            for dir_l in (-1, 1):
                mid_l = l + dir_l
                next_l = l + 2 * dir_l
                if not (0 <= mid_l < L and 0 <= next_l < L):
                    continue
                if allowed_layers[mid_l] or not allowed_layers[next_l]:
                    continue
                t1 = float(transition_costs[l, mid_l])
                t2 = float(transition_costs[mid_l, next_l])
                if t1 < 0.0 or t2 < 0.0 or not (math.isfinite(t1) and math.isfinite(t2)):
                    continue
                lo1, hi1 = min(l, mid_l), max(l, mid_l)
                if via_blocked_layers[lo1, y, x] or via_blocked_layers[hi1, y, x]:
                    continue
                if stagger_blocked is not None and stagger_blocked[lo1, y, x] != 0:
                    continue
                if blocked[mid_l, y, x]:
                    continue
                lo2, hi2 = min(mid_l, next_l), max(mid_l, next_l)
                for dx, dy, step_d in _OCTILINEAR_DIRS:
                    steps_b = s_diag if (dx != 0 and dy != 0) else s_ortho
                    nx = x + dx * steps_b
                    ny = y + dy * steps_b
                    if not (0 <= nx < W and 0 <= ny < H):
                        continue
                    n_idx = (next_l * H + ny) * W + nx
                    if closed[n_idx]:
                        continue
                    if via_blocked_layers[lo2, ny, nx] or via_blocked_layers[hi2, ny, nx]:
                        continue
                    if stagger_blocked is not None and stagger_blocked[lo2, ny, nx] != 0:
                        continue
                    dog_ok = True
                    dog_cost = 0.0
                    for k_s in range(1, steps_b + 1):
                        cx_k = x + dx * k_s
                        cy_k = y + dy * k_s
                        if blocked[mid_l, cy_k, cx_k]:
                            dog_ok = False
                            break
                        if dx != 0 and dy != 0:
                            px_k, py_k = cx_k - dx, cy_k - dy
                            if blocked[mid_l, py_k, cx_k] and blocked[mid_l, cy_k, px_k]:
                                dog_ok = False
                                break
                            occ_a = int(occ_layers[mid_l, py_k, cx_k])
                            occ_b = int(occ_layers[mid_l, cy_k, px_k])
                            if occ_a > 0 and occ_a != net_id and occ_b > 0 and occ_b != net_id:
                                dog_ok = False
                                break
                        dog_cost += step_d * (
                            1.0 + float(cost_field[mid_l, cy_k, cx_k]) + float(soft_penalty[mid_l, cy_k, cx_k])
                        )
                    if not dog_ok:
                        continue
                    tentative_g = cur_g + t1 + t2 + dog_cost + float(soft_penalty[next_l, ny, nx])
                    if tentative_g < g_score[n_idx]:
                        g_score[n_idx] = tentative_g
                        parent[n_idx] = idx
                        heapq.heappush(heap, (tentative_g + h_flat[ny * W + nx], tentative_g, n_idx))
    return None


def _astar_multilayer(
    L: int,
    H: int,
    W: int,
    start_nodes: list[tuple[int, int, int]],
    target_nodes: list[tuple[int, int, int]],
    blocked: np.ndarray,
    via_blocked_layers: np.ndarray,
    soft_penalty: np.ndarray,
    cost_field: np.ndarray,
    occ_layers: np.ndarray,
    allowed_layers: np.ndarray,
    transition_costs: np.ndarray,
    net_id: int,
    max_expansions: int = 150_000,
    stagger_blocked: np.ndarray | None = None,
    min_stagger_cells: float = 0.0,
) -> list[tuple[int, int, int]] | None:
    """Run multi-layer octilinear A* using compiled C++20 kernel with Python fallback."""
    if not target_nodes or not start_nodes:
        return None
    if native_astar_available():
        return run_native_astar(
            L=L,
            H=H,
            W=W,
            start_nodes=start_nodes,
            target_nodes=target_nodes,
            blocked=blocked,
            via_blocked=via_blocked_layers,
            soft_penalty=soft_penalty,
            cost_field=cost_field,
            occ_layers=occ_layers,
            allowed_layers=allowed_layers,
            transition_costs=transition_costs,
            net_id=net_id,
            max_expansions=max_expansions,
            stagger_blocked=stagger_blocked,
            min_stagger_cells=min_stagger_cells,
        )
    return _astar_multilayer_python(
        L=L,
        H=H,
        W=W,
        start_nodes=start_nodes,
        target_nodes=target_nodes,
        blocked=blocked,
        via_blocked_layers=via_blocked_layers,
        soft_penalty=soft_penalty,
        cost_field=cost_field,
        occ_layers=occ_layers,
        allowed_layers=allowed_layers,
        transition_costs=transition_costs,
        net_id=net_id,
        max_expansions=max_expansions,
        stagger_blocked=stagger_blocked,
        min_stagger_cells=min_stagger_cells,
    )


def _route_plane_net_4layer(
    state: BoardState,
    net_id: int,
    plane_layer: int,
    trace_width_mm: float,
    blocked: np.ndarray,
    via_blocked_layers: np.ndarray,
    soft_blocked: np.ndarray,
    fallback_via_blocked: np.ndarray,
    scaled_cost: np.ndarray,
    occ_layers: np.ndarray,
    soft_penalty_zero: np.ndarray,
    soft_penalty_high: np.ndarray,
    allow_soft_collision: bool,
    plane_layers: list[int] | None = None,
    stagger_blocked: np.ndarray | None = None,
    min_stagger_cells: float = 0.0,
) -> tuple[list[TraceSegment], list[Via], bool]:
    """Route an internal plane net (In1.GND / In2.PWR / 8L 2+4+2) via legal vias + short fanouts."""
    cfg = state.config
    res = state.res
    H, W = state.H, state.W
    num_layers = state.num_layers
    bottom_layer = state.bottom_layer
    micro_diam = getattr(cfg, "microvia_diameter_mm", 0.40)
    micro_drill = getattr(cfg, "microvia_drill_mm", 0.20)
    buried_diam = getattr(cfg, "buried_via_diameter_mm", 0.45)
    buried_drill = getattr(cfg, "buried_via_drill_mm", 0.20)
    gamma_micro = float(getattr(cfg, "gamma_microvia_cost", 4.5))
    gamma_buried = float(getattr(cfg, "gamma_buried_via_cost", 9.0))
    gamma_thru = float(cfg.gamma_via_cost)
    p_layers_list = sorted(set(plane_layers)) if plane_layers else [plane_layer]
    p_layers_set = set(p_layers_list)
    legal_spans = get_legal_via_spans(
        num_layers,
        int(getattr(cfg, "max_lamination_cycles", 2)),
        str(getattr(cfg, "hdi_buildup_type", "2+4+2")),
    )

    endpoints = state.get_net_pin_endpoints(net_id)
    if not endpoints:
        return ([], [], True)

    all_segments: list[TraceSegment] = []
    all_vias: list[Via] = []
    connected_planar: set[tuple[int, int, int]] = set()
    all_connected = True

    # THT barrels (layer_mask == 3) already tie straight into internal planes with 0 extra vias
    smd_endpoints: list[tuple[float, float, int]] = []
    for wx, wy, lmask in endpoints:
        gx, gy = state.mm_to_grid(wx, wy)
        if lmask == 3:
            connected_planar.add((0, gx, gy))
            connected_planar.add((bottom_layer, gx, gy))
        else:
            smd_endpoints.append((wx, wy, lmask))

    if not smd_endpoints:
        return ([], [], True)

    allowed_layers = np.zeros(num_layers, dtype=np.uint8)
    allowed_layers[0] = 1
    allowed_layers[bottom_layer] = 1
    if num_layers >= 8:
        # Allow staggered dogbones on L2/L7 and core routing on L3/L6 if falling back to buried core via
        allowed_layers[1] = 1
        allowed_layers[2] = 1
        allowed_layers[5] = 1
        allowed_layers[6] = 1

    # Order SMD endpoints deterministically for repeatable micro-via placement
    smd_endpoints.sort(key=lambda p: (round(p[1], 2), round(p[0], 2)))

    r_self_via_cells = max(2, int(math.ceil((micro_diam + cfg.clearance_mm) / res)))

    for wx, wy, lmask in smd_endpoints:
        src_l = 0 if (lmask & 1) else bottom_layer
        gx, gy = state.mm_to_grid(wx, wy)
        if (src_l, gx, gy) in connected_planar:
            continue

        blocked[src_l, gy, gx] = False
        soft_blocked[src_l, gy, gx] = False

        # Choose nearest legal plane layer for this side (e.g., In1.GND from F.Cu, In6.GND3 from B.Cu)
        direct_planes = [
            pl for pl in p_layers_list if (min(src_l, pl), max(src_l, pl)) in legal_spans
        ]
        if direct_planes:
            target_plane = min(direct_planes, key=lambda pl: abs(pl - src_l))
            lo_l, hi_l = min(src_l, target_plane), max(src_l, target_plane)
            use_thru_fallback = False
        elif num_layers >= 8 and src_l == bottom_layer and any(2 <= pl <= 5 for pl in p_layers_list):
            # Rev 14 Under-BGA B.Cu -> In3.PWR1 low-inductance drop via (6,7)->(5,6)->(2,5) buried core via
            # Never touches F.Cu (layer 0) so F.Cu BGA pads stay 100% unobstructed.
            target_plane = 2
            lo_l, hi_l = 2, bottom_layer
            use_thru_fallback = True
        else:
            # 8L PWR plane (In3.PWR1, layer 3): under 2 lamination cycles, 0<->3 skip-via is forbidden.
            # Tie into In3.PWR1 cleanly via legal (0<->7) PTH via (or 0->1->2->5 buried core via).
            target_plane = bottom_layer if src_l == 0 else 0
            lo_l, hi_l = 0, bottom_layer
            use_thru_fallback = True

        def _collect_drop_targets(
            v_block: np.ndarray, t_block: np.ndarray, max_rad: int
        ) -> list[tuple[int, int, int]]:
            x0 = max(0, gx - max_rad)
            x1 = min(W - 1, gx + max_rad)
            y0 = max(0, gy - max_rad)
            y1 = min(H - 1, gy + max_rad)
            span_blk = v_block[lo_l : hi_l + 1, y0 : y1 + 1, x0 : x1 + 1].any(axis=0)
            valid = (~span_blk) & (~t_block[src_l, y0 : y1 + 1, x0 : x1 + 1])
            ys_idx, xs_idx = np.nonzero(valid)
            if len(xs_idx) == 0:
                return []
            vx_all = xs_idx + x0
            vy_all = ys_idx + y0
            d2 = (vx_all - gx) ** 2 + (vy_all - gy) ** 2
            k_best = min(24, len(d2))
            if len(d2) > k_best:
                part = np.argpartition(d2, k_best - 1)[:k_best]
                order = part[np.argsort(d2[part])]
            else:
                order = np.argsort(d2)
            return [(target_plane, int(vx_all[i]), int(vy_all[i])) for i in order]

        target_cands = _collect_drop_targets(via_blocked_layers, blocked, max_rad=35)
        if not target_cands:
            target_cands = _collect_drop_targets(via_blocked_layers, blocked, max_rad=max(H, W))

        # Allow sharing a nearby existing planar stub/micro-via on same side
        local_connected = [
            node
            for node in connected_planar
            if node[0] == src_l and (abs(node[1] - gx) + abs(node[2] - gy)) <= 20
        ]
        all_targets = list(dict.fromkeys(target_cands + local_connected))

        trans = np.full((num_layers, num_layers), -1.0, dtype=np.float32)
        if not use_thru_fallback:
            trans[src_l, target_plane] = gamma_micro
        else:
            trans[0, bottom_layer] = gamma_thru
            trans[bottom_layer, 0] = gamma_thru
            if num_layers >= 8:
                for la, lb in legal_spans:
                    if (la == 0 and lb == bottom_layer):
                        continue
                    cost_v = gamma_micro if (lb - la == 1) else gamma_buried
                    trans[la, lb] = cost_v
                    trans[lb, la] = cost_v
        if num_layers < 8:
            trans[0, bottom_layer] = gamma_thru
            trans[bottom_layer, 0] = gamma_thru

        path = _astar_multilayer(
            L=num_layers,
            H=H,
            W=W,
            start_nodes=[(src_l, gx, gy)],
            target_nodes=all_targets,
            blocked=blocked,
            via_blocked_layers=via_blocked_layers,
            soft_penalty=soft_penalty_zero,
            cost_field=scaled_cost,
            occ_layers=occ_layers,
            allowed_layers=allowed_layers,
            transition_costs=trans,
            net_id=net_id,
            stagger_blocked=stagger_blocked,
            min_stagger_cells=min_stagger_cells,
        )

        if path is None and allow_soft_collision:
            fb_targets = _collect_drop_targets(fallback_via_blocked, soft_blocked, max_rad=max(H, W))
            fb_all = list(dict.fromkeys(fb_targets + local_connected))
            path = _astar_multilayer(
                L=num_layers,
                H=H,
                W=W,
                start_nodes=[(src_l, gx, gy)],
                target_nodes=fb_all,
                blocked=soft_blocked,
                via_blocked_layers=fallback_via_blocked,
                soft_penalty=soft_penalty_high,
                cost_field=scaled_cost,
                occ_layers=occ_layers,
                allowed_layers=allowed_layers,
                transition_costs=trans,
                net_id=net_id,
                stagger_blocked=stagger_blocked,
                min_stagger_cells=min_stagger_cells,
            )

        if path is None:
            all_connected = False
            continue

        segs, vias = _compress_collinear_segments(
            path_cells=path,
            res=res,
            net_id=net_id,
            trace_width_mm=trace_width_mm,
            via_diameter_mm=cfg.via_diameter_mm,
            via_drill_mm=cfg.via_drill_mm,
            num_layers=num_layers,
            microvia_diameter_mm=micro_diam,
            microvia_drill_mm=micro_drill,
            buried_via_diameter_mm=buried_diam,
            buried_via_drill_mm=buried_drill,
            plane_layers=p_layers_set,
        )
        all_segments.extend(segs)
        all_vias.extend(vias)

        for pl, px_c, py_c in path:
            if pl in (0, bottom_layer):
                connected_planar.add((pl, px_c, py_c))
                occ_layers[pl, py_c, px_c] = net_id

        for v in vias:
            vgx, vgy = state.mm_to_grid(v.x_mm, v.y_mm)
            x0 = max(0, vgx - r_self_via_cells)
            x1 = min(W - 1, vgx + r_self_via_cells)
            y0 = max(0, vgy - r_self_via_cells)
            y1 = min(H - 1, vgy + r_self_via_cells)
            v_lo = min(v.start_layer, v.end_layer)
            v_hi = max(v.start_layer, v.end_layer)
            via_blocked_layers[v_lo : v_hi + 1, y0 : y1 + 1, x0 : x1 + 1] = True
            if stagger_blocked is not None and v_hi - v_lo == 1:
                _mark_microvia_stagger_zone(
                    stagger_blocked, vgx, vgy, v_lo, v_hi, min_stagger_cells, num_layers
                )

    return (all_segments, all_vias, all_connected)


def octilinear_astar_route(
    state: BoardState,
    net_id: int,
    cost_field: np.ndarray,
    alpha: float | None = None,
    beta: float = 1.0,
    gamma: float | None = None,
    allow_soft_collision: bool = True,
) -> tuple[list[TraceSegment], list[Via], bool]:
    """Route all pins of `net_id` via 45-degree octilinear 2-layer or 4-layer Steiner-MST A*.

    Args:
        state: Current BoardState (other nets already committed).
        net_id: Target net ID to route.
        cost_field: Spatial cost field `[2, H, W]` or `[L, H, W]` (`congestion + penalty_field`).
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
    num_layers = getattr(state, "num_layers", 2)
    bottom_layer = getattr(state, "bottom_layer", 1)
    alpha_val = cfg.alpha_congestion if alpha is None else alpha
    gamma_val = cfg.gamma_via_cost if gamma is None else gamma

    net = state.netlist.nets.get(net_id)
    is_pwr = bool(net and net.is_power_ground)
    sig_w = getattr(cfg, "signal_trace_width_mm", cfg.trace_width_mm)
    trace_width_mm = cfg.power_trace_width_mm if is_pwr else sig_w

    pairs = compute_net_mst_pairs(state.netlist, net_id)
    if not pairs:
        return ([], [], True)

    plane_layers_for_net: list[int] = []
    if num_layers >= 4 and getattr(state, "plane_assignments", None):
        for l_idx, (pnid, _) in state.plane_assignments.items():
            if pnid == net_id:
                plane_layers_for_net.append(l_idx)
    plane_layer: int | None = plane_layers_for_net[0] if plane_layers_for_net else None

    if num_layers >= 8:
        # On 8L (2+4+2) HDI: GND/signals escape via 0.30mm microvias; PWR In3.PWR1 drops 0.50mm PTH vias.
        if plane_layer is not None and 1 not in plane_layers_for_net and 6 not in plane_layers_for_net:
            via_diam_for_mask = cfg.via_diameter_mm
        else:
            via_diam_for_mask = getattr(cfg, "microvia_diameter_mm", 0.30)
    else:
        via_diam_for_mask = (
            getattr(cfg, "microvia_diameter_mm", 0.40) if plane_layer is not None else cfg.via_diameter_mm
        )
    blocked, via_blocked_layers, soft_blocked, own_pads_mask = _build_multilayer_clearance_masks(
        state=state,
        net_id=net_id,
        trace_width_mm=trace_width_mm,
        via_diameter_mm=via_diam_for_mask,
    )
    out_of_board = state.outline_grid < 0.5
    fallback_via_layers = via_blocked_layers

    # Expand [2, H, W] cost field to [L, H, W] cleanly when routing on 4-layer / 8-layer stackup
    if cost_field.shape[0] == num_layers:
        cf_full = cost_field
    elif num_layers >= 4 and cost_field.shape[0] == 2:
        cf_full = np.zeros((num_layers, H, W), dtype=np.float32)
        cf_full[0] = cost_field[0]
        cf_full[bottom_layer] = cost_field[1]
    else:
        cf_full = np.zeros((num_layers, H, W), dtype=np.float32)
        cf_full[: min(num_layers, cost_field.shape[0])] = cost_field[: min(num_layers, cost_field.shape[0])]

    scaled_cost = (alpha_val * np.clip(cf_full, 0.0, 100.0)).astype(np.float32)
    courtyard_pref = (beta * 0.15 * state.courtyard_grid).astype(np.float32)
    scaled_cost[0] += courtyard_pref
    if num_layers >= 8:
        # 8L (2+4+2): Prefer F.Cu / In2.SIG1 / In5.SIG2 / B.Cu for signal runs;
        # gentle bias on L2 (In1.GND) and L7 (In6.GND3) keeps them for short staggered microvia dogbones.
        scaled_cost[1] += 0.55
        scaled_cost[2] += courtyard_pref * 0.15 + 0.02
        scaled_cost[5] += courtyard_pref * 0.15 + 0.05
        scaled_cost[6] += 0.55
        scaled_cost[bottom_layer] += courtyard_pref * 0.25 + 0.08
    elif num_layers >= 4:
        # On 4-layer boards, B.Cu is a primary signal plane; gentle offset avoids crowding F.Cu IC escapes
        scaled_cost[bottom_layer] += courtyard_pref * 0.35 + 0.02
    else:
        scaled_cost[bottom_layer] += courtyard_pref * 0.5 + 0.15

    # On dense / multi-layer / miniature boards, add a soft collar around foreign pads so long traces don't fence off IC escapes
    if num_layers >= 4 or len(state.netlist.routable_net_ids) >= 16 or max(state.netlist.width_mm, state.netlist.height_mm) <= 35.0:
        cached_halo = getattr(state, "_pad_escape_halo_cache", None)
        if cached_halo is None:
            cached_entry = getattr(state, "_pad_dist_cache", None)
            if cached_entry is not None:
                all_halo = np.zeros((num_layers, H, W), dtype=np.float32)
                for _, p_lmask, gy0, gy1, gx0, gx1, d_pad, _ in cached_entry[1]:
                    h_patch = np.where((d_pad > 0.1) & (d_pad < 0.9), 0.75, 0.0).astype(np.float32)
                    if p_lmask & 1:
                        all_halo[0, gy0:gy1, gx0:gx1] = np.maximum(
                            all_halo[0, gy0:gy1, gx0:gx1], h_patch
                        )
                    if p_lmask & 2:
                        all_halo[bottom_layer, gy0:gy1, gx0:gx1] = np.maximum(
                            all_halo[bottom_layer, gy0:gy1, gx0:gx1], h_patch
                        )
                cached_halo = all_halo
                state._pad_escape_halo_cache = cached_halo
        if cached_halo is not None:
            # Do not penalize approaching our own pads
            scaled_cost += np.where(own_pads_mask, 0.0, cached_halo)

    # Rev 10 Preferred-Margin Soft Repulsion Halo (0.32mm open-board bus spacing + mounting-hole buffer + anti-tombstone flank guard)
    if getattr(cfg, "enable_dfm_craftsmanship", True):
        pref_clr = float(getattr(cfg, "preferred_clearance_mm", 0.32))
        if pref_clr > cfg.clearance_mm + 0.02:
            xs_all = np.arange(W, dtype=np.float32) * res
            ys_all = np.arange(H, dtype=np.float32) * res
            r_hard = cfg.clearance_mm + trace_width_mm * 0.5 + 0.36 * res
            r_pref = pref_clr + trace_width_mm * 0.5 + 0.36 * res
            pref_field = np.zeros((num_layers, H, W), dtype=np.float32)

            for other_nid, segs in state.routed_traces.items():
                if other_nid == net_id:
                    continue
                for seg in segs:
                    half_w = seg.width_mm * 0.5
                    margin_p = r_pref + half_w + res
                    gx0 = max(0, int(math.floor((min(seg.x1_mm, seg.x2_mm) - margin_p) / res)))
                    gx1 = min(W - 1, int(math.ceil((max(seg.x1_mm, seg.x2_mm) + margin_p) / res)))
                    gy0 = max(0, int(math.floor((min(seg.y1_mm, seg.y2_mm) - margin_p) / res)))
                    gy1 = min(H - 1, int(math.ceil((max(seg.y1_mm, seg.y2_mm) + margin_p) / res)))
                    if gx0 > gx1 or gy0 > gy1:
                        continue
                    xx_mm = xs_all[gx0 : gx1 + 1][None, :]
                    yy_mm = ys_all[gy0 : gy1 + 1][:, None]
                    vx = seg.x2_mm - seg.x1_mm
                    vy = seg.y2_mm - seg.y1_mm
                    lsq = vx * vx + vy * vy
                    if lsq <= 1e-12:
                        d_s = np.hypot(xx_mm - seg.x1_mm, yy_mm - seg.y1_mm)
                    else:
                        t_s = np.clip(((xx_mm - seg.x1_mm) * vx + (yy_mm - seg.y1_mm) * vy) / lsq, 0.0, 1.0)
                        d_s = np.hypot(xx_mm - (seg.x1_mm + t_s * vx), yy_mm - (seg.y1_mm + t_s * vy))
                    edge_d = d_s - half_w
                    lyr = max(0, min(bottom_layer, int(seg.layer)))
                    band = (edge_d >= r_hard) & (edge_d < r_pref)
                    if band.any():
                        pen = np.where(band, 0.55 * (1.0 - (edge_d - r_hard) / max(0.05, r_pref - r_hard)), 0.0).astype(np.float32)
                        pref_field[lyr, gy0 : gy1 + 1, gx0 : gx1 + 1] = np.maximum(
                            pref_field[lyr, gy0 : gy1 + 1, gx0 : gx1 + 1], pen
                        )

            # Soft collar around mounting holes so traces route wide in open space
            for comp in state.netlist.components.values():
                if comp.comp_type == "mounting_hole":
                    drill_r = float(comp.electrical_meta.get("drill_mm", min(comp.width, comp.height))) * 0.5
                    mh_rad = drill_r + 0.75
                    gx0 = max(0, int(math.floor((comp.x - mh_rad) / res)))
                    gx1 = min(W - 1, int(math.ceil((comp.x + mh_rad) / res)))
                    gy0 = max(0, int(math.floor((comp.y - mh_rad) / res)))
                    gy1 = min(H - 1, int(math.ceil((comp.y + mh_rad) / res)))
                    if gx0 <= gx1 and gy0 <= gy1:
                        xx_mm = xs_all[gx0 : gx1 + 1][None, :]
                        yy_mm = ys_all[gy0 : gy1 + 1][:, None]
                        dmh = np.hypot(xx_mm - comp.x, yy_mm - comp.y)
                        mh_pen = np.where(dmh < mh_rad, 0.85, 0.0).astype(np.float32)
                        pref_field[:, gy0 : gy1 + 1, gx0 : gx1 + 1] = np.maximum(
                            pref_field[:, gy0 : gy1 + 1, gx0 : gx1 + 1], mh_pen[None, :, :]
                        )
                elif (
                    comp.comp_type in ("res", "cap", "diode")
                    and len(comp.pads) == 2
                    and any(p.net_id == net_id for p in comp.pads)
                ):
                    # Anti-tombstone soft flank guard on 2-pin SMD passives for this net
                    p0, p1 = comp.pads[0], comp.pads[1]
                    w0x, w0y = comp.pad_world_pos(p0)
                    w1x, w1y = comp.pad_world_pos(p1)
                    ax = w1x - w0x
                    ay = w1y - w0y
                    alen = math.hypot(ax, ay)
                    if alen > 0.2:
                        ux, uy = ax / alen, ay / alen
                        px_n, py_n = -uy, ux  # lateral perpendicular unit vector
                        for pad_obj, sign in ((p0, -1.0), (p1, 1.0)):
                            if pad_obj.net_id != net_id:
                                continue
                            wx, wy = comp.pad_world_pos(pad_obj)
                            pw, ph = comp.pad_effective_size(pad_obj)
                            box_r = max(pw, ph) + 0.45
                            gx0 = max(0, int(math.floor((wx - box_r) / res)))
                            gx1 = min(W - 1, int(math.ceil((wx + box_r) / res)))
                            gy0 = max(0, int(math.floor((wy - box_r) / res)))
                            gy1 = min(H - 1, int(math.ceil((wy + box_r) / res)))
                            if gx0 <= gx1 and gy0 <= gy1:
                                dx_g = xs_all[gx0 : gx1 + 1][None, :] - wx
                                dy_g = ys_all[gy0 : gy1 + 1][:, None] - wy
                                proj_ax = dx_g * (ux * sign) + dy_g * (uy * sign)
                                proj_lat = np.abs(dx_g * px_n + dy_g * py_n)
                                # Penalize entering pad laterally (side-entry) outside pad center
                                flank = (proj_lat > 0.18) & (proj_lat < 0.75) & (np.abs(proj_ax) < 0.35)
                                if flank.any():
                                    lyr = bottom_layer if pad_obj.layer_mask == 2 else 0
                                    pref_field[lyr, gy0 : gy1 + 1, gx0 : gx1 + 1] += np.where(
                                        flank, 1.6, 0.0
                                    ).astype(np.float32)

            scaled_cost += np.where(own_pads_mask, 0.0, pref_field)

    # Differential pair coupling corridor when complementary net is already routed
    mate_id = _find_diff_pair_mate_net(state, net_id)
    if mate_id is not None and mate_id in state.routed_traces and state.routed_traces[mate_id]:
        xs_all = np.arange(W, dtype=np.float32) * res
        ys_all = np.arange(H, dtype=np.float32) * res
        corridor = np.zeros((num_layers, H, W), dtype=bool)
        for mseg in state.routed_traces[mate_id]:
            l_idx = max(0, min(bottom_layer, int(mseg.layer)))
            pad_r = 0.95
            bx0 = max(0, int(math.floor((min(mseg.x1_mm, mseg.x2_mm) - pad_r) / res)))
            bx1 = min(W - 1, int(math.ceil((max(mseg.x1_mm, mseg.x2_mm) + pad_r) / res)))
            by0 = max(0, int(math.floor((min(mseg.y1_mm, mseg.y2_mm) - pad_r) / res)))
            by1 = min(H - 1, int(math.ceil((max(mseg.y1_mm, mseg.y2_mm) + pad_r) / res)))
            if bx0 > bx1 or by0 > by1:
                continue
            xx_mm = xs_all[bx0 : bx1 + 1][None, :]
            yy_mm = ys_all[by0 : by1 + 1][:, None]
            vx = mseg.x2_mm - mseg.x1_mm
            vy = mseg.y2_mm - mseg.y1_mm
            lsq = vx * vx + vy * vy
            if lsq <= 1e-12:
                dm = np.hypot(xx_mm - mseg.x1_mm, yy_mm - mseg.y1_mm)
            else:
                t = np.clip(((xx_mm - mseg.x1_mm) * vx + (yy_mm - mseg.y1_mm) * vy) / lsq, 0.0, 1.0)
                dm = np.hypot(xx_mm - (mseg.x1_mm + t * vx), yy_mm - (mseg.y1_mm + t * vy))
            corridor[l_idx, by0 : by1 + 1, bx0 : bx1 + 1] |= dm <= pad_r
        # Encourage differential mate to stay parallel inside the 0.95mm coupled corridor
        scaled_cost += np.where(corridor | own_pads_mask, 0.0, 0.35).astype(np.float32)

    # Rev 13: Z-Axis Broadside Crosstalk Avoidance + Mixed-Signal Corridor & Thermal Plume Guard
    net_dom = getattr(net, "noise_domain", "neutral") if net else "neutral"
    if getattr(cfg, "enable_mixed_signal_physics", True) and net_dom in (
        "analog_precision",
        "digital_hs",
        "power_switching",
    ):
        xs_all = np.arange(W, dtype=np.float32) * res
        ys_all = np.arange(H, dtype=np.float32) * res
        p_assign = getattr(state, "plane_assignments", {}) or {}
        gnd_layers = {l_idx for l_idx, (_, pname) in p_assign.items() if "GND" in pname.upper()}
        opp_domains = (
            ("digital_hs", "power_switching") if net_dom == "analog_precision" else ("analog_precision",)
        )
        em_penalty = np.zeros((num_layers, H, W), dtype=np.float32)

        # 1. Z-axis broadside parallel-run repulsion on adjacent unshielded layers
        broadside_r = 0.65
        for other_nid, segs in state.routed_traces.items():
            if other_nid == net_id:
                continue
            other_net = state.netlist.nets.get(other_nid)
            if other_net is None or getattr(other_net, "noise_domain", "neutral") not in opp_domains:
                continue
            for seg in segs:
                s_lyr = max(0, min(bottom_layer, int(seg.layer)))
                adj_targets = [
                    al for al in (s_lyr - 1, s_lyr + 1)
                    if 0 <= al < num_layers and (s_lyr not in gnd_layers) and (al not in gnd_layers)
                ]
                if not adj_targets:
                    continue
                bx0 = max(0, int(math.floor((min(seg.x1_mm, seg.x2_mm) - broadside_r) / res)))
                bx1 = min(W - 1, int(math.ceil((max(seg.x1_mm, seg.x2_mm) + broadside_r) / res)))
                by0 = max(0, int(math.floor((min(seg.y1_mm, seg.y2_mm) - broadside_r) / res)))
                by1 = min(H - 1, int(math.ceil((max(seg.y1_mm, seg.y2_mm) + broadside_r) / res)))
                if bx0 > bx1 or by0 > by1:
                    continue
                xx_mm = xs_all[bx0 : bx1 + 1][None, :]
                yy_mm = ys_all[by0 : by1 + 1][:, None]
                vx = seg.x2_mm - seg.x1_mm
                vy = seg.y2_mm - seg.y1_mm
                lsq = vx * vx + vy * vy
                if lsq <= 1e-12:
                    ds = np.hypot(xx_mm - seg.x1_mm, yy_mm - seg.y1_mm)
                else:
                    t = np.clip(((xx_mm - seg.x1_mm) * vx + (yy_mm - seg.y1_mm) * vy) / lsq, 0.0, 1.0)
                    ds = np.hypot(xx_mm - (seg.x1_mm + t * vx), yy_mm - (seg.y1_mm + t * vy))
                b_pen = np.where(ds <= broadside_r, 28.0 * (1.0 - ds / 0.72), 0.0).astype(np.float32)
                for al in adj_targets:
                    em_penalty[al, by0 : by1 + 1, bx0 : bx1 + 1] = np.maximum(
                        em_penalty[al, by0 : by1 + 1, bx0 : bx1 + 1], b_pen
                    )

        # 2. Mixed-signal spatial domain corridor guard & hot-device thermal plume avoidance
        if net_dom in ("analog_precision", "power_switching"):
            avoid_dom = "power_switching" if net_dom == "analog_precision" else "analog_precision"
            halo_guard = float(getattr(cfg, "analog_guard_halo_mm", 3.0))
            for comp in state.netlist.components.values():
                cdom = getattr(comp, "noise_domain", "neutral")
                pw_watts = float(getattr(comp, "electrical_meta", {}).get("power_watts", 0.0) or 0.0)
                is_thermal_src = (net_dom == "analog_precision") and pw_watts >= 2.0
                if cdom != avoid_dom and not is_thermal_src:
                    continue
                ew, eh = comp.effective_size()
                rad_c = max(halo_guard, 3.5) if is_thermal_src else halo_guard
                cx0, cx1 = comp.x - ew * 0.5, comp.x + ew * 0.5
                cy0, cy1 = comp.y - eh * 0.5, comp.y + eh * 0.5
                gx0 = max(0, int(math.floor((cx0 - rad_c) / res)))
                gx1 = min(W - 1, int(math.ceil((cx1 + rad_c) / res)))
                gy0 = max(0, int(math.floor((cy0 - rad_c) / res)))
                gy1 = min(H - 1, int(math.ceil((cy1 + rad_c) / res)))
                if gx0 <= gx1 and gy0 <= gy1:
                    xx_mm = xs_all[gx0 : gx1 + 1][None, :]
                    yy_mm = ys_all[gy0 : gy1 + 1][:, None]
                    dx = np.maximum(0.0, np.maximum(cx0 - xx_mm, xx_mm - cx1))
                    dy = np.maximum(0.0, np.maximum(cy0 - yy_mm, yy_mm - cy1))
                    dc = np.hypot(dx, dy)
                    c_pen = np.where(dc <= rad_c, 18.0 * (1.0 - dc / (rad_c + 0.3)), 0.0).astype(np.float32)
                    em_penalty[:, gy0 : gy1 + 1, gx0 : gx1 + 1] = np.maximum(
                        em_penalty[:, gy0 : gy1 + 1, gx0 : gx1 + 1], c_pen[None, :, :]
                    )
            for other_nid, segs in state.routed_traces.items():
                if other_nid == net_id:
                    continue
                other_net = state.netlist.nets.get(other_nid)
                if other_net is None or getattr(other_net, "noise_domain", "neutral") != avoid_dom:
                    continue
                for seg in segs:
                    bx0 = max(0, int(math.floor((min(seg.x1_mm, seg.x2_mm) - halo_guard) / res)))
                    bx1 = min(W - 1, int(math.ceil((max(seg.x1_mm, seg.x2_mm) + halo_guard) / res)))
                    by0 = max(0, int(math.floor((min(seg.y1_mm, seg.y2_mm) - halo_guard) / res)))
                    by1 = min(H - 1, int(math.ceil((max(seg.y1_mm, seg.y2_mm) + halo_guard) / res)))
                    if bx0 > bx1 or by0 > by1:
                        continue
                    xx_mm = xs_all[bx0 : bx1 + 1][None, :]
                    yy_mm = ys_all[by0 : by1 + 1][:, None]
                    vx = seg.x2_mm - seg.x1_mm
                    vy = seg.y2_mm - seg.y1_mm
                    lsq = vx * vx + vy * vy
                    if lsq <= 1e-12:
                        ds = np.hypot(xx_mm - seg.x1_mm, yy_mm - seg.y1_mm)
                    else:
                        t = np.clip(((xx_mm - seg.x1_mm) * vx + (yy_mm - seg.y1_mm) * vy) / lsq, 0.0, 1.0)
                        ds = np.hypot(xx_mm - (seg.x1_mm + t * vx), yy_mm - (seg.y1_mm + t * vy))
                    s_pen = np.where(ds <= halo_guard, 16.0 * (1.0 - ds / (halo_guard + 0.3)), 0.0).astype(np.float32)
                    em_penalty[:, by0 : by1 + 1, bx0 : bx1 + 1] = np.maximum(
                        em_penalty[:, by0 : by1 + 1, bx0 : bx1 + 1], s_pen[None, :, :]
                    )

        scaled_cost += np.where(own_pads_mask, 0.0, em_penalty)

    occ_layers = state.occ_layers.copy()
    soft_penalty_zero = np.zeros((num_layers, H, W), dtype=np.float32)
    soft_penalty_high = np.where(blocked & (~soft_blocked), 85.0, 0.0).astype(np.float32)

    ep_vias = _generate_thermal_ep_vias(state, net_id)

    enforce_stagger = bool(getattr(cfg, "enforce_staggered_microvias", True)) and num_layers >= 6
    min_stagger_cells = (
        float(getattr(cfg, "microvia_stagger_offset_mm", 0.25)) / res if enforce_stagger else 0.0
    )
    stagger_blocked = _build_stagger_blocked_mask(state, min_stagger_cells)

    if plane_layer is not None:
        segs, vias, ok = _route_plane_net_4layer(
            state=state,
            net_id=net_id,
            plane_layer=plane_layer,
            trace_width_mm=trace_width_mm,
            blocked=blocked,
            via_blocked_layers=via_blocked_layers,
            soft_blocked=soft_blocked,
            fallback_via_blocked=fallback_via_layers,
            scaled_cost=scaled_cost,
            occ_layers=occ_layers,
            soft_penalty_zero=soft_penalty_zero,
            soft_penalty_high=soft_penalty_high,
            allow_soft_collision=allow_soft_collision,
            plane_layers=plane_layers_for_net,
            stagger_blocked=stagger_blocked,
            min_stagger_cells=min_stagger_cells,
        )
        return (segs, ep_vias + vias, ok)

    via_blocked_2d = via_blocked_layers.any(axis=0)
    fallback_via_2d = fallback_via_layers.any(axis=0)

    allowed_layers = np.zeros(num_layers, dtype=np.uint8)
    allowed_layers[0] = 1
    allowed_layers[bottom_layer] = 1
    if num_layers >= 8:
        # True signal layers on 8L (2+4+2): F.Cu (0), In2.SIG1 (2), In5.SIG2 (5), B.Cu (7).
        # Buildup layers L2 (1) and L7 (6) stay 0 here so signals only traverse them via
        # atomic >=0.25mm staggered microvia dogbone bridges (0->1->2 / 5->6->7).
        allowed_layers[2] = 1
        allowed_layers[5] = 1
    elif num_layers >= 6:
        allowed_layers[2] = 1
        allowed_layers[3] = 1

    trans_costs = np.full((num_layers, num_layers), -1.0, dtype=np.float32)
    if num_layers >= 8:
        gamma_micro = float(getattr(cfg, "gamma_microvia_cost", 4.5))
        gamma_buried = float(getattr(cfg, "gamma_buried_via_cost", 9.0))
        legal_spans = get_legal_via_spans(
            num_layers,
            int(getattr(cfg, "max_lamination_cycles", 2)),
            str(getattr(cfg, "hdi_buildup_type", "2+4+2")),
        )
        for la, lb in legal_spans:
            if la == 0 and lb == bottom_layer:
                # Signal nets on 8L (2+4+2) HDI escape cleanly via 0.30mm microvias + 0.45mm buried core vias,
                # avoiding bulky 0.50mm through-vias inside dense BGA arrays.
                continue
            elif lb - la == 1:
                c_v = gamma_micro
            else:
                c_v = gamma_buried
            trans_costs[la, lb] = c_v
            trans_costs[lb, la] = c_v
    else:
        for la in range(num_layers):
            if not allowed_layers[la]:
                continue
            for lb in range(num_layers):
                if la != lb and allowed_layers[lb]:
                    trans_costs[la, lb] = float(gamma_val)

    plane_layers_set = set(getattr(state, "plane_assignments", {}).keys())

    all_segments: list[TraceSegment] = []
    all_vias: list[Via] = list(ep_vias)
    connected_cells: list[tuple[int, int, int]] = []
    for ev in ep_vias:
        vgx, vgy = state.mm_to_grid(ev.x_mm, ev.y_mm)
        for l_idx in range(ev.start_layer, ev.end_layer + 1):
            if allowed_layers[l_idx]:
                connected_cells.append((l_idx, vgx, vgy))
            occ_layers[l_idx, vgy, vgx] = net_id
    all_connected = True

    def endpoint_nodes(pt: tuple[float, float, int]) -> list[tuple[int, int, int]]:
        px, py, lmask = pt
        gx, gy = state.mm_to_grid(px, py)
        nodes: list[tuple[int, int, int]] = []
        if lmask & 1:
            nodes.append((0, gx, gy))
        if lmask & 2:
            nodes.append((bottom_layer, gx, gy))
        if not nodes:
            nodes.append((0, gx, gy))
        return nodes

    # Route MST pairs sequentially, reusing all already-routed Steiner copper of `net_id`
    for idx_pair, (src_pt, dst_pt) in enumerate(pairs):
        src_nodes = endpoint_nodes(src_pt)
        dst_nodes = endpoint_nodes(dst_pt)

        start_cands = src_nodes if idx_pair == 0 else dst_nodes
        target_cands = dst_nodes if idx_pair == 0 else list(set(connected_cells + src_nodes))

        for l, gx, gy in start_cands + target_cands:
            blocked[l, gy, gx] = False
            soft_blocked[l, gy, gx] = False

        if num_layers == 2:
            path = _astar_2pin(
                H=H,
                W=W,
                start_nodes=start_cands,
                target_nodes=target_cands,
                blocked=blocked,
                via_blocked=via_blocked_2d,
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
                    via_blocked=fallback_via_2d,
                    soft_penalty=soft_penalty_high,
                    cost_field=scaled_cost,
                    occ_layers=occ_layers,
                    net_id=net_id,
                    gamma_via=gamma_val,
                )
        else:
            max_exp_ml = 260_000 if num_layers >= 8 else 150_000
            path = _astar_multilayer(
                L=num_layers,
                H=H,
                W=W,
                start_nodes=start_cands,
                target_nodes=target_cands,
                blocked=blocked,
                via_blocked_layers=via_blocked_layers,
                soft_penalty=soft_penalty_zero,
                cost_field=scaled_cost,
                occ_layers=occ_layers,
                allowed_layers=allowed_layers,
                transition_costs=trans_costs,
                net_id=net_id,
                max_expansions=max_exp_ml,
                stagger_blocked=stagger_blocked,
                min_stagger_cells=min_stagger_cells,
            )
            if path is None and allow_soft_collision:
                path = _astar_multilayer(
                    L=num_layers,
                    H=H,
                    W=W,
                    start_nodes=start_cands,
                    target_nodes=target_cands,
                    blocked=soft_blocked,
                    via_blocked_layers=fallback_via_layers,
                    soft_penalty=soft_penalty_high,
                    cost_field=scaled_cost,
                    occ_layers=occ_layers,
                    allowed_layers=allowed_layers,
                    transition_costs=trans_costs,
                    net_id=net_id,
                    max_expansions=max_exp_ml,
                    stagger_blocked=stagger_blocked,
                    min_stagger_cells=min_stagger_cells,
                )

        if path is None:
            all_connected = False
            continue

        segs, vias = _compress_collinear_segments(
            path_cells=path,
            res=res,
            net_id=net_id,
            trace_width_mm=trace_width_mm,
            via_diameter_mm=cfg.via_diameter_mm,
            via_drill_mm=cfg.via_drill_mm,
            num_layers=num_layers,
            microvia_diameter_mm=getattr(cfg, "microvia_diameter_mm", 0.40),
            microvia_drill_mm=getattr(cfg, "microvia_drill_mm", 0.20),
            buried_via_diameter_mm=getattr(cfg, "buried_via_diameter_mm", 0.45),
            buried_via_drill_mm=getattr(cfg, "buried_via_drill_mm", 0.20),
            plane_layers=plane_layers_set,
        )
        all_segments.extend(segs)
        all_vias.extend(vias)

        path_set = set(path)
        for pl, px_c, py_c in path:
            connected_cells.append((pl, px_c, py_c))
            occ_layers[pl, py_c, px_c] = net_id
        for ep_nodes in (src_nodes, dst_nodes):
            if len(ep_nodes) > 1 and any(node in path_set for node in ep_nodes):
                for el, ex_c, ey_c in ep_nodes:
                    if (el, ex_c, ey_c) not in path_set:
                        connected_cells.append((el, ex_c, ey_c))
                    occ_layers[el, ey_c, ex_c] = net_id

        for v in vias:
            vgx, vgy = state.mm_to_grid(v.x_mm, v.y_mm)
            x0, x1 = max(0, vgx - 2), min(W - 1, vgx + 2)
            y0, y1 = max(0, vgy - 2), min(H - 1, vgy + 2)
            v_lo = min(v.start_layer, v.end_layer)
            v_hi = max(v.start_layer, v.end_layer)
            via_blocked_2d[y0 : y1 + 1, x0 : x1 + 1] = True
            via_blocked_layers[v_lo : v_hi + 1, y0 : y1 + 1, x0 : x1 + 1] = True
            if stagger_blocked is not None and (v_hi - v_lo) == 1:
                _mark_microvia_stagger_zone(
                    stagger_blocked, vgx, vgy, v_lo, v_hi, min_stagger_cells, num_layers
                )

    return (all_segments, all_vias, all_connected)


def _split_segment_at_flex_seams(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    flex_zones: list[Any],
    taper_mm: float = 0.50,
) -> list[tuple[float, float, float, float, str]]:
    """Split a segment at Rigid<->DynamicFlex boundaries and insert smooth seam taper zones."""
    if not flex_zones:
        return [(x1, y1, x2, y2, "rigid")]

    dx = x2 - x1
    dy = y2 - y1
    seg_len = math.hypot(dx, dy)
    if seg_len < 1e-4:
        return []

    ts: list[float] = [0.0, 1.0]
    for fz in flex_zones:
        if abs(dx) > 1e-6:
            for xb in (float(fz.x0_mm), float(fz.x1_mm)):
                t = (xb - x1) / dx
                if 0.001 < t < 0.999:
                    ts.append(t)
        if abs(dy) > 1e-6:
            for yb in (float(fz.y0_mm), float(fz.y1_mm)):
                t = (yb - y1) / dy
                if 0.001 < t < 0.999:
                    ts.append(t)
                    # Add smooth taper transition just on the rigid side of the seam
                    dt_taper = min(0.25, taper_mm / seg_len)
                    if 0.001 < t - dt_taper < 0.999:
                        ts.append(t - dt_taper)
                    if 0.001 < t + dt_taper < 0.999:
                        ts.append(t + dt_taper)

    ts = sorted(set(round(t, 6) for t in ts))
    out: list[tuple[float, float, float, float, str]] = []
    for i in range(len(ts) - 1):
        ta, tb = ts[i], ts[i + 1]
        if tb - ta < 1e-4:
            continue
        ax, ay = x1 + ta * dx, y1 + ta * dy
        bx, by = x1 + tb * dx, y1 + tb * dy
        mx, my = 0.5 * (ax + bx), 0.5 * (ay + by)
        in_flex = any(
            float(fz.x0_mm) <= mx <= float(fz.x1_mm) and float(fz.y0_mm) <= my <= float(fz.y1_mm)
            for fz in flex_zones
        )
        if in_flex:
            kind = "flex"
        else:
            near_seam = any(
                (float(fz.x0_mm) - taper_mm - 0.05 <= mx <= float(fz.x1_mm) + taper_mm + 0.05)
                and (float(fz.y0_mm) - taper_mm - 0.05 <= my <= float(fz.y1_mm) + taper_mm + 0.05)
                for fz in flex_zones
            )
            kind = "taper" if (near_seam and math.hypot(bx - ax, by - ay) <= taper_mm + 0.15) else "rigid"
        out.append((ax, ay, bx, by, kind))
    return out


def _apply_trombone_skew_match(
    shorter_segs: list[TraceSegment],
    mate_segs: list[TraceSegment],
    delta_mm: float,
    flex_zones: list[Any],
) -> list[TraceSegment]:
    """Insert an octilinear 135-deg trombone meander on a rigid segment to match diff-pair length."""
    if delta_mm <= 0.03 or not shorter_segs:
        return shorter_segs

    def _in_flex(x: float, y: float) -> bool:
        return any(
            float(fz.x0_mm) - 0.3 <= x <= float(fz.x1_mm) + 0.3
            and float(fz.y0_mm) - 0.3 <= y <= float(fz.y1_mm) + 0.3
            for fz in flex_zones
        )

    best_idx = -1
    best_len = 0.0
    for idx, s in enumerate(shorter_segs):
        slen = math.hypot(s.x2_mm - s.x1_mm, s.y2_mm - s.y1_mm)
        mx, my = 0.5 * (s.x1_mm + s.x2_mm), 0.5 * (s.y1_mm + s.y2_mm)
        if slen >= 1.4 and not _in_flex(mx, my) and not _in_flex(s.x1_mm, s.y1_mm) and not _in_flex(s.x2_mm, s.y2_mm):
            if slen > best_len:
                best_len = slen
                best_idx = idx
    if best_idx < 0:
        return shorter_segs

    seg = shorter_segs[best_idx]
    ux = (seg.x2_mm - seg.x1_mm) / best_len
    uy = (seg.y2_mm - seg.y1_mm) / best_len
    px_n, py_n = -uy, ux

    # Choose outward perpendicular direction away from mate_segs
    mx, my = 0.5 * (seg.x1_mm + seg.x2_mm), 0.5 * (seg.y1_mm + seg.y2_mm)
    def _min_dist_to_mate(qx: float, qy: float) -> float:
        d_min = 1e9
        for ms in mate_segs:
            if ms.layer != seg.layer:
                continue
            dx, dy = ms.x2_mm - ms.x1_mm, ms.y2_mm - ms.y1_mm
            lsq = dx * dx + dy * dy
            t = 0.0 if lsq <= 1e-12 else max(0.0, min(1.0, ((qx - ms.x1_mm) * dx + (qy - ms.y1_mm) * dy) / lsq))
            d_min = min(d_min, math.hypot(qx - (ms.x1_mm + t * dx), qy - (ms.y1_mm + t * dy)))
        return d_min

    d_plus = _min_dist_to_mate(mx + 0.35 * px_n, my + 0.35 * py_n)
    d_minus = _min_dist_to_mate(mx - 0.35 * px_n, my - 0.35 * py_n)
    if d_minus > d_plus:
        px_n, py_n = -px_n, -py_n

    # Build octilinear 135-degree trombone jog adding exact `delta_mm`
    r_max = 0.15
    pure_45_max = 2.0 * r_max * (math.sqrt(2.0) - 1.0)
    if delta_mm <= pure_45_max:
        r = delta_mm / (2.0 * (math.sqrt(2.0) - 1.0))
        h_perp = 0.0
    else:
        r = r_max
        h_perp = max(0.0, (delta_mm - pure_45_max) * 0.5)

    d_flat = min(0.65, max(0.25, best_len - 2.0 * r - 0.45))
    span_total = 2.0 * r + d_flat
    if span_total >= best_len - 0.15:
        return shorter_segs

    start_off = 0.5 * (best_len - span_total)
    p1x = seg.x1_mm + start_off * ux
    p1y = seg.y1_mm + start_off * uy
    q1x = p1x + r * ux + r * px_n
    q1y = p1y + r * uy + r * py_n
    q2x = q1x + h_perp * px_n
    q2y = q1y + h_perp * py_n
    q3x = q2x + d_flat * ux
    q3y = q2y + d_flat * uy
    q4x = q3x - h_perp * px_n
    q4y = q3y - h_perp * py_n
    p2x = p1x + span_total * ux
    p2y = p1y + span_total * uy

    pts = [(seg.x1_mm, seg.y1_mm), (p1x, p1y), (q1x, q1y)]
    if h_perp > 1e-4:
        pts.extend([(q2x, q2y), (q3x, q3y), (q4x, q4y)])
    else:
        pts.append((q3x, q3y))
    pts.extend([(p2x, p2y), (seg.x2_mm, seg.y2_mm)])

    replacement: list[TraceSegment] = []
    for i in range(len(pts) - 1):
        ax, ay = pts[i]
        bx, by = pts[i + 1]
        if math.hypot(bx - ax, by - ay) > 1e-4:
            replacement.append(
                TraceSegment(seg.net_id, seg.layer, round(ax, 4), round(ay, 4), round(bx, 4), round(by, 4), seg.width_mm)
            )
    return shorter_segs[:best_idx] + replacement + shorter_segs[best_idx + 1 :]


def route_differential_pair(
    state: BoardState,
    net_p: int,
    net_n: int,
    cost_field: np.ndarray | None = None,
    z_diff_target: float = 85.0,
    allow_soft_collision: bool = True,
) -> tuple[list[TraceSegment], list[Via], list[TraceSegment], list[Via], bool]:
    """Coupled differential-pair router with Rigid<->Cross-Hatched Flex impedance compensation & skew tuning (rev14)."""
    from env.metrics import FLEX_COPPER_THICKNESS_MM, FR4_ER, POLYIMIDE_ER, solve_diff_pair_geometry

    cfg = state.config
    res = state.res
    H, W = state.H, state.W
    num_layers = getattr(state, "num_layers", 2)
    bottom_layer = getattr(state, "bottom_layer", 1)

    p_pts = state.get_net_pin_endpoints(net_p)
    n_pts = state.get_net_pin_endpoints(net_n)
    if len(p_pts) != 2 or len(n_pts) != 2:
        cf = cost_field if cost_field is not None else np.zeros((num_layers, H, W), dtype=np.float32)
        sp, vp, okp = octilinear_astar_route(state, net_p, cf, allow_soft_collision=allow_soft_collision)
        sn, vn, okn = octilinear_astar_route(state, net_n, cf, allow_soft_collision=allow_soft_collision)
        return (sp, vp, sn, vn, okp and okn)

    # Pair endpoints by minimal Euclidean separation
    d_direct = math.hypot(p_pts[0][0] - n_pts[0][0], p_pts[0][1] - n_pts[0][1]) + math.hypot(
        p_pts[1][0] - n_pts[1][0], p_pts[1][1] - n_pts[1][1]
    )
    d_swap = math.hypot(p_pts[0][0] - n_pts[1][0], p_pts[0][1] - n_pts[1][1]) + math.hypot(
        p_pts[1][0] - n_pts[0][0], p_pts[1][1] - n_pts[0][1]
    )
    if d_direct <= d_swap:
        p_src, p_dst = p_pts[0], p_pts[1]
        n_src, n_dst = n_pts[0], n_pts[1]
    else:
        p_src, p_dst = p_pts[0], p_pts[1]
        n_src, n_dst = n_pts[1], n_pts[0]

    zones = list(getattr(state.netlist, "substrate_zones", []) or [])
    rigid_z = next((z for z in zones if getattr(z, "zone_type", "rigid") == "rigid"), None)
    flex_zones = [z for z in zones if getattr(z, "zone_type", "rigid") == "dynamic_flex"]
    flex_z = flex_zones[0] if flex_zones else None

    rigid_er = float(rigid_z.er) if rigid_z else float(getattr(cfg, "rigid_er", FR4_ER))
    rigid_h = float(rigid_z.prepreg_h_mm) if rigid_z else float(getattr(cfg, "prepreg_thickness_mm", 0.10))
    w_rigid, s_rigid = solve_diff_pair_geometry(
        z_diff_target, er=rigid_er, h_mm=rigid_h, hatch_fill=1.0, copper_thickness_mm=getattr(cfg, "copper_thickness_mm", 0.035)
    )
    if flex_z is not None:
        w_flex, s_flex = solve_diff_pair_geometry(
            z_diff_target,
            er=float(flex_z.er),
            h_mm=float(flex_z.prepreg_h_mm),
            hatch_fill=float(flex_z.hatch_fill_ratio),
            copper_thickness_mm=float(getattr(cfg, "flex_copper_thickness_mm", FLEX_COPPER_THICKNESS_MM)),
        )
    else:
        w_flex, s_flex = w_rigid, s_rigid

    pitch_rigid = w_rigid + s_rigid
    pitch_flex = w_flex + s_flex
    w_taper = round(0.5 * (w_rigid + w_flex), 4)

    c0_x = 0.5 * (p_src[0] + n_src[0])
    c0_y = 0.5 * (p_src[1] + n_src[1])
    c1_x = 0.5 * (p_dst[0] + n_dst[0])
    c1_y = 0.5 * (p_dst[1] + n_dst[1])

    # Build centerline A* masks permitting both net_p and net_n own pads
    via_diam_m = getattr(cfg, "microvia_diameter_mm", 0.30) if num_layers >= 8 else cfg.via_diameter_mm
    blk_p, via_blk_p, soft_p, own_p = _build_multilayer_clearance_masks(
        state, net_p, trace_width_mm=max(w_rigid, w_flex), via_diameter_mm=via_diam_m
    )
    blk_n, _, soft_n, own_n = _build_multilayer_clearance_masks(
        state, net_n, trace_width_mm=max(w_rigid, w_flex), via_diameter_mm=via_diam_m
    )
    # A cell is clear for the bundle if neither net's foreign-obstacle mask blocks it (except each other's own pads)
    blocked = (blk_p & blk_n) | (blk_p & (~own_n)) & (blk_n & (~own_p))
    # Ensure board outline & keepout layers are strictly respected
    blocked |= (state.outline_grid < 0.5)[None, :, :]
    keepout_mask = getattr(state, "keepout_layers", None)
    if keepout_mask is not None and keepout_mask.shape == blocked.shape:
        blocked |= keepout_mask
    soft_blocked = (soft_p & (~own_n)) | (soft_n & (~own_p))
    if keepout_mask is not None and keepout_mask.shape == soft_blocked.shape:
        soft_blocked |= keepout_mask
    soft_blocked |= (state.outline_grid < 0.5)[None, :, :]

    def _ep_nodes(x_mm: float, y_mm: float, lmask: int) -> list[tuple[int, int, int]]:
        gx, gy = state.mm_to_grid(x_mm, y_mm)
        nodes: list[tuple[int, int, int]] = []
        if lmask & 1:
            nodes.append((0, gx, gy))
        if lmask & 2:
            nodes.append((bottom_layer, gx, gy))
        return nodes or [(0, gx, gy)]

    start_nodes = _ep_nodes(c0_x, c0_y, p_src[2])
    target_nodes = _ep_nodes(c1_x, c1_y, p_dst[2])

    # Open narrow breakout collar around source/destination pad midpoints so centerline can escape cleanly
    for cx_m, cy_m, lmask in ((c0_x, c0_y, p_src[2]), (c1_x, c1_y, p_dst[2])):
        cgx, cgy = state.mm_to_grid(cx_m, cy_m)
        r_col = max(2, int(round(0.45 / res)))
        for lyr in ([0] if (lmask & 1) else []) + ([bottom_layer] if (lmask & 2) else []):
            blocked[lyr, max(0, cgy - r_col) : min(H, cgy + r_col + 1), max(0, cgx - r_col) : min(W, cgx + r_col + 1)] = False
            soft_blocked[lyr, max(0, cgy - r_col) : min(H, cgy + r_col + 1), max(0, cgx - r_col) : min(W, cgx + r_col + 1)] = False

    cf_full = np.zeros((num_layers, H, W), dtype=np.float32)
    if cost_field is not None:
        if cost_field.shape[0] == num_layers:
            cf_full = cost_field
        elif cost_field.shape[0] >= 2:
            cf_full[0] = cost_field[0]
            cf_full[bottom_layer] = cost_field[1]
    scaled_cost = (cfg.alpha_congestion * np.clip(cf_full, 0.0, 100.0)).astype(np.float32)
    scaled_cost[0] += 0.15 * state.courtyard_grid

    occ_layers = state.occ_layers.copy()
    soft_zero = np.zeros((num_layers, H, W), dtype=np.float32)
    soft_high = np.where(blocked & (~soft_blocked), 85.0, 0.0).astype(np.float32)
    via_blk_2d = via_blk_p.any(axis=0)

    if num_layers == 2:
        path = _astar_2pin(
            H=H,
            W=W,
            start_nodes=start_nodes,
            target_nodes=target_nodes,
            blocked=blocked,
            via_blocked=via_blk_2d,
            soft_penalty=soft_zero,
            cost_field=scaled_cost,
            occ_layers=occ_layers,
            net_id=net_p,
            gamma_via=cfg.gamma_via_cost,
        )
        if path is None and allow_soft_collision:
            path = _astar_2pin(
                H=H,
                W=W,
                start_nodes=start_nodes,
                target_nodes=target_nodes,
                blocked=soft_blocked,
                via_blocked=via_blk_2d,
                soft_penalty=soft_high,
                cost_field=scaled_cost,
                occ_layers=occ_layers,
                net_id=net_p,
                gamma_via=cfg.gamma_via_cost,
            )
    else:
        allowed_layers = np.zeros(num_layers, dtype=np.uint8)
        allowed_layers[0] = 1
        allowed_layers[bottom_layer] = 1
        if num_layers >= 8:
            allowed_layers[2] = 1
            allowed_layers[5] = 1
        elif num_layers >= 6:
            allowed_layers[2] = 1
            allowed_layers[3] = 1

        trans = np.full((num_layers, num_layers), -1.0, dtype=np.float32)
        if num_layers >= 8:
            legal_spans = get_legal_via_spans(num_layers, int(getattr(cfg, "max_lamination_cycles", 2)), str(getattr(cfg, "hdi_buildup_type", "2+4+2")))
            for la, lb in legal_spans:
                if la == 0 and lb == bottom_layer:
                    continue
                trans[la, lb] = trans[lb, la] = float(getattr(cfg, "gamma_microvia_cost", 3.5)) if (lb - la == 1) else float(getattr(cfg, "gamma_buried_via_cost", 7.5))
        else:
            for la in range(num_layers):
                for lb in range(num_layers):
                    if la != lb and allowed_layers[la] and allowed_layers[lb]:
                        trans[la, lb] = float(cfg.gamma_via_cost)

        path = _astar_multilayer(
            L=num_layers,
            H=H,
            W=W,
            start_nodes=start_nodes,
            target_nodes=target_nodes,
            blocked=blocked,
            via_blocked_layers=via_blk_p,
            soft_penalty=soft_zero,
            cost_field=scaled_cost,
            occ_layers=occ_layers,
            allowed_layers=allowed_layers,
            transition_costs=trans,
            net_id=net_p,
            max_expansions=260_000,
        )
        if path is None and allow_soft_collision:
            path = _astar_multilayer(
                L=num_layers,
                H=H,
                W=W,
                start_nodes=start_nodes,
                target_nodes=target_nodes,
                blocked=soft_blocked,
                via_blocked_layers=via_blk_p,
                soft_penalty=soft_high,
                cost_field=scaled_cost,
                occ_layers=occ_layers,
                allowed_layers=allowed_layers,
                transition_costs=trans,
                net_id=net_p,
                max_expansions=260_000,
            )

    if path is None:
        return ([], [], [], [], False)

    center_segs, center_vias = _compress_collinear_segments(
        path_cells=path,
        res=res,
        net_id=net_p,
        trace_width_mm=w_rigid,
        via_diameter_mm=cfg.via_diameter_mm,
        via_drill_mm=cfg.via_drill_mm,
        num_layers=num_layers,
        microvia_diameter_mm=getattr(cfg, "microvia_diameter_mm", 0.30),
        microvia_drill_mm=getattr(cfg, "microvia_drill_mm", 0.15),
        buried_via_diameter_mm=getattr(cfg, "buried_via_diameter_mm", 0.45),
        buried_via_drill_mm=getattr(cfg, "buried_via_drill_mm", 0.20),
    )

    # Split centerline segments at Rigid<->Flex seams and insert smooth taper spans
    split_segs: list[tuple[int, float, float, float, float, str]] = []
    for cs in center_segs:
        for ax, ay, bx, by, kind in _split_segment_at_flex_seams(cs.x1_mm, cs.y1_mm, cs.x2_mm, cs.y2_mm, flex_zones):
            split_segs.append((cs.layer, ax, ay, bx, by, kind))

    if not split_segs:
        return ([], [], [], [], False)

    # Determine consistent sign (+1 or -1 normal) matching source pad side (p_src vs n_src)
    _, ax0, ay0, bx0, by0, _ = split_segs[0]
    tl0 = math.hypot(bx0 - ax0, by0 - ay0) or 1.0
    n0x, n0y = -(by0 - ay0) / tl0, (bx0 - ax0) / tl0
    sign_p = 1.0 if ((p_src[0] - n_src[0]) * n0x + (p_src[1] - n_src[1]) * n0y) >= 0.0 else -1.0

    # Offset each centerline segment into parallel P and N polylines with exact mitered corners & smooth seam flares
    segs_p: list[TraceSegment] = []
    segs_n: list[TraceSegment] = []

    def _seg_params(kind: str) -> tuple[float, float]:
        if kind == "flex":
            return (w_flex, pitch_flex)
        if kind == "taper":
            return (w_taper, 0.5 * (pitch_rigid + pitch_flex))
        return (w_rigid, pitch_rigid)

    # Filter degenerate zero-length segments
    clean_splits = [
        (lyr, ax, ay, bx, by, kind)
        for (lyr, ax, ay, bx, by, kind) in split_segs
        if math.hypot(bx - ax, by - ay) >= 1e-4
    ]
    n_sp = len(clean_splits)
    seg_normals: list[tuple[float, float]] = []
    for _, ax, ay, bx, by, _ in clean_splits:
        sl = math.hypot(bx - ax, by - ay)
        seg_normals.append((-(by - ay) / sl, (bx - ax) / sl))

    # Compute mitered vertex offsets for start/end of every segment
    p_starts: list[tuple[float, float]] = []
    p_ends: list[tuple[float, float]] = []
    n_starts: list[tuple[float, float]] = []
    n_ends: list[tuple[float, float]] = []
    widths_list: list[float] = []

    for idx, (lyr, ax, ay, bx, by, kind) in enumerate(clean_splits):
        w_seg, pitch = _seg_params(kind)
        widths_list.append(w_seg)
        nax, nay = seg_normals[idx]

        # Start vertex offset (miter with previous same-layer segment if present)
        if idx > 0 and clean_splits[idx - 1][0] == lyr:
            pnx, pny = seg_normals[idx - 1]
            _, prev_pitch = _seg_params(clean_splits[idx - 1][5])
            avg_pitch = 0.5 * (prev_pitch + pitch)
            mx, my = 0.5 * (pnx + nax), 0.5 * (pny + nay)
            ml2 = mx * mx + my * my
            if ml2 > 0.25:
                mvx, mvy = mx / ml2, my / ml2
            else:
                mvx, mvy = nax, nay
            hp = 0.5 * avg_pitch * sign_p
        else:
            mvx, mvy = nax, nay
            hp = 0.5 * pitch * sign_p
        p_starts.append((ax + hp * mvx, ay + hp * mvy))
        n_starts.append((ax - hp * mvx, ay - hp * mvy))

        # End vertex offset (miter with next same-layer segment if present)
        if idx + 1 < n_sp and clean_splits[idx + 1][0] == lyr:
            nnx, nny = seg_normals[idx + 1]
            _, next_pitch = _seg_params(clean_splits[idx + 1][5])
            avg_pitch = 0.5 * (pitch + next_pitch)
            mx, my = 0.5 * (nax + nnx), 0.5 * (nay + nny)
            ml2 = mx * mx + my * my
            if ml2 > 0.25:
                mvx, mvy = mx / ml2, my / ml2
            else:
                mvx, mvy = nax, nay
            hp = 0.5 * avg_pitch * sign_p
        else:
            mvx, mvy = nax, nay
            hp = 0.5 * pitch * sign_p
        p_ends.append((bx + hp * mvx, by + hp * mvy))
        n_ends.append((bx - hp * mvx, by - hp * mvy))

    for idx, (lyr, _, _, _, _, _) in enumerate(clean_splits):
        pax, pay = p_starts[idx]
        pbx, pby = p_ends[idx]
        nax, nay = n_starts[idx]
        nbx, nby = n_ends[idx]
        w_seg = widths_list[idx]
        if math.hypot(pbx - pax, pby - pay) > 1e-4:
            segs_p.append(TraceSegment(net_p, lyr, round(pax, 4), round(pay, 4), round(pbx, 4), round(pby, 4), w_seg))
        if math.hypot(nbx - nax, nby - nay) > 1e-4:
            segs_n.append(TraceSegment(net_n, lyr, round(nax, 4), round(nay, 4), round(nbx, 4), round(nby, 4), w_seg))

    # Connect source & destination pads cleanly to the coupled P/N trunk (trimming back monotonically to avoid hairpin stubs)
    if segs_p and segs_n:
        def _trim_ends_from_pads(s_list: list[TraceSegment], sx: float, sy: float, dx: float, dy: float) -> None:
            # Remove tiny end segments that lie within 0.60mm of source/dest pad so entry stub is monotonic >=120 deg
            while len(s_list) > 1 and math.hypot(s_list[0].x2_mm - sx, s_list[0].y2_mm - sy) < 0.65:
                s_list.pop(0)
            while len(s_list) > 1 and math.hypot(s_list[-1].x1_mm - dx, s_list[-1].y1_mm - dy) < 0.65:
                s_list.pop()
            first = s_list[0]
            flen = math.hypot(first.x2_mm - first.x1_mm, first.y2_mm - first.y1_mm)
            if flen > 1.1:
                t_trim = 0.55 / flen
                first.x1_mm = round(first.x1_mm + t_trim * (first.x2_mm - first.x1_mm), 4)
                first.y1_mm = round(first.y1_mm + t_trim * (first.y2_mm - first.y1_mm), 4)
            last = s_list[-1]
            llen = math.hypot(last.x2_mm - last.x1_mm, last.y2_mm - last.y1_mm)
            if llen > 1.1:
                t_trim = 0.55 / llen
                last.x2_mm = round(last.x2_mm - t_trim * (last.x2_mm - last.x1_mm), 4)
                last.y2_mm = round(last.y2_mm - t_trim * (last.y2_mm - last.y1_mm), 4)

        _trim_ends_from_pads(segs_p, p_src[0], p_src[1], p_dst[0], p_dst[1])
        _trim_ends_from_pads(segs_n, n_src[0], n_src[1], n_dst[0], n_dst[1])
        if math.hypot(p_src[0] - segs_p[0].x1_mm, p_src[1] - segs_p[0].y1_mm) > 1e-3:
            segs_p.insert(0, TraceSegment(net_p, segs_p[0].layer, round(p_src[0], 4), round(p_src[1], 4), segs_p[0].x1_mm, segs_p[0].y1_mm, w_rigid))
        if math.hypot(n_src[0] - segs_n[0].x1_mm, n_src[1] - segs_n[0].y1_mm) > 1e-3:
            segs_n.insert(0, TraceSegment(net_n, segs_n[0].layer, round(n_src[0], 4), round(n_src[1], 4), segs_n[0].x1_mm, segs_n[0].y1_mm, w_rigid))
        if math.hypot(p_dst[0] - segs_p[-1].x2_mm, p_dst[1] - segs_p[-1].y2_mm) > 1e-3:
            segs_p.append(TraceSegment(net_p, segs_p[-1].layer, segs_p[-1].x2_mm, segs_p[-1].y2_mm, round(p_dst[0], 4), round(p_dst[1], 4), w_rigid))
        if math.hypot(n_dst[0] - segs_n[-1].x2_mm, n_dst[1] - segs_n[-1].y2_mm) > 1e-3:
            segs_n.append(TraceSegment(net_n, segs_n[-1].layer, segs_n[-1].x2_mm, segs_n[-1].y2_mm, round(n_dst[0], 4), round(n_dst[1], 4), w_rigid))

    # Offset any centerline vias symmetrically
    vias_p: list[Via] = []
    vias_n: list[Via] = []
    for cv in center_vias:
        half_p = 0.5 * pitch_rigid * sign_p
        vias_p.append(Via(net_p, round(cv.x_mm + half_p * n0x, 4), round(cv.y_mm + half_p * n0y, 4), cv.diameter_mm, cv.drill_mm, cv.start_layer, cv.end_layer, cv.via_type))
        vias_n.append(Via(net_n, round(cv.x_mm - half_p * n0x, 4), round(cv.y_mm - half_p * n0y, 4), cv.diameter_mm, cv.drill_mm, cv.start_layer, cv.end_layer, cv.via_type))

    # Phase/Skew Matching (<= 0.125mm) via localized trombone meander in rigid breakout
    len_p = sum(math.hypot(s.x2_mm - s.x1_mm, s.y2_mm - s.y1_mm) for s in segs_p)
    len_n = sum(math.hypot(s.x2_mm - s.x1_mm, s.y2_mm - s.y1_mm) for s in segs_n)
    skew = abs(len_p - len_n)
    if skew > 0.04:
        if len_p < len_n:
            segs_p = _apply_trombone_skew_match(segs_p, segs_n, skew, flex_zones)
        else:
            segs_n = _apply_trombone_skew_match(segs_n, segs_p, skew, flex_zones)

    return (segs_p, vias_p, segs_n, vias_n, True)
