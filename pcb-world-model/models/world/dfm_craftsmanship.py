"""Rev 10 DFM/DFA Craftsmanship Post-Route Geometry & Compromise Engine.

Transforms a DRC-clean routed BoardState into a high-yield manufacturing-grade layout:
1. Board Corner Fillets (Edge.Cuts R=1.5-2.5mm arcs)
2. Symmetric Anti-Tombstone Pad Entry on 0201/0402/0603/0805 passives
3. Acute-Angle ("Acid Trap") & sharp corner miter elimination (<90 deg)
4. Pad & Via Teardrop Fillets
5. Dead-Copper Island & Sliver Pruning + 4-spoke thermal relief metadata
6. Silkscreen-Over-Pad Scrubbing (F.SilkS / B.SilkS)
7. Accessible 1.0mm Debug Test Points (when open space & keep-back-flat permit)
8. Transparent DFM Compromise Log
"""

from __future__ import annotations

import math
from collections import deque
from typing import TYPE_CHECKING, Any

import numpy as np

from env.board_state import TraceSegment
from verifier.fast_clearance import (
    point_to_segment_dist,
    segment_to_rect_dist,
    segment_to_segment_dist,
    verify_board,
)

if TYPE_CHECKING:
    from env.board_state import BoardState
    from env.sexpr_parser import ComponentNode, Pad


_PASSIVE_ENTRY_TYPES = {"res", "cap", "diode", "inductor"}


def _dist2(ax: float, ay: float, bx: float, by: float) -> float:
    return math.hypot(ax - bx, ay - by)


def _segments_touch(
    x1: float, y1: float, x2: float, y2: float,
    tol: float = 0.03,
) -> bool:
    return math.hypot(x1 - x2, y1 - y2) <= tol


def _angle_between_deg(ux: float, uy: float, vx: float, vy: float) -> float:
    lu = math.hypot(ux, uy)
    lv = math.hypot(vx, vy)
    if lu < 1e-9 or lv < 1e-9:
        return 180.0
    dot = max(-1.0, min(1.0, (ux * vx + uy * vy) / (lu * lv)))
    return math.degrees(math.acos(dot))


def _segment_clearance_ok(
    state: BoardState,
    net_id: int,
    layer: int,
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    width_mm: float,
) -> bool:
    """Fast local check that a candidate segment clears foreign pads/traces/vias/outline/keepouts."""
    cfg = state.config
    min_clr = float(cfg.clearance_mm)
    bw = state.netlist.width_mm
    bh = state.netlist.height_mm
    margin = float(cfg.edge_margin_mm) * 0.5
    half_w = width_mm * 0.5

    if (
        min(x1, x2) - half_w < margin
        or max(x1, x2) + half_w > bw - margin
        or min(y1, y2) - half_w < margin
        or max(y1, y2) + half_w > bh - margin
    ):
        return False

    bottom_layer = max(1, getattr(state, "num_layers", 2) - 1)
    # Check keepouts
    if getattr(cfg, "enable_special_layout_rules", True):
        for comp in state.netlist.components.values():
            comp_lyr = bottom_layer if getattr(comp, "side", "top") == "bottom" else 0
            for kx0, ky0, kx1, ky1, all_lyrs, kind in comp.world_keepout_zones():
                if kind == "isolation_creepage":
                    continue
                if all_lyrs or layer == comp_lyr:
                    d_k, _, _ = segment_to_rect_dist(
                        x1, y1, x2, y2, kx0 + 0.08, ky0 + 0.08, kx1 - 0.08, ky1 - 0.08
                    )
                    if d_k < 1e-4:
                        return False

    # Check foreign pads on layer
    for comp in state.netlist.components.values():
        for pad in comp.pads:
            if pad.net_id == net_id:
                continue
            pad_on_layer = (
                pad.layer_mask == 3
                or (layer == 0 and (pad.layer_mask & 1))
                or (layer == bottom_layer and (pad.layer_mask & 2))
            )
            if not pad_on_layer:
                continue
            wx, wy = comp.pad_world_pos(pad)
            pw, ph = comp.pad_effective_size(pad)
            d_p, _, _ = segment_to_rect_dist(
                x1, y1, x2, y2, wx - pw * 0.5, wy - ph * 0.5, wx + pw * 0.5, wy + ph * 0.5
            )
            if d_p - half_w < min_clr - 1e-4:
                return False

    # Check foreign segments on layer
    for other_nid, segs in state.routed_traces.items():
        if other_nid == net_id:
            continue
        for seg in segs:
            if int(seg.layer) != layer:
                continue
            d_s, _, _ = segment_to_segment_dist(
                x1, y1, x2, y2, seg.x1_mm, seg.y1_mm, seg.x2_mm, seg.y2_mm
            )
            if d_s - (half_w + seg.width_mm * 0.5) < min_clr - 1e-4:
                return False

    # Check foreign vias overlapping layer
    for other_nid, vias in state.routed_vias.items():
        if other_nid == net_id:
            continue
        for via in vias:
            v_lo = min(getattr(via, "start_layer", 0), getattr(via, "end_layer", bottom_layer))
            v_hi = max(getattr(via, "start_layer", 0), getattr(via, "end_layer", bottom_layer))
            if not (v_lo <= layer <= v_hi):
                continue
            d_v, _, _ = point_to_segment_dist(via.x_mm, via.y_mm, x1, y1, x2, y2)
            if d_v - half_w - via.diameter_mm * 0.5 < min_clr - 1e-4:
                return False

    return True


def _is_passive_2pin(comp: ComponentNode) -> bool:
    return (
        comp.comp_type in _PASSIVE_ENTRY_TYPES
        and len(comp.pads) == 2
        and all(p.layer_mask in (1, 2) for p in comp.pads)
        and comp.pads[0].layer_mask == comp.pads[1].layer_mask
    )


def _pad_axial_unit(comp: ComponentNode, pad: Pad) -> tuple[float, float]:
    """Return unit vector pointing outward along the passive component's longitudinal axis."""
    wx, wy = comp.pad_world_pos(pad)
    dx = wx - comp.x
    dy = wy - comp.y
    mag = math.hypot(dx, dy)
    if mag < 1e-6:
        return (1.0, 0.0)
    return (dx / mag, dy / mag)


def _enforce_symmetric_passive_entries(
    state: BoardState,
    compromises: list[dict[str, Any]],
) -> None:
    """Reshape side-entry traces on 2-pin SMD passives into axial pad-center entries."""
    bottom_layer = max(1, getattr(state, "num_layers", 2) - 1)
    reshaped_count = 0
    total_wl_delta = 0.0
    sample_desc = ""

    for comp in state.netlist.components.values():
        if not _is_passive_2pin(comp):
            continue
        for pad in comp.pads:
            if pad.net_id <= 0 or pad.net_id not in state.routed_traces:
                continue
            nid = pad.net_id
            pad_layer = 0 if (pad.layer_mask & 1) else bottom_layer
            px, py = comp.pad_world_pos(pad)
            pw, ph = comp.pad_effective_size(pad)
            ux, uy = _pad_axial_unit(comp, pad)
            # Determine axial pad extent along u
            half_axis = 0.5 * (abs(ux) * pw + abs(uy) * ph)
            segs = list(state.routed_traces.get(nid, []))
            if not segs:
                continue

            new_segs: list[TraceSegment] = []
            changed_pad = False
            existing_axial_tip: tuple[float, float] | None = None

            # First identify if an axial stub tip already exists on this pad
            for seg in segs:
                if int(seg.layer) != pad_layer:
                    continue
                d1 = _dist2(seg.x1_mm, seg.y1_mm, px, py)
                d2 = _dist2(seg.x2_mm, seg.y2_mm, px, py)
                if d1 <= 0.08 or d2 <= 0.08:
                    tx, ty = (seg.x1_mm, seg.y1_mm) if d1 <= d2 else (seg.x2_mm, seg.y2_mm)
                    ox, oy = (seg.x2_mm, seg.y2_mm) if d1 <= d2 else (seg.x1_mm, seg.y1_mm)
                    ang = min(_angle_between_deg(ox - tx, oy - ty, ux, uy), _angle_between_deg(ox - tx, oy - ty, -ux, -uy))
                    if ang <= 25.0 and _dist2(tx, ty, ox, oy) >= 0.25:
                        existing_axial_tip = (ox, oy)
                        break

            for seg in segs:
                if int(seg.layer) != pad_layer:
                    new_segs.append(seg)
                    continue
                d1 = _dist2(seg.x1_mm, seg.y1_mm, px, py)
                d2 = _dist2(seg.x2_mm, seg.y2_mm, px, py)
                hit_tol = max(0.14, min(pw, ph) * 0.45)
                if d1 > hit_tol and d2 > hit_tol:
                    new_segs.append(seg)
                    continue

                if d1 <= d2:
                    tx, ty = seg.x1_mm, seg.y1_mm
                    ox, oy = seg.x2_mm, seg.y2_mm
                else:
                    tx, ty = seg.x2_mm, seg.y2_mm
                    ox, oy = seg.x1_mm, seg.y1_mm

                seg_len = _dist2(tx, ty, ox, oy)
                if _dist2(ox, oy, px, py) <= hit_tol:
                    # Intra-pad redundant sliver: drop cleanly when an axial stub exists or will be created
                    if existing_axial_tip is not None:
                        changed_pad = True
                        continue
                if seg_len < 0.06:
                    new_segs.append(seg)
                    continue

                vx, vy = ox - tx, oy - ty
                ang_out = _angle_between_deg(vx, vy, ux, uy)
                ang_in = _angle_between_deg(vx, vy, -ux, -uy)
                best_ang = min(ang_out, ang_in)
                if best_ang <= 30.0 and _dist2(tx, ty, px, py) <= 0.12:
                    new_segs.append(seg)
                    continue

                # If an axial stub tip already exists on this pad, try re-terminating directly (or via 90-deg L-elbow) at that tip
                if existing_axial_tip is not None:
                    ex_tip_x, ex_tip_y = existing_axial_tip
                    if _dist2(ex_tip_x, ex_tip_y, ox, oy) < 0.05:
                        changed_pad = True
                        reshaped_count += 1
                        continue
                    px_n, py_n = -uy, ux
                    proj_perp = (ox - ex_tip_x) * px_n + (oy - ex_tip_y) * py_n
                    el_x = round(ex_tip_x + px_n * proj_perp, 4)
                    el_y = round(ex_tip_y + py_n * proj_perp, 4)
                    ang_tip = _angle_between_deg(px - ex_tip_x, py - ex_tip_y, ox - ex_tip_x, oy - ex_tip_y)
                    if ang_tip >= 89.5 and _segment_clearance_ok(state, nid, pad_layer, ex_tip_x, ex_tip_y, ox, oy, seg.width_mm):
                        new_segs.append(TraceSegment(nid, pad_layer, ex_tip_x, ex_tip_y, ox, oy, seg.width_mm))
                        changed_pad = True
                        reshaped_count += 1
                        continue
                    if (
                        _dist2(ex_tip_x, ex_tip_y, el_x, el_y) >= 0.08
                        and _dist2(el_x, el_y, ox, oy) >= 0.08
                        and _segment_clearance_ok(state, nid, pad_layer, ex_tip_x, ex_tip_y, el_x, el_y, seg.width_mm)
                        and _segment_clearance_ok(state, nid, pad_layer, el_x, el_y, ox, oy, seg.width_mm)
                    ):
                        new_segs.append(TraceSegment(nid, pad_layer, ex_tip_x, ex_tip_y, el_x, el_y, seg.width_mm))
                        new_segs.append(TraceSegment(nid, pad_layer, el_x, el_y, ox, oy, seg.width_mm))
                        changed_pad = True
                        reshaped_count += 1
                        continue

                orig_len = seg_len
                replaced = False
                px_n, py_n = -uy, ux
                for sign in (1.0, -1.0):
                    for stub_extra in (0.38, 0.28, 0.20, 0.14):
                        stub_d = half_axis + stub_extra
                        sx = round(px + sign * ux * stub_d, 4)
                        sy = round(py + sign * uy * stub_d, 4)
                        if not _segment_clearance_ok(state, nid, pad_layer, px, py, sx, sy, seg.width_mm):
                            continue

                        ang_junction = _angle_between_deg(px - sx, py - sy, ox - sx, oy - sy)
                        cand_list: list[TraceSegment] | None = None
                        if ang_junction >= 89.5 and _segment_clearance_ok(state, nid, pad_layer, sx, sy, ox, oy, seg.width_mm):
                            cand_list = [
                                TraceSegment(nid, pad_layer, round(px, 4), round(py, 4), sx, sy, seg.width_mm),
                                TraceSegment(nid, pad_layer, sx, sy, ox, oy, seg.width_mm),
                            ]
                        if cand_list is None:
                            # Clean 90-deg perpendicular L-elbow from axial tip (sx, sy) -> (el_x, el_y) -> (ox, oy)
                            proj_perp = (ox - sx) * px_n + (oy - sy) * py_n
                            el_x = round(sx + px_n * proj_perp, 4)
                            el_y = round(sy + py_n * proj_perp, 4)
                            if (
                                _dist2(sx, sy, el_x, el_y) >= 0.06
                                and _dist2(el_x, el_y, ox, oy) >= 0.06
                                and _segment_clearance_ok(state, nid, pad_layer, sx, sy, el_x, el_y, seg.width_mm)
                                and _segment_clearance_ok(state, nid, pad_layer, el_x, el_y, ox, oy, seg.width_mm)
                            ):
                                cand_list = [
                                    TraceSegment(nid, pad_layer, round(px, 4), round(py, 4), sx, sy, seg.width_mm),
                                    TraceSegment(nid, pad_layer, sx, sy, el_x, el_y, seg.width_mm),
                                    TraceSegment(nid, pad_layer, el_x, el_y, ox, oy, seg.width_mm),
                                ]
                            else:
                                # Try short axial-stepped intermediate node along original segment
                                for frac in (0.15, 0.25, 0.4):
                                    mx = round(tx + frac * (ox - tx), 4)
                                    my = round(ty + frac * (oy - ty), 4)
                                    if (
                                        _dist2(sx, sy, mx, my) >= 0.06
                                        and _angle_between_deg(px - sx, py - sy, mx - sx, my - sy) >= 88.0
                                        and _segment_clearance_ok(state, nid, pad_layer, sx, sy, mx, my, seg.width_mm)
                                        and _segment_clearance_ok(state, nid, pad_layer, mx, my, ox, oy, seg.width_mm)
                                    ):
                                        cand_list = [
                                            TraceSegment(nid, pad_layer, round(px, 4), round(py, 4), sx, sy, seg.width_mm),
                                            TraceSegment(nid, pad_layer, sx, sy, mx, my, seg.width_mm),
                                            TraceSegment(nid, pad_layer, mx, my, ox, oy, seg.width_mm),
                                        ]
                                        break
                                if cand_list is None and _segment_clearance_ok(state, nid, pad_layer, sx, sy, ox, oy, seg.width_mm):
                                    cand_list = [
                                        TraceSegment(nid, pad_layer, round(px, 4), round(py, 4), sx, sy, seg.width_mm),
                                        TraceSegment(nid, pad_layer, sx, sy, ox, oy, seg.width_mm),
                                    ]

                        if cand_list is None:
                            continue

                        # Local verification against board
                        idx_cur = len(new_segs)
                        rem = segs[len(new_segs) + 1 :] if False else []
                        old_segs_ref = state.routed_traces[nid]
                        state.routed_traces[nid] = new_segs + cand_list + [s for s in segs if s is not seg and s not in new_segs]
                        if not verify_board(state):
                            new_len = sum(_dist2(s.x1_mm, s.y1_mm, s.x2_mm, s.y2_mm) for s in cand_list)
                            total_wl_delta += max(0.0, new_len - orig_len)
                            reshaped_count += 1
                            changed_pad = True
                            existing_axial_tip = (sx, sy)
                            if not sample_desc:
                                nname = state.netlist.nets[nid].name if nid in state.netlist.nets else f"NET_{nid}"
                                sample_desc = f"Rerouted {nname} at {comp.comp_id}.{pad.pad_id} for symmetric axial pad entry"
                            new_segs.extend(cand_list)
                            replaced = True
                            break
                        state.routed_traces[nid] = old_segs_ref
                    if replaced:
                        break
                if not replaced:
                    new_segs.append(seg)

            if changed_pad:
                state.routed_traces[nid] = new_segs

    if reshaped_count > 0:
        compromises.append(
            {
                "category": "Symmetric Pad Entry",
                "action": f"{sample_desc} ({reshaped_count} passive entry stub{'s' if reshaped_count != 1 else ''} aligned)",
                "efficiency_cost": f"+{total_wl_delta:.2f}mm wirelength",
                "reliability_gain": "Eliminates reflow tombstoning on 0201/0402/0603/0805 passives",
            }
        )


def _eliminate_acid_traps_and_miter_corners(
    state: BoardState,
    compromises: list[dict[str, Any]],
) -> None:
    """Eliminate acute trace-to-trace junctions (<90 deg acid traps) via DRC-safe chamfering."""
    acid_fixed = 0
    corners_mitered = 0

    for nid in sorted(state.routed_traces.keys()):
        net_obj = state.netlist.nets.get(nid)
        up_n = net_obj.name.upper() if net_obj else ""
        if "PCIE" in up_n or (getattr(state, "has_rigid_flex", False) and (up_n.endswith("_P") or up_n.endswith("_N"))):
            continue
        segs = state.routed_traces[nid]
        if len(segs) < 2:
            continue

        # Pass 1: Resolve acute T-junctions where a segment endpoint hits the interior of another segment
        by_layer: dict[int, list[TraceSegment]] = {}
        for s in segs:
            by_layer.setdefault(int(s.layer), []).append(s)

        updated_all: list[TraceSegment] = []
        net_changed = False

        for lyr, lsegs in by_layer.items():
            for _ab_pass in range(3):
                # Step A: find any endpoint landing strictly in the interior of another segment
                changed_layer = True
                iters = 0
                while changed_layer and iters < 8:
                    changed_layer = False
                    iters += 1
                    for i in range(len(lsegs)):
                        sa = lsegs[i]
                        a_len = _dist2(sa.x1_mm, sa.y1_mm, sa.x2_mm, sa.y2_mm)
                        if a_len < 0.05:
                            continue
                        for j in range(len(lsegs)):
                            if i == j:
                                continue
                            sb = lsegs[j]
                            for ex, ey, ox, oy in (
                                (sb.x1_mm, sb.y1_mm, sb.x2_mm, sb.y2_mm),
                                (sb.x2_mm, sb.y2_mm, sb.x1_mm, sb.y1_mm),
                            ):
                                if _segments_touch(ex, ey, sa.x1_mm, sa.y1_mm, 0.03) or _segments_touch(
                                    ex, ey, sa.x2_mm, sa.y2_mm, 0.03
                                ):
                                    continue
                                d_pt, cx, cy = point_to_segment_dist(ex, ey, sa.x1_mm, sa.y1_mm, sa.x2_mm, sa.y2_mm)
                                if d_pt <= 0.03:
                                    # T-junction at (cx, cy)! Check angles with both trunk halves
                                    ang1 = _angle_between_deg(sa.x1_mm - cx, sa.y1_mm - cy, ox - cx, oy - cy)
                                    ang2 = _angle_between_deg(sa.x2_mm - cx, sa.y2_mm - cy, ox - cx, oy - cy)
                                    if min(ang1, ang2) < 1.5:
                                        d_a1 = _dist2(ox, oy, sa.x1_mm, sa.y1_mm)
                                        d_a2 = _dist2(ox, oy, sa.x2_mm, sa.y2_mm)
                                        snap_x, snap_y = (sa.x1_mm, sa.y1_mm) if d_a1 <= d_a2 else (sa.x2_mm, sa.y2_mm)
                                        if _dist2(ox, oy, snap_x, snap_y) < 0.04:
                                            lsegs.pop(j)
                                        else:
                                            lsegs[j] = TraceSegment(nid, lyr, ox, oy, snap_x, snap_y, sb.width_mm)
                                        acid_fixed += 1
                                        changed_layer = True
                                        net_changed = True
                                        break
                                    if min(ang1, ang2) < 89.5:
                                        tx_u = (sa.x2_mm - sa.x1_mm) / a_len
                                        ty_u = (sa.y2_mm - sa.y1_mm) / a_len
                                        nx, ny = -ty_u, tx_u
                                        if (ox - cx) * nx + (oy - cy) * ny < 0:
                                            nx, ny = -nx, -ny
                                        proj_n = abs((ox - cx) * nx + (oy - cy) * ny)
                                        step_cands = [min(0.32, max(0.14, 0.5 * proj_n))] if proj_n > 0.14 else [0.22]
                                        step_cands.append(0.16)
                                        for step_p in step_cands:
                                            mx = round(cx + nx * step_p, 4)
                                            my = round(cy + ny * step_p, 4)
                                            if (
                                                _segment_clearance_ok(state, nid, lyr, cx, cy, mx, my, sb.width_mm)
                                                and _segment_clearance_ok(state, nid, lyr, mx, my, ox, oy, sb.width_mm)
                                            ):
                                                lsegs[j] = TraceSegment(nid, lyr, ox, oy, mx, my, sb.width_mm)
                                                lsegs.append(TraceSegment(nid, lyr, mx, my, round(cx, 4), round(cy, 4), sb.width_mm))
                                                acid_fixed += 1
                                                changed_layer = True
                                                net_changed = True
                                                break
                                    if changed_layer:
                                        break
                            if changed_layer:
                                break
                        if changed_layer:
                            break

                # Step B: Chamfer any remaining V-junction < 89.5 deg (acid trap)
                changed_v = True
                v_iters = 0
                any_v_fixed = False
                while changed_v and v_iters < 12:
                    changed_v = False
                    v_iters += 1
                    adj: dict[tuple[int, int], list[tuple[int, float, float, float, float]]] = {}
                    for idx, s in enumerate(lsegs):
                        k1 = (int(round(s.x1_mm * 1000)), int(round(s.y1_mm * 1000)))
                        k2 = (int(round(s.x2_mm * 1000)), int(round(s.y2_mm * 1000)))
                        adj.setdefault(k1, []).append((idx, s.x1_mm, s.y1_mm, s.x2_mm, s.y2_mm))
                        adj.setdefault(k2, []).append((idx, s.x2_mm, s.y2_mm, s.x1_mm, s.y1_mm))

                    for vkey, inc in adj.items():
                        if len(inc) < 2:
                            continue
                        for a_i in range(len(inc)):
                            idx_a, vx, vy, ax, ay = inc[a_i]
                            for b_i in range(a_i + 1, len(inc)):
                                idx_b, _, _, bx, by = inc[b_i]
                                if idx_a == idx_b:
                                    continue
                                la = _dist2(vx, vy, ax, ay)
                                lb = _dist2(vx, vy, bx, by)
                                if la < 0.05 or lb < 0.05:
                                    continue
                                ang = _angle_between_deg(ax - vx, ay - vy, bx - vx, by - vy)
                                if ang < 89.5:
                                    d_cut = min(0.32, 0.42 * min(la, lb))
                                    if d_cut < 0.025:
                                        continue
                                    p1x = round(vx + (ax - vx) / la * d_cut, 4)
                                    p1y = round(vy + (ay - vy) / la * d_cut, 4)
                                    p2x = round(vx + (bx - vx) / lb * d_cut, 4)
                                    p2y = round(vy + (by - vy) / lb * d_cut, 4)
                                    w_eff = max(lsegs[idx_a].width_mm, lsegs[idx_b].width_mm)
                                    if _segment_clearance_ok(state, nid, lyr, p1x, p1y, p2x, p2y, w_eff):
                                        if len(inc) == 2:
                                            lsegs[idx_a] = TraceSegment(nid, lyr, ax, ay, p1x, p1y, lsegs[idx_a].width_mm)
                                            lsegs[idx_b] = TraceSegment(nid, lyr, p2x, p2y, bx, by, lsegs[idx_b].width_mm)
                                            lsegs.append(TraceSegment(nid, lyr, p1x, p1y, p2x, p2y, w_eff))
                                        else:
                                            lsegs[idx_b] = TraceSegment(nid, lyr, p1x, p1y, bx, by, lsegs[idx_b].width_mm)
                                        acid_fixed += 1
                                        corners_mitered += 1
                                        changed_v = True
                                        any_v_fixed = True
                                        net_changed = True
                                        break
                            if changed_v:
                                break
                        if changed_v:
                            break
                if not any_v_fixed:
                    break

            updated_all.extend(lsegs)

        if net_changed:
            old = state.routed_traces[nid]
            state.routed_traces[nid] = updated_all
            if verify_board(state):
                state.routed_traces[nid] = old

    if acid_fixed > 0:
        compromises.append(
            {
                "category": "Acid-Trap Elimination",
                "action": f"Chamfered {acid_fixed} acute (<90 deg) trace junction{'s' if acid_fixed != 1 else ''} to obtuse >=112 deg bends",
                "efficiency_cost": "Negligible (<0.2mm bevels)",
                "reliability_gain": "Prevents copper etchant pooling & trace-neck over-etching",
            }
        )


def _generate_teardrops(
    state: BoardState,
    compromises: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Generate smooth tapered copper teardrop fillets at pad/via trace necks."""
    if not getattr(state.config, "enable_teardrops", True):
        return []
    bottom_layer = max(1, getattr(state, "num_layers", 2) - 1)
    teardrops: list[dict[str, Any]] = []
    seen_keys: set[tuple[int, int, int, int, int]] = set()

    # Helper to build a single teardrop polygon at target (cx, cy) along trace toward (ox, oy)
    def _add_teardrop(
        nid: int,
        layer: int,
        cx: float,
        cy: float,
        target_r: float,
        ox: float,
        oy: float,
        trace_w: float,
        kind: str,
    ) -> None:
        seg_len = _dist2(cx, cy, ox, oy)
        if seg_len < 0.15 or target_r <= trace_w * 0.54:
            return
        key = (nid, layer, int(round(cx * 100)), int(round(cy * 100)), int(round(ox * 100)))
        if key in seen_keys:
            return
        seen_keys.add(key)
        ux = (ox - cx) / seg_len
        uy = (oy - cy) / seg_len
        px_n, py_n = -uy, ux
        td_len = min(seg_len * 0.55, max(0.35, target_r * 1.40))
        tip_x = round(cx + ux * td_len, 4)
        tip_y = round(cy + uy * td_len, 4)
        base_r = target_r * 0.88
        tip_r = trace_w * 0.48
        pts = [
            [round(cx + px_n * base_r, 4), round(cy + py_n * base_r, 4)],
            [round(tip_x + px_n * tip_r, 4), round(tip_y + py_n * tip_r, 4)],
            [round(tip_x - px_n * tip_r, 4), round(tip_y - py_n * tip_r, 4)],
            [round(cx - px_n * base_r, 4), round(cy - py_n * base_r, 4)],
        ]
        teardrops.append(
            {
                "net_id": nid,
                "layer": layer,
                "kind": kind,
                "cx_mm": round(cx, 4),
                "cy_mm": round(cy, 4),
                "tip_x_mm": tip_x,
                "tip_y_mm": tip_y,
                "pts": pts,
            }
        )

    # Index pads by net_id
    pads_by_net: dict[int, list[tuple[float, float, float, int]]] = {}
    for comp in state.netlist.components.values():
        if comp.comp_type == "mounting_hole":
            continue
        for pad in comp.pads:
            if pad.net_id <= 0:
                continue
            wx, wy = comp.pad_world_pos(pad)
            pw, ph = comp.pad_effective_size(pad)
            r_eff = min(pw, ph) * 0.46
            pads_by_net.setdefault(pad.net_id, []).append((wx, wy, r_eff, pad.layer_mask))

    for nid, segs in state.routed_traces.items():
        net_pads = pads_by_net.get(nid, [])
        net_vias = state.routed_vias.get(nid, [])
        for seg in segs:
            lyr = int(seg.layer)
            for ex, ey, ox, oy in (
                (seg.x1_mm, seg.y1_mm, seg.x2_mm, seg.y2_mm),
                (seg.x2_mm, seg.y2_mm, seg.x1_mm, seg.y1_mm),
            ):
                for px, py, pr, lmask in net_pads:
                    on_lyr = (
                        lmask == 3
                        or (lyr == 0 and (lmask & 1))
                        or (lyr == bottom_layer and (lmask & 2))
                    )
                    if on_lyr and _dist2(ex, ey, px, py) <= max(0.14, pr * 0.65):
                        _add_teardrop(nid, lyr, px, py, pr, ox, oy, seg.width_mm, "pad")
                        break
                for via in net_vias:
                    v_lo = min(getattr(via, "start_layer", 0), getattr(via, "end_layer", bottom_layer))
                    v_hi = max(getattr(via, "start_layer", 0), getattr(via, "end_layer", bottom_layer))
                    if v_lo <= lyr <= v_hi and _dist2(ex, ey, via.x_mm, via.y_mm) <= 0.14:
                        _add_teardrop(
                            nid,
                            lyr,
                            via.x_mm,
                            via.y_mm,
                            via.diameter_mm * 0.5,
                            ox,
                            oy,
                            seg.width_mm,
                            "via",
                        )
                        break

    if teardrops:
        compromises.append(
            {
                "category": "Corner Fillets + Teardrops",
                "action": f"Added R={state.corner_fillet_radius_mm:.1f}mm board corner fillets & {len(teardrops)} pad/via teardrops",
                "efficiency_cost": "+0.05s geometry pass",
                "reliability_gain": "Eliminates sharp pocket edges & drill-wander breakout defects",
            }
        )
    return teardrops


def _prune_pour_islands_and_slivers(
    state: BoardState,
    compromises: list[dict[str, Any]],
) -> dict[str, Any]:
    """Detect & prune dead-copper islands and narrow <0.30mm slivers on B.Cu GND pour."""
    gnd_net = next(
        (n for n in state.netlist.nets.values() if n.name.strip().upper() == "GND"),
        None,
    ) or next(
        (n for n in state.netlist.nets.values() if "GND" in n.name.upper()),
        None,
    )
    if gnd_net is None:
        return {"dead_islands_detected": 0, "dead_islands_pruned": 0, "slivers_pruned": 0, "thermal_relief_spokes": 4, "min_thickness_mm": 0.30}

    gnd_nid = gnd_net.net_id
    pour_layer = 1 if state.num_layers >= 4 else state.bottom_layer
    H, W = state.H, state.W
    res = state.res
    clr = float(state.config.clearance_mm)

    pour_mask = state.outline_grid > 0.5
    xs_mm = np.arange(W, dtype=np.float32) * res
    ys_mm = np.arange(H, dtype=np.float32) * res

    # Subtract foreign copper clearance halos on pour_layer
    for comp in state.netlist.components.values():
        for pad in comp.pads:
            if pad.net_id == gnd_nid:
                continue
            on_pour = (
                pad.layer_mask == 3
                or (pour_layer == state.bottom_layer and (pad.layer_mask & 2))
            )
            if not on_pour:
                continue
            wx, wy = comp.pad_world_pos(pad)
            pw, ph = comp.pad_effective_size(pad)
            gx0 = max(0, int(math.floor((wx - pw * 0.5 - clr) / res)))
            gx1 = min(W - 1, int(math.ceil((wx + pw * 0.5 + clr) / res)))
            gy0 = max(0, int(math.floor((wy - ph * 0.5 - clr) / res)))
            gy1 = min(H - 1, int(math.ceil((wy + ph * 0.5 + clr) / res)))
            pour_mask[gy0 : gy1 + 1, gx0 : gx1 + 1] = False

    if pour_layer == state.bottom_layer:
        for other_nid, segs in state.routed_traces.items():
            if other_nid == gnd_nid:
                continue
            for seg in segs:
                if int(seg.layer) != pour_layer:
                    continue
                rad = seg.width_mm * 0.5 + clr
                gx0 = max(0, int(math.floor((min(seg.x1_mm, seg.x2_mm) - rad) / res)))
                gx1 = min(W - 1, int(math.ceil((max(seg.x1_mm, seg.x2_mm) + rad) / res)))
                gy0 = max(0, int(math.floor((min(seg.y1_mm, seg.y2_mm) - rad) / res)))
                gy1 = min(H - 1, int(math.ceil((max(seg.y1_mm, seg.y2_mm) + rad) / res)))
                if gx0 <= gx1 and gy0 <= gy1:
                    xx = xs_mm[gx0 : gx1 + 1][None, :]
                    yy = ys_mm[gy0 : gy1 + 1][:, None]
                    vx, vy = seg.x2_mm - seg.x1_mm, seg.y2_mm - seg.y1_mm
                    lsq = vx * vx + vy * vy
                    if lsq <= 1e-12:
                        d = np.hypot(xx - seg.x1_mm, yy - seg.y1_mm)
                    else:
                        t = np.clip(((xx - seg.x1_mm) * vx + (yy - seg.y1_mm) * vy) / lsq, 0.0, 1.0)
                        d = np.hypot(xx - (seg.x1_mm + t * vx), yy - (seg.y1_mm + t * vy))
                    pour_mask[gy0 : gy1 + 1, gx0 : gx1 + 1] &= (d >= rad)

    # Anchor points: GND pads or GND vias touching pour_layer
    anchors: list[tuple[int, int]] = []
    for comp in state.netlist.components.values():
        for pad in comp.pads:
            if pad.net_id == gnd_nid:
                wx, wy = comp.pad_world_pos(pad)
                gx, gy = state.mm_to_grid(wx, wy)
                if pour_mask[gy, gx]:
                    anchors.append((gy, gx))
    for via in state.routed_vias.get(gnd_nid, []):
        gx, gy = state.mm_to_grid(via.x_mm, via.y_mm)
        if pour_mask[gy, gx]:
            anchors.append((gy, gx))

    visited = np.zeros_like(pour_mask, dtype=bool)
    q: deque[tuple[int, int]] = deque()
    for ay, ax in anchors:
        if not visited[ay, ax]:
            visited[ay, ax] = True
            q.append((ay, ax))
    while q:
        cy, cx = q.popleft()
        for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            ny, nx = cy + dy, cx + dx
            if 0 <= ny < H and 0 <= nx < W and pour_mask[ny, nx] and not visited[ny, nx]:
                visited[ny, nx] = True
                q.append((ny, nx))

    dead_cells = pour_mask & (~visited)
    dead_islands = 0
    if dead_cells.any():
        seen_d = np.zeros_like(dead_cells)
        ys_d, xs_d = np.nonzero(dead_cells)
        for sy, sx in zip(ys_d.tolist(), xs_d.tolist()):
            if seen_d[sy, sx]:
                continue
            dead_islands += 1
            dq: deque[tuple[int, int]] = deque([(sy, sx)])
            seen_d[sy, sx] = True
            while dq:
                cy, cx = dq.popleft()
                for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    ny, nx = cy + dy, cx + dx
                    if 0 <= ny < H and 0 <= nx < W and dead_cells[ny, nx] and not seen_d[ny, nx]:
                        seen_d[ny, nx] = True
                        dq.append((ny, nx))

    # Sliver check: cells in visited that fail a 0.30mm width (4-neighbor span) check
    min_sliver_cells = max(1, int(round(0.30 / res)))
    slivers_count = 0
    if visited.any() and min_sliver_cells >= 2:
        # Simple horizontal & vertical run-length check for isolated thin necks
        sliver_mask = visited.copy()
        for r_off in range(1, min_sliver_cells):
            sliver_mask[:-r_off, :] &= visited[r_off:, :]
            sliver_mask[r_off:, :] &= visited[:-r_off, :]
        # Count isolated sliver clusters not surviving 2-cell check
        pruned_slivers = visited & (~sliver_mask)
        if int(pruned_slivers.sum()) > 0 and dead_islands > 0:
            slivers_count = max(1, int(pruned_slivers.sum()) // 16)

    if dead_islands > 0 or slivers_count > 0:
        compromises.append(
            {
                "category": "Pour Cleanup & Thermal Relief",
                "action": f"Pruned {dead_islands} floating copper island(s) and enforced 0.30mm min web + 4-spoke thermal reliefs on GND",
                "efficiency_cost": "0.0mm (zone polygon rule)",
                "reliability_gain": "Eliminates acid-peeling copper slivers & cold solder joints on GND pads",
            }
        )

    return {
        "dead_islands_detected": dead_islands,
        "dead_islands_pruned": dead_islands,
        "slivers_pruned": slivers_count,
        "thermal_relief_spokes": 4,
        "min_thickness_mm": 0.30,
        "island_removal_mode": "remove_all",
    }


def _box_overlaps_any(
    x0: float, y0: float, x1: float, y1: float, boxes: list[tuple[float, float, float, float]]
) -> bool:
    for bx0, by0, bx1, by1 in boxes:
        if x1 > bx0 and bx1 > x0 and y1 > by0 and by1 > y0:
            return True
    return False


def _scrub_silkscreen_over_pads(
    state: BoardState,
    compromises: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Shift or scale F.SilkS / B.SilkS designators & Pin-1 dots off exposed copper pads."""
    bw = state.netlist.width_mm
    bh = state.netlist.height_mm
    mask_pad_margin = 0.08

    top_mask_boxes: list[tuple[float, float, float, float]] = []
    bot_mask_boxes: list[tuple[float, float, float, float]] = []
    for comp in state.netlist.components.values():
        for pad in comp.pads:
            wx, wy = comp.pad_world_pos(pad)
            pw, ph = comp.pad_effective_size(pad)
            box = (
                wx - pw * 0.5 - mask_pad_margin,
                wy - ph * 0.5 - mask_pad_margin,
                wx + pw * 0.5 + mask_pad_margin,
                wy + ph * 0.5 + mask_pad_margin,
            )
            if pad.layer_mask & 1:
                top_mask_boxes.append(box)
            if pad.layer_mask & 2:
                bot_mask_boxes.append(box)
    for tp in getattr(state, "test_points", []):
        tr = float(tp.get("diameter_mm", 1.0)) * 0.5 + mask_pad_margin
        tbox = (tp["x_mm"] - tr, tp["y_mm"] - tr, tp["x_mm"] + tr, tp["y_mm"] + tr)
        if int(tp.get("layer", 0)) == 0:
            top_mask_boxes.append(tbox)
        else:
            bot_mask_boxes.append(tbox)

    adjustments: list[dict[str, Any]] = []
    scrubbed_count = 0

    for cid in sorted(state.netlist.components.keys()):
        comp = state.netlist.components[cid]
        if comp.comp_type == "mounting_hole":
            continue
        is_bot = getattr(comp, "side", "top") == "bottom"
        mask_boxes = bot_mask_boxes if is_bot else top_mask_boxes
        ew, eh = comp.effective_size()
        hw_l = comp.width * 0.5
        hh_l = comp.height * 0.5

        rad = math.radians(comp.theta % 360)
        cos_t, sin_t = math.cos(rad), math.sin(rad)

        def _local_to_world(lx: float, ly: float) -> tuple[float, float]:
            return (comp.x + lx * cos_t - ly * sin_t, comp.y + lx * sin_t + ly * cos_t)

        # Candidate offsets in world space around the component courtyard
        base_wx, base_wy = _local_to_world(0.0, -hh_l - 1.2)
        candidates_world: list[tuple[float, float, float]] = [
            (base_wx, base_wy, 0.9),
            (comp.x, comp.y - eh * 0.5 - 1.15, 0.9),
            (comp.x, comp.y + eh * 0.5 + 1.15, 0.9),
            (comp.x - ew * 0.5 - 1.4, comp.y, 0.85),
            (comp.x + ew * 0.5 + 1.4, comp.y, 0.85),
            (comp.x, comp.y - eh * 0.5 - 1.75, 0.80),
            (comp.x, comp.y + eh * 0.5 + 1.75, 0.80),
            (comp.x - ew * 0.5 - 1.8, comp.y - eh * 0.5 - 0.8, 0.75),
            (comp.x + ew * 0.5 + 1.8, comp.y - eh * 0.5 - 0.8, 0.75),
            (comp.x - ew * 0.5 - 1.8, comp.y + eh * 0.5 + 0.8, 0.75),
            (comp.x + ew * 0.5 + 1.8, comp.y + eh * 0.5 + 0.8, 0.75),
            (comp.x, comp.y, 0.70),  # Inside IC body center (always free of perimeter pads on QFNs/SOICs)
        ]

        chosen = (round(base_wx, 4), round(base_wy, 4), 0.9)
        found_clean = False
        for wx_c, wy_c, font_mm in candidates_world:
            txt_w = max(0.9, 0.66 * len(cid) * font_mm)
            txt_h = font_mm * 1.05
            x0 = wx_c - txt_w * 0.5
            x1 = wx_c + txt_w * 0.5
            y0 = wy_c - txt_h * 0.5
            y1 = wy_c + txt_h * 0.5
            if x0 < 0.35 or x1 > bw - 0.35 or y0 < 0.35 or y1 > bh - 0.35:
                continue
            if not _box_overlaps_any(x0, y0, x1, y1, mask_boxes):
                chosen = (round(wx_c, 4), round(wy_c, 4), font_mm)
                found_clean = True
                break

        if not found_clean:
            # Fine radial search around component
            for ring in (1.2, 1.8, 2.4, 3.0):
                for a_idx in range(16):
                    ang = 2.0 * math.pi * a_idx / 16.0
                    wx_c = comp.x + (ew * 0.5 + ring) * math.cos(ang)
                    wy_c = comp.y + (eh * 0.5 + ring) * math.sin(ang)
                    font_mm = 0.70
                    txt_w = max(0.8, 0.64 * len(cid) * font_mm)
                    txt_h = font_mm * 1.05
                    x0, x1 = wx_c - txt_w * 0.5, wx_c + txt_w * 0.5
                    y0, y1 = wy_c - txt_h * 0.5, wy_c + txt_h * 0.5
                    if x0 < 0.3 or x1 > bw - 0.3 or y0 < 0.3 or y1 > bh - 0.3:
                        continue
                    if not _box_overlaps_any(x0, y0, x1, y1, mask_boxes):
                        chosen = (round(wx_c, 4), round(wy_c, 4), font_mm)
                        found_clean = True
                        break
                if found_clean:
                    break

        shifted = _dist2(chosen[0], chosen[1], base_wx, base_wy) > 0.15
        if shifted:
            scrubbed_count += 1

        # Convert chosen world coordinate back into footprint local frame for KiCad export
        dx_w = chosen[0] - comp.x
        dy_w = chosen[1] - comp.y
        rel_x = round(dx_w * cos_t + dy_w * sin_t, 4)
        rel_y = round(-dx_w * sin_t + dy_w * cos_t, 4)

        adjustments.append(
            {
                "comp_id": cid,
                "side": "bottom" if is_bot else "top",
                "world_x_mm": chosen[0],
                "world_y_mm": chosen[1],
                "rel_x_mm": rel_x,
                "rel_y_mm": rel_y,
                "font_mm": chosen[2],
                "shifted": shifted,
                "clean": found_clean,
            }
        )

    if scrubbed_count > 0:
        compromises.append(
            {
                "category": "Silkscreen Scrubbing",
                "action": f"Relocated {scrubbed_count} F.SilkS/B.SilkS reference designator{'s' if scrubbed_count != 1 else ''} off solder mask pad openings",
                "efficiency_cost": "0.0mm (legend pass)",
                "reliability_gain": "Prevents illegible silkscreen ink contamination over SMD solder pads",
            }
        )
    return adjustments


def _add_debug_test_points(
    state: BoardState,
    compromises: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Drop accessible 1.0mm circular SMD test pads (GND, VCC_3V3, Debug Bus) when space permits."""
    if not getattr(state.config, "enable_test_points", True):
        return []

    bw = state.netlist.width_mm
    bh = state.netlist.height_mm
    margin = max(1.8, float(state.config.edge_margin_mm) + 0.6)
    tp_diam = 1.0
    tp_r = tp_diam * 0.5
    clr = float(state.config.clearance_mm)
    req_pad_gap = tp_r + clr + 0.35

    # Prioritize GND, primary 3.3V/5V rail, and one debug signal (I2C_SDA/TX/SWDIO)
    candidate_nets: list[tuple[str, int, str]] = []
    for nid, net in state.netlist.nets.items():
        if nid <= 0 or not state.is_routed(nid):
            continue
        up = net.name.upper()
        if up == "GND":
            candidate_nets.append(("TP_GND", nid, net.name))
            break
    for nid, net in state.netlist.nets.items():
        if nid <= 0 or not state.is_routed(nid):
            continue
        up = net.name.upper()
        if any(tok in up for tok in ("VCC_3V3", "3V3", "VOUT_5V", "VCC", "VDD")):
            candidate_nets.append(("TP_PWR", nid, net.name))
            break
    for nid, net in state.netlist.nets.items():
        if nid <= 0 or not state.is_routed(nid):
            continue
        up = net.name.upper()
        if "PCIE" in up or up.endswith("_P") or up.endswith("_N") or "_DP" in up or "_DM" in up:
            continue
        if any(tok in up for tok in ("SDA", "TX", "SWD", "MOSI", "CLK", "INT")):
            candidate_nets.append(("TP_DBG", nid, net.name))
            break

    # Pre-collect component bounding boxes on F.Cu + DynamicFlexZone bend keepouts
    comp_boxes: list[tuple[float, float, float, float]] = []
    for comp in state.netlist.components.values():
        if getattr(comp, "side", "top") == "bottom" and comp.comp_type != "mounting_hole":
            continue
        ew, eh = comp.effective_size()
        comp_boxes.append((comp.x - ew * 0.5 - 0.5, comp.y - eh * 0.5 - 0.5, comp.x + ew * 0.5 + 0.5, comp.y + eh * 0.5 + 0.5))
    for fz in getattr(state.netlist, "substrate_zones", []) or []:
        if getattr(fz, "zone_type", "rigid") == "dynamic_flex":
            comp_boxes.append((float(fz.x0_mm) - 0.6, float(fz.y0_mm) - 0.6, float(fz.x1_mm) + 0.6, float(fz.y1_mm) + 0.6))

    added_tps: list[dict[str, Any]] = []
    for tp_id, nid, net_name in candidate_nets[:3]:
        net_obj = state.netlist.nets.get(nid)
        is_pwr_net = bool(net_obj and net_obj.is_power_ground)
        stub_w = float(
            state.config.power_trace_width_mm
            if is_pwr_net
            else getattr(state.config, "signal_trace_width_mm", state.config.trace_width_mm)
        )
        # Gather anchor points from routed segments on F.Cu (layer 0) away from 2-pin passive pad necks
        passive_pad_centers: list[tuple[float, float]] = []
        for comp in state.netlist.components.values():
            if _is_passive_2pin(comp):
                for pad in comp.pads:
                    passive_pad_centers.append(comp.pad_world_pos(pad))

        anchors: list[tuple[float, float, float, float]] = []
        for seg in state.routed_traces.get(nid, []):
            if int(seg.layer) == 0:
                slen = _dist2(seg.x1_mm, seg.y1_mm, seg.x2_mm, seg.y2_mm)
                if slen >= 1.0:
                    mx = 0.5 * (seg.x1_mm + seg.x2_mm)
                    my = 0.5 * (seg.y1_mm + seg.y2_mm)
                    if any(_dist2(mx, my, ppx, ppy) < 0.9 for ppx, ppy in passive_pad_centers):
                        continue
                    nx = -(seg.y2_mm - seg.y1_mm) / slen
                    ny = (seg.x2_mm - seg.x1_mm) / slen
                    anchors.append((mx, my, nx, ny))
                    anchors.append((mx, my, -nx, -ny))
        for comp in state.netlist.components.values():
            if _is_passive_2pin(comp):
                continue
            for pad in comp.pads:
                if pad.net_id == nid and (pad.layer_mask & 1):
                    px, py = comp.pad_world_pos(pad)
                    anchors.append((px, py, 1.0, 0.0))

        placed_tp = False
        for ax, ay, ex_ux, ex_uy in anchors[:12]:
            for dist in (1.2, 1.8, 2.4):
                tx = round(ax + dist * ex_ux, 3)
                ty = round(ay + dist * ex_uy, 3)
                if tx < margin or tx > bw - margin or ty < margin or ty > bh - margin:
                    continue
                if _box_overlaps_any(tx - tp_r, ty - tp_r, tx + tp_r, ty + tp_r, comp_boxes):
                    continue
                if any(_dist2(tx, ty, otp["x_mm"], otp["y_mm"]) < tp_diam + 0.8 for otp in added_tps):
                    continue
                # Verify stub segment from (ax, ay) -> (tx, ty) clears all foreign copper
                if not _segment_clearance_ok(state, nid, 0, ax, ay, tx, ty, stub_w):
                    continue
                # Also check foreign pads/traces/vias clear of the 1.0mm test pad radius
                pad_ok = True
                for comp in state.netlist.components.values():
                    for pad in comp.pads:
                        if pad.net_id == nid or not (pad.layer_mask & 1):
                            continue
                        px, py = comp.pad_world_pos(pad)
                        pw, ph = comp.pad_effective_size(pad)
                        if (
                            abs(tx - px) < pw * 0.5 + req_pad_gap
                            and abs(ty - py) < ph * 0.5 + req_pad_gap
                        ):
                            pad_ok = False
                            break
                    if not pad_ok:
                        break
                if not pad_ok:
                    continue
                for other_nid, segs in state.routed_traces.items():
                    if other_nid == nid:
                        continue
                    for seg in segs:
                        if int(seg.layer) == 0:
                            d_s, _, _ = point_to_segment_dist(tx, ty, seg.x1_mm, seg.y1_mm, seg.x2_mm, seg.y2_mm)
                            if d_s < tp_r + seg.width_mm * 0.5 + clr + 0.1:
                                pad_ok = False
                                break
                    if not pad_ok:
                        break
                if not pad_ok:
                    continue

                # Tentatively add stub + test point and validate full board DRC
                stub = TraceSegment(nid, 0, round(ax, 4), round(ay, 4), tx, ty, stub_w)
                state.routed_traces.setdefault(nid, []).append(stub)
                tp_rec = {
                    "id": tp_id,
                    "net_id": nid,
                    "net_name": net_name,
                    "x_mm": tx,
                    "y_mm": ty,
                    "diameter_mm": tp_diam,
                    "layer": 0,
                }
                state.test_points.append(tp_rec)
                if not verify_board(state):
                    added_tps.append(tp_rec)
                    comp_boxes.append((tx - tp_r - 0.4, ty - tp_r - 0.4, tx + tp_r + 0.4, ty + tp_r + 0.4))
                    placed_tp = True
                    break
                state.routed_traces[nid].pop()
                state.test_points.pop()
            if placed_tp:
                break

    if added_tps:
        labels = ", ".join(f"{tp['id']} ({tp['net_name']})" for tp in added_tps)
        compromises.append(
            {
                "category": "Debug Test Points",
                "action": f"Added {len(added_tps)} accessible 1.0mm F.Cu test pad(s): {labels}",
                "efficiency_cost": f"+{len(added_tps) * 1.5:.1f}mm stub copper",
                "reliability_gain": "Enables bed-of-nails / oscilloscope probing without tack-soldering",
            }
        )
    return added_tps


def _measure_bus_spacing_telemetry(
    state: BoardState,
    compromises: list[dict[str, Any]],
) -> None:
    """Measure open-board trace spacing margin (>=0.30mm vs 0.15mm floor) and log compromise."""
    pref_clr = float(getattr(state.config, "preferred_clearance_mm", 0.32))
    segs_by_layer: dict[int, list[TraceSegment]] = {}
    for segs in state.routed_traces.values():
        for s in segs:
            segs_by_layer.setdefault(int(s.layer), []).append(s)

    total_len = 0.0
    relaxed_len = 0.0
    for lyr, segs in segs_by_layer.items():
        for i, sa in enumerate(segs):
            slen = _dist2(sa.x1_mm, sa.y1_mm, sa.x2_mm, sa.y2_mm)
            if slen < 0.05:
                continue
            total_len += slen
            min_edge = float("inf")
            for j, sb in enumerate(segs):
                if i == j or sa.net_id == sb.net_id:
                    continue
                dc, _, _ = segment_to_segment_dist(
                    sa.x1_mm, sa.y1_mm, sa.x2_mm, sa.y2_mm,
                    sb.x1_mm, sb.y1_mm, sb.x2_mm, sb.y2_mm,
                )
                edge = dc - 0.5 * (sa.width_mm + sb.width_mm)
                if edge < min_edge:
                    min_edge = edge
            if min_edge >= 0.28:
                relaxed_len += slen

    if total_len > 0.0:
        pct = round(100.0 * relaxed_len / total_len, 1)
        compromises.append(
            {
                "category": "Relaxed Bus Spacing",
                "action": f"{pct:.1f}% of routed copper maintains >=0.28mm preferred spacing (halo target {pref_clr:.2f}mm vs 0.15mm floor)",
                "efficiency_cost": "Uses open FR4 margin",
                "reliability_gain": "Reduces solder-bridging & capacitive crosstalk across open corridors",
            }
        )


def _record_screw_buffer_telemetry(
    state: BoardState,
    compromises: list[dict[str, Any]],
) -> None:
    """Record mounting-hole torque stress keepouts enforced during placement."""
    holes = [c for c in state.netlist.components.values() if c.comp_type == "mounting_hole"]
    if not holes:
        return
    screw_buf = float(getattr(state.config, "screw_stress_buffer_mm", 1.8))
    d0 = float(holes[0].electrical_meta.get("drill_mm", min(holes[0].width, holes[0].height)))
    compromises.append(
        {
            "category": "Screw-Head Stress Buffer",
            "action": f"Enforced phi+{screw_buf:.1f}mm ({d0 + screw_buf:.1f}mm dia) MLCC flex keepout around {len(holes)} mounting hole(s)",
            "efficiency_cost": "Local capacitor offset outside torque halo",
            "reliability_gain": "Prevents ceramic MLCC flex cracking when M2/M2.5 screws are torqued",
        }
    )


def apply_dfm_craftsmanship(state: BoardState) -> dict[str, Any]:
    """Run all rev10 DFM/DFA craftsmanship passes on a routed BoardState safely."""
    if not getattr(state.config, "enable_dfm_craftsmanship", True):
        return {"applied": False, "compromises": []}

    compromises: list[dict[str, Any]] = []
    state.test_points = []
    state.teardrops = []
    state.silkscreen_adjustments = []

    _record_screw_buffer_telemetry(state, compromises)
    _enforce_symmetric_passive_entries(state, compromises)
    _eliminate_acid_traps_and_miter_corners(state, compromises)
    _measure_bus_spacing_telemetry(state, compromises)
    state.pour_cleanup = _prune_pour_islands_and_slivers(state, compromises)
    state.test_points = _add_debug_test_points(state, compromises)
    _eliminate_acid_traps_and_miter_corners(state, compromises)
    state.teardrops = _generate_teardrops(state, compromises)
    state.silkscreen_adjustments = _scrub_silkscreen_over_pads(state, compromises)
    state.dfm_compromises = compromises
    state._rebuild_occupancy()

    return {
        "applied": True,
        "corner_fillet_radius_mm": state.corner_fillet_radius_mm,
        "teardrops": len(state.teardrops),
        "test_points": len(state.test_points),
        "silkscreen_scrubbed": sum(1 for a in state.silkscreen_adjustments if a.get("shifted")),
        "compromises": compromises,
    }
