"""Millisecond 2D geometric polygon/segment clearance & collision verifier."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from env.board_state import BoardState, TraceSegment, Via


@dataclass
class DRCError:
    """Design Rule Check violation with exact physical and grid coordinates."""

    error_type: str  # "clearance" | "short" | "via_clearance" | "courtyard_overlap" | "out_of_bounds"
    layer: int  # 0 = F.Cu, 1 = B.Cu
    x_mm: float
    y_mm: float
    x_grid: int
    y_grid: int
    offending_net_id: int
    secondary_net_id: int = 0
    clearance_mm: float = 0.0
    message: str = ""

    def to_dict(self) -> dict[str, object]:
        """Serialize error for WebSocket/JSON telemetry."""
        return {
            "error_type": self.error_type,
            "layer": self.layer,
            "x_mm": round(self.x_mm, 3),
            "y_mm": round(self.y_mm, 3),
            "x_grid": self.x_grid,
            "y_grid": self.y_grid,
            "offending_net_id": self.offending_net_id,
            "secondary_net_id": self.secondary_net_id,
            "clearance_mm": round(self.clearance_mm, 4),
            "message": self.message,
        }


def point_to_segment_dist(
    px: float, py: float, x1: float, y1: float, x2: float, y2: float
) -> tuple[float, float, float]:
    """Return `(min_distance, closest_x, closest_y)` from point `(px, py)` to segment."""
    dx = x2 - x1
    dy = y2 - y1
    len_sq = dx * dx + dy * dy
    if len_sq <= 1e-12:
        return (math.hypot(px - x1, py - y1), x1, y1)
    t = max(0.0, min(1.0, ((px - x1) * dx + (py - y1) * dy) / len_sq))
    cx = x1 + t * dx
    cy = y1 + t * dy
    return (math.hypot(px - cx, py - cy), cx, cy)


def _segments_intersect(
    ax1: float, ay1: float, ax2: float, ay2: float,
    bx1: float, by1: float, bx2: float, by2: float,
) -> tuple[bool, float, float]:
    """Check whether segments A and B intersect and return an approximate crossing point."""
    r_x, r_y = ax2 - ax1, ay2 - ay1
    s_x, s_y = bx2 - bx1, by2 - by1
    denom = r_x * s_y - r_y * s_x
    qp_x, qp_y = bx1 - ax1, by1 - ay1
    if abs(denom) < 1e-12:
        return (False, 0.0, 0.0)
    t = (qp_x * s_y - qp_y * s_x) / denom
    u = (qp_x * r_y - qp_y * r_x) / denom
    if 0.0 <= t <= 1.0 and 0.0 <= u <= 1.0:
        return (True, ax1 + t * r_x, ay1 + t * r_y)
    return (False, 0.0, 0.0)


def segment_to_segment_dist(
    ax1: float, ay1: float, ax2: float, ay2: float,
    bx1: float, by1: float, bx2: float, by2: float,
) -> tuple[float, float, float]:
    """Return `(min_distance, midpoint_x, midpoint_y)` between two 2D line segments."""
    hit, ix, iy = _segments_intersect(ax1, ay1, ax2, ay2, bx1, by1, bx2, by2)
    if hit:
        return (0.0, ix, iy)

    candidates = [
        point_to_segment_dist(ax1, ay1, bx1, by1, bx2, by2),
        point_to_segment_dist(ax2, ay2, bx1, by1, bx2, by2),
        point_to_segment_dist(bx1, by1, ax1, ay1, ax2, ay2),
        point_to_segment_dist(bx2, by2, ax1, ay1, ax2, ay2),
    ]
    best_d, cx, cy = min(candidates, key=lambda item: item[0])
    # Report midpoint between the two closest points for spatial penalty centering
    if best_d == candidates[0][0]:
        return (best_d, 0.5 * (ax1 + cx), 0.5 * (ay1 + cy))
    if best_d == candidates[1][0]:
        return (best_d, 0.5 * (ax2 + cx), 0.5 * (ay2 + cy))
    if best_d == candidates[2][0]:
        return (best_d, 0.5 * (bx1 + cx), 0.5 * (by1 + cy))
    return (best_d, 0.5 * (bx2 + cx), 0.5 * (by2 + cy))


def segment_to_rect_dist(
    x1: float, y1: float, x2: float, y2: float,
    rx0: float, ry0: float, rx1: float, ry1: float,
) -> tuple[float, float, float]:
    """Exact minimum distance from a 2D segment to an axis-aligned rectangle."""
    if (rx0 <= x1 <= rx1 and ry0 <= y1 <= ry1) or (rx0 <= x2 <= rx1 and ry0 <= y2 <= ry1):
        mx = min(max(0.5 * (x1 + x2), rx0), rx1)
        my = min(max(0.5 * (y1 + y2), ry0), ry1)
        return (0.0, mx, my)

    edges = (
        (rx0, ry0, rx1, ry0),
        (rx1, ry0, rx1, ry1),
        (rx1, ry1, rx0, ry1),
        (rx0, ry1, rx0, ry0),
    )
    best = (float("inf"), 0.5 * (rx0 + rx1), 0.5 * (ry0 + ry1))
    for ex1, ey1, ex2, ey2 in edges:
        d, cx, cy = segment_to_segment_dist(x1, y1, x2, y2, ex1, ey1, ex2, ey2)
        if d < best[0]:
            best = (d, cx, cy)
            if d == 0.0:
                return best
    return best


def check_courtyard_overlaps(board_state: BoardState) -> list[DRCError]:
    """Detect courtyard overlaps and out-of-bounds component placements."""
    errors: list[DRCError] = []
    comps = list(board_state.netlist.components.values())
    margin = board_state.config.edge_margin_mm
    bw = board_state.netlist.width_mm
    bh = board_state.netlist.height_mm

    for c in comps:
        ew, eh = c.effective_size()
        x0, x1 = c.x - ew / 2.0, c.x + ew / 2.0
        y0, y1 = c.y - eh / 2.0, c.y + eh / 2.0
        if x0 < margin - 0.05 or x1 > bw - margin + 0.05 or y0 < margin - 0.05 or y1 > bh - margin + 0.05:
            gx, gy = board_state.mm_to_grid(c.x, c.y)
            errors.append(
                DRCError(
                    error_type="out_of_bounds",
                    layer=0,
                    x_mm=c.x,
                    y_mm=c.y,
                    x_grid=gx,
                    y_grid=gy,
                    offending_net_id=0,
                    message=f"Component {c.comp_id} exceeds board margin ({margin}mm)",
                )
            )

    for i in range(len(comps)):
        a = comps[i]
        aw, ah = a.effective_size()
        ax0, ax1 = a.x - aw / 2.0, a.x + aw / 2.0
        ay0, ay1 = a.y - ah / 2.0, a.y + ah / 2.0
        for j in range(i + 1, len(comps)):
            b = comps[j]
            bw_c, bh_c = b.effective_size()
            bx0, bx1 = b.x - bw_c / 2.0, b.x + bw_c / 2.0
            by0, by1 = b.y - bh_c / 2.0, b.y + bh_c / 2.0
            ox = min(ax1, bx1) - max(ax0, bx0)
            oy = min(ay1, by1) - max(ay0, by0)
            if ox > 0.05 and oy > 0.05:
                cx = 0.5 * (max(ax0, bx0) + min(ax1, bx1))
                cy = 0.5 * (max(ay0, by0) + min(ay1, by1))
                gx, gy = board_state.mm_to_grid(cx, cy)
                errors.append(
                    DRCError(
                        error_type="courtyard_overlap",
                        layer=0,
                        x_mm=cx,
                        y_mm=cy,
                        x_grid=gx,
                        y_grid=gy,
                        offending_net_id=0,
                        clearance_mm=-(ox * oy),
                        message=f"Courtyard overlap between {a.comp_id} and {b.comp_id}",
                    )
                )
    return errors


def check_copper_clearances(board_state: BoardState) -> list[DRCError]:
    """Verify 2D edge-to-edge copper clearances between traces, pads, and vias."""
    errors: list[DRCError] = []
    min_clr = board_state.config.clearance_mm

    # Collect segments per layer with bounding boxes for fast spatial pruning
    segs_by_layer: dict[int, list[tuple[TraceSegment, float, float, float, float]]] = {0: [], 1: []}
    for net_id, segs in board_state.routed_traces.items():
        for seg in segs:
            half_w = seg.width_mm / 2.0
            bx0 = min(seg.x1_mm, seg.x2_mm) - half_w
            bx1 = max(seg.x1_mm, seg.x2_mm) + half_w
            by0 = min(seg.y1_mm, seg.y2_mm) - half_w
            by1 = max(seg.y1_mm, seg.y2_mm) + half_w
            segs_by_layer.setdefault(seg.layer, []).append((seg, bx0, by0, bx1, by1))

    # 1. Segment-to-Segment clearance on the same layer
    for layer, entries in segs_by_layer.items():
        n = len(entries)
        for i in range(n):
            sa, ax0, ay0, ax1, ay1 = entries[i]
            for j in range(i + 1, n):
                sb, bx0, by0, bx1, by1 = entries[j]
                if sa.net_id == sb.net_id:
                    continue
                if ax1 + min_clr <= bx0 or bx1 + min_clr <= ax0:
                    continue
                if ay1 + min_clr <= by0 or by1 + min_clr <= ay0:
                    continue
                center_d, cx, cy = segment_to_segment_dist(
                    sa.x1_mm, sa.y1_mm, sa.x2_mm, sa.y2_mm,
                    sb.x1_mm, sb.y1_mm, sb.x2_mm, sb.y2_mm,
                )
                edge_clr = center_d - 0.5 * (sa.width_mm + sb.width_mm)
                if edge_clr < min_clr - 1e-4:
                    gx, gy = board_state.mm_to_grid(cx, cy)
                    err_type = "short" if edge_clr <= 0.0 else "clearance"
                    errors.append(
                        DRCError(
                            error_type=err_type,
                            layer=layer,
                            x_mm=cx,
                            y_mm=cy,
                            x_grid=gx,
                            y_grid=gy,
                            offending_net_id=sb.net_id,
                            secondary_net_id=sa.net_id,
                            clearance_mm=edge_clr,
                            message=(
                                f"Trace {err_type} on L{layer + 1}: net {sa.net_id} vs net {sb.net_id} "
                                f"(clr={edge_clr:.3f}mm < {min_clr:.3f}mm)"
                            ),
                        )
                    )

    # 2. Segment-to-Foreign-Pad clearance
    pads_by_layer: dict[int, list[tuple[int, float, float, float, float, str]]] = {0: [], 1: []}
    for comp in board_state.netlist.components.values():
        for pad in comp.pads:
            wx, wy = comp.pad_world_pos(pad)
            pw, ph = comp.pad_effective_size(pad)
            rx0, rx1 = wx - pw / 2.0, wx + pw / 2.0
            ry0, ry1 = wy - ph / 2.0, wy + ph / 2.0
            label = f"{comp.comp_id}.{pad.pad_id}"
            if pad.layer_mask & 1:
                pads_by_layer[0].append((pad.net_id, rx0, ry0, rx1, ry1, label))
            if pad.layer_mask & 2:
                pads_by_layer[1].append((pad.net_id, rx0, ry0, rx1, ry1, label))

    for layer, entries in segs_by_layer.items():
        pads = pads_by_layer.get(layer, [])
        for seg, ax0, ay0, ax1, ay1 in entries:
            half_w = seg.width_mm / 2.0
            for pad_net, px0, py0, px1, py1, label in pads:
                if pad_net == seg.net_id:
                    continue
                if ax1 + min_clr <= px0 or px1 + min_clr <= ax0:
                    continue
                if ay1 + min_clr <= py0 or py1 + min_clr <= ay0:
                    continue
                rect_d, cx, cy = segment_to_rect_dist(
                    seg.x1_mm, seg.y1_mm, seg.x2_mm, seg.y2_mm,
                    px0, py0, px1, py1,
                )
                edge_clr = rect_d - half_w
                if edge_clr < min_clr - 1e-4:
                    gx, gy = board_state.mm_to_grid(cx, cy)
                    err_type = "short" if edge_clr <= 0.0 else "clearance"
                    errors.append(
                        DRCError(
                            error_type=err_type,
                            layer=layer,
                            x_mm=cx,
                            y_mm=cy,
                            x_grid=gx,
                            y_grid=gy,
                            offending_net_id=seg.net_id,
                            secondary_net_id=pad_net,
                            clearance_mm=edge_clr,
                            message=(
                                f"Trace-to-Pad {err_type} on L{layer + 1}: net {seg.net_id} vs {label} "
                                f"(net {pad_net}, clr={edge_clr:.3f}mm)"
                            ),
                        )
                    )

    # 3. Via-to-Via and Via-to-Foreign-Segment clearances (Vias span both layers)
    all_vias: list[Via] = [v for vs in board_state.routed_vias.values() for v in vs]
    for i in range(len(all_vias)):
        va = all_vias[i]
        for j in range(i + 1, len(all_vias)):
            vb = all_vias[j]
            if va.net_id == vb.net_id:
                continue
            d = math.hypot(vb.x_mm - va.x_mm, vb.y_mm - va.y_mm)
            edge_clr = d - 0.5 * (va.diameter_mm + vb.diameter_mm)
            if edge_clr < min_clr - 1e-4:
                mx, my = 0.5 * (va.x_mm + vb.x_mm), 0.5 * (va.y_mm + vb.y_mm)
                gx, gy = board_state.mm_to_grid(mx, my)
                errors.append(
                    DRCError(
                        error_type="via_clearance",
                        layer=0,
                        x_mm=mx,
                        y_mm=my,
                        x_grid=gx,
                        y_grid=gy,
                        offending_net_id=vb.net_id,
                        secondary_net_id=va.net_id,
                        clearance_mm=edge_clr,
                        message=f"Via clearance violation: net {va.net_id} vs net {vb.net_id}",
                    )
                )

        for layer, entries in segs_by_layer.items():
            for seg, ax0, ay0, ax1, ay1 in entries:
                if seg.net_id == va.net_id:
                    continue
                vr = va.diameter_mm / 2.0
                if ax1 + vr + min_clr <= va.x_mm or va.x_mm + vr + min_clr <= ax0:
                    continue
                if ay1 + vr + min_clr <= va.y_mm or va.y_mm + vr + min_clr <= ay0:
                    continue
                d, cx, cy = point_to_segment_dist(
                    va.x_mm, va.y_mm, seg.x1_mm, seg.y1_mm, seg.x2_mm, seg.y2_mm
                )
                edge_clr = d - vr - seg.width_mm / 2.0
                if edge_clr < min_clr - 1e-4:
                    gx, gy = board_state.mm_to_grid(cx, cy)
                    errors.append(
                        DRCError(
                            error_type="via_clearance",
                            layer=layer,
                            x_mm=cx,
                            y_mm=cy,
                            x_grid=gx,
                            y_grid=gy,
                            offending_net_id=seg.net_id,
                            secondary_net_id=va.net_id,
                            clearance_mm=edge_clr,
                            message=f"Trace-to-Via clearance on L{layer + 1}: net {seg.net_id} vs via net {va.net_id}",
                        )
                    )

        for layer in (0, 1):
            for pad_net, px0, py0, px1, py1, label in pads_by_layer.get(layer, []):
                if pad_net == va.net_id:
                    continue
                cx_rect = max(px0, min(va.x_mm, px1))
                cy_rect = max(py0, min(va.y_mm, py1))
                d = math.hypot(va.x_mm - cx_rect, va.y_mm - cy_rect)
                edge_clr = d - va.diameter_mm / 2.0
                if edge_clr < min_clr - 1e-4:
                    gx, gy = board_state.mm_to_grid(va.x_mm, va.y_mm)
                    errors.append(
                        DRCError(
                            error_type="via_clearance",
                            layer=layer,
                            x_mm=va.x_mm,
                            y_mm=va.y_mm,
                            x_grid=gx,
                            y_grid=gy,
                            offending_net_id=va.net_id,
                            secondary_net_id=pad_net,
                            clearance_mm=edge_clr,
                            message=f"Via-to-Pad clearance on L{layer + 1}: via net {va.net_id} vs {label}",
                        )
                    )

    return errors


def verify_board(board_state: BoardState) -> list[DRCError]:
    """Run complete fast 2D geometric DRC verification on a BoardState."""
    errors = check_courtyard_overlaps(board_state)
    errors.extend(check_copper_clearances(board_state))
    return errors
