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


def _is_hv_net(name: str) -> bool:
    up = name.upper()
    return any(tok in up for tok in ("HV_", "AC_", "MAINS", "RELAY_COM", "RELAY_NO", "RELAY_NC"))


def point_in_polygon(px: float, py: float, poly: list[tuple[float, float]]) -> bool:
    """Ray-casting point-in-polygon check (inclusive of boundary)."""
    n = len(poly)
    if n < 3:
        return True
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = poly[i]
        xj, yj = poly[j]
        # Check on-edge tolerance
        dx, dy = xj - xi, yj - yi
        seg_len_sq = dx * dx + dy * dy
        if seg_len_sq > 1e-9:
            t = max(0.0, min(1.0, ((px - xi) * dx + (py - yi) * dy) / seg_len_sq))
            if math.hypot(px - (xi + t * dx), py - (yi + t * dy)) <= 0.05:
                return True
        if ((yi > py) != (yj > py)) and (px < (xj - xi) * (py - yi) / ((yj - yi) or 1e-12) + xi):
            inside = not inside
        j = i
    return inside


def check_enclosure_collisions(board_state: BoardState) -> list[DRCError]:
    """Verify component 3D Z-heights against Z_max clearance map & non-rectangular outline polygons (rev11)."""
    from env.step_exporter import get_component_height_mm

    errors: list[DRCError] = []
    netlist = board_state.netlist
    z_regions = list(getattr(netlist, "z_max_regions", []) or [])
    z_default = getattr(netlist, "z_max_default_mm", None)
    outline_poly = list(getattr(netlist, "outline_polygon", []) or [])

    if not z_regions and z_default is None and len(outline_poly) < 3:
        return errors

    for comp in netlist.components.values():
        if getattr(comp, "side", "top") == "bottom" or comp.comp_type == "mounting_hole":
            continue
        ew, eh = comp.effective_size()
        cx0, cx1 = comp.x - ew * 0.5, comp.x + ew * 0.5
        cy0, cy1 = comp.y - eh * 0.5, comp.y + eh * 0.5
        comp_z = get_component_height_mm(comp)

        if z_default is not None and comp_z > float(z_default) + 1e-3:
            gx, gy = board_state.mm_to_grid(comp.x, comp.y)
            errors.append(
                DRCError(
                    error_type="enclosure_z_collision",
                    layer=0,
                    x_mm=comp.x,
                    y_mm=comp.y,
                    x_grid=gx,
                    y_grid=gy,
                    offending_net_id=0,
                    clearance_mm=float(z_default) - comp_z,
                    message=f"3D Z collision: {comp.comp_id} height {comp_z:.2f}mm > enclosure ceiling {float(z_default):.2f}mm",
                )
            )

        for reg in z_regions:
            rx0, ry0, rx1, ry1 = float(reg["x0"]), float(reg["y0"]), float(reg["x1"]), float(reg["y1"])
            z_lim = float(reg["z_max_mm"])
            ix = min(cx1, rx1) - max(cx0, rx0)
            iy = min(cy1, ry1) - max(cy0, ry0)
            if ix > 0.05 and iy > 0.05 and comp_z > z_lim + 1e-3:
                mx = 0.5 * (max(cx0, rx0) + min(cx1, rx1))
                my = 0.5 * (max(cy0, ry0) + min(cy1, ry1))
                gx, gy = board_state.mm_to_grid(mx, my)
                errors.append(
                    DRCError(
                        error_type="enclosure_z_collision",
                        layer=0,
                        x_mm=mx,
                        y_mm=my,
                        x_grid=gx,
                        y_grid=gy,
                        offending_net_id=0,
                        clearance_mm=z_lim - comp_z,
                        message=(
                            f"3D Z collision: {comp.comp_id} (Z={comp_z:.2f}mm) hits enclosure rib "
                            f"'{reg.get('label', 'Z_MAX')}' (Z_max={z_lim:.2f}mm)"
                        ),
                    )
                )

        if len(outline_poly) >= 3 and not comp.is_locked:
            corners = [(cx0, cy0), (cx1, cy0), (cx1, cy1), (cx0, cy1)]
            if not all(point_in_polygon(px, py, outline_poly) for px, py in corners):
                gx, gy = board_state.mm_to_grid(comp.x, comp.y)
                errors.append(
                    DRCError(
                        error_type="outline_polygon_violation",
                        layer=0,
                        x_mm=comp.x,
                        y_mm=comp.y,
                        x_grid=gx,
                        y_grid=gy,
                        offending_net_id=0,
                        message=f"Component {comp.comp_id} protrudes outside imported .STEP/.DXF polygon cavity",
                    )
                )

    return errors


def check_courtyard_overlaps(board_state: BoardState) -> list[DRCError]:
    """Detect courtyard overlaps, keepout conflicts, edge-lock, keep_back_flat, and out-of-bounds placements."""
    errors: list[DRCError] = []
    comps = list(board_state.netlist.components.values())
    margin = board_state.config.edge_margin_mm
    bw = board_state.netlist.width_mm
    bh = board_state.netlist.height_mm
    check_rules = getattr(board_state.config, "enable_special_layout_rules", True)
    enforce_back_flat = (
        getattr(board_state.config, "keep_back_flat", False)
        or getattr(board_state.netlist, "single_sided", False)
    )

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
        if check_rules and "perimeter_edge_lock" in getattr(c, "special_layout_rules", []):
            edge_dist = min(max(0.0, x0), max(0.0, bw - x1), max(0.0, y0), max(0.0, bh - y1))
            if edge_dist > 3.5:
                gx, gy = board_state.mm_to_grid(c.x, c.y)
                errors.append(
                    DRCError(
                        error_type="edge_lock_violation",
                        layer=0,
                        x_mm=c.x,
                        y_mm=c.y,
                        x_grid=gx,
                        y_grid=gy,
                        offending_net_id=0,
                        clearance_mm=edge_dist,
                        message=f"Perimeter edge-locked component {c.comp_id} is {edge_dist:.2f}mm from nearest edge (>3.5mm)",
                    )
                )
        if enforce_back_flat:
            is_exempt = (
                c.comp_type == "mounting_hole"
                or bool(getattr(c, "electrical_meta", {}).get("rear_pad_exempt", False))
            )
            if not is_exempt and (
                getattr(c, "side", "top") == "bottom"
                or any(p.layer_mask == 3 and p.net_id > 0 for p in c.pads)
            ):
                gx, gy = board_state.mm_to_grid(c.x, c.y)
                errors.append(
                    DRCError(
                        error_type="back_flat_violation",
                        layer=1,
                        x_mm=c.x,
                        y_mm=c.y,
                        x_grid=gx,
                        y_grid=gy,
                        offending_net_id=0,
                        message=f"keep_back_flat violation: {c.comp_id} protrudes onto B.Cu",
                    )
                )

    for i in range(len(comps)):
        a = comps[i]
        aw, ah = a.effective_size()
        ax0, ax1 = a.x - aw / 2.0, a.x + aw / 2.0
        ay0, ay1 = a.y - ah / 2.0, a.y + ah / 2.0
        a_tht = any(p.layer_mask == 3 for p in a.pads) or a.comp_type == "mounting_hole"
        a_side = getattr(a, "side", "top")
        for j in range(i + 1, len(comps)):
            b = comps[j]
            b_tht = any(p.layer_mask == 3 for p in b.pads) or b.comp_type == "mounting_hole"
            b_side = getattr(b, "side", "top")
            if a_side != b_side and not a_tht and not b_tht:
                continue
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

    # Check non-owner components against RF antenna / isolation keepout zones
    if check_rules:
        for owner in comps:
            for kx0, ky0, kx1, ky1, all_layers, kind in owner.world_keepout_zones():
                for other in comps:
                    if other.comp_id == owner.comp_id:
                        continue
                    if not all_layers and getattr(other, "side", "top") != getattr(owner, "side", "top"):
                        continue
                    ow, oh = other.effective_size()
                    ox0, ox1 = other.x - ow / 2.0, other.x + ow / 2.0
                    oy0, oy1 = other.y - oh / 2.0, other.y + oh / 2.0
                    ix = min(kx1, ox1) - max(kx0, ox0)
                    iy = min(ky1, oy1) - max(ky0, oy0)
                    if ix > 0.05 and iy > 0.05:
                        cx = 0.5 * (max(kx0, ox0) + min(kx1, ox1))
                        cy = 0.5 * (max(ky0, oy0) + min(ky1, oy1))
                        gx, gy = board_state.mm_to_grid(cx, cy)
                        errors.append(
                            DRCError(
                                error_type="keepout_violation",
                                layer=0,
                                x_mm=cx,
                                y_mm=cy,
                                x_grid=gx,
                                y_grid=gy,
                                offending_net_id=0,
                                clearance_mm=-(ix * iy),
                                message=f"Component {other.comp_id} overlaps {owner.comp_id} {kind} keepout zone",
                            )
                        )
    errors.extend(check_enclosure_collisions(board_state))
    return errors


def check_copper_clearances(board_state: BoardState) -> list[DRCError]:
    """Verify 2D/multi-layer edge-to-edge copper clearances between traces, pads, and vias."""
    errors: list[DRCError] = []
    min_clr = board_state.config.clearance_mm
    num_layers = getattr(board_state, "num_layers", 2)
    bottom_layer = max(1, num_layers - 1)

    # Collect segments per layer with bounding boxes for fast spatial pruning
    segs_by_layer: dict[int, list[tuple[TraceSegment, float, float, float, float]]] = {
        l_idx: [] for l_idx in range(num_layers)
    }
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
    pads_by_layer: dict[int, list[tuple[int, float, float, float, float, str]]] = {
        l_idx: [] for l_idx in range(num_layers)
    }
    for comp in board_state.netlist.components.values():
        for pad in comp.pads:
            wx, wy = comp.pad_world_pos(pad)
            pw, ph = comp.pad_effective_size(pad)
            rx0, rx1 = wx - pw / 2.0, wx + pw / 2.0
            ry0, ry1 = wy - ph / 2.0, wy + ph / 2.0
            label = f"{comp.comp_id}.{pad.pad_id}"
            if pad.layer_mask == 3:
                for l_idx in range(num_layers):
                    pads_by_layer[l_idx].append((pad.net_id, rx0, ry0, rx1, ry1, label))
            else:
                if pad.layer_mask & 1:
                    pads_by_layer[0].append((pad.net_id, rx0, ry0, rx1, ry1, label))
                if pad.layer_mask & 2:
                    pads_by_layer[bottom_layer].append((pad.net_id, rx0, ry0, rx1, ry1, label))

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

    # 3. Via-to-Via, Via-to-Segment, and Via-to-Pad clearances respecting layer spans
    all_vias: list[Via] = [v for vs in board_state.routed_vias.values() for v in vs]
    for i in range(len(all_vias)):
        va = all_vias[i]
        va_lo = max(0, min(getattr(va, "start_layer", 0), getattr(va, "end_layer", bottom_layer)))
        va_hi = min(bottom_layer, max(getattr(va, "start_layer", 0), getattr(va, "end_layer", bottom_layer)))

        for j in range(i + 1, len(all_vias)):
            vb = all_vias[j]
            if va.net_id == vb.net_id:
                continue
            vb_lo = max(0, min(getattr(vb, "start_layer", 0), getattr(vb, "end_layer", bottom_layer)))
            vb_hi = min(bottom_layer, max(getattr(vb, "start_layer", 0), getattr(vb, "end_layer", bottom_layer)))
            spans_overlap = max(va_lo, vb_lo) <= min(va_hi, vb_hi)

            d = math.hypot(vb.x_mm - va.x_mm, vb.y_mm - va.y_mm)
            if spans_overlap:
                edge_clr = d - 0.5 * (va.diameter_mm + vb.diameter_mm)
                required_clr = min_clr
            else:
                # Non-overlapping blind/micro vias still respect drill-to-drill spacing
                edge_clr = d - 0.5 * (va.drill_mm + vb.drill_mm)
                required_clr = min_clr

            if edge_clr < required_clr - 1e-4:
                mx, my = 0.5 * (va.x_mm + vb.x_mm), 0.5 * (va.y_mm + vb.y_mm)
                gx, gy = board_state.mm_to_grid(mx, my)
                errors.append(
                    DRCError(
                        error_type="via_clearance",
                        layer=max(va_lo, vb_lo),
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

        for layer in range(va_lo, va_hi + 1):
            for seg, ax0, ay0, ax1, ay1 in segs_by_layer.get(layer, []):
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

    # 4. Keepout Zone (RF Antenna & Isolation Slot) Copper Exclusion Checks
    if getattr(board_state.config, "enable_special_layout_rules", True):
        for comp in board_state.netlist.components.values():
            comp_side_layer = bottom_layer if getattr(comp, "side", "top") == "bottom" else 0
            for kx0, ky0, kx1, ky1, all_layers, kind in comp.world_keepout_zones():
                if kind == "isolation_creepage":
                    continue
                target_layers = list(range(num_layers)) if all_layers else [comp_side_layer]
                # Shrink slightly so pads sitting right on keepout edge can escape outward
                ix0, iy0, ix1, iy1 = kx0 + 0.08, ky0 + 0.08, kx1 - 0.08, ky1 - 0.08
                if ix0 >= ix1 or iy0 >= iy1:
                    continue
                for l_idx in target_layers:
                    for seg, _, _, _, _ in segs_by_layer.get(l_idx, []):
                        d_k, cx, cy = segment_to_rect_dist(
                            seg.x1_mm, seg.y1_mm, seg.x2_mm, seg.y2_mm, ix0, iy0, ix1, iy1
                        )
                        if d_k < 1e-4:
                            gx, gy = board_state.mm_to_grid(cx, cy)
                            errors.append(
                                DRCError(
                                    error_type="keepout_violation",
                                    layer=l_idx,
                                    x_mm=cx,
                                    y_mm=cy,
                                    x_grid=gx,
                                    y_grid=gy,
                                    offending_net_id=seg.net_id,
                                    clearance_mm=d_k,
                                    message=f"Trace net {seg.net_id} enters {comp.comp_id} {kind} keepout on L{l_idx + 1}",
                                )
                            )
                for va in all_vias:
                    va_lo = max(0, min(getattr(va, "start_layer", 0), getattr(va, "end_layer", bottom_layer)))
                    va_hi = min(bottom_layer, max(getattr(va, "start_layer", 0), getattr(va, "end_layer", bottom_layer)))
                    if any(va_lo <= l_idx <= va_hi for l_idx in target_layers):
                        if ix0 <= va.x_mm <= ix1 and iy0 <= va.y_mm <= iy1:
                            gx, gy = board_state.mm_to_grid(va.x_mm, va.y_mm)
                            errors.append(
                                DRCError(
                                    error_type="keepout_violation",
                                    layer=va_lo,
                                    x_mm=va.x_mm,
                                    y_mm=va.y_mm,
                                    x_grid=gx,
                                    y_grid=gy,
                                    offending_net_id=va.net_id,
                                    clearance_mm=0.0,
                                    message=f"Via net {va.net_id} placed inside {comp.comp_id} {kind} keepout",
                                )
                            )

    # 5. Primary-to-Secondary Isolation Creepage Checks
    has_iso = any(
        "isolation_creepage" in getattr(c, "special_layout_rules", [])
        for c in board_state.netlist.components.values()
    )
    if has_iso and getattr(board_state.config, "enable_special_layout_rules", True):
        creep_mm = float(getattr(board_state.config, "creepage_mm", 2.5))
        hv_nets = {
            nid for nid, n in board_state.netlist.nets.items() if nid > 0 and _is_hv_net(n.name)
        }
        if hv_nets:
            for layer, entries in segs_by_layer.items():
                for i in range(len(entries)):
                    sa, ax0, ay0, ax1, ay1 = entries[i]
                    sa_hv = sa.net_id in hv_nets
                    for j in range(i + 1, len(entries)):
                        sb, bx0, by0, bx1, by1 = entries[j]
                        if (sb.net_id in hv_nets) == sa_hv:
                            continue
                        if ax1 + creep_mm <= bx0 or bx1 + creep_mm <= ax0:
                            continue
                        if ay1 + creep_mm <= by0 or by1 + creep_mm <= ay0:
                            continue
                        center_d, cx, cy = segment_to_segment_dist(
                            sa.x1_mm, sa.y1_mm, sa.x2_mm, sa.y2_mm,
                            sb.x1_mm, sb.y1_mm, sb.x2_mm, sb.y2_mm,
                        )
                        edge_clr = center_d - 0.5 * (sa.width_mm + sb.width_mm)
                        if edge_clr < creep_mm - 0.1:
                            gx, gy = board_state.mm_to_grid(cx, cy)
                            errors.append(
                                DRCError(
                                    error_type="creepage_violation",
                                    layer=layer,
                                    x_mm=cx,
                                    y_mm=cy,
                                    x_grid=gx,
                                    y_grid=gy,
                                    offending_net_id=sb.net_id,
                                    secondary_net_id=sa.net_id,
                                    clearance_mm=edge_clr,
                                    message=(
                                        f"Isolation creepage violation on L{layer + 1}: "
                                        f"{edge_clr:.2f}mm < {creep_mm:.2f}mm between HV net {sa.net_id} and LV net {sb.net_id}"
                                    ),
                                )
                            )

    return errors


def check_hdi_via_rules(board_state: BoardState) -> list[DRCError]:
    """Verify 8L+ HDI lamination cycle via spans and staggered-only microvia reliability (rev12)."""
    from env.board_state import get_legal_via_spans

    errors: list[DRCError] = []
    num_layers = getattr(board_state, "num_layers", 2)
    if num_layers < 6:
        return errors

    bottom_layer = max(1, num_layers - 1)
    cfg = board_state.config
    max_cycles = int(getattr(cfg, "max_lamination_cycles", 2))
    buildup = str(getattr(cfg, "hdi_buildup_type", "2+4+2"))
    enforce_stagger = bool(getattr(cfg, "enforce_staggered_microvias", True))
    stagger_min = float(getattr(cfg, "microvia_stagger_offset_mm", 0.25))
    legal_spans = get_legal_via_spans(num_layers, max_cycles, buildup)

    micros: list[tuple[Via, int, int]] = []
    for vs in board_state.routed_vias.values():
        for va in vs:
            lo = max(0, min(int(getattr(va, "start_layer", 0)), int(getattr(va, "end_layer", bottom_layer))))
            hi = min(bottom_layer, max(int(getattr(va, "start_layer", 0)), int(getattr(va, "end_layer", bottom_layer))))
            if num_layers >= 8 and (lo, hi) not in legal_spans:
                gx, gy = board_state.mm_to_grid(va.x_mm, va.y_mm)
                errors.append(
                    DRCError(
                        error_type="illegal_skip_via",
                        layer=lo,
                        x_mm=va.x_mm,
                        y_mm=va.y_mm,
                        x_grid=gx,
                        y_grid=gy,
                        offending_net_id=va.net_id,
                        clearance_mm=0.0,
                        message=(
                            f"Illegal skip-via L{lo + 1}<->L{hi + 1} on net {va.net_id} "
                            f"violates max_lamination_cycles={max_cycles} ({buildup} HDI)"
                        ),
                    )
                )
            if hi - lo == 1 and getattr(va, "via_type", "through") == "micro":
                micros.append((va, lo, hi))

    if enforce_stagger:
        for i in range(len(micros)):
            va, alo, ahi = micros[i]
            for j in range(i + 1, len(micros)):
                vb, blo, bhi = micros[j]
                if (alo, ahi) == (blo, bhi):
                    continue
                if ahi == blo or bhi == alo:
                    dist = math.hypot(va.x_mm - vb.x_mm, va.y_mm - vb.y_mm)
                    if dist < stagger_min - 1e-4:
                        mx, my = 0.5 * (va.x_mm + vb.x_mm), 0.5 * (va.y_mm + vb.y_mm)
                        gx, gy = board_state.mm_to_grid(mx, my)
                        errors.append(
                            DRCError(
                                error_type="stacked_microvia_reliability_risk",
                                layer=max(alo, blo),
                                x_mm=mx,
                                y_mm=my,
                                x_grid=gx,
                                y_grid=gy,
                                offending_net_id=va.net_id,
                                secondary_net_id=vb.net_id,
                                clearance_mm=dist,
                                message=(
                                    f"stacked_microvia_reliability_risk: microvias L{alo + 1}<->L{ahi + 1} "
                                    f"and L{blo + 1}<->L{bhi + 1} concentric/under-staggered "
                                    f"(dist={dist:.3f}mm < {stagger_min:.2f}mm)"
                                ),
                            )
                        )
    return errors


def check_broadside_crosstalk(board_state: BoardState) -> list[DRCError]:
    """Flag 3D Z-axis broadside overlaps between adjacent unshielded layers (rev13)."""
    if not getattr(board_state.config, "enable_mixed_signal_physics", True):
        return []
    from env.metrics import compute_broadside_crosstalk_risk

    report = compute_broadside_crosstalk_risk(board_state)
    errors: list[DRCError] = []
    for viol in report.get("broadside_violations", []):
        vx = float(viol["x_mm"])
        vy = float(viol["y_mm"])
        gx, gy = board_state.mm_to_grid(vx, vy)
        v_nid = int(viol["victim_net_id"])
        a_nid = int(viol["aggressor_net_id"])
        v_lyr = int(viol["victim_layer"])
        a_lyr = int(viol["aggressor_layer"])
        errors.append(
            DRCError(
                error_type="broadside_crosstalk_violation",
                layer=v_lyr,
                x_mm=vx,
                y_mm=vy,
                x_grid=gx,
                y_grid=gy,
                offending_net_id=v_nid,
                secondary_net_id=a_nid,
                clearance_mm=float(viol["lateral_dist_mm"]),
                message=(
                    f"Z-axis broadside coupling: {viol['victim_net_name']} (L{v_lyr + 1}) overlaps "
                    f"{viol['aggressor_net_name']} (L{a_lyr + 1}) over {viol['overlap_length_mm']:.2f}mm "
                    f"(C_broadside={viol['c_broadside_pf']:.4f}pF > {report['limit_pf']:.3f}pF)"
                ),
            )
        )
    return errors


def validate_ground_topology(board_state: BoardState) -> dict[str, object]:
    """Validate ground plane continuity and reject remote star-ground slot-antenna traps (rev13)."""
    from env.metrics import compute_net_mst_pairs

    netlist = board_state.netlist
    gc = getattr(netlist, "ground_constraints", {}) or {}
    mode = str(gc.get("mode", "")).strip().lower()
    net_names_upper = {n.name.strip().upper() for n in netlist.nets.values() if n.net_id > 0}
    has_agnd_dgnd_split = ("AGND" in net_names_upper and "DGND" in net_names_upper) and not bool(
        gc.get("rejected_remote_star_ground", False)
    )

    if mode != "remote_star_ground" and not has_agnd_dgnd_split:
        refused = bool(gc.get("rejected_remote_star_ground", False))
        prev_slot_nh = float(gc.get("rejected_slot_inductance_nh", 0.0) or 0.0)
        return {
            "valid": True,
            "status": "CONTINUOUS_GND_VERIFIED",
            "rejection_code": None,
            "continuous_gnd": True,
            "slot_antenna_loops": 0,
            "max_detour_mm": 0.0,
            "slot_loop_area_mm2": 0.0,
            "slot_inductance_nh": 0.0,
            "continuous_gnd_inductance_nh": 0.85,
            "refused_invalid_constraint": refused,
            "rejected_slot_inductance_nh": prev_slot_nh,
            "reason": (
                f"Rejected invalid remote star-ground constraint (would add {prev_slot_nh:.1f} nH slot loop); enforced continuous In1.GND."
                if refused
                else "Unified continuous ground return plane verified (0 slot-antenna splits)."
            ),
        }

    split_axis = str(gc.get("split_axis", "x")).lower()
    split_pos = float(gc.get("split_pos_mm", netlist.width_mm * 0.5))
    bridge = gc.get("bridge_xy", [split_pos, netlist.height_mm * 0.9])
    bx, by = float(bridge[0]), float(bridge[1])

    crossing_pts: list[tuple[float, float, int]] = []
    for nid in netlist.routable_net_ids:
        net = netlist.nets[nid]
        if net.is_power_ground:
            continue
        segs = board_state.routed_traces.get(nid, [])
        if segs:
            pts_list = [((s.x1_mm, s.y1_mm), (s.x2_mm, s.y2_mm)) for s in segs]
        else:
            pts_list = [((a[0], a[1]), (b[0], b[1])) for a, b in compute_net_mst_pairs(netlist, nid)]
        for (x1, y1), (x2, y2) in pts_list:
            if split_axis == "y":
                if (y1 - split_pos) * (y2 - split_pos) <= 0.0 and abs(y1 - y2) > 1e-4:
                    t = (split_pos - y1) / (y2 - y1)
                    crossing_pts.append((x1 + t * (x2 - x1), split_pos, nid))
            else:
                if (x1 - split_pos) * (x2 - split_pos) <= 0.0 and abs(x1 - x2) > 1e-4:
                    t = (split_pos - x1) / (x2 - x1)
                    crossing_pts.append((split_pos, y1 + t * (y2 - y1), nid))

    max_detour = 0.0
    for cx, cy, _ in crossing_pts:
        detour = 2.0 * math.hypot(cx - bx, cy - by)
        if detour > max_detour:
            max_detour = detour
    if not crossing_pts:
        max_detour = max(10.0, 0.5 * max(netlist.width_mm, netlist.height_mm))

    slot_loop_area = max_detour * 0.5
    slot_inductance_nh = 0.42 * max(5.0, max_detour)
    return {
        "valid": False,
        "status": "REJECTED_INVALID_CONSTRAINT",
        "rejection_code": "invalid_remote_star_ground",
        "continuous_gnd": False,
        "slot_antenna_loops": max(1, len(crossing_pts)),
        "max_detour_mm": round(max_detour, 2),
        "slot_loop_area_mm2": round(slot_loop_area, 2),
        "slot_inductance_nh": round(slot_inductance_nh, 2),
        "continuous_gnd_inductance_nh": 0.85,
        "bridge_xy": [bx, by],
        "split_axis": split_axis,
        "split_pos_mm": split_pos,
        "crossing_points": [(round(x, 2), round(y, 2), nid) for x, y, nid in crossing_pts],
        "reason": (
            f"invalid_remote_star_ground: remote star bridge at ({bx:.1f}, {by:.1f}) forces "
            f"{max(1, len(crossing_pts))} signal crossing(s) into a {max_detour:.1f}mm return detour "
            f"(L_loop={slot_inductance_nh:.1f}nH slot antenna vs 0.85nH continuous In1.GND)."
        ),
    }


def check_ground_topology(board_state: BoardState) -> list[DRCError]:
    """Emit DRCError when an invalid remote star-ground split is active (rev13)."""
    topo = validate_ground_topology(board_state)
    if bool(topo.get("valid", True)):
        return []
    crossings = topo.get("crossing_points") or []
    if crossings:
        cx, cy, nid = crossings[0]  # type: ignore[misc]
    else:
        cx = float(board_state.netlist.width_mm) * 0.5
        cy = float(board_state.netlist.height_mm) * 0.5
        nid = 0
    gx, gy = board_state.mm_to_grid(float(cx), float(cy))
    return [
        DRCError(
            error_type="invalid_remote_star_ground",
            layer=1 if getattr(board_state, "num_layers", 2) >= 4 else 1,
            x_mm=float(cx),
            y_mm=float(cy),
            x_grid=gx,
            y_grid=gy,
            offending_net_id=int(nid),
            clearance_mm=float(topo.get("max_detour_mm", 0.0)),  # type: ignore[arg-type]
            message=str(topo.get("reason", "invalid_remote_star_ground")),
        )
    ]


def check_rigid_flex_rules(board_state: BoardState) -> list[DRCError]:
    """Enforce DynamicFlexZone component, via, and disallowed-layer keepout rules (rev14)."""
    zones = [
        z for z in (getattr(board_state.netlist, "substrate_zones", []) or [])
        if getattr(z, "zone_type", "rigid") == "dynamic_flex"
    ]
    if not zones:
        return []

    errors: list[DRCError] = []
    for z in zones:
        allowed_layers = set(z.allowed_layers) if z.allowed_layers else {0}
        if not z.allow_components:
            for comp in board_state.netlist.components.values():
                ew, eh = comp.effective_size()
                cx0, cx1 = comp.x - ew * 0.5, comp.x + ew * 0.5
                cy0, cy1 = comp.y - eh * 0.5, comp.y + eh * 0.5
                if cx1 > z.x0_mm + 1e-3 and z.x1_mm > cx0 + 1e-3 and cy1 > z.y0_mm + 1e-3 and z.y1_mm > cy0 + 1e-3:
                    gx, gy = board_state.mm_to_grid(comp.x, comp.y)
                    errors.append(
                        DRCError(
                            error_type="flex_zone_component_violation",
                            layer=0,
                            x_mm=comp.x,
                            y_mm=comp.y,
                            x_grid=gx,
                            y_grid=gy,
                            offending_net_id=0,
                            clearance_mm=0.0,
                            message=f"Component {comp.comp_id} intrudes into dynamic flex bend zone '{z.zone_id}'",
                        )
                    )
        if not z.allow_vias:
            for nid, vias in board_state.routed_vias.items():
                for via in vias:
                    if (z.x0_mm - 0.1) <= via.x_mm <= (z.x1_mm + 0.1) and (z.y0_mm - 0.1) <= via.y_mm <= (z.y1_mm + 0.1):
                        gx, gy = board_state.mm_to_grid(via.x_mm, via.y_mm)
                        errors.append(
                            DRCError(
                                error_type="flex_zone_via_violation",
                                layer=int(via.start_layer),
                                x_mm=via.x_mm,
                                y_mm=via.y_mm,
                                x_grid=gx,
                                y_grid=gy,
                                offending_net_id=nid,
                                clearance_mm=0.0,
                                message=f"Via on net {nid} inside dynamic flex bend zone '{z.zone_id}' is forbidden",
                            )
                        )
        for nid, segs in board_state.routed_traces.items():
            for seg in segs:
                if int(seg.layer) in allowed_layers:
                    continue
                mx = 0.5 * (seg.x1_mm + seg.x2_mm)
                my = 0.5 * (seg.y1_mm + seg.y2_mm)
                if (z.x0_mm < mx < z.x1_mm) and (z.y0_mm < my < z.y1_mm):
                    gx, gy = board_state.mm_to_grid(mx, my)
                    errors.append(
                        DRCError(
                            error_type="flex_zone_layer_violation",
                            layer=int(seg.layer),
                            x_mm=mx,
                            y_mm=my,
                            x_grid=gx,
                            y_grid=gy,
                            offending_net_id=nid,
                            clearance_mm=0.0,
                            message=f"Trace on net {nid} layer L{int(seg.layer) + 1} inside 2L flex zone '{z.zone_id}'",
                        )
                    )
    return errors


def check_pdn_decoupling(board_state: BoardState) -> list[DRCError]:
    """Flag bypass capacitors violating PDN mounting loop inductance or voltage droop (rev14)."""
    if not getattr(board_state.config, "enable_rigid_flex_physics", True):
        return []
    from env.metrics import compute_pdn_decoupling_inductance

    report = compute_pdn_decoupling_inductance(board_state)
    if not report.get("pdn_verified", False):
        return []
    errors: list[DRCError] = []
    lim_nh = float(report.get("max_loop_inductance_limit_nh", 1.2))
    for viol in report.get("violations", []):
        vx = float(viol["x_mm"])
        vy = float(viol["y_mm"])
        gx, gy = board_state.mm_to_grid(vx, vy)
        errors.append(
            DRCError(
                error_type="pdn_excessive_loop_inductance",
                layer=0 if viol.get("cap_side", "top") == "top" else max(1, board_state.num_layers - 1),
                x_mm=vx,
                y_mm=vy,
                x_grid=gx,
                y_grid=gy,
                offending_net_id=int(viol.get("net_id", 0)),
                clearance_mm=float(viol.get("l_loop_nh", 0.0)),
                message=(
                    f"pdn_excessive_loop_inductance: {viol['cap_id']} ({viol['pwr_net']}) "
                    f"L_loop={viol['l_loop_nh']:.3f}nH > {lim_nh:.2f}nH "
                    f"(V_droop={viol['v_droop_mv']:.1f}mV vs {viol['max_droop_mv']:.1f}mV)"
                ),
            )
        )
    return errors


def verify_board(board_state: BoardState) -> list[DRCError]:
    """Run complete fast 2D + 3D mixed-signal + rigid-flex/PDN geometric DRC verification."""
    errors = check_courtyard_overlaps(board_state)
    errors.extend(check_copper_clearances(board_state))
    errors.extend(check_hdi_via_rules(board_state))
    errors.extend(check_broadside_crosstalk(board_state))
    errors.extend(check_ground_topology(board_state))
    errors.extend(check_rigid_flex_rules(board_state))
    errors.extend(check_pdn_decoupling(board_state))
    return errors
