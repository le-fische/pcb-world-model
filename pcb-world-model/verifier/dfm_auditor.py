"""Rev 10 DFM/DFA Craftsmanship Auditor & Transparent Compromise Report Generator."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from verifier.fast_clearance import (
    point_to_segment_dist,
    segment_to_segment_dist,
    verify_board,
)

if TYPE_CHECKING:
    from env.board_state import BoardState, TraceSegment


@dataclass
class DFMIssue:
    """Single DFM/DFA craftsmanship violation or advisory item."""

    rule: str
    severity: str  # "error" | "advisory"
    x_mm: float
    y_mm: float
    layer: int
    message: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "rule": self.rule,
            "severity": self.severity,
            "x_mm": round(self.x_mm, 3),
            "y_mm": round(self.y_mm, 3),
            "layer": self.layer,
            "message": self.message,
        }


def _angle_deg(ux: float, uy: float, vx: float, vy: float) -> float:
    lu = math.hypot(ux, uy)
    lv = math.hypot(vx, vy)
    if lu < 1e-9 or lv < 1e-9:
        return 180.0
    dot = max(-1.0, min(1.0, (ux * vx + uy * vy) / (lu * lv)))
    return math.degrees(math.acos(dot))


def _rect_intersects_circle(
    rx: float, ry: float, rw: float, rh: float, cx: float, cy: float, rad: float
) -> bool:
    hw, hh = rw * 0.5, rh * 0.5
    nx = max(rx - hw, min(cx, rx + hw))
    ny = max(ry - hh, min(cy, ry + hh))
    return math.hypot(cx - nx, cy - ny) < rad - 1e-4


def _boxes_overlap(
    a: tuple[float, float, float, float], b: tuple[float, float, float, float]
) -> bool:
    return a[2] > b[0] and b[2] > a[0] and a[3] > b[1] and b[3] > a[1]


def audit_board_dfm(
    board_state: BoardState,
    kicad_pcb_text: str | None = None,
) -> dict[str, Any]:
    """Run full rev10 DFM/DFA Craftsmanship Audit against a routed BoardState."""
    issues: list[DFMIssue] = []
    cfg = board_state.config
    bottom_layer = max(1, getattr(board_state, "num_layers", 2) - 1)

    # 1. Hard DRC gate check
    drc_errors = verify_board(board_state)

    # Pre-collect own pad rects by (net_id, layer) so we don't confuse multi-branch pad center ties with open-copper acid traps
    pad_zones: list[tuple[int, int, float, float, float, float]] = []
    for comp in board_state.netlist.components.values():
        for pad in comp.pads:
            if pad.net_id <= 0:
                continue
            wx, wy = comp.pad_world_pos(pad)
            pw, ph = comp.pad_effective_size(pad)
            for lyr in range(board_state.num_layers):
                on_l = pad.layer_mask == 3 or (lyr == 0 and (pad.layer_mask & 1)) or (lyr == bottom_layer and (pad.layer_mask & 2))
                if on_l:
                    pad_zones.append((pad.net_id, lyr, wx - pw * 0.55, wy - ph * 0.55, wx + pw * 0.55, wy + ph * 0.55))

    def _inside_own_pad(nid: int, lyr: int, x: float, y: float) -> bool:
        for pnid, plyr, x0, y0, x1, y1 in pad_zones:
            if pnid == nid and plyr == lyr and x0 <= x <= x1 and y0 <= y <= y1:
                return True
        return False

    # 2. Acute-angle ("Acid Trap" < 90 deg) junction check
    acid_trap_count = 0
    for nid, segs in board_state.routed_traces.items():
        by_layer: dict[int, list[TraceSegment]] = {}
        for s in segs:
            by_layer.setdefault(int(s.layer), []).append(s)
        for lyr, lsegs in by_layer.items():
            adj: dict[tuple[int, int], list[tuple[int, float, float, float, float]]] = {}
            for idx, s in enumerate(lsegs):
                if math.hypot(s.x2_mm - s.x1_mm, s.y2_mm - s.y1_mm) < 0.04:
                    continue
                k1 = (int(round(s.x1_mm * 1000)), int(round(s.y1_mm * 1000)))
                k2 = (int(round(s.x2_mm * 1000)), int(round(s.y2_mm * 1000)))
                adj.setdefault(k1, []).append((idx, s.x1_mm, s.y1_mm, s.x2_mm, s.y2_mm))
                adj.setdefault(k2, []).append((idx, s.x2_mm, s.y2_mm, s.x1_mm, s.y1_mm))

            for _, inc in adj.items():
                if len(inc) < 2:
                    continue
                for i in range(len(inc)):
                    idx_a, vx, vy, ax, ay = inc[i]
                    if _inside_own_pad(nid, lyr, vx, vy):
                        continue
                    for j in range(i + 1, len(inc)):
                        idx_b, _, _, bx, by = inc[j]
                        if idx_a == idx_b:
                            continue
                        ang = _angle_deg(ax - vx, ay - vy, bx - vx, by - vy)
                        if ang < 89.0:
                            acid_trap_count += 1
                            issues.append(
                                DFMIssue(
                                    rule="acid_trap",
                                    severity="error",
                                    x_mm=vx,
                                    y_mm=vy,
                                    layer=lyr,
                                    message=f"Acute {ang:.1f} deg trace junction (<90 deg) on net {nid} L{lyr+1}",
                                )
                            )

            # Check interior T-junctions
            for i in range(len(lsegs)):
                sa = lsegs[i]
                if math.hypot(sa.x2_mm - sa.x1_mm, sa.y2_mm - sa.y1_mm) < 0.05:
                    continue
                for j in range(len(lsegs)):
                    if i == j:
                        continue
                    sb = lsegs[j]
                    for ex, ey, ox, oy in (
                        (sb.x1_mm, sb.y1_mm, sb.x2_mm, sb.y2_mm),
                        (sb.x2_mm, sb.y2_mm, sb.x1_mm, sb.y1_mm),
                    ):
                        if math.hypot(ex - sa.x1_mm, ey - sa.y1_mm) <= 0.03 or math.hypot(ex - sa.x2_mm, ey - sa.y2_mm) <= 0.03:
                            continue
                        d_pt, cx, cy = point_to_segment_dist(ex, ey, sa.x1_mm, sa.y1_mm, sa.x2_mm, sa.y2_mm)
                        if d_pt <= 0.03 and not _inside_own_pad(nid, lyr, cx, cy):
                            ang1 = _angle_deg(sa.x1_mm - cx, sa.y1_mm - cy, ox - cx, oy - cy)
                            ang2 = _angle_deg(sa.x2_mm - cx, sa.y2_mm - cy, ox - cx, oy - cy)
                            if min(ang1, ang2) < 89.0:
                                acid_trap_count += 1
                                issues.append(
                                    DFMIssue(
                                        rule="acid_trap",
                                        severity="error",
                                        x_mm=cx,
                                        y_mm=cy,
                                        layer=lyr,
                                        message=f"Acute T-junction ({min(ang1, ang2):.1f} deg) on net {nid} L{lyr+1}",
                                    )
                                )

    # 3. Asymmetric side-entry check on 0201/0402/0603/0805 passives
    asymmetric_pad_entries = 0
    for comp in board_state.netlist.components.values():
        if comp.comp_type not in ("res", "cap", "diode", "inductor") or len(comp.pads) != 2:
            continue
        if not all(p.layer_mask in (1, 2) for p in comp.pads):
            continue
        for pad in comp.pads:
            if pad.net_id <= 0 or pad.net_id not in board_state.routed_traces:
                continue
            pad_layer = 0 if (pad.layer_mask & 1) else bottom_layer
            px, py = comp.pad_world_pos(pad)
            pw, ph = comp.pad_effective_size(pad)
            dx_ax, dy_ax = px - comp.x, py - comp.y
            mag = math.hypot(dx_ax, dy_ax)
            if mag < 1e-6:
                continue
            ux, uy = dx_ax / mag, dy_ax / mag
            hit_tol = max(0.14, min(pw, ph) * 0.45)

            for seg in board_state.routed_traces.get(pad.net_id, []):
                if int(seg.layer) != pad_layer:
                    continue
                d1 = math.hypot(seg.x1_mm - px, seg.y1_mm - py)
                d2 = math.hypot(seg.x2_mm - px, seg.y2_mm - py)
                if d1 > hit_tol and d2 > hit_tol:
                    continue
                if d1 <= d2:
                    tx, ty = seg.x1_mm, seg.y1_mm
                    vx, vy = seg.x2_mm - seg.x1_mm, seg.y2_mm - seg.y1_mm
                else:
                    tx, ty = seg.x2_mm, seg.y2_mm
                    vx, vy = seg.x1_mm - seg.x2_mm, seg.y1_mm - seg.y2_mm
                if math.hypot(vx, vy) < 0.06:
                    continue
                # If endpoint sits on the longitudinal centerline near the outer pad tip, copper enters axially along the centerline
                lat_off = abs((tx - px) * (-uy) + (ty - py) * ux)
                ax_out = (tx - px) * ux + (ty - py) * uy
                if lat_off <= 0.08 and ax_out >= 0.20:
                    continue
                ang = min(_angle_deg(vx, vy, ux, uy), _angle_deg(vx, vy, -ux, -uy))
                if ang > 35.0:
                    asymmetric_pad_entries += 1
                    issues.append(
                        DFMIssue(
                            rule="asymmetric_pad_entry",
                            severity="error",
                            x_mm=px,
                            y_mm=py,
                            layer=pad_layer,
                            message=f"Off-axis pad entry ({ang:.1f} deg) on {comp.comp_id}.{pad.pad_id}",
                        )
                    )

    # 4. Screw-head & torque stress zone check for ceramic MLCC capacitors
    screw_stress_buf = float(getattr(cfg, "screw_stress_buffer_mm", 1.8))
    screw_zones: list[tuple[str, float, float, float]] = []
    for comp in board_state.netlist.components.values():
        if comp.comp_type == "mounting_hole":
            drill = float(comp.electrical_meta.get("drill_mm", min(comp.width, comp.height)))
            screw_zones.append((comp.comp_id, comp.x, comp.y, 0.5 * (drill + screw_stress_buf)))

    screw_stress_mlcc_violations = 0
    for comp in board_state.netlist.components.values():
        if comp.comp_type != "cap" or "ELEC" in (comp.footprint or "").upper():
            continue
        ew, eh = comp.effective_size()
        for mh_id, mhx, mhy, s_rad in screw_zones:
            if _rect_intersects_circle(comp.x, comp.y, ew, eh, mhx, mhy, s_rad):
                screw_stress_mlcc_violations += 1
                issues.append(
                    DFMIssue(
                        rule="screw_stress_mlcc",
                        severity="error",
                        x_mm=comp.x,
                        y_mm=comp.y,
                        layer=0,
                        message=f"Ceramic MLCC {comp.comp_id} inside {mh_id} screw-head torque zone (R={s_rad:.2f}mm)",
                    )
                )

    # 5. Silkscreen-over-pad check (F.Mask / B.Mask)
    mask_pad_margin = 0.08
    top_mask_boxes: list[tuple[float, float, float, float]] = []
    bot_mask_boxes: list[tuple[float, float, float, float]] = []
    for comp in board_state.netlist.components.values():
        for pad in comp.pads:
            wx, wy = comp.pad_world_pos(pad)
            pw, ph = comp.pad_effective_size(pad)
            b = (wx - pw * 0.5 - mask_pad_margin, wy - ph * 0.5 - mask_pad_margin, wx + pw * 0.5 + mask_pad_margin, wy + ph * 0.5 + mask_pad_margin)
            if pad.layer_mask & 1:
                top_mask_boxes.append(b)
            if pad.layer_mask & 2:
                bot_mask_boxes.append(b)
    for tp in getattr(board_state, "test_points", []):
        tr = float(tp.get("diameter_mm", 1.0)) * 0.5 + mask_pad_margin
        tb = (tp["x_mm"] - tr, tp["y_mm"] - tr, tp["x_mm"] + tr, tp["y_mm"] + tr)
        if int(tp.get("layer", 0)) == 0:
            top_mask_boxes.append(tb)
        else:
            bot_mask_boxes.append(tb)

    silk_entries = getattr(board_state, "silkscreen_adjustments", [])
    if not silk_entries:
        for cid, comp in board_state.netlist.components.items():
            if comp.comp_type == "mounting_hole":
                continue
            _, eh = comp.effective_size()
            silk_entries.append(
                {
                    "comp_id": cid,
                    "side": getattr(comp, "side", "top"),
                    "world_x_mm": comp.x,
                    "world_y_mm": comp.y - eh * 0.5 - 1.2,
                    "font_mm": 0.9,
                }
            )

    silkscreen_pad_overlaps = 0
    for entry in silk_entries:
        cid = str(entry["comp_id"])
        wx = float(entry["world_x_mm"])
        wy = float(entry["world_y_mm"])
        font_mm = float(entry.get("font_mm", 0.9))
        txt_w = max(0.8, 0.64 * len(cid) * font_mm)
        txt_h = font_mm * 1.05
        tbox = (wx - txt_w * 0.5, wy - txt_h * 0.5, wx + txt_w * 0.5, wy + txt_h * 0.5)
        mboxes = bot_mask_boxes if entry.get("side") == "bottom" else top_mask_boxes
        if _boxes_overlap_list(tbox, mboxes):
            silkscreen_pad_overlaps += 1
            issues.append(
                DFMIssue(
                    rule="silkscreen_over_pad",
                    severity="error",
                    x_mm=wx,
                    y_mm=wy,
                    layer=bottom_layer if entry.get("side") == "bottom" else 0,
                    message=f"Silkscreen designator {cid} overlaps exposed solder mask pad",
                )
            )

    # 6. Corner Fillets & Teardrops & Preferred Bus Spacing
    fillet_r = float(getattr(board_state, "corner_fillet_radius_mm", 0.0))
    has_ext_poly = len(getattr(board_state.netlist, "outline_polygon", []) or []) >= 3
    corner_fillet_valid = (fillet_r >= 1.0) or has_ext_poly
    if kicad_pcb_text is not None:
        if has_ext_poly:
            corner_fillet_valid = ('(gr_line' in kicad_pcb_text) and ('"Edge.Cuts"' in kicad_pcb_text)
        else:
            corner_fillet_valid = corner_fillet_valid and ('(gr_arc' in kicad_pcb_text) and ('"Edge.Cuts"' in kicad_pcb_text)

    total_segments = sum(len(s) for s in board_state.routed_traces.values())
    teardrop_count = len(getattr(board_state, "teardrops", []))
    teardrops_valid = (teardrop_count > 0) or (total_segments == 0)

    # Open-board preferred spacing telemetry
    total_len = 0.0
    relaxed_len = 0.0
    segs_by_layer: dict[int, list[TraceSegment]] = {}
    for segs in board_state.routed_traces.values():
        for s in segs:
            segs_by_layer.setdefault(int(s.layer), []).append(s)
    for lyr, segs in segs_by_layer.items():
        for i, sa in enumerate(segs):
            slen = math.hypot(sa.x2_mm - sa.x1_mm, sa.y2_mm - sa.y1_mm)
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
                min_edge = min(min_edge, dc - 0.5 * (sa.width_mm + sb.width_mm))
            if min_edge >= 0.28:
                relaxed_len += slen
    preferred_clearance_pct = round(100.0 * relaxed_len / total_len, 1) if total_len > 0 else 100.0

    pour_cleanup = dict(getattr(board_state, "pour_cleanup", {}))
    dead_islands_pruned = int(pour_cleanup.get("dead_islands_pruned", 0))
    test_point_count = len(getattr(board_state, "test_points", []))

    passed = (
        len(drc_errors) == 0
        and acid_trap_count == 0
        and asymmetric_pad_entries == 0
        and screw_stress_mlcc_violations == 0
        and silkscreen_pad_overlaps == 0
        and corner_fillet_valid
        and teardrops_valid
    )

    # Multi-Objective Craftsmanship Score J_rev10 in [0, 100]
    penalties = (
        18.0 * len(drc_errors)
        + 12.0 * acid_trap_count
        + 10.0 * asymmetric_pad_entries
        + 12.0 * screw_stress_mlcc_violations
        + 8.0 * silkscreen_pad_overlaps
        + (0.0 if corner_fillet_valid else 8.0)
        + (0.0 if teardrops_valid else 8.0)
    )
    margin_bonus = 0.04 * preferred_clearance_pct  # up to +4.0 pts
    tp_bonus = min(2.0, 0.7 * test_point_count)
    craftsmanship_score = round(max(0.0, min(100.0, 94.0 + margin_bonus + tp_bonus - penalties)), 2)

    return {
        "passed": passed,
        "status": "PASS" if passed else "FAIL",
        "craftsmanship_score": craftsmanship_score,
        "drc_errors": len(drc_errors),
        "acid_trap_count": acid_trap_count,
        "asymmetric_pad_entries": asymmetric_pad_entries,
        "screw_stress_mlcc_violations": screw_stress_mlcc_violations,
        "silkscreen_pad_overlaps": silkscreen_pad_overlaps,
        "corner_fillet_radius_mm": round(fillet_r, 3),
        "corner_fillet_valid": corner_fillet_valid,
        "teardrop_count": teardrop_count,
        "teardrops_valid": teardrops_valid,
        "test_point_count": test_point_count,
        "dead_islands_pruned": dead_islands_pruned,
        "preferred_clearance_pct": preferred_clearance_pct,
        "issues": [iss.to_dict() for iss in issues],
        "compromises": list(getattr(board_state, "dfm_compromises", [])),
    }


def _boxes_overlap_list(
    box: tuple[float, float, float, float],
    boxes: list[tuple[float, float, float, float]],
) -> bool:
    for b in boxes:
        if _boxes_overlap(box, b):
            return True
    return False


def generate_dfm_report_markdown(
    board_state: BoardState,
    audit: dict[str, Any] | None = None,
) -> str:
    """Generate the transparent rev10 DFM/DFA Engineering Compromise Report (`dfm_report.md`)."""
    rep = audit or getattr(board_state, "dfm_audit", None) or audit_board_dfm(board_state)
    bname = board_state.netlist.board_name
    compromises = rep.get("compromises") or getattr(board_state, "dfm_compromises", [])

    lines: list[str] = [
        f"# DFM / DFA Craftsmanship & Engineering Compromise Report — `{bname}`",
        "",
        f"* **Audit Gate Status:** **{rep['status']}** (`rev10` Craftsmanship Score: **{rep['craftsmanship_score']:.2f} / 100**)",
        f"* **Board Envelope:** `{board_state.netlist.width_mm:.1f} x {board_state.netlist.height_mm:.1f} mm` · `{board_state.num_layers}-Layer` · Corner Fillet Radius `R = {rep['corner_fillet_radius_mm']:.2f} mm`",
        "",
        "## 1. Acceptance Gate Verification (`verifier/dfm_auditor.py`)",
        "",
        "| Check | Measured | Requirement | Status |",
        "| :--- | :---: | :---: | :---: |",
        f"| **Hard KiCad / 2D DRC** | `{rep['drc_errors']}` | `0` | {'PASS' if rep['drc_errors'] == 0 else 'FAIL'} |",
        f"| **Acute Acid-Trap Junctions (<90 deg)** | `{rep['acid_trap_count']}` | `0` | {'PASS' if rep['acid_trap_count'] == 0 else 'FAIL'} |",
        f"| **Asymmetric Passive Pad Entries** | `{rep['asymmetric_pad_entries']}` | `0` | {'PASS' if rep['asymmetric_pad_entries'] == 0 else 'FAIL'} |",
        f"| **MLCCs in Screw-Head Stress Zones** | `{rep['screw_stress_mlcc_violations']}` | `0` | {'PASS' if rep['screw_stress_mlcc_violations'] == 0 else 'FAIL'} |",
        f"| **Silkscreen-Over-Pad Overlaps** | `{rep['silkscreen_pad_overlaps']}` | `0` | {'PASS' if rep['silkscreen_pad_overlaps'] == 0 else 'FAIL'} |",
        f"| **Board Corner Fillets (`Edge.Cuts` Arcs)** | `R={rep['corner_fillet_radius_mm']:.2f}mm` | `>= 1.0mm` | {'PASS' if rep['corner_fillet_valid'] else 'FAIL'} |",
        f"| **Pad & Via Teardrop Fillets** | `{rep['teardrop_count']}` | `> 0` | {'PASS' if rep['teardrops_valid'] else 'FAIL'} |",
        f"| **Open-Board Preferred Spacing (>=0.28mm)** | `{rep['preferred_clearance_pct']:.1f}%` | `Telemetry` | INFO |",
        f"| **Accessible 1.0mm Debug Test Points** | `{rep['test_point_count']}` | `Best-Effort` | INFO |",
        "",
        "## 2. Transparent Engineering Compromises",
        "",
        "| Compromise Category | Action Taken by `rev10` | Efficiency Cost | Manufacturing / Reliability Gain |",
        "| :--- | :--- | :---: | :--- |",
    ]

    if compromises:
        for c in compromises:
            lines.append(
                f"| **{c['category']}** | {c['action']} | {c['efficiency_cost']} | {c['reliability_gain']} |"
            )
    else:
        lines.append("| **Baseline** | No intrusive compromises required | `0.0mm` | Clean native routing |")

    if rep.get("issues"):
        lines.extend(["", "## 3. Remaining Audit Findings", ""])
        for iss in rep["issues"]:
            lines.append(
                f"- `[{iss['severity'].upper()}]` `{iss['rule']}` at `({iss['x_mm']:.2f}, {iss['y_mm']:.2f})mm` L{iss['layer']+1}: {iss['message']}"
            )

    lines.append("")
    return "\n".join(lines)
