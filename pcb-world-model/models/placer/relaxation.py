"""Differentiable overlap repulsion relaxation, orientation optimization, and grid snapping."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np

from env.metrics import compute_hpwl, compute_mst_wirelength, compute_subset_wire_cost
from env.step_exporter import get_component_height_mm
from verifier.fast_clearance import point_in_polygon

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


def _placement_wire_cost(
    netlist_graph: NetlistGraph,
    placement: dict[str, tuple[float, float, int]],
) -> float:
    """Combined wire objective J = 0.5 * HPWL + 0.5 * MST_Wirelength."""
    return 0.5 * compute_hpwl(netlist_graph, placement) + 0.5 * compute_mst_wirelength(
        netlist_graph, placement
    )


def _is_placement_legal(
    placement: dict[str, tuple[float, float, int]],
    netlist_graph: NetlistGraph,
    margin: float,
    bw: float,
    bh: float,
    pad: float = 1.0,
) -> bool:
    """Verify board boundary margins and zero pairwise courtyard overlaps with padding."""
    boxes: list[tuple[float, float, float, float]] = []
    for cid, comp in netlist_graph.components.items():
        cx, cy, th = placement[cid]
        ew, eh = (comp.height, comp.width) if (th % 360) in (90, 270) else (comp.width, comp.height)
        if not comp.is_locked:
            if (
                cx - ew * 0.5 < margin - 0.1
                or cx + ew * 0.5 > bw - margin + 0.1
                or cy - eh * 0.5 < margin - 0.1
                or cy + eh * 0.5 > bh - margin + 0.1
            ):
                return False
        boxes.append((cx, cy, ew, eh))

    for i in range(len(boxes)):
        ax, ay, aw, ah = boxes[i]
        for j in range(i + 1, len(boxes)):
            bx, by, bw_c, bh_c = boxes[j]
            if _rects_overlap(ax, ay, aw, ah, bx, by, bw_c, bh_c, pad=pad):
                return False
    return True


def compute_spectral_anchor_placement(
    netlist_graph: NetlistGraph,
    board_config: BoardConfig,
) -> dict[str, tuple[float, float, int]]:
    """Compute a zero-hint initial placement anchored to locked connectors without reading unlocked coordinates."""
    bw = netlist_graph.width_mm if netlist_graph.width_mm > 0 else board_config.board_width_mm
    bh = netlist_graph.height_mm if netlist_graph.height_mm > 0 else board_config.board_height_mm
    margin = board_config.edge_margin_mm + 2.0
    comp_ids = sorted(netlist_graph.components.keys())

    # Accumulate weighted anchor pull from locked components along shared nets
    anchor_sum: dict[str, list[float]] = {cid: [0.0, 0.0, 0.0] for cid in comp_ids}
    for nid in netlist_graph.routable_net_ids:
        net = netlist_graph.nets[nid]
        locked_pts: list[tuple[float, float]] = []
        unlocked_cids: list[str] = []
        for cid, _ in net.pins:
            comp = netlist_graph.components.get(cid)
            if comp is None:
                continue
            if comp.is_locked:
                locked_pts.append((float(comp.x), float(comp.y)))
            else:
                unlocked_cids.append(cid)
        if locked_pts and unlocked_cids:
            lx = sum(p[0] for p in locked_pts) / len(locked_pts)
            ly = sum(p[1] for p in locked_pts) / len(locked_pts)
            for ucid in unlocked_cids:
                anchor_sum[ucid][0] += lx * net.weight
                anchor_sum[ucid][1] += ly * net.weight
                anchor_sum[ucid][2] += net.weight

    golden_angle = math.pi * (3.0 - math.sqrt(5.0))
    unlocked_ids = [cid for cid in comp_ids if not netlist_graph.components[cid].is_locked]
    total_unlocked = max(1, len(unlocked_ids))
    placement: dict[str, tuple[float, float, int]] = {}

    for cid in comp_ids:
        comp = netlist_graph.components[cid]
        if comp.is_locked:
            placement[cid] = (float(comp.x), float(comp.y), int(comp.theta) % 360)

    dom_set = {getattr(c, "noise_domain", "neutral") for c in netlist_graph.components.values()}
    has_mixed_signal = ("power_switching" in dom_set) and ("analog_precision" in dom_set)
    pwr_locked = [
        (c.x, c.y)
        for c in netlist_graph.components.values()
        if c.is_locked and getattr(c, "noise_domain", "neutral") == "power_switching"
    ]
    ana_locked = [
        (c.x, c.y)
        for c in netlist_graph.components.values()
        if c.is_locked and getattr(c, "noise_domain", "neutral") == "analog_precision"
    ]

    pwr_slot = 0
    ana_slot = 0
    ordered_unlocked = sorted(
        unlocked_ids,
        key=lambda c_id: (
            0 if netlist_graph.components[c_id].comp_type in ("ic", "inductor") else 1,
            c_id,
        ),
    )
    for rank, cid in enumerate(ordered_unlocked):
        comp = netlist_graph.components[cid]
        ax, ay, aw = anchor_sum[cid]
        if aw > 0.0:
            base_x = 0.65 * (bw * 0.5) + 0.35 * (ax / aw)
            base_y = 0.65 * (bh * 0.5) + 0.35 * (ay / aw)
        else:
            base_x = bw * 0.5
            base_y = bh * 0.5

        if has_mixed_signal:
            flex_y0 = min(
                (float(z.y0_mm) for z in (getattr(netlist_graph, "substrate_zones", []) or []) if getattr(z, "zone_type", "rigid") == "dynamic_flex" and z.y0_mm > 5.0),
                default=None,
            )
            dom = getattr(comp, "noise_domain", "neutral")
            if flex_y0 is not None:
                if dom == "power_switching":
                    base_x = min(4.8, 0.16 * bw)
                    base_y = 4.5 + 6.5 * (pwr_slot % 3)
                    pwr_slot += 1
                    placement[cid] = (max(margin, min(bw - margin, base_x)), max(margin, min(flex_y0 - 3.0, base_y)), 0)
                    continue
                elif dom == "analog_precision":
                    base_x = max(bw - 4.8, 0.84 * bw)
                    base_y = max(margin, flex_y0 - 7.5 - 6.0 * (ana_slot % 3))
                    ana_slot += 1
                    placement[cid] = (max(margin, min(bw - margin, base_x)), max(margin, min(flex_y0 - 3.0, base_y)), 0)
                    continue
                else:
                    base_y = min(base_y, flex_y0 - 3.5)
            elif dom == "power_switching":
                if pwr_locked:
                    base_x = sum(p[0] for p in pwr_locked) / len(pwr_locked)
                    base_y = sum(p[1] for p in pwr_locked) / len(pwr_locked)
                elif bh >= bw:
                    base_y = 0.22 * bh
                else:
                    base_x = 0.22 * bw
            elif dom == "analog_precision":
                if ana_locked:
                    base_x = sum(p[0] for p in ana_locked) / len(ana_locked)
                    base_y = sum(p[1] for p in ana_locked) / len(ana_locked)
                elif bh >= bw:
                    base_y = 0.78 * bh
                else:
                    base_x = 0.78 * bw

        radius = (0.10 if has_mixed_signal else 0.28) * min(bw, bh) * math.sqrt((rank + 0.5) / total_unlocked)
        angle = rank * golden_angle
        px = max(margin, min(bw - margin, base_x + radius * math.cos(angle)))
        py = max(margin, min(bh - margin, base_y + radius * math.sin(angle)))
        placement[cid] = (px, py, 0)

    return placement


def relax_and_snap_to_grid(
    placement: dict[str, tuple[float, float, int]],
    netlist_graph: NetlistGraph,
    board_config: BoardConfig,
    iterations: int = 50,
    lr: float = 0.45,
    routing_buffer_mm: float = 1.6,
    blind_anneal: bool = False,
) -> dict[str, tuple[float, float, int]]:
    """Run 50 iterations of differentiable overlap repulsion and snap to 0.5mm manufacturing grid."""
    comp_ids = sorted(netlist_graph.components.keys())
    comp_idx = {cid: i for i, cid in enumerate(comp_ids)}
    n = len(comp_ids)
    if n == 0:
        return {}

    bw = netlist_graph.width_mm if netlist_graph.width_mm > 0 else board_config.board_width_mm
    bh = netlist_graph.height_mm if netlist_graph.height_mm > 0 else board_config.board_height_mm
    is_compact_board = max(bw, bh) <= 35.0
    margin = board_config.edge_margin_mm + (0.0 if is_compact_board else 0.5)
    eff_routing_buffer = min(routing_buffer_mm, 0.65) if is_compact_board else routing_buffer_mm
    eff_spiral_pad = 0.40 if is_compact_board else 1.2
    eff_swap_pad = 0.35 if is_compact_board else 1.0

    # Collect static keepout boxes (e.g., rf_antenna_keepout / isolation_creepage) from locked components
    static_keepouts: list[tuple[float, float, float, float]] = []
    iso_barrier_x0: list[float] = []
    screw_stress_zones: list[tuple[str, float, float, float]] = []
    enable_dfm = bool(getattr(board_config, "enable_dfm_craftsmanship", True))
    screw_buf = float(getattr(board_config, "screw_stress_buffer_mm", 1.8))
    outline_poly = list(getattr(netlist_graph, "outline_polygon", []) or [])
    z_regions = list(getattr(netlist_graph, "z_max_regions", []) or [])
    cutout_boxes: list[tuple[float, float, float, float]] = []
    for cpoly in getattr(netlist_graph, "enclosure_cutouts", []) or []:
        if len(cpoly) >= 3:
            cxs = [p[0] for p in cpoly]
            cys = [p[1] for p in cpoly]
            cutout_boxes.append((min(cxs), min(cys), max(cxs), max(cys)))
    flex_zones = [
        z for z in (getattr(netlist_graph, "substrate_zones", []) or [])
        if getattr(z, "zone_type", "rigid") == "dynamic_flex"
    ]
    for fz in flex_zones:
        if not getattr(fz, "allow_components", False):
            static_keepouts.append((float(fz.x0_mm), float(fz.y0_mm), float(fz.x1_mm), float(fz.y1_mm)))
    rigid_main_max_y: float | None = min((float(fz.y0_mm) for fz in flex_zones if fz.y0_mm > 5.0), default=None)
    if flex_zones and rigid_main_max_y is not None:
        # Reserve direct F.Cu breakout corridor for locked flex-crossing diff-pair components
        fz0 = flex_zones[0]
        locked_pre = [
            c for c in netlist_graph.components.values()
            if c.is_locked and c.y < fz0.y0_mm and c.comp_type == "ic"
        ]
        locked_post = [
            c for c in netlist_graph.components.values()
            if c.is_locked and c.y > fz0.y1_mm
        ]
        for lc_a in locked_pre:
            for lc_b in locked_post:
                shared = {
                    p.net_id for p in lc_a.pads if p.net_id > 0
                } & {
                    p.net_id for p in lc_b.pads if p.net_id > 0
                }
                sig_shared = [
                    nid for nid in shared
                    if nid in netlist_graph.nets and not netlist_graph.nets[nid].is_power_ground
                ]
                if sig_shared:
                    xs_a = [lc_a.pad_world_pos(p)[0] for p in lc_a.pads if p.net_id in sig_shared]
                    ys_a = [lc_a.pad_world_pos(p)[1] for p in lc_a.pads if p.net_id in sig_shared]
                    xs_b = [lc_b.pad_world_pos(p)[0] for p in lc_b.pads if p.net_id in sig_shared]
                    cx_lo = min(xs_a + xs_b) - 0.8
                    cx_hi = max(xs_a + xs_b) + 0.8
                    cy_lo = max(ys_a) + 0.3
                    static_keepouts.append((cx_lo, cy_lo, cx_hi, float(fz0.y0_mm)))
    poly_cx = (sum(p[0] for p in outline_poly) / len(outline_poly)) if outline_poly else bw * 0.5
    poly_cy = (sum(p[1] for p in outline_poly) / len(outline_poly)) if outline_poly else bh * 0.5
    for comp in netlist_graph.components.values():
        if comp.comp_type == "mounting_hole" and enable_dfm:
            drill_mm = float(comp.electrical_meta.get("drill_mm", min(comp.width, comp.height)))
            stress_rad = 0.5 * (drill_mm + screw_buf)
            screw_stress_zones.append((comp.comp_id, float(comp.x), float(comp.y), stress_rad))
        if comp.is_locked:
            for kx0, ky0, kx1, ky1, _, kind in comp.world_keepout_zones():
                static_keepouts.append((kx0, ky0, kx1, ky1))
                if kind == "isolation_creepage":
                    iso_barrier_x0.append(comp.x - comp.effective_size()[0] * 0.5)

    def _rect_intersects_circle(
        rx: float, ry: float, rw: float, rh: float, cx: float, cy: float, rad: float
    ) -> bool:
        half_w, half_h = rw * 0.5, rh * 0.5
        nx = max(rx - half_w, min(cx, rx + half_w))
        ny = max(ry - half_h, min(cy, ry + half_h))
        return math.hypot(cx - nx, cy - ny) < rad - 1e-4

    # Rev 10 DFA rework buffer around tall connectors & fine-pitch QFNs/LGAs
    rework_extra = 0.0
    if enable_dfm:
        is_dense = is_compact_board or len(comp_ids) > 30
        rework_extra = 0.25 if is_dense else min(0.8, float(getattr(board_config, "rework_courtyard_margin_mm", 0.8))) * 0.5
    rework_pad = np.zeros(n, dtype=np.float32)
    is_mlcc_cap = np.zeros(n, dtype=bool)
    is_passive_align = np.zeros(n, dtype=bool)
    comp_z_keepouts: list[list[tuple[float, float, float, float]]] = [[] for _ in range(n)]
    for idx_c, cid_c in enumerate(comp_ids):
        comp_c = netlist_graph.components[cid_c]
        fp_u = (comp_c.footprint or "").upper()
        is_mlcc_cap[idx_c] = comp_c.comp_type == "cap" and "ELEC" not in fp_u
        is_passive_align[idx_c] = comp_c.comp_type in ("res", "cap", "diode")
        if enable_dfm and (
            comp_c.comp_type in ("connector", "switch")
            or ("QFN" in fp_u or "LGA" in fp_u) and len(comp_c.pads) >= 16
        ):
            rework_pad[idx_c] = rework_extra
        comp_z = get_component_height_mm(comp_c)
        if getattr(comp_c, "side", "top") != "bottom" and comp_c.comp_type != "mounting_hole":
            for reg in z_regions:
                if comp_z > float(reg["z_max_mm"]) + 1e-3:
                    comp_z_keepouts[idx_c].append(
                        (float(reg["x0"]), float(reg["y0"]), float(reg["x1"]), float(reg["y1"]))
                    )
        comp_z_keepouts[idx_c].extend(cutout_boxes)
    lv_zone_max_x: float | None = (min(iso_barrier_x0) - 0.5) if iso_barrier_x0 else None
    hv_tokens = ("HV_", "AC_", "MAINS", "RELAY_COM", "RELAY_NO", "RELAY_NC")
    comp_max_x: dict[str, float] = {}
    comp_max_y: dict[str, float] = {}
    for cid in comp_ids:
        comp = netlist_graph.components[cid]
        if lv_zone_max_x is not None and not comp.is_locked:
            comp_has_hv = any(
                p.net_id in netlist_graph.nets and any(t in netlist_graph.nets[p.net_id].name.upper() for t in hv_tokens)
                for p in comp.pads
            )
            if not comp_has_hv:
                comp_max_x[cid] = lv_zone_max_x
        if rigid_main_max_y is not None and not comp.is_locked and comp.y < rigid_main_max_y + 1.0:
            comp_max_y[cid] = rigid_main_max_y - 0.4

    # Prepare initial tensors
    init_xy: list[list[float]] = []
    sizes: list[list[float]] = []
    locked: list[bool] = []
    thetas: list[int] = []
    sides: list[str] = []
    has_tht: list[bool] = []

    for cid in comp_ids:
        comp = netlist_graph.components[cid]
        if comp.is_locked:
            x, y, th = comp.x, comp.y, int(comp.theta) % 360
        else:
            x, y, th = placement.get(cid, (comp.x, comp.y, comp.theta))
            th = int(round((th % 360) / 90.0) * 90) % 360
        ew, eh = (comp.height, comp.width) if th in (90, 270) else (comp.width, comp.height)
        x_lim_hi = comp_max_x.get(cid, bw - margin) - ew * 0.5
        y_lim_hi = comp_max_y.get(cid, bh - margin) - eh * 0.5
        x = max(margin + ew * 0.5, min(max(margin + ew * 0.5, x_lim_hi), float(x)))
        y = max(margin + eh * 0.5, min(max(margin + eh * 0.5, y_lim_hi), float(y)))
        init_xy.append([x, y])
        sizes.append([ew, eh])
        locked.append(bool(comp.is_locked))
        thetas.append(th)
        sides.append(getattr(comp, "side", "top"))
        has_tht.append(any(p.layer_mask == 3 for p in comp.pads) or comp.comp_type == "mounting_hole")

    xy = np.array(init_xy, dtype=np.float32)
    fixed_xy = xy.copy()
    size_arr = np.array(sizes, dtype=np.float32)
    locked_arr = np.array(locked, dtype=bool)
    unlocked_arr = ~locked_arr

    # Build vectorized component-component net spring pairs for fast wirelength cohesion
    comp_domains = [getattr(netlist_graph.components[cid], "noise_domain", "neutral") for cid in comp_ids]
    has_mixed_signal = ("power_switching" in comp_domains) and ("analog_precision" in comp_domains)
    min_dom_sep = min(
        max(float(getattr(board_config, "min_domain_separation_mm", 20.0)) + 5.0, 25.0),
        0.74 * max(bw, bh),
    )

    high_priority_comps = {
        cid
        for cid, comp in netlist_graph.components.items()
        if any(r in getattr(comp, "special_layout_rules", []) for r in ("high_di_dt_loop", "kelvin_sense_pair"))
    }
    u_list: list[int] = []
    v_list: list[int] = []
    w_list: list[float] = []
    comp_nets: dict[str, set[int]] = {cid: set() for cid in comp_ids}
    for nid in netlist_graph.routable_net_ids:
        net = netlist_graph.nets[nid]
        cset = sorted({c for c, _ in net.pins if c in comp_idx})
        if len(net.pins) <= 16:
            for c in cset:
                comp_nets[c].add(nid)
        spring_w = float(net.weight) if len(cset) <= 12 else float(net.weight) / max(1.0, 0.25 * len(cset))
        if any(c in high_priority_comps for c in cset):
            spring_w *= 1.8
        for i in range(len(cset)):
            for j in range(i + 1, len(cset)):
                ui = comp_idx[cset[i]]
                vi = comp_idx[cset[j]]
                pair_w = spring_w
                if has_mixed_signal:
                    di, dj = comp_domains[ui], comp_domains[vi]
                    if di == dj and di in ("power_switching", "analog_precision"):
                        pair_w *= 2.4
                    elif {di, dj} == {"power_switching", "analog_precision"}:
                        pair_w *= 0.02
                    elif net.is_power_ground and di != dj:
                        pair_w *= 0.02
                u_list.append(ui)
                v_list.append(vi)
                w_list.append(pair_w)

    has_springs = len(u_list) > 0
    if has_springs:
        u_idx_np = np.array(u_list, dtype=np.int32)
        v_idx_np = np.array(v_list, dtype=np.int32)
        w_np = np.array(w_list, dtype=np.float32)

    half_sizes = size_arr * 0.5
    half_eff_w = (size_arr[:, 0] + eff_routing_buffer + rework_pad) * 0.5
    half_eff_h = (size_arr[:, 1] + eff_routing_buffer + rework_pad) * 0.5
    eye_mask = np.zeros((n, n), dtype=bool)
    cross_domain_mask = np.zeros((n, n), dtype=bool)
    for i in range(n):
        for j in range(n):
            if i != j and (sides[i] == sides[j] or has_tht[i] or has_tht[j]):
                eye_mask[i, j] = True
            if (
                has_mixed_signal
                and i != j
                and sides[i] == sides[j]
                and {comp_domains[i], comp_domains[j]} == {"power_switching", "analog_precision"}
            ):
                cross_domain_mask[i, j] = True

    # Analytical Adam optimizer (avoids PyTorch autograd overhead in tight inner loop)
    m_adam = np.zeros_like(xy)
    v_adam = np.zeros_like(xy)
    beta1, beta2, eps_adam = 0.9, 0.999, 1e-8
    eff_iters = 0 if (not blind_anneal and has_mixed_signal and rigid_main_max_y is not None) else iterations

    for step in range(eff_iters):
        xy[locked_arr] = fixed_xy[locked_arr]

        x0 = xy[:, 0] - half_eff_w
        x1 = xy[:, 0] + half_eff_w
        y0 = xy[:, 1] - half_eff_h
        y1 = xy[:, 1] + half_eff_h

        ox = np.maximum(0.0, np.minimum(x1[:, None], x1[None, :]) - np.maximum(x0[:, None], x0[None, :]))
        oy = np.maximum(0.0, np.minimum(y1[:, None], y1[None, :]) - np.maximum(y0[:, None], y0[None, :]))
        overlap_active = (ox > 0.0) & (oy > 0.0) & eye_mask

        grad = np.zeros_like(xy)
        if overlap_active.any():
            d_ox = (
                (x1[:, None] < x1[None, :]).astype(np.float32)
                + 0.5 * (x1[:, None] == x1[None, :]).astype(np.float32)
                - (x0[:, None] > x0[None, :]).astype(np.float32)
                - 0.5 * (x0[:, None] == x0[None, :]).astype(np.float32)
            )
            d_oy = (
                (y1[:, None] < y1[None, :]).astype(np.float32)
                + 0.5 * (y1[:, None] == y1[None, :]).astype(np.float32)
                - (y0[:, None] > y0[None, :]).astype(np.float32)
                - 0.5 * (y0[:, None] == y0[None, :]).astype(np.float32)
            )
            grad[:, 0] += 20.0 * np.sum(np.where(overlap_active, oy * d_ox, 0.0), axis=1)
            grad[:, 1] += 20.0 * np.sum(np.where(overlap_active, ox * d_oy, 0.0), axis=1)

        diff_all = xy[:, None, :] - xy[None, :, :]
        dists_sq = (diff_all * diff_all).sum(axis=-1)
        repel_active = (dists_sq < 9.0) & eye_mask
        if repel_active.any():
            grad += -0.30 * np.sum(np.where(repel_active[:, :, None], diff_all, 0.0), axis=1)

        # Rev 13: Cross-domain repulsion between power_switching (GaN) and analog_precision (24-bit ADC)
        if has_mixed_signal and cross_domain_mask.any():
            dist_mat = np.sqrt(dists_sq + 1e-4)
            too_close = (dist_mat < min_dom_sep) & cross_domain_mask
            if too_close.any():
                # Push apart along vector
                unit_vec = diff_all / dist_mat[:, :, None]
                strength = np.where(too_close, 26.0 * (1.0 - dist_mat / max(1.0, min_dom_sep)) + 8.0, 0.0)
                grad += -np.sum(strength[:, :, None] * unit_vec, axis=1)
            # Strong pole bias along longest board axis (or diagonal corners in Rigid_Main) to prevent local stalls
            eff_bh = rigid_main_max_y if rigid_main_max_y is not None else bh
            major_ax = 1 if eff_bh >= bw else 0
            span_ax = eff_bh if major_ax == 1 else bw
            pole_f = 24.0 if rigid_main_max_y is not None else 10.0
            for idx_d, dom_d in enumerate(comp_domains):
                if locked_arr[idx_d]:
                    continue
                if dom_d == "power_switching":
                    if xy[idx_d, major_ax] > (0.38 if rigid_main_max_y else 0.25) * span_ax:
                        grad[idx_d, major_ax] += pole_f
                    if rigid_main_max_y is not None and xy[idx_d, 0] > 0.25 * bw:
                        grad[idx_d, 0] += pole_f * 1.4
                elif dom_d == "analog_precision":
                    if xy[idx_d, major_ax] < (0.62 if rigid_main_max_y else 0.72) * span_ax:
                        grad[idx_d, major_ax] -= pole_f
                    if rigid_main_max_y is not None and xy[idx_d, 0] < 0.75 * bw:
                        grad[idx_d, 0] -= pole_f * 1.4

        if screw_stress_zones and is_mlcc_cap.any():
            for _, mhx, mhy, s_rad in screw_stress_zones:
                dx_m = xy[:, 0] - mhx
                dy_m = xy[:, 1] - mhy
                dist_m = np.sqrt(dx_m * dx_m + dy_m * dy_m) + 1e-4
                req_d = s_rad + np.maximum(half_sizes[:, 0], half_sizes[:, 1]) + 0.15
                viol = is_mlcc_cap & unlocked_arr & (dist_m < req_d)
                if viol.any():
                    grad[viol, 0] += -25.0 * (dx_m[viol] / dist_m[viol])
                    grad[viol, 1] += -25.0 * (dy_m[viol] / dist_m[viol])

        grad[:, 0] += 30.0 * (
            -(xy[:, 0] - half_sizes[:, 0] < margin).astype(np.float32)
            + (xy[:, 0] + half_sizes[:, 0] > bw - margin).astype(np.float32)
        )
        grad[:, 1] += 30.0 * (
            -(xy[:, 1] - half_sizes[:, 1] < margin).astype(np.float32)
            + (xy[:, 1] + half_sizes[:, 1] > bh - margin).astype(np.float32)
        )

        # Rev 11: Repel unlocked components from 3D Z_max rib keepouts & outside outline_polygon
        for idx_c in range(n):
            if locked_arr[idx_c]:
                continue
            cx_c, cy_c = float(xy[idx_c, 0]), float(xy[idx_c, 1])
            hw_c, hh_c = float(half_sizes[idx_c, 0]), float(half_sizes[idx_c, 1])
            for kx0, ky0, kx1, ky1 in comp_z_keepouts[idx_c]:
                kcx, kcy = 0.5 * (kx0 + kx1), 0.5 * (ky0 + ky1)
                if _rects_overlap(cx_c, cy_c, hw_c * 2.0, hh_c * 2.0, kcx, kcy, kx1 - kx0, ky1 - ky0, pad=0.15):
                    dx_k = cx_c - kcx
                    dy_k = cy_c - kcy
                    d_k = math.hypot(dx_k, dy_k) + 1e-4
                    grad[idx_c, 0] += -26.0 * (dx_k / d_k)
                    grad[idx_c, 1] += -26.0 * (dy_k / d_k)
            if len(outline_poly) >= 3:
                corners = [
                    (cx_c - hw_c, cy_c - hh_c),
                    (cx_c + hw_c, cy_c - hh_c),
                    (cx_c + hw_c, cy_c + hh_c),
                    (cx_c - hw_c, cy_c + hh_c),
                ]
                if not all(point_in_polygon(px, py, outline_poly) for px, py in corners):
                    dx_p = cx_c - poly_cx
                    dy_p = cy_c - poly_cy
                    d_p = math.hypot(dx_p, dy_p) + 1e-4
                    grad[idx_c, 0] += 28.0 * (dx_p / d_p)
                    grad[idx_c, 1] += 28.0 * (dy_p / d_p)

        hpwl_w = (
            0.28 - 0.20 * (step / max(1, iterations - 1))
            if blind_anneal
            else 0.08
        )
        if has_springs:
            diff_e = xy[u_idx_np] - xy[v_idx_np]
            dist_e = np.sqrt((diff_e * diff_e).sum(axis=-1) + 1e-4)
            g_edge = (hpwl_w * w_np / dist_e)[:, None] * diff_e
            np.add.at(grad, u_idx_np, g_edge)
            np.add.at(grad, v_idx_np, -g_edge)

        grad[locked_arr] = 0.0

        t_step = step + 1
        m_adam = beta1 * m_adam + (1.0 - beta1) * grad
        v_adam = beta2 * v_adam + (1.0 - beta2) * (grad * grad)
        m_hat = m_adam / (1.0 - beta1**t_step)
        v_hat = v_adam / (1.0 - beta2**t_step)
        xy[unlocked_arr] -= lr * m_hat[unlocked_arr] / (np.sqrt(v_hat[unlocked_arr]) + eps_adam)

    xy[locked_arr] = fixed_xy[locked_arr]
    final_xy = xy

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
        x_lim_hi = _snap(max(margin + ew * 0.5, comp_max_x.get(cid, bw - margin) - ew * 0.5), grid)
        y_lim_hi = _snap(max(margin + eh * 0.5, comp_max_y.get(cid, bh - margin) - eh * 0.5), grid)
        cx = max(_snap(margin + ew * 0.5, grid), min(x_lim_hi, cx))
        cy = max(_snap(margin + eh * 0.5, grid), min(y_lim_hi, cy))
        legal[cid] = (cx, cy, thetas[idx])

    # Deterministic spiral grid search so `legal` starts completely overlap-free on the 0.5mm grid
    placed_ids = [cid for cid in comp_ids if netlist_graph.components[cid].is_locked]
    unlocked_ids = [cid for cid in comp_ids if not netlist_graph.components[cid].is_locked]
    for cid in unlocked_ids:
        comp = netlist_graph.components[cid]
        cx, cy, th = legal[cid]
        ew, eh = (comp.height, comp.width) if th in (90, 270) else (comp.width, comp.height)
        cid_x_hi = comp_max_x.get(cid, bw - margin)
        cid_y_hi = comp_max_y.get(cid, bh - margin)

        def is_valid(tx: float, ty: float) -> bool:
            if (
                tx - ew * 0.5 < margin - 0.01
                or tx + ew * 0.5 > cid_x_hi + 0.01
                or ty - eh * 0.5 < margin - 0.01
                or ty + eh * 0.5 > cid_y_hi + 0.01
            ):
                return False
            c_idx = comp_idx[cid]
            if len(outline_poly) >= 3:
                corners = [
                    (tx - ew * 0.5, ty - eh * 0.5),
                    (tx + ew * 0.5, ty - eh * 0.5),
                    (tx + ew * 0.5, ty + eh * 0.5),
                    (tx - ew * 0.5, ty + eh * 0.5),
                ]
                if not all(point_in_polygon(px, py, outline_poly) for px, py in corners):
                    return False
            for kx0, ky0, kx1, ky1 in comp_z_keepouts[c_idx]:
                if _rects_overlap(
                    tx, ty, ew, eh,
                    0.5 * (kx0 + kx1), 0.5 * (ky0 + ky1),
                    kx1 - kx0, ky1 - ky0, pad=0.1,
                ):
                    return False
            for kx0, ky0, kx1, ky1 in static_keepouts:
                if _rects_overlap(
                    tx, ty, ew, eh,
                    0.5 * (kx0 + kx1), 0.5 * (ky0 + ky1),
                    kx1 - kx0, ky1 - ky0, pad=0.2,
                ):
                    return False
            if is_mlcc_cap[c_idx]:
                for _, mhx, mhy, s_rad in screw_stress_zones:
                    if _rect_intersects_circle(tx, ty, ew, eh, mhx, mhy, s_rad):
                        return False
            for pid in placed_ids:
                p_idx = comp_idx[pid]
                if not eye_mask[c_idx, p_idx]:
                    continue
                pc = netlist_graph.components[pid]
                px, py, pth = legal[pid]
                pw, ph = (pc.height, pc.width) if pth in (90, 270) else (pc.width, pc.height)
                pair_pad = eff_spiral_pad + max(float(rework_pad[c_idx]), float(rework_pad[p_idx]))
                if _rects_overlap(tx, ty, ew, eh, px, py, pw, ph, pad=pair_pad):
                    return False
                if has_mixed_signal and cross_domain_mask[c_idx, p_idx]:
                    if math.hypot(tx - px, ty - py) < min(8.0, 0.40 * min_dom_sep):
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

    # 2-pass joint refinement using exact incremental net delta evaluation
    # Vectorized courtyard bounding boxes for fast O(N) legality checks during 2-opt/orientation
    cid_to_idx = {cid: idx for idx, cid in enumerate(comp_ids)}
    box_arr = np.zeros((len(comp_ids), 4), dtype=np.float64)
    for idx, cid in enumerate(comp_ids):
        cx_i, cy_i, th_i = legal[cid]
        c_i = netlist_graph.components[cid]
        ew_i, eh_i = (c_i.height, c_i.width) if (th_i % 360) in (90, 270) else (c_i.width, c_i.height)
        box_arr[idx] = (cx_i, cy_i, ew_i, eh_i)

    def _sync_box(cid: str, pose: tuple[float, float, int]) -> None:
        idx = cid_to_idx[cid]
        c_i = netlist_graph.components[cid]
        px, py, pth = pose
        ew_i, eh_i = (c_i.height, c_i.width) if (pth % 360) in (90, 270) else (c_i.width, c_i.height)
        box_arr[idx] = (px, py, ew_i, eh_i)

    def _pair_is_legal(moved: dict[str, tuple[float, float, int]]) -> bool:
        moved_idxs = {cid_to_idx[m_cid] for m_cid in moved}
        moved_boxes: list[tuple[int, float, float, float, float]] = []
        for m_cid, (mx, my, mth) in moved.items():
            mc = netlist_graph.components[m_cid]
            mew, meh = (mc.height, mc.width) if (mth % 360) in (90, 270) else (mc.width, mc.height)
            if (
                mx - mew * 0.5 < margin - 0.1
                or mx + mew * 0.5 > bw - margin + 0.1
                or my - meh * 0.5 < margin - 0.1
                or my + meh * 0.5 > bh - margin + 0.1
            ):
                return False
            moved_boxes.append((cid_to_idx[m_cid], mx, my, mew, meh))

        # Check moved-vs-keepouts, enclosure Z_max / polygon, and screw-head torque stress zones
        for m_idx, mx, my, mew, meh in moved_boxes:
            if len(outline_poly) >= 3:
                corners = [
                    (mx - mew * 0.5, my - meh * 0.5),
                    (mx + mew * 0.5, my - meh * 0.5),
                    (mx + mew * 0.5, my + meh * 0.5),
                    (mx - mew * 0.5, my + meh * 0.5),
                ]
                if not all(point_in_polygon(px, py, outline_poly) for px, py in corners):
                    return False
            for kx0, ky0, kx1, ky1 in comp_z_keepouts[m_idx]:
                if _rects_overlap(
                    mx, my, mew, meh,
                    0.5 * (kx0 + kx1), 0.5 * (ky0 + ky1),
                    kx1 - kx0, ky1 - ky0, pad=0.1,
                ):
                    return False
            for kx0, ky0, kx1, ky1 in static_keepouts:
                if _rects_overlap(
                    mx, my, mew, meh,
                    0.5 * (kx0 + kx1), 0.5 * (ky0 + ky1),
                    kx1 - kx0, ky1 - ky0, pad=0.2,
                ):
                    return False
            if is_mlcc_cap[m_idx]:
                for _, mhx, mhy, s_rad in screw_stress_zones:
                    if _rect_intersects_circle(mx, my, mew, meh, mhx, mhy, s_rad):
                        return False

        # Check moved-vs-moved
        for i_a in range(len(moved_boxes)):
            idx_a, ax, ay, aw, ah = moved_boxes[i_a]
            for i_b in range(i_a + 1, len(moved_boxes)):
                idx_b, bx, by, bw_c, bh_c = moved_boxes[i_b]
                if not eye_mask[idx_a, idx_b]:
                    continue
                if _rects_overlap(ax, ay, aw, ah, bx, by, bw_c, bh_c, pad=eff_swap_pad):
                    return False

        # Check moved-vs-stationary using vectorized NumPy comparisons
        for m_idx, mx, my, mew, meh in moved_boxes:
            overlap_x = np.abs(mx - box_arr[:, 0]) < (0.5 * (mew + box_arr[:, 2]) + eff_swap_pad - 1e-4)
            overlap_y = np.abs(my - box_arr[:, 1]) < (0.5 * (meh + box_arr[:, 3]) + eff_swap_pad - 1e-4)
            hits = np.flatnonzero(overlap_x & overlap_y & eye_mask[m_idx])
            for h_idx in hits:
                if int(h_idx) not in moved_idxs:
                    return False
        return True

    for _pass in range(2):
        # 2-Opt Component Swap Loop over unlocked components with compatible dimensions
        for i in range(len(unlocked_ids)):
            cid_a = unlocked_ids[i]
            comp_a = netlist_graph.components[cid_a]
            for j in range(i + 1, len(unlocked_ids)):
                cid_b = unlocked_ids[j]
                comp_b = netlist_graph.components[cid_b]
                if getattr(comp_a, "side", "top") != getattr(comp_b, "side", "top"):
                    continue
                if has_mixed_signal and (
                    getattr(comp_a, "noise_domain", "neutral") != getattr(comp_b, "noise_domain", "neutral")
                ):
                    continue
                if (
                    abs(comp_a.width - comp_b.width) > 2.5
                    or abs(comp_a.height - comp_b.height) > 2.5
                ):
                    continue
                xa, ya, tha = legal[cid_a]
                xb, yb, thb = legal[cid_b]
                moved = {
                    cid_a: (_snap(xb, grid), _snap(yb, grid), tha),
                    cid_b: (_snap(xa, grid), _snap(ya, grid), thb),
                }
                if not _pair_is_legal(moved):
                    continue
                affected = comp_nets[cid_a] | comp_nets[cid_b]
                if not affected:
                    continue
                before_cost = compute_subset_wire_cost(netlist_graph, affected, legal)
                trial = dict(legal)
                trial.update(moved)
                after_cost = compute_subset_wire_cost(netlist_graph, affected, trial)
                if after_cost + 0.05 < before_cost:
                    legal = trial
                    _sync_box(cid_a, moved[cid_a])
                    _sync_box(cid_b, moved[cid_b])

        # Pin-Aware Discrete Orientation Search (0, 90, 180, 270) with 0/90 passive family preference
        for cid in unlocked_ids:
            affected = comp_nets[cid]
            if not affected:
                continue
            cur_x, cur_y, best_th = legal[cid]
            cur_cost = compute_subset_wire_cost(netlist_graph, affected, legal)
            c_idx = cid_to_idx[cid]
            canon_bias = 0.25 if (enable_dfm and is_passive_align[c_idx]) else 0.0
            for cand_th in (0, 90, 180, 270):
                if cand_th == best_th:
                    continue
                moved = {cid: (cur_x, cur_y, cand_th)}
                if not _pair_is_legal(moved):
                    continue
                trial = dict(legal)
                trial[cid] = (cur_x, cur_y, cand_th)
                trial_cost = compute_subset_wire_cost(netlist_graph, affected, trial)
                penalty_delta = (
                    (0.0 if cand_th in (0, 90) else canon_bias)
                    - (0.0 if best_th in (0, 90) else canon_bias)
                )
                if (trial_cost + penalty_delta) + 0.05 < cur_cost:
                    legal = trial
                    _sync_box(cid, (cur_x, cur_y, cand_th))
                    cur_cost = trial_cost
                    best_th = cand_th

    return legal


def migrate_pdn_caps_under_bga(
    graph: NetlistGraph,
    board_config: BoardConfig,
) -> list[dict[str, Any]]:
    """Rip up hostile remote decoupling traces & relocate bypass caps under BGA on B.Cu (rev14)."""
    from env.board_state import BoardState
    from env.metrics import compute_pdn_decoupling_inductance

    probe_state = BoardState(netlist=graph, config=board_config)
    pdn_rep = compute_pdn_decoupling_inductance(probe_state)
    violations = pdn_rep.get("violations", [])
    if not violations:
        return []

    slots = [
        (-1.6, -1.2),
        (1.6, 1.2),
        (-1.6, 1.2),
        (1.6, -1.2),
        (0.0, -2.0),
        (0.0, 2.0),
    ]
    used_slots_by_ic: dict[str, int] = {}
    migrated: list[dict[str, Any]] = []
    old_pad_pts: list[tuple[float, float]] = []
    ripped_pwr_nids: set[int] = set()

    for viol in violations:
        cid = str(viol["cap_id"])
        ic_id = str(viol["ic_id"])
        cap = graph.components.get(cid)
        ic = graph.components.get(ic_id)
        if cap is None or ic is None or cap.is_locked:
            continue

        for p in cap.pads:
            old_pad_pts.append(cap.pad_world_pos(p))
        ripped_pwr_nids.add(int(viol.get("net_id", 0)))

        slot_idx = used_slots_by_ic.get(ic_id, 0)
        used_slots_by_ic[ic_id] = slot_idx + 1
        dx, dy = slots[slot_idx % len(slots)]
        before_xy = [round(cap.x, 3), round(cap.y, 3)]

        cap.to_bottom_side()
        cap.theta = 0
        cap.x = round(ic.x + dx, 2)
        cap.y = round(ic.y + dy, 2)
        cap.is_locked = True
        cap.electrical_meta["under_bga_migrated"] = True
        cap.electrical_meta["migrated_from"] = before_xy
        cap.electrical_meta["before_l_loop_nh"] = float(viol["l_loop_nh"])

        migrated.append(
            {
                "cap_id": cid,
                "ic_id": ic_id,
                "pwr_net": viol["pwr_net"],
                "before_l_loop_nh": float(viol["l_loop_nh"]),
                "before_v_droop_mv": float(viol["v_droop_mv"]),
                "before_xy": before_xy,
                "after_xy": [cap.x, cap.y],
                "side": cap.side,
            }
        )

    # Rip up pre-seeded remote PDN traces attached to the old capacitor pad coordinates
    if migrated and getattr(graph, "raw_segments", None):
        kept_segs: list[dict[str, Any]] = []
        for sd in graph.raw_segments:
            snid = int(sd.get("net_id", 0))
            sx1, sy1 = float(sd["x1_mm"]), float(sd["y1_mm"])
            sx2, sy2 = float(sd["x2_mm"]), float(sd["y2_mm"])
            touches_old = any(
                math.hypot(sx1 - ox, sy1 - oy) <= 0.6 or math.hypot(sx2 - ox, sy2 - oy) <= 0.6
                for ox, oy in old_pad_pts
            )
            if touches_old or snid in ripped_pwr_nids:
                continue
            kept_segs.append(sd)
        graph.raw_segments = kept_segs
        if not kept_segs:
            graph.enclosure_meta.pop("preserve_preseeded_traces", None)

    if migrated:
        graph.enclosure_meta["pdn_violations_remediated"] = migrated
        graph.design_notes.append(
            f"[rev14 PDN Migration] Flagged pdn_excessive_loop_inductance on "
            f"{', '.join(m['cap_id'] for m in migrated)}; ripped up remote F.Cu traces & relocated caps under BGA on B.Cu via-in-pad."
        )

    return migrated
