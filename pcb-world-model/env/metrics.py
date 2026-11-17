"""Routing and placement metrics: HPWL, MST wirelength, congestion, and net ranking."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from env.board_state import BoardState
    from env.sexpr_parser import NetlistGraph

# Standard 1 oz/ft^2 copper electrical constants (annealed copper IACS at 20 C)
COPPER_RESISTIVITY_OHM_MM: float = 1.724e-5
DEFAULT_COPPER_THICKNESS_MM: float = 0.035
FLEX_COPPER_THICKNESS_MM: float = 0.018
DEFAULT_VIA_PLATING_MM: float = 0.020
FR4_ER: float = 4.5
POLYIMIDE_ER: float = 3.4


def _net_pin_world_coords(
    netlist_graph: NetlistGraph,
    net_id: int,
    placement: dict[str, tuple[float, float, int]] | None = None,
) -> list[tuple[float, float, int]]:
    """Return `(x_mm, y_mm, layer_mask)` coordinates for all pads on `net_id`."""
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
            norm_t = int(ctheta) % 360
            if norm_t == 0:
                wx = cx + pad.rel_x
                wy = cy + pad.rel_y
            elif norm_t == 90:
                wx = cx - pad.rel_y
                wy = cy + pad.rel_x
            elif norm_t == 180:
                wx = cx - pad.rel_x
                wy = cy - pad.rel_y
            elif norm_t == 270:
                wx = cx + pad.rel_y
                wy = cy - pad.rel_x
            else:
                rad = math.radians(norm_t)
                wx = cx + (pad.rel_x * math.cos(rad) - pad.rel_y * math.sin(rad))
                wy = cy + (pad.rel_x * math.sin(rad) + pad.rel_y * math.cos(rad))
        else:
            wx, wy = comp.pad_world_pos(pad)
        pin_coords.append((wx, wy, pad.layer_mask))
    return pin_coords


def compute_net_mst_pairs(
    netlist_graph: NetlistGraph,
    net_id: int,
    placement: dict[str, tuple[float, float, int]] | None = None,
) -> list[tuple[tuple[float, float, int], tuple[float, float, int]]]:
    """Decompose a multi-pin net into 2-pin Minimum Spanning Tree segments via Kruskal's MST.

    Each pin endpoint is `(x_mm, y_mm, layer_mask)`.
    """
    pin_coords = _net_pin_world_coords(netlist_graph, net_id, placement)
    n = len(pin_coords)
    if n < 2:
        return []
    if n == 2:
        return [(pin_coords[0], pin_coords[1])]

    edges: list[tuple[float, int, int]] = []
    for i in range(n):
        x1, y1, _ = pin_coords[i]
        for j in range(i + 1, n):
            x2, y2, _ = pin_coords[j]
            dx = x2 - x1
            dy = y2 - y1
            edges.append((dx * dx + dy * dy, i, j))
    edges.sort()

    parent = list(range(n))

    def _find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    adj: list[list[int]] = [[] for _ in range(n)]
    edges_added = 0
    for _, u, v in edges:
        ru = _find(u)
        rv = _find(v)
        if ru != rv:
            parent[ru] = rv
            adj[u].append(v)
            edges_added += 1
            if edges_added == n - 1:
                break

    pairs: list[tuple[tuple[float, float, int], tuple[float, float, int]]] = []
    for u in range(n):
        if adj[u]:
            adj[u].sort()
            pu = pin_coords[u]
            for v in adj[u]:
                pairs.append((pu, pin_coords[v]))
    return pairs


def compute_net_hpwl(
    netlist_graph: NetlistGraph,
    net_id: int,
    placement: dict[str, tuple[float, float, int]] | None = None,
) -> float:
    """Calculate weighted Half-Perimeter Wirelength (HPWL) in mm for a single net."""
    pin_coords = _net_pin_world_coords(netlist_graph, net_id, placement)
    if len(pin_coords) < 2:
        return 0.0
    xs = [p[0] for p in pin_coords]
    ys = [p[1] for p in pin_coords]
    weight = netlist_graph.nets[net_id].weight
    return float(weight * ((max(xs) - min(xs)) + (max(ys) - min(ys))))


def compute_hpwl(
    netlist_graph: NetlistGraph,
    placement: dict[str, tuple[float, float, int]] | None = None,
) -> float:
    """Calculate total weighted Half-Perimeter Wirelength (HPWL) in mm."""
    return float(
        sum(
            compute_net_hpwl(netlist_graph, nid, placement)
            for nid in netlist_graph.routable_net_ids
        )
    )


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
    """Order nets for sequential routing: switching/short signals first, high-fanout rails balanced."""
    scored: list[tuple[tuple[int, int, float, int], int]] = []
    for net_id in netlist_graph.routable_net_ids:
        net = netlist_graph.nets[net_id]
        pairs = compute_net_mst_pairs(netlist_graph, net_id)
        mst_len = sum(math.hypot(b[0] - a[0], b[1] - a[1]) for a, b in pairs)
        fanout = len(net.pins)
        upper_name = net.name.upper()
        # Prioritize high-di/dt switching nodes before power rails so they stay short and via-free on F.Cu
        is_switching = 0 if ("SW_NODE" in upper_name or "SWNODE" in upper_name) else 1
        # Route short, tight 2-3 pin signal nets first so high-fanout power/GND buses can weave around via L2
        is_high_fanout_rail = 1 if (net.is_power_ground and fanout >= 4 and is_switching == 1) else 0
        priority_score = (is_high_fanout_rail, is_switching, mst_len + 1.5 * fanout, net_id)
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


def compute_subset_wire_cost(
    netlist_graph: NetlistGraph,
    net_ids: list[int] | set[int],
    placement: dict[str, tuple[float, float, int]] | None = None,
) -> float:
    """Compute partial objective `0.5 * HPWL + 0.5 * MST` over a subset of nets."""
    total = 0.0
    for nid in net_ids:
        hpwl_n = compute_net_hpwl(netlist_graph, nid, placement)
        mst_n = sum(
            math.hypot(b[0] - a[0], b[1] - a[1])
            for a, b in compute_net_mst_pairs(netlist_graph, nid, placement)
        )
        total += 0.5 * hpwl_n + 0.5 * mst_n
    return float(total)


def compute_trace_dc_resistance_ohm(
    length_mm: float,
    width_mm: float,
    copper_thickness_mm: float = DEFAULT_COPPER_THICKNESS_MM,
) -> float:
    """Compute trace DC resistance R_DC = rho * L / (W * t_Cu) in Ohms (1 oz Cu default)."""
    if length_mm <= 0.0:
        return 0.0
    area_mm2 = max(1e-9, width_mm * copper_thickness_mm)
    return float(COPPER_RESISTIVITY_OHM_MM * (length_mm / area_mm2))


def compute_trace_ampacity_a(
    width_mm: float,
    copper_thickness_mm: float = DEFAULT_COPPER_THICKNESS_MM,
    is_internal: bool = False,
    delta_t_c: float = 10.0,
) -> float:
    """Compute IPC-2221 trace current capacity in Amperes for a given temperature rise."""
    area_mil2 = max(1e-6, (width_mm * copper_thickness_mm) / (0.0254 * 0.0254))
    k_coeff = 0.024 if is_internal else 0.048
    return float(k_coeff * (max(1.0, delta_t_c) ** 0.44) * (area_mil2 ** 0.725))


def compute_via_ampacity_a(
    drill_mm: float,
    plating_thickness_mm: float = DEFAULT_VIA_PLATING_MM,
    delta_t_c: float = 10.0,
) -> float:
    """Compute IPC-2221 cylindrical via barrel current capacity in Amperes."""
    barrel_area_mm2 = math.pi * max(0.05, drill_mm) * max(0.005, plating_thickness_mm)
    area_mil2 = barrel_area_mm2 / (0.0254 * 0.0254)
    return float(0.048 * (max(1.0, delta_t_c) ** 0.44) * (area_mil2 ** 0.725))


def compute_via_resistance_ohm(
    drill_mm: float,
    span_mm: float = 1.6,
    plating_thickness_mm: float = DEFAULT_VIA_PLATING_MM,
) -> float:
    """Compute cylindrical copper barrel DC resistance of a via in Ohms."""
    barrel_area_mm2 = math.pi * max(0.05, drill_mm) * max(0.005, plating_thickness_mm)
    return float(COPPER_RESISTIVITY_OHM_MM * (max(0.05, span_mm) / barrel_area_mm2))


def compute_microstrip_inductance_nh_per_mm(
    width_mm: float,
    height_mm: float,
    copper_thickness_mm: float = DEFAULT_COPPER_THICKNESS_MM,
) -> float:
    """Estimate microstrip inductance per unit length (nH/mm) over nearest reference plane."""
    ratio = (5.98 * max(0.05, height_mm)) / max(0.05, 0.8 * width_mm + copper_thickness_mm)
    return float(0.2 * math.log(max(1.15, ratio)))


def compute_microstrip_z0_ohm(
    width_mm: float,
    height_mm: float,
    copper_thickness_mm: float = DEFAULT_COPPER_THICKNESS_MM,
    er: float = FR4_ER,
) -> float:
    """Estimate single-ended microstrip characteristic impedance Z0 in Ohms."""
    ratio = (5.98 * max(0.05, height_mm)) / max(0.05, 0.8 * width_mm + copper_thickness_mm)
    return float((87.0 / math.sqrt(er + 1.41)) * math.log(max(1.15, ratio)))


def compute_via_inductance_nh(drill_mm: float, span_mm: float) -> float:
    """Goldfarb/Pucel parasitic inductance estimate (nH) for a via barrel."""
    h = max(0.05, span_mm)
    d = max(0.05, drill_mm)
    return float(0.2 * h * (math.log(max(1.1, (4.0 * h) / d)) + 1.0))


def _estimate_net_design_current_a(net_name: str, is_power_ground: bool) -> float:
    upper = net_name.upper()
    if any(tok in upper for tok in ("SW_", "SWNODE", "SW_NODE", "VBAT", "VIN_12V", "VOUT_5V", "VCC_5V", "VBUS")):
        return 1.5
    if is_power_ground:
        return 0.8
    return 0.1


def compute_signal_integrity_telemetry(board_state: BoardState) -> dict[str, Any]:
    """Compute electrical signal-integrity and thermal telemetry across all routed nets.

    Evaluates:
    - Trace DC resistance R_DC = rho * L / (W * t_Cu) for 1 oz (0.035mm) copper
    - Switching-node (SW_NODE / BOOT) & high-speed clock loop inductance L_loop (nH)
    - IPC-2221 trace & via barrel current capacity checks
    - Microstrip characteristic impedance Z0 over nearest reference plane
    """
    cfg = board_state.config
    t_cu = getattr(cfg, "copper_thickness_mm", DEFAULT_COPPER_THICKNESS_MM)
    board_h = getattr(cfg, "board_thickness_mm", 1.6)
    prepreg_h = getattr(cfg, "prepreg_thickness_mm", 0.20)
    num_layers = getattr(board_state, "num_layers", 2)
    ref_h = prepreg_h if num_layers >= 4 else board_h

    plane_nets: set[int] = set()
    if hasattr(board_state, "plane_assignments"):
        plane_nets = {nid for nid, _ in board_state.plane_assignments.values()}

    per_net: list[dict[str, Any]] = []
    switching_nodes: list[dict[str, Any]] = []
    all_r_mohm: list[float] = []
    all_l_nh: list[float] = []
    via_capacities_seen: list[float] = []
    trace_capacities_seen: list[float] = []
    via_all_pass = True
    trace_all_pass = True

    for nid in board_state.netlist.routable_net_ids:
        net = board_state.netlist.nets[nid]
        segs = board_state.routed_traces.get(nid, [])
        vias = board_state.routed_vias.get(nid, [])
        is_plane_net = nid in plane_nets

        seg_len_mm = 0.0
        r_dc_ohm = 0.0
        l_trace_nh = 0.0
        min_width_mm = cfg.power_trace_width_mm if net.is_power_ground else getattr(
            cfg, "signal_trace_width_mm", cfg.trace_width_mm
        )
        widths: list[float] = []

        for s in segs:
            slen = math.hypot(s.x2_mm - s.x1_mm, s.y2_mm - s.y1_mm)
            if slen <= 0.0:
                continue
            w_mm = s.width_mm if s.width_mm > 0.0 else min_width_mm
            widths.append(w_mm)
            seg_len_mm += slen
            r_dc_ohm += compute_trace_dc_resistance_ohm(slen, w_mm, t_cu)
            l_trace_nh += slen * compute_microstrip_inductance_nh_per_mm(w_mm, ref_h, t_cu)

        eff_w = min(widths) if widths else min_width_mm
        mean_w = (sum(widths) / len(widths)) if widths else min_width_mm
        trace_ampacity = compute_trace_ampacity_a(eff_w, t_cu, is_internal=False, delta_t_c=20.0)

        l_vias_nh = 0.0
        r_vias_ohm = 0.0
        single_via_amps: list[float] = []
        for v in vias:
            span_layers = max(1, abs(getattr(v, "end_layer", 1) - getattr(v, "start_layer", 0)))
            if num_layers >= 4:
                span_mm = prepreg_h if span_layers == 1 else (board_h * (span_layers / max(1, num_layers - 1)))
            else:
                span_mm = board_h
            l_vias_nh += compute_via_inductance_nh(v.drill_mm, span_mm)
            r_vias_ohm += compute_via_resistance_ohm(v.drill_mm, span_mm)
            single_via_amps.append(compute_via_ampacity_a(v.drill_mm, delta_t_c=20.0))

        # Total copper resistance (for plane nets, micro-vias are parallel return drops)
        if vias:
            if is_plane_net:
                r_total_ohm = r_dc_ohm + (r_vias_ohm / max(1, len(vias) ** 2))
                loop_inductance_nh = l_trace_nh + (l_vias_nh / max(1, len(vias)))
                via_ampacity = sum(single_via_amps)
            else:
                r_total_ohm = r_dc_ohm + r_vias_ohm
                loop_inductance_nh = l_trace_nh + l_vias_nh
                via_ampacity = min(single_via_amps)
        else:
            r_total_ohm = r_dc_ohm
            loop_inductance_nh = l_trace_nh
            via_ampacity = compute_via_ampacity_a(cfg.via_drill_mm, delta_t_c=20.0)

        design_current = _estimate_net_design_current_a(net.name, net.is_power_ground)
        via_ok = (len(vias) == 0) or (via_ampacity >= design_current)
        trace_ok = (seg_len_mm == 0.0) or (trace_ampacity >= design_current)
        if not via_ok:
            via_all_pass = False
        if not trace_ok:
            trace_all_pass = False

        r_mohm = r_total_ohm * 1000.0
        z0_ohm = compute_microstrip_z0_ohm(mean_w, ref_h, t_cu)

        if seg_len_mm > 0.0 or vias:
            all_r_mohm.append(r_mohm)
            all_l_nh.append(loop_inductance_nh)
            trace_capacities_seen.append(trace_ampacity)
            if vias:
                via_capacities_seen.append(via_ampacity)

        uname = net.name.upper()
        is_switching = any(tok in uname for tok in ("SW_NODE", "SWNODE", "BOOT"))
        is_clock_hs = any(
            tok in uname for tok in ("XIN", "XOUT", "XTAL", "HSE", "USB_D", "QSPI_CLK", "SWCLK")
        )

        net_entry = {
            "net_id": nid,
            "net_name": net.name,
            "is_power_ground": net.is_power_ground,
            "is_plane_net": is_plane_net,
            "is_switching": is_switching,
            "is_high_speed": is_clock_hs,
            "length_mm": round(seg_len_mm, 3),
            "width_mm": round(mean_w, 3),
            "via_count": len(vias),
            "dc_resistance_mohm": round(r_mohm, 3),
            "loop_inductance_nh": round(loop_inductance_nh, 3),
            "z0_ohm": round(z0_ohm, 1),
            "trace_ampacity_a": round(trace_ampacity, 3),
            "via_ampacity_a": round(via_ampacity, 3),
            "design_current_a": round(design_current, 2),
            "via_current_ok": via_ok,
            "trace_current_ok": trace_ok,
        }
        per_net.append(net_entry)
        if is_switching:
            switching_nodes.append(net_entry)

    sw_inductance: float | None = None
    if switching_nodes:
        sw_inductance = round(
            max(node["loop_inductance_nh"] for node in switching_nodes), 3
        )

    # Compute differential pair length skew across coupled pairs (e.g. USB_DP / USB_DM, PCIE_TX0_P / PCIE_TX0_N)
    by_name = {n["net_name"].upper(): n["length_mm"] for n in per_net if n["length_mm"] > 0}
    diff_pairs: list[dict[str, Any]] = []
    seen_pairs: set[tuple[str, str]] = set()
    for pos_suf, neg_suf in (("_DP", "_DM"), ("D+", "D-"), ("CANH", "CANL"), ("TX+", "TX-")):
        for pname, plen in by_name.items():
            if pos_suf in pname:
                nname = pname.replace(pos_suf, neg_suf, 1)
                if nname in by_name and (pname, nname) not in seen_pairs:
                    seen_pairs.add((pname, nname))
                    diff_pairs.append(
                        {
                            "pos_net": pname,
                            "neg_net": nname,
                            "pos_len_mm": round(plen, 3),
                            "neg_len_mm": round(by_name[nname], 3),
                            "skew_mm": round(abs(plen - by_name[nname]), 3),
                        }
                    )
    for pname, plen in by_name.items():
        if pname.endswith("_P"):
            nname = pname[:-2] + "_N"
            if nname in by_name and (pname, nname) not in seen_pairs:
                seen_pairs.add((pname, nname))
                diff_pairs.append(
                    {
                        "pos_net": pname,
                        "neg_net": nname,
                        "pos_len_mm": round(plen, 3),
                        "neg_len_mm": round(by_name[nname], 3),
                        "skew_mm": round(abs(plen - by_name[nname]), 3),
                    }
                )

    active_rules = sorted(
        {
            rule
            for comp in board_state.netlist.components.values()
            for rule in getattr(comp, "special_layout_rules", [])
        }
    )

    return {
        "copper_thickness_mm": t_cu,
        "copper_weight_oz": round(t_cu / 0.035, 2),
        "reference_plane_height_mm": round(ref_h, 3),
        "max_dc_resistance_mohm": round(max(all_r_mohm), 3) if all_r_mohm else 0.0,
        "mean_dc_resistance_mohm": round(sum(all_r_mohm) / len(all_r_mohm), 3)
        if all_r_mohm
        else 0.0,
        "sw_node_inductance_nh": sw_inductance,
        "max_loop_inductance_nh": round(max(all_l_nh), 3) if all_l_nh else 0.0,
        "min_via_ampacity_a": round(min(via_capacities_seen), 3)
        if via_capacities_seen
        else round(compute_via_ampacity_a(cfg.via_drill_mm, delta_t_c=20.0), 3),
        "min_trace_ampacity_a": round(min(trace_capacities_seen), 3)
        if trace_capacities_seen
        else round(compute_trace_ampacity_a(cfg.trace_width_mm, t_cu, delta_t_c=20.0), 3),
        "via_capacity_all_pass": via_all_pass,
        "trace_capacity_all_pass": trace_all_pass,
        "si_thermal_status": "PASS" if (via_all_pass and trace_all_pass) else "WARN",
        "switching_nodes": switching_nodes,
        "diff_pairs": diff_pairs,
        "active_special_rules": active_rules,
        "per_net": per_net,
    }


def _segment_distance_to_grid_box(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    xs_mm: np.ndarray,
    ys_mm: np.ndarray,
    radius_mm: float,
    res: float,
    W: int,
    H: int,
) -> tuple[int, int, int, int, np.ndarray] | None:
    """Compute Euclidean distance from segment to grid coordinates inside its dilated bbox."""
    gx0 = max(0, int(math.floor((min(x1, x2) - radius_mm) / res)))
    gx1 = min(W - 1, int(math.ceil((max(x1, x2) + radius_mm) / res)))
    gy0 = max(0, int(math.floor((min(y1, y2) - radius_mm) / res)))
    gy1 = min(H - 1, int(math.ceil((max(y1, y2) + radius_mm) / res)))
    if gx0 > gx1 or gy0 > gy1:
        return None
    xx = xs_mm[gx0 : gx1 + 1][None, :]
    yy = ys_mm[gy0 : gy1 + 1][:, None]
    vx = x2 - x1
    vy = y2 - y1
    lsq = vx * vx + vy * vy
    if lsq <= 1e-12:
        dist = np.hypot(xx - x1, yy - y1)
    else:
        t = np.clip(((xx - x1) * vx + (yy - y1) * vy) / lsq, 0.0, 1.0)
        dist = np.hypot(xx - (x1 + t * vx), yy - (y1 + t * vy))
    return (gy0, gy1 + 1, gx0, gx1 + 1, dist)


def compute_ground_return_isolation(board_state: BoardState) -> dict[str, Any]:
    """Verify continuous In1.GND return physics (J_return) and noisy-vs-quiet domain isolation (rev13)."""
    cfg = board_state.config
    netlist = board_state.netlist
    H, W = board_state.H, board_state.W
    res = board_state.res
    halo_mm = float(getattr(cfg, "analog_guard_halo_mm", 3.0))
    prepreg_h = max(0.05, float(getattr(cfg, "prepreg_thickness_mm", 0.20)))
    num_layers = getattr(board_state, "num_layers", 2)

    pwr_comps = [
        c for c in netlist.components.values() if getattr(c, "noise_domain", "neutral") == "power_switching"
    ]
    ana_comps = [
        c for c in netlist.components.values() if getattr(c, "noise_domain", "neutral") == "analog_precision"
    ]
    dig_comps = [
        c for c in netlist.components.values() if getattr(c, "noise_domain", "neutral") == "digital_hs"
    ]

    pwr_nets = [
        nid for nid, n in netlist.nets.items()
        if nid > 0 and getattr(n, "noise_domain", "neutral") == "power_switching"
    ]
    ana_nets = [
        nid for nid, n in netlist.nets.items()
        if nid > 0 and getattr(n, "noise_domain", "neutral") == "analog_precision"
    ]

    has_mixed = bool(pwr_comps or pwr_nets) and bool(ana_comps or ana_nets)

    # Compute physical center-to-center separation between primary power_switching and analog_precision devices
    pwr_primary = [c for c in pwr_comps if c.comp_type in ("ic", "inductor")] or pwr_comps
    ana_primary = [c for c in ana_comps if c.comp_type == "ic"] or ana_comps
    min_sep_mm: float | None = None
    if pwr_primary and ana_primary:
        for pc in pwr_primary:
            for ac in ana_primary:
                d = math.hypot(pc.x - ac.x, pc.y - ac.y)
                if min_sep_mm is None or d < min_sep_mm:
                    min_sep_mm = d

    # Continuous GND check
    gnd_constraints = getattr(netlist, "ground_constraints", {}) or {}
    has_split_constraint = str(gnd_constraints.get("mode", "")).lower() == "remote_star_ground"
    if num_layers >= 4:
        p_assign = getattr(board_state, "plane_assignments", {}) or {}
        has_gnd_plane = any("GND" in pname.upper() for _, pname in p_assign.values())
        continuous_gnd = has_gnd_plane and not has_split_constraint
    else:
        continuous_gnd = not has_split_constraint

    if not has_mixed:
        return {
            "has_mixed_signal_domains": False,
            "power_switching_components": [c.comp_id for c in pwr_comps],
            "analog_precision_components": [c.comp_id for c in ana_comps],
            "digital_hs_components": [c.comp_id for c in dig_comps],
            "gan_to_adc_separation_mm": round(min_sep_mm, 3) if min_sep_mm is not None else None,
            "j_return_overlap_pct": 0.0,
            "j_return_max_in_analog_norm": 0.0,
            "switching_under_analog_count": 0,
            "analog_guard_halo_mm": halo_mm,
            "continuous_gnd": continuous_gnd,
            "isolation_passed": continuous_gnd,
        }

    xs_mm = np.arange(W, dtype=np.float32) * res
    ys_mm = np.arange(H, dtype=np.float32) * res

    # 1. Build analog_precision exclusion corridor mask (component courtyards + analog traces + halo)
    analog_mask = np.zeros((H, W), dtype=bool)
    for comp in ana_comps:
        ew, eh = comp.effective_size()
        ax0, ax1 = comp.x - ew * 0.5, comp.x + ew * 0.5
        ay0, ay1 = comp.y - eh * 0.5, comp.y + eh * 0.5
        gx0 = max(0, int(math.floor((ax0 - halo_mm) / res)))
        gx1 = min(W - 1, int(math.ceil((ax1 + halo_mm) / res)))
        gy0 = max(0, int(math.floor((ay0 - halo_mm) / res)))
        gy1 = min(H - 1, int(math.ceil((ay1 + halo_mm) / res)))
        if gx0 <= gx1 and gy0 <= gy1:
            xx = xs_mm[gx0 : gx1 + 1][None, :]
            yy = ys_mm[gy0 : gy1 + 1][:, None]
            dx = np.maximum(0.0, np.maximum(ax0 - xx, xx - ax1))
            dy = np.maximum(0.0, np.maximum(ay0 - yy, yy - ay1))
            analog_mask[gy0 : gy1 + 1, gx0 : gx1 + 1] |= np.hypot(dx, dy) <= halo_mm

    for nid in ana_nets:
        for seg in board_state.routed_traces.get(nid, []):
            box = _segment_distance_to_grid_box(
                seg.x1_mm, seg.y1_mm, seg.x2_mm, seg.y2_mm,
                xs_mm, ys_mm, halo_mm + seg.width_mm * 0.5, res, W, H,
            )
            if box is not None:
                gy0, gy1, gx0, gx1, dist = box
                analog_mask[gy0:gy1, gx0:gx1] |= (dist - seg.width_mm * 0.5) <= halo_mm

    # 2. Build J_return(x, y) switching ground-return density envelope
    j_return = np.zeros((H, W), dtype=np.float32)
    ret_radius = max(1.0, 5.0 * prepreg_h)
    switching_segs: list[tuple[float, float, float, float, float]] = []
    for nid in pwr_nets:
        for seg in board_state.routed_traces.get(nid, []):
            switching_segs.append((seg.x1_mm, seg.y1_mm, seg.x2_mm, seg.y2_mm, seg.width_mm))

    # Include switching loop MST segments when routed traces haven't been committed yet
    if not switching_segs and pwr_nets:
        for nid in pwr_nets:
            for (x1, y1, _), (x2, y2, _) in compute_net_mst_pairs(netlist, nid):
                switching_segs.append((x1, y1, x2, y2, 0.25))

    for x1, y1, x2, y2, w_mm in switching_segs:
        box = _segment_distance_to_grid_box(
            x1, y1, x2, y2, xs_mm, ys_mm, ret_radius + w_mm * 0.5, res, W, H
        )
        if box is not None:
            gy0, gy1, gx0, gx1, dist = box
            d_eff = np.maximum(0.0, dist - w_mm * 0.5)
            # Lorentzian image-current density on In1.GND plane at height prepreg_h
            density = np.where(
                d_eff <= ret_radius,
                1.0 / (1.0 + (d_eff / prepreg_h) ** 2),
                0.0,
            ).astype(np.float32)
            j_return[gy0:gy1, gx0:gx1] = np.maximum(j_return[gy0:gy1, gx0:gx1], density)

    total_j = float(j_return.sum())
    overlap_j = float(j_return[analog_mask].sum()) if total_j > 0.0 else 0.0
    overlap_pct = (100.0 * overlap_j / total_j) if total_j > 1e-9 else 0.0
    max_in_analog = float(j_return[analog_mask].max()) if analog_mask.any() else 0.0

    # 3. Direct proximity count of switching copper vs analog corridor
    switching_under_analog = 0
    for nid in pwr_nets:
        for seg in board_state.routed_traces.get(nid, []):
            box = _segment_distance_to_grid_box(
                seg.x1_mm, seg.y1_mm, seg.x2_mm, seg.y2_mm, xs_mm, ys_mm, 0.2, res, W, H
            )
            if box is not None:
                gy0, gy1, gx0, gx1, dist = box
                on_seg = dist <= max(0.15, seg.width_mm * 0.5)
                if (on_seg & analog_mask[gy0:gy1, gx0:gx1]).any():
                    switching_under_analog += 1

    passed = continuous_gnd and overlap_pct <= 0.1 and switching_under_analog == 0
    return {
        "has_mixed_signal_domains": True,
        "power_switching_components": [c.comp_id for c in pwr_comps],
        "analog_precision_components": [c.comp_id for c in ana_comps],
        "digital_hs_components": [c.comp_id for c in dig_comps],
        "gan_to_adc_separation_mm": round(min_sep_mm, 3) if min_sep_mm is not None else None,
        "j_return_overlap_pct": round(overlap_pct, 3),
        "j_return_max_in_analog_norm": round(max_in_analog, 4),
        "switching_under_analog_count": switching_under_analog,
        "analog_guard_halo_mm": halo_mm,
        "continuous_gnd": continuous_gnd,
        "isolation_passed": passed,
    }


def compute_broadside_crosstalk_risk(board_state: BoardState) -> dict[str, Any]:
    """Detect 3D Z-axis broadside coupling between adjacent unshielded layers (analog <1mV vs aggressor)."""
    cfg = board_state.config
    netlist = board_state.netlist
    num_layers = getattr(board_state, "num_layers", 2)
    h_core = max(0.03, float(getattr(cfg, "thin_core_thickness_mm", 0.10)))
    limit_pf = float(getattr(cfg, "max_broadside_cap_pf", 0.05))

    if num_layers < 2:
        return {
            "broadside_violations": [],
            "violation_count": 0,
            "max_broadside_cap_pf": 0.0,
            "limit_pf": limit_pf,
            "h_core_mm": h_core,
            "broadside_passed": True,
        }

    p_assign = getattr(board_state, "plane_assignments", {}) or {}
    gnd_layers = {
        l_idx for l_idx, (_, pname) in p_assign.items() if "GND" in pname.upper()
    }

    analog_segs: list[tuple[int, str, Any]] = []
    aggressor_segs: list[tuple[int, str, Any]] = []
    for nid, segs in board_state.routed_traces.items():
        net = netlist.nets.get(nid)
        if net is None:
            continue
        nd = getattr(net, "noise_domain", "neutral")
        if nd == "analog_precision":
            for s in segs:
                analog_segs.append((nid, net.name, s))
        elif nd in ("digital_hs", "power_switching"):
            for s in segs:
                aggressor_segs.append((nid, net.name, s))

    violations: list[dict[str, Any]] = []
    max_cap_pf = 0.0
    eps0_pf_per_mm = 8.8541878e-3

    for v_nid, v_name, vs in analog_segs:
        vl = int(vs.layer)
        v_dx = vs.x2_mm - vs.x1_mm
        v_dy = vs.y2_mm - vs.y1_mm
        v_len = math.hypot(v_dx, v_dy)
        if v_len < 0.05:
            continue
        ux, uy = v_dx / v_len, v_dy / v_len

        for a_nid, a_name, ag in aggressor_segs:
            al = int(ag.layer)
            if abs(vl - al) != 1:
                continue
            # Solid GND plane on either layer acts as an electrostatic shield
            if vl in gnd_layers or al in gnd_layers:
                continue

            a_dx = ag.x2_mm - ag.x1_mm
            a_dy = ag.y2_mm - ag.y1_mm
            a_len = math.hypot(a_dx, a_dy)
            if a_len < 0.05:
                continue
            ax, ay = a_dx / a_len, a_dy / a_len
            dot = abs(ux * ax + uy * ay)
            ang_deg = math.degrees(math.acos(max(0.0, min(1.0, dot))))

            # Near-perpendicular crossing (>= 60 deg) has negligible parallel overlap area
            if ang_deg >= 60.0:
                c_cross = eps0_pf_per_mm * FR4_ER * (vs.width_mm * ag.width_mm) / h_core
                max_cap_pf = max(max_cap_pf, c_cross)
                continue

            # Project aggressor endpoints onto victim's local (along, perp) frame
            t1 = (ag.x1_mm - vs.x1_mm) * ux + (ag.y1_mm - vs.y1_mm) * uy
            t2 = (ag.x2_mm - vs.x1_mm) * ux + (ag.y2_mm - vs.y1_mm) * uy
            ov_lo = max(0.0, min(t1, t2))
            ov_hi = min(v_len, max(t1, t2))
            ov_len = ov_hi - ov_lo
            if ov_len < 0.4:
                continue

            # Lateral perpendicular distance at overlap midpoint
            t_mid = 0.5 * (ov_lo + ov_hi)
            vx_m = vs.x1_mm + t_mid * ux
            vy_m = vs.y1_mm + t_mid * uy
            # Closest point on aggressor to (vx_m, vy_m)
            ta = max(0.0, min(1.0, ((vx_m - ag.x1_mm) * a_dx + (vy_m - ag.y1_mm) * a_dy) / (a_len * a_len)))
            ax_m = ag.x1_mm + ta * a_dx
            ay_m = ag.y1_mm + ta * a_dy
            d_perp = math.hypot(vx_m - ax_m, vy_m - ay_m)
            edge_gap = max(0.0, d_perp - 0.5 * (vs.width_mm + ag.width_mm))

            area_eff = ov_len * min(vs.width_mm, ag.width_mm) * math.exp(-edge_gap / (2.0 * h_core))
            c_broadside_pf = eps0_pf_per_mm * FR4_ER * (area_eff / h_core)
            max_cap_pf = max(max_cap_pf, c_broadside_pf)

            if c_broadside_pf > limit_pf:
                violations.append(
                    {
                        "victim_net_id": v_nid,
                        "victim_net_name": v_name,
                        "victim_layer": vl,
                        "aggressor_net_id": a_nid,
                        "aggressor_net_name": a_name,
                        "aggressor_layer": al,
                        "overlap_length_mm": round(ov_len, 3),
                        "lateral_dist_mm": round(d_perp, 3),
                        "angle_deg": round(ang_deg, 1),
                        "c_broadside_pf": round(c_broadside_pf, 4),
                        "x_mm": round(0.5 * (vx_m + ax_m), 3),
                        "y_mm": round(0.5 * (vy_m + ay_m), 3),
                    }
                )

    return {
        "broadside_violations": violations,
        "violation_count": len(violations),
        "max_broadside_cap_pf": round(max_cap_pf, 4),
        "limit_pf": limit_pf,
        "h_core_mm": h_core,
        "broadside_passed": len(violations) == 0,
    }


def compute_hatch_em_factors(hatch_fill: float = 1.0) -> tuple[float, float]:
    """Return `(c_eff_ratio, l_eff_ratio)` for a cross-hatched reference plane vs solid plane (rev14).

    Removing reference copper (`hatch_fill < 1.0`) decreases capacitive coupling `C_eff` to the
    reference mesh and forces return image currents to detour around mesh apertures (`L_eff` up).
    """
    phi = max(0.15, min(1.0, float(hatch_fill)))
    if phi >= 0.999:
        return (1.0, 1.0)
    c_eff_ratio = phi ** 0.85
    l_eff_ratio = 1.0 + 0.64 * ((1.0 - phi) ** 0.85)
    return (float(c_eff_ratio), float(l_eff_ratio))


def compute_diff_pair_z_diff_ohm(
    width_mm: float,
    gap_mm: float,
    er: float = FR4_ER,
    h_mm: float = 0.10,
    hatch_fill: float = 1.0,
    copper_thickness_mm: float | None = None,
) -> float:
    """Compute edge-coupled microstrip differential impedance Z_diff (Ohms) over solid or hatched ref."""
    w = max(0.05, float(width_mm))
    s = max(0.05, float(gap_mm))
    h = max(0.03, float(h_mm))
    if copper_thickness_mm is None:
        t_cu = FLEX_COPPER_THICKNESS_MM if (hatch_fill < 0.99 or er <= 3.6) else DEFAULT_COPPER_THICKNESS_MM
    else:
        t_cu = max(0.009, float(copper_thickness_mm))

    ratio = (5.98 * h) / max(0.04, 0.8 * w + t_cu)
    z0_solid = (87.0 / math.sqrt(max(1.5, float(er)) + 1.41)) * math.log(max(1.15, ratio))
    coupling = 1.0 - 0.48 * math.exp(-0.96 * (s / h))
    c_ratio, l_ratio = compute_hatch_em_factors(hatch_fill)
    hatch_z_factor = math.sqrt(l_ratio / max(1e-6, c_ratio))
    return float(2.0 * z0_solid * coupling * hatch_z_factor)


def solve_diff_pair_geometry(
    z_diff_target_ohm: float = 85.0,
    er: float = FR4_ER,
    h_mm: float = 0.10,
    hatch_fill: float = 1.0,
    copper_thickness_mm: float | None = None,
    min_width_mm: float = 0.10,
    min_gap_mm: float = 0.115,
) -> tuple[float, float]:
    """Solve `(trace_width_mm, intra_pair_gap_mm)` to hit `z_diff_target_ohm` in a substrate zone (rev14)."""
    target = float(z_diff_target_ohm)
    best_w = round(max(min_width_mm, 0.128), 4)
    best_s = round(max(min_gap_mm, 0.120), 4)
    best_err = float("inf")

    # Search compact manufacturable gap window near tight differential coupling first
    gap_candidates = [round(min_gap_mm + 0.005 * k, 4) for k in range(25)]
    w_min = max(0.07, float(min_width_mm))
    w_max = 0.45

    for s in gap_candidates:
        lo, hi = w_min, w_max
        # Z_diff decreases monotonically with increasing trace width w
        for _ in range(36):
            mid = 0.5 * (lo + hi)
            z_mid = compute_diff_pair_z_diff_ohm(
                mid, s, er=er, h_mm=h_mm, hatch_fill=hatch_fill, copper_thickness_mm=copper_thickness_mm
            )
            if z_mid > target:
                lo = mid
            else:
                hi = mid
        w_sol = round(0.5 * (lo + hi), 4)
        z_sol = compute_diff_pair_z_diff_ohm(
            w_sol, s, er=er, h_mm=h_mm, hatch_fill=hatch_fill, copper_thickness_mm=copper_thickness_mm
        )
        # Small preference for tighter coupling near 0.120mm gap when impedance error is tied
        err = abs(z_sol - target) + 0.15 * abs(s - max(min_gap_mm, 0.120))
        if err < best_err:
            best_err = err
            best_w = w_sol
            best_s = s

    return (round(best_w, 4), round(best_s, 4))


def _find_diff_pairs_in_netlist(netlist: NetlistGraph) -> list[tuple[int, int, str]]:
    """Return `(net_id_p, net_id_n, base_name)` for differential pairs in `netlist`."""
    by_name = {n.name.upper(): nid for nid, n in netlist.nets.items() if nid > 0 and len(n.pins) >= 2}
    pairs: list[tuple[int, int, str]] = []
    seen: set[tuple[int, int]] = set()
    for pname, pid in sorted(by_name.items()):
        nname: str | None = None
        base = pname
        for pos_suf, neg_suf in (("_DP", "_DM"), ("D+", "D-"), ("CANH", "CANL"), ("TX+", "TX-"), ("RX+", "RX-")):
            if pos_suf in pname:
                nname = pname.replace(pos_suf, neg_suf, 1)
                base = pname.replace(pos_suf, "", 1)
                break
        if nname is None and pname.endswith("_P"):
            nname = pname[:-2] + "_N"
            base = pname[:-2]
        if nname and nname in by_name:
            nid = by_name[nname]
            key = (min(pid, nid), max(pid, nid))
            if key not in seen:
                seen.add(key)
                pairs.append((pid, nid, base))
    return pairs


def compute_rigid_flex_impedance_telemetry(board_state: BoardState) -> dict[str, Any]:
    """Compute per-zone differential impedance (Rigid FR4 vs Cross-Hatched Flex) and skew (rev14)."""
    cfg = board_state.config
    zones = list(getattr(board_state.netlist, "substrate_zones", []) or [])
    has_flex = any(getattr(z, "zone_type", "rigid") == "dynamic_flex" for z in zones)
    target_ohm = float(getattr(cfg, "pcie_diff_target_ohm", 85.0))
    tol_pct = float(getattr(cfg, "pcie_diff_tolerance_pct", 5.0))
    max_skew = float(getattr(cfg, "pcie_max_skew_mm", 0.125))

    rigid_z = next((z for z in zones if getattr(z, "zone_type", "rigid") == "rigid"), None)
    flex_z = next((z for z in zones if getattr(z, "zone_type", "rigid") == "dynamic_flex"), None)

    rigid_er = float(rigid_z.er) if rigid_z else float(getattr(cfg, "rigid_er", FR4_ER))
    rigid_h = float(rigid_z.prepreg_h_mm) if rigid_z else float(getattr(cfg, "prepreg_thickness_mm", 0.10))
    flex_er = float(flex_z.er) if flex_z else float(getattr(cfg, "flex_er", POLYIMIDE_ER))
    flex_h = float(flex_z.prepreg_h_mm) if flex_z else float(getattr(cfg, "flex_prepreg_thickness_mm", 0.050))
    flex_hatch = float(flex_z.hatch_fill_ratio) if flex_z else float(getattr(cfg, "flex_hatch_fill_ratio", 0.50))

    w_rigid_opt, s_rigid_opt = solve_diff_pair_geometry(
        target_ohm, er=rigid_er, h_mm=rigid_h, hatch_fill=1.0, copper_thickness_mm=DEFAULT_COPPER_THICKNESS_MM
    )
    w_flex_opt, s_flex_opt = solve_diff_pair_geometry(
        target_ohm,
        er=flex_er,
        h_mm=flex_h,
        hatch_fill=flex_hatch,
        copper_thickness_mm=float(getattr(cfg, "flex_copper_thickness_mm", FLEX_COPPER_THICKNESS_MM)),
    )
    z_uncomp_flex = compute_diff_pair_z_diff_ohm(
        w_rigid_opt,
        s_rigid_opt,
        er=flex_er,
        h_mm=flex_h,
        hatch_fill=flex_hatch,
        copper_thickness_mm=float(getattr(cfg, "flex_copper_thickness_mm", FLEX_COPPER_THICKNESS_MM)),
    )

    pair_reports: list[dict[str, Any]] = []
    all_pairs_pass = True

    def _seg_in_flex(seg: Any) -> bool:
        if flex_z is None:
            return False
        mx = 0.5 * (seg.x1_mm + seg.x2_mm)
        my = 0.5 * (seg.y1_mm + seg.y2_mm)
        return (flex_z.x0_mm <= mx <= flex_z.x1_mm) and (flex_z.y0_mm <= my <= flex_z.y1_mm)

    for pid, nid, base in _find_diff_pairs_in_netlist(board_state.netlist):
        psegs = board_state.routed_traces.get(pid, [])
        nsegs = board_state.routed_traces.get(nid, [])
        if not psegs or not nsegs:
            continue

        len_p = sum(math.hypot(s.x2_mm - s.x1_mm, s.y2_mm - s.y1_mm) for s in psegs)
        len_n = sum(math.hypot(s.x2_mm - s.x1_mm, s.y2_mm - s.y1_mm) for s in nsegs)
        skew_mm = abs(len_p - len_n)

        rigid_ws = [s.width_mm for s in (psegs + nsegs) if not _seg_in_flex(s) and math.hypot(s.x2_mm - s.x1_mm, s.y2_mm - s.y1_mm) > 0.02]
        flex_ws = [s.width_mm for s in (psegs + nsegs) if _seg_in_flex(s) and math.hypot(s.x2_mm - s.x1_mm, s.y2_mm - s.y1_mm) > 0.02]

        # Estimate realized intra-pair gap per zone from parallel opposing segments
        def _measure_zone_gap(in_flex: bool, default_gap: float) -> float:
            p_z = [s for s in psegs if _seg_in_flex(s) == in_flex and math.hypot(s.x2_mm - s.x1_mm, s.y2_mm - s.y1_mm) >= 0.3]
            n_z = [s for s in nsegs if _seg_in_flex(s) == in_flex and math.hypot(s.x2_mm - s.x1_mm, s.y2_mm - s.y1_mm) >= 0.3]
            gaps: list[float] = []
            for sp in p_z:
                mx = 0.5 * (sp.x1_mm + sp.x2_mm)
                my = 0.5 * (sp.y1_mm + sp.y2_mm)
                best_c = None
                for sn in n_z:
                    if sp.layer != sn.layer:
                        continue
                    dx, dy = sn.x2_mm - sn.x1_mm, sn.y2_mm - sn.y1_mm
                    lsq = dx * dx + dy * dy
                    t = 0.0 if lsq <= 1e-12 else max(0.0, min(1.0, ((mx - sn.x1_mm) * dx + (my - sn.y1_mm) * dy) / lsq))
                    d = math.hypot(mx - (sn.x1_mm + t * dx), my - (sn.y1_mm + t * dy))
                    edge_g = d - 0.5 * (sp.width_mm + sn.width_mm)
                    if 0.04 <= edge_g <= 0.45 and (best_c is None or edge_g < best_c):
                        best_c = edge_g
                if best_c is not None:
                    gaps.append(best_c)
            return float(np.median(gaps)) if gaps else default_gap

        w_r_meas = float(np.median(rigid_ws)) if rigid_ws else w_rigid_opt
        s_r_meas = _measure_zone_gap(False, s_rigid_opt)
        z_rigid = compute_diff_pair_z_diff_ohm(
            w_r_meas, s_r_meas, er=rigid_er, h_mm=rigid_h, hatch_fill=1.0, copper_thickness_mm=DEFAULT_COPPER_THICKNESS_MM
        )
        rigid_ok = abs(z_rigid - target_ohm) <= target_ohm * (tol_pct / 100.0)

        z_flex: float | None = None
        w_f_meas: float | None = None
        s_f_meas: float | None = None
        flex_ok = True
        if flex_ws:
            w_f_meas = float(np.median(flex_ws))
            s_f_meas = _measure_zone_gap(True, s_flex_opt)
            z_flex = compute_diff_pair_z_diff_ohm(
                w_f_meas,
                s_f_meas,
                er=flex_er,
                h_mm=flex_h,
                hatch_fill=flex_hatch,
                copper_thickness_mm=float(getattr(cfg, "flex_copper_thickness_mm", FLEX_COPPER_THICKNESS_MM)),
            )
            flex_ok = abs(z_flex - target_ohm) <= target_ohm * (tol_pct / 100.0)

        skew_ok = skew_mm <= max_skew + 1e-4
        pair_pass = rigid_ok and flex_ok and skew_ok
        if not pair_pass:
            all_pairs_pass = False

        pair_reports.append(
            {
                "pair_name": base,
                "pos_net": board_state.netlist.nets[pid].name,
                "neg_net": board_state.netlist.nets[nid].name,
                "pos_len_mm": round(len_p, 3),
                "neg_len_mm": round(len_n, 3),
                "skew_mm": round(skew_mm, 4),
                "skew_passed": skew_ok,
                "rigid_width_mm": round(w_r_meas, 4),
                "rigid_gap_mm": round(s_r_meas, 4),
                "rigid_z_diff_ohm": round(z_rigid, 2),
                "rigid_impedance_passed": rigid_ok,
                "crosses_flex": bool(flex_ws),
                "flex_width_mm": round(w_f_meas, 4) if w_f_meas is not None else None,
                "flex_gap_mm": round(s_f_meas, 4) if s_f_meas is not None else None,
                "flex_z_diff_ohm": round(z_flex, 2) if z_flex is not None else None,
                "flex_impedance_passed": flex_ok,
                "passed": pair_pass,
            }
        )

    return {
        "has_rigid_flex": has_flex,
        "target_z_diff_ohm": target_ohm,
        "tolerance_pct": tol_pct,
        "max_skew_mm": max_skew,
        "rigid_target_geometry": {"width_mm": w_rigid_opt, "gap_mm": s_rigid_opt, "er": rigid_er, "h_mm": rigid_h},
        "flex_target_geometry": {
            "width_mm": w_flex_opt,
            "gap_mm": s_flex_opt,
            "er": flex_er,
            "h_mm": flex_h,
            "hatch_fill_ratio": flex_hatch,
            "uncompensated_z_diff_ohm": round(z_uncomp_flex, 2),
        },
        "pairs": pair_reports,
        "all_pairs_passed": all_pairs_pass,
    }


def _cap_esl_nh(comp: Any) -> float:
    meta = getattr(comp, "electrical_meta", {}) or {}
    if "esl_nh" in meta:
        return float(meta["esl_nh"])
    fp = (getattr(comp, "footprint", "") or "").upper()
    if "0201" in fp:
        return 0.35
    if "0402" in fp:
        return 0.45
    if "0603" in fp:
        return 0.60
    if "0805" in fp:
        return 0.85
    return 0.50


def compute_pdn_decoupling_inductance(board_state: BoardState) -> dict[str, Any]:
    """Compute physical decoupling mounting loop inductance L_loop and transient droop V_droop (rev14)."""
    cfg = board_state.config
    netlist = board_state.netlist
    num_layers = getattr(board_state, "num_layers", 2)
    t_cu = float(getattr(cfg, "copper_thickness_mm", DEFAULT_COPPER_THICKNESS_MM))
    board_h = float(getattr(cfg, "board_thickness_mm", 1.6))
    prepreg_h = float(getattr(cfg, "prepreg_thickness_mm", 0.10))
    ref_h = prepreg_h if num_layers >= 4 else board_h
    max_loop_nh = float(getattr(cfg, "pdn_max_loop_inductance_nh", 1.2))
    default_max_droop = float(getattr(cfg, "pdn_default_max_droop_mv", 65.0))

    p_assign = getattr(board_state, "plane_assignments", {}) or {}
    plane_nets = {nid for nid, _ in p_assign.values()}

    # Identify ICs with explicit PDN edge-rate / transient droop verification metadata
    pdn_ics = [
        c
        for c in netlist.components.values()
        if c.comp_type == "ic"
        and (
            "edge_rate_ns" in (c.electrical_meta or {})
            or "di_dt_a_per_ns" in (c.electrical_meta or {})
            or bool((c.electrical_meta or {}).get("pdn_verify", False))
        )
    ]
    if not pdn_ics:
        return {
            "pdn_verified": False,
            "max_loop_inductance_limit_nh": max_loop_nh,
            "cap_reports": [],
            "violations": [],
            "violation_count": 0,
            "max_loop_inductance_nh": 0.0,
            "pdn_passed": True,
        }

    # Build available segments per net (routed_traces or preseeded raw_segments)
    segs_by_net: dict[int, list[tuple[float, float, float, float, float, int]]] = {}
    for nid, segs in board_state.routed_traces.items():
        if segs:
            segs_by_net[nid] = [(s.x1_mm, s.y1_mm, s.x2_mm, s.y2_mm, s.width_mm, int(s.layer)) for s in segs]
    if not segs_by_net and getattr(netlist, "raw_segments", None):
        for sd in netlist.raw_segments:
            snid = int(sd.get("net_id", 0))
            if snid > 0:
                segs_by_net.setdefault(snid, []).append(
                    (
                        float(sd["x1_mm"]),
                        float(sd["y1_mm"]),
                        float(sd["x2_mm"]),
                        float(sd["y2_mm"]),
                        float(sd.get("width_mm", cfg.signal_trace_width_mm)),
                        int(sd.get("layer", 0)),
                    )
                )

    gnd_nids = {
        nid
        for nid, n in netlist.nets.items()
        if nid > 0 and "GND" in n.name.upper()
    }

    cap_reports: list[dict[str, Any]] = []
    violations: list[dict[str, Any]] = []

    def _trace_inductance_from_pad(
        pad_x: float,
        pad_y: float,
        net_id: int,
        target_pts: list[tuple[float, float]],
        cap_vias: list[Any],
        is_under_bga_vip: bool,
    ) -> tuple[float, float]:
        """Measure actual connected trace length & inductance from cap pad toward IC pad or plane via."""
        # If cap has a via-in-pad directly on/next to the pad AND net is on an internal plane, loop is local via-in-pad
        has_local_vip = any(math.hypot(v.x_mm - pad_x, v.y_mm - pad_y) <= 0.45 for v in cap_vias)
        net_segs = segs_by_net.get(net_id, [])

        # Walk connected segments starting from (pad_x, pad_y)
        visited: set[int] = set()
        frontier = [(pad_x, pad_y)]
        total_len = 0.0
        total_l_nh = 0.0
        reached_ic = False
        while frontier:
            cx, cy = frontier.pop(0)
            for idx, (x1, y1, x2, y2, w_mm, lyr) in enumerate(net_segs):
                if idx in visited:
                    continue
                d1 = math.hypot(x1 - cx, y1 - cy)
                d2 = math.hypot(x2 - cx, y2 - cy)
                if min(d1, d2) <= 0.35:
                    visited.add(idx)
                    slen = math.hypot(x2 - x1, y2 - y1)
                    if slen > 0.01:
                        total_len += slen
                        l_per_mm = compute_microstrip_inductance_nh_per_mm(w_mm, ref_h, t_cu)
                        total_l_nh += slen * l_per_mm
                    nx, ny = (x2, y2) if d1 <= d2 else (x1, y1)
                    if any(math.hypot(nx - tx, ny - ty) <= 0.45 for tx, ty in target_pts):
                        reached_ic = True
                    else:
                        frontier.append((nx, ny))
                # Stop after reaching IC or collecting clear remote trace run
                if reached_ic:
                    break
            if reached_ic:
                break

        if (is_under_bga_vip or has_local_vip) and total_len <= 1.2:
            # Short via-in-pad micro-stub (~0.2mm) into internal plane
            eff_len = max(0.15, total_len) if total_len > 0 else 0.15
            return (eff_len, eff_len * compute_microstrip_inductance_nh_per_mm(0.25, ref_h, t_cu))

        if total_len > 0.2:
            return (total_len, total_l_nh)

        # Unrouted state estimate: if under-BGA backside with internal planes, via-in-pad path is ~0.2mm; else surface distance
        if is_under_bga_vip and num_layers >= 4 and (net_id in plane_nets or net_id in gnd_nids):
            return (0.20, 0.20 * compute_microstrip_inductance_nh_per_mm(0.25, ref_h, t_cu))
        if target_pts:
            min_d = min(math.hypot(pad_x - tx, pad_y - ty) for tx, ty in target_pts)
            return (min_d, min_d * compute_microstrip_inductance_nh_per_mm(cfg.power_trace_width_mm, ref_h, t_cu))
        return (0.25, 0.08)

    for ic in pdn_ics:
        ic_pwr_nets = {p.net_id for p in ic.pads if p.net_id > 0 and p.net_id not in gnd_nids and netlist.nets[p.net_id].is_power_ground}
        if not ic_pwr_nets:
            ic_pwr_nets = {
                p.net_id
                for p in ic.pads
                if p.net_id > 0 and p.net_id not in gnd_nids and any(k in netlist.nets[p.net_id].name.upper() for k in ("VCC", "VDD", "PWR", "3V3", "1V8", "1V2"))
            }
        ic_pwr_pts_by_net: dict[int, list[tuple[float, float]]] = {}
        for p in ic.pads:
            if p.net_id in ic_pwr_nets:
                ic_pwr_pts_by_net.setdefault(p.net_id, []).append(ic.pad_world_pos(p))
        ic_gnd_pts = [ic.pad_world_pos(p) for p in ic.pads if p.net_id in gnd_nids]

        ic_ew, ic_eh = ic.effective_size()
        meta = ic.electrical_meta or {}
        edge_rate_ns = float(meta.get("edge_rate_ns", 0.5))
        max_droop_mv = float(meta.get("max_droop_mv", default_max_droop))
        di_dt_total = float(meta.get("di_dt_a_per_ns", 0.0))

        # Find decoupling caps paired with this IC's power & ground nets (exclude switching-stage top_only bulk caps)
        bypass_caps = [
            c
            for c in netlist.components.values()
            if c.comp_type == "cap"
            and getattr(c, "noise_domain", "neutral") != "power_switching"
            and "top_only" not in getattr(c, "special_layout_rules", [])
            and any(p.net_id in ic_pwr_nets for p in c.pads)
            and any(p.net_id in gnd_nids for p in c.pads)
        ]
        if not bypass_caps:
            continue
        if di_dt_total <= 0.0:
            delta_i_per_cap = float(meta.get("delta_i_per_cap_a", 0.025))
            di_dt_per_cap = delta_i_per_cap / max(0.1, edge_rate_ns)
        else:
            di_dt_per_cap = di_dt_total / max(1, len(bypass_caps))

        for cap in bypass_caps:
            pwr_pad = next(p for p in cap.pads if p.net_id in ic_pwr_nets)
            gnd_pad = next(p for p in cap.pads if p.net_id in gnd_nids)
            px, py = cap.pad_world_pos(pwr_pad)
            gx, gy = cap.pad_world_pos(gnd_pad)

            inside_bga_xy = (
                abs(cap.x - ic.x) <= ic_ew * 0.5 + 0.4
                and abs(cap.y - ic.y) <= ic_eh * 0.5 + 0.4
            )
            is_under_bga_vip = bool(
                (getattr(cap, "side", "top") != getattr(ic, "side", "top"))
                and inside_bga_xy
            )

            pwr_vias = board_state.routed_vias.get(pwr_pad.net_id, [])
            gnd_vias = board_state.routed_vias.get(gnd_pad.net_id, [])

            pwr_len, l_pwr_trace = _trace_inductance_from_pad(
                px, py, pwr_pad.net_id, ic_pwr_pts_by_net.get(pwr_pad.net_id, []), pwr_vias, is_under_bga_vip
            )
            gnd_len, l_gnd_trace = _trace_inductance_from_pad(
                gx, gy, gnd_pad.net_id, ic_gnd_pts, gnd_vias, is_under_bga_vip
            )
            l_trace_total = l_pwr_trace + l_gnd_trace

            # Via pair inductance
            if num_layers >= 4 and (is_under_bga_vip or (pwr_pad.net_id in plane_nets and gnd_pad.net_id in plane_nets)):
                # Local microvia + buried plane pair with mutual flux cancellation (0.65x)
                v_drill = float(getattr(cfg, "microvia_drill_mm", 0.15))
                l_v_gnd = compute_via_inductance_nh(v_drill, prepreg_h)
                l_v_pwr = compute_via_inductance_nh(v_drill, min(board_h * 0.45, prepreg_h * 4.5))
                pad_pitch = math.hypot(px - gx, py - gy)
                k_mut = 0.35 if pad_pitch <= 1.5 else 0.15
                l_via_pair = (l_v_gnd + l_v_pwr) * (1.0 - k_mut)
            else:
                v_drill = float(getattr(cfg, "via_drill_mm", 0.25))
                # Same-side remote cap still incurs via inductance when connecting to BGA / planes
                l_via_pair = 0.25 if (pwr_len > 1.5 and gnd_len > 1.5) else 2.0 * compute_via_inductance_nh(v_drill, board_h * 0.5)

            l_esl = _cap_esl_nh(cap)
            l_loop_nh = l_trace_total + l_via_pair + l_esl
            v_droop_mv = l_loop_nh * di_dt_per_cap * 1000.0
            cap_pass = (l_loop_nh <= max_loop_nh + 1e-4) and (v_droop_mv <= max_droop_mv + 1e-3)

            entry = {
                "ic_id": ic.comp_id,
                "cap_id": cap.comp_id,
                "cap_side": getattr(cap, "side", "top"),
                "under_bga": is_under_bga_vip,
                "pwr_net": netlist.nets[pwr_pad.net_id].name,
                "trace_path_mm": round(pwr_len + gnd_len, 3),
                "l_trace_nh": round(l_trace_total, 4),
                "l_via_pair_nh": round(l_via_pair, 4),
                "l_cap_esl_nh": round(l_esl, 4),
                "l_loop_nh": round(l_loop_nh, 4),
                "v_droop_mv": round(v_droop_mv, 2),
                "max_droop_mv": max_droop_mv,
                "passed": cap_pass,
            }
            cap_reports.append(entry)
            if not cap_pass:
                violations.append(
                    {
                        **entry,
                        "error_type": "pdn_excessive_loop_inductance",
                        "x_mm": round(cap.x, 3),
                        "y_mm": round(cap.y, 3),
                        "net_id": pwr_pad.net_id,
                    }
                )

    max_l = max((r["l_loop_nh"] for r in cap_reports), default=0.0)
    return {
        "pdn_verified": True,
        "max_loop_inductance_limit_nh": max_loop_nh,
        "cap_reports": cap_reports,
        "violations": violations,
        "violation_count": len(violations),
        "max_loop_inductance_nh": round(max_l, 4),
        "pdn_passed": len(violations) == 0,
    }
