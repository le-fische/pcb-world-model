"""Exporter translating BoardState back into valid KiCad 8 `.kicad_pcb` S-expressions."""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from env.board_state import BoardState


def _uid(seed_str: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, seed_str))


def _find_gnd_net(board_state: BoardState) -> tuple[int, str] | None:
    """Locate the primary GND net ID and name for copper zone pour generation."""
    for nid in sorted(board_state.netlist.nets.keys()):
        net = board_state.netlist.nets[nid]
        if net.name.strip().upper() == "GND":
            return (nid, net.name.replace('"', ""))
    for nid in sorted(board_state.netlist.nets.keys()):
        net = board_state.netlist.nets[nid]
        if "GND" in net.name.upper():
            return (nid, net.name.replace('"', ""))
    return None


def _kicad_layer_name(layer_idx: int, num_layers: int) -> str:
    """Map discrete internal layer index to canonical KiCad 8 copper layer name."""
    if num_layers >= 8:
        return {
            0: "F.Cu",
            1: "In1.Cu",
            2: "In2.Cu",
            3: "In3.Cu",
            4: "In4.Cu",
            5: "In5.Cu",
            6: "In6.Cu",
            7: "B.Cu",
        }.get(layer_idx, "F.Cu")
    if num_layers >= 6:
        return {
            0: "F.Cu",
            1: "In1.Cu",
            2: "In2.Cu",
            3: "In3.Cu",
            4: "In4.Cu",
            5: "B.Cu",
        }.get(layer_idx, "F.Cu")
    if num_layers >= 4:
        return {0: "F.Cu", 1: "In1.Cu", 2: "In2.Cu", 3: "B.Cu"}.get(layer_idx, "F.Cu")
    return "B.Cu" if layer_idx == 1 else "F.Cu"


def _emit_zone_block(
    lines: list[str],
    net_id: int,
    net_name: str,
    kicad_layer: str,
    board_name: str,
    x0: float,
    y0: float,
    x1: float,
    y1: float,
    clearance_mm: float,
) -> None:
    """Append a KiCad 8 copper pour zone S-expression block."""
    safe_net = net_name.replace('"', "")
    pts_str = (
        f"(xy {x0:.4f} {y0:.4f}) (xy {x1:.4f} {y0:.4f}) "
        f"(xy {x1:.4f} {y1:.4f}) (xy {x0:.4f} {y1:.4f})"
    )
    lines.extend(
        [
            f'\t(zone (net {net_id}) (net_name "{safe_net}") (layer "{kicad_layer}") '
            f'(uuid "{_uid(f"{board_name}-zone-{kicad_layer}-{net_id}")}")',
            "\t\t(hatch edge 0.5)",
            f"\t\t(connect_pads (clearance {clearance_mm:.4f}))",
            "\t\t(min_thickness 0.30)",
            "\t\t(island_removal_mode 1)",
            "\t\t(fill yes (thermal_gap 0.3000) (thermal_bridge_width 0.4000))",
            f"\t\t(polygon (pts {pts_str}))",
            f'\t\t(filled_polygon (layer "{kicad_layer}") (pts {pts_str}))',
            "\t)",
        ]
    )


def serialize_kicad_pcb(board_state: BoardState, include_gnd_pour: bool = True) -> str:
    """Serialize a BoardState into a manufacturing-ready KiCad `.kicad_pcb` string."""
    netlist = board_state.netlist
    cfg = board_state.config
    w = netlist.width_mm
    h = netlist.height_mm
    num_layers = getattr(board_state, "num_layers", 2)

    safe_board = netlist.board_name.replace('"', "")
    layer_defs = [
        '\t\t(0 "F.Cu" signal)',
    ]
    if num_layers >= 8:
        layer_defs.extend(
            [
                '\t\t(1 "In1.Cu" power "In1.GND")',
                '\t\t(2 "In2.Cu" signal "In2.SIG1")',
                '\t\t(3 "In3.Cu" power "In3.PWR1")',
                '\t\t(4 "In4.Cu" power "In4.GND2")',
                '\t\t(5 "In5.Cu" signal "In5.SIG2")',
                '\t\t(6 "In6.Cu" power "In6.GND3")',
            ]
        )
    elif num_layers >= 6:
        layer_defs.extend(
            [
                '\t\t(1 "In1.Cu" power "In1.GND")',
                '\t\t(2 "In2.Cu" signal "In2.Sig1")',
                '\t\t(3 "In3.Cu" signal "In3.Sig2")',
                '\t\t(4 "In4.Cu" power "In4.PWR")',
            ]
        )
    elif num_layers >= 4:
        layer_defs.extend(
            [
                '\t\t(1 "In1.Cu" power "In1.GND")',
                '\t\t(2 "In2.Cu" power "In2.PWR")',
            ]
        )
    layer_defs.extend(
        [
            '\t\t(31 "B.Cu" signal)',
            '\t\t(32 "B.Adhes" user "B.Adhesive")',
            '\t\t(33 "F.Adhes" user "F.Adhesive")',
            '\t\t(34 "B.Paste" user)',
            '\t\t(35 "F.Paste" user)',
            '\t\t(36 "B.SilkS" user "B.Silkscreen")',
            '\t\t(37 "F.SilkS" user "F.Silkscreen")',
            '\t\t(38 "B.Mask" user)',
            '\t\t(39 "F.Mask" user)',
            '\t\t(44 "Edge.Cuts" user)',
            '\t\t(46 "B.CrtYd" user "B.Courtyard")',
            '\t\t(47 "F.CrtYd" user "F.Courtyard")',
            '\t\t(48 "B.Fab" user)',
            '\t\t(49 "F.Fab" user)',
        ]
    )

    setup_lines = [
        "\t(setup",
        "\t\t(pad_to_mask_clearance 0.05)",
    ]
    if num_layers >= 4:
        u_size = getattr(cfg, "microvia_diameter_mm", 0.40)
        u_drill = getattr(cfg, "microvia_drill_mm", 0.20)
        setup_lines.extend(
            [
                "\t\t(uvias_allowed yes)",
                f"\t\t(uvia_size {u_size:.4f})",
                f"\t\t(uvia_min_drill {u_drill:.4f})",
            ]
        )
        if num_layers >= 8:
            setup_lines.append("\t\t(blind_buried_vias_allowed yes)")
    setup_lines.extend(
        [
            "\t\t(pcbplotparams",
            "\t\t\t(layerselection 0x00010fc_ffffffff)",
            "\t\t)",
            "\t)",
        ]
    )

    lines: list[str] = [
        '(kicad_pcb (version 20240108) (generator "pcb-world-model") (generator_version "rev10")',
        "\t(general",
        f"\t\t(thickness {getattr(netlist, 'thickness_mm', getattr(cfg, 'board_thickness_mm', 1.6)):.2f})",
        "\t)",
        '\t(paper "A4")',
        "\t(title_block",
        f'\t\t(title "{safe_board}")',
        '\t\t(rev "10")',
        "\t)",
        "\t(layers",
        *layer_defs,
        "\t)",
        *setup_lines,
        '\t(net 0 "")',
    ]

    for nid in sorted(netlist.nets.keys()):
        net = netlist.nets[nid]
        safe_name = net.name.replace('"', "")
        lines.append(f'\t(net {nid} "{safe_name}")')

    # Board Edge.Cuts outline (rev11 external .STEP/.DXF polygon OR rev10 corner fillets via gr_line + gr_arc)
    outline_poly = list(getattr(netlist, "outline_polygon", []) or [])
    r_fillet = float(getattr(board_state, "corner_fillet_radius_mm", 0.0))
    if len(outline_poly) >= 3:
        for p_idx in range(len(outline_poly)):
            p0 = outline_poly[p_idx]
            p1 = outline_poly[(p_idx + 1) % len(outline_poly)]
            lines.append(
                f"\t(gr_line (start {p0[0]:.4f} {p0[1]:.4f}) (end {p1[0]:.4f} {p1[1]:.4f}) "
                f'(stroke (width 0.1) (type default)) (layer "Edge.Cuts") '
                f'(uuid "{_uid(f"{netlist.board_name}-edge-poly-{p_idx}")}"))'
            )
    elif r_fillet >= 0.2 and r_fillet * 2.5 < min(w, h):
        d_mid = r_fillet * (1.0 - 1.0 / 1.4142135623730951)
        edge_segments = [
            f"\t(gr_line (start {r_fillet:.4f} 0.0000) (end {w - r_fillet:.4f} 0.0000) "
            f'(stroke (width 0.1) (type default)) (layer "Edge.Cuts") (uuid "{_uid(f"{netlist.board_name}-edge-top")}"))',
            f"\t(gr_arc (start {w - r_fillet:.4f} 0.0000) (mid {w - d_mid:.4f} {d_mid:.4f}) (end {w:.4f} {r_fillet:.4f}) "
            f'(stroke (width 0.1) (type default)) (layer "Edge.Cuts") (uuid "{_uid(f"{netlist.board_name}-edge-tr")}"))',
            f"\t(gr_line (start {w:.4f} {r_fillet:.4f}) (end {w:.4f} {h - r_fillet:.4f}) "
            f'(stroke (width 0.1) (type default)) (layer "Edge.Cuts") (uuid "{_uid(f"{netlist.board_name}-edge-right")}"))',
            f"\t(gr_arc (start {w:.4f} {h - r_fillet:.4f}) (mid {w - d_mid:.4f} {h - d_mid:.4f}) (end {w - r_fillet:.4f} {h:.4f}) "
            f'(stroke (width 0.1) (type default)) (layer "Edge.Cuts") (uuid "{_uid(f"{netlist.board_name}-edge-br")}"))',
            f"\t(gr_line (start {w - r_fillet:.4f} {h:.4f}) (end {r_fillet:.4f} {h:.4f}) "
            f'(stroke (width 0.1) (type default)) (layer "Edge.Cuts") (uuid "{_uid(f"{netlist.board_name}-edge-bot")}"))',
            f"\t(gr_arc (start {r_fillet:.4f} {h:.4f}) (mid {d_mid:.4f} {h - d_mid:.4f}) (end 0.0000 {h - r_fillet:.4f}) "
            f'(stroke (width 0.1) (type default)) (layer "Edge.Cuts") (uuid "{_uid(f"{netlist.board_name}-edge-bl")}"))',
            f"\t(gr_line (start 0.0000 {h - r_fillet:.4f}) (end 0.0000 {r_fillet:.4f}) "
            f'(stroke (width 0.1) (type default)) (layer "Edge.Cuts") (uuid "{_uid(f"{netlist.board_name}-edge-left")}"))',
            f"\t(gr_arc (start 0.0000 {r_fillet:.4f}) (mid {d_mid:.4f} {d_mid:.4f}) (end {r_fillet:.4f} 0.0000) "
            f'(stroke (width 0.1) (type default)) (layer "Edge.Cuts") (uuid "{_uid(f"{netlist.board_name}-edge-tl")}"))',
        ]
        lines.extend(edge_segments)
    else:
        lines.extend(
            [
                "\t(gr_rect (start 0.0 0.0) "
                f"(end {w:.4f} {h:.4f}) "
                '(stroke (width 0.1) (type default)) (fill none) (layer "Edge.Cuts") '
                f'(uuid "{_uid(f"{netlist.board_name}-edge")}"))'
            ]
        )

    silk_by_comp = {
        str(a["comp_id"]): a
        for a in getattr(board_state, "silkscreen_adjustments", [])
    }
    enable_td = bool(getattr(cfg, "enable_teardrops", True))
    td_pad_block = (
        " (teardrops (best_length_ratio 0.5) (max_length 1.0) "
        "(best_width_ratio 1.0) (max_width 2.0) (curvy no) "
        "(filter_ratio 0.9) (allow_two_segments yes) (prefer_zone_connections yes))"
        if enable_td
        else ""
    )

    # Footprints & Pads
    for comp_id in sorted(netlist.components.keys()):
        comp = netlist.components[comp_id]
        hw = comp.width / 2.0
        hh = comp.height / 2.0
        is_bot = getattr(comp, "side", "top") == "bottom"
        fp_lyr = "B.Cu" if is_bot else "F.Cu"
        silks_lyr = "B.SilkS" if is_bot else "F.SilkS"
        fab_lyr = "B.Fab" if is_bot else "F.Fab"
        crtyd_lyr = "B.CrtYd" if is_bot else "F.CrtYd"
        locked_str = " (locked)" if comp.is_locked else ""
        silk_adj = silk_by_comp.get(comp_id)
        ref_rx = float(silk_adj["rel_x_mm"]) if silk_adj else 0.0
        ref_ry = float(silk_adj["rel_y_mm"]) if silk_adj else (-hh - 1.2)
        ref_font = float(silk_adj.get("font_mm", 1.0)) if silk_adj else 1.0
        lines.append(
            f'\t(footprint "{comp.footprint}" (layer "{fp_lyr}") (uuid "{_uid(comp.comp_id)}")'
        )
        lines.append(f"\t\t(at {comp.x:.4f} {comp.y:.4f} {comp.theta}){locked_str}")
        lines.append(
            f'\t\t(property "Reference" "{comp.comp_id}" (at {ref_rx:.4f} {ref_ry:.4f} 0) '
            f'(layer "{silks_lyr}") (uuid "' + _uid(f"{comp.comp_id}-ref") + '") '
            f"(effects (font (size {ref_font:.2f} {ref_font:.2f}) (thickness 0.15))))"
        )
        lines.append(
            f'\t\t(property "Value" "{comp.value or comp.comp_id}" (at 0 {hh + 1.2:.4f} 0) '
            f'(layer "{fab_lyr}") (uuid "' + _uid(f"{comp.comp_id}-val") + '") '
            "(effects (font (size 1 1) (thickness 0.15))))"
        )
        lines.append(
            f"\t\t(fp_rect (start {-hw:.4f} {-hh:.4f}) (end {hw:.4f} {hh:.4f}) "
            f'(stroke (width 0.05) (type default)) (fill none) (layer "{crtyd_lyr}") '
            f'(uuid "{_uid(f"{comp.comp_id}-crtyd")}"))'
        )
        is_npth_hole = (
            comp.comp_type == "mounting_hole"
            and not bool(getattr(comp, "electrical_meta", {}).get("plated", False))
        )
        for pad in comp.pads:
            if is_npth_hole:
                ptype = "np_thru_hole"
                players = '"*.Cu" "*.Mask"'
                drill_str = f" (drill {min(pad.width, pad.height):.4f})"
            elif pad.layer_mask == 3:
                ptype = "thru_hole"
                players = '"*.Cu" "*.Mask"'
                drill_str = f" (drill {min(pad.width, pad.height) * 0.55:.4f})"
            elif pad.layer_mask == 2:
                ptype = "smd"
                players = '"B.Cu" "B.Paste" "B.Mask"'
                drill_str = ""
            else:
                ptype = "smd"
                players = '"F.Cu" "F.Paste" "F.Mask"'
                drill_str = ""

            net_name = (
                netlist.nets[pad.net_id].name if pad.net_id in netlist.nets else ""
            )
            lines.append(
                f'\t\t(pad "{pad.pad_id}" {ptype} {pad.shape} '
                f"(at {pad.rel_x:.4f} {pad.rel_y:.4f}) "
                f"(size {pad.width:.4f} {pad.height:.4f}){drill_str} "
                f"(layers {players}) "
                f'(net {pad.net_id} "{net_name}"){"" if is_npth_hole else td_pad_block} '
                f'(uuid "{_uid(f"{comp.comp_id}-{pad.pad_id}")}"))'
            )
        lines.append("\t)")

    # Accessible 1.0mm Debug Test Points (rev10)
    for tp in getattr(board_state, "test_points", []):
        tp_id = str(tp.get("id", "TP1"))
        tp_nid = int(tp.get("net_id", 0))
        tp_nname = str(tp.get("net_name", "")).replace('"', "")
        tp_x = float(tp.get("x_mm", 0.0))
        tp_y = float(tp.get("y_mm", 0.0))
        tp_d = float(tp.get("diameter_mm", 1.0))
        tp_lyr = "B.Cu" if int(tp.get("layer", 0)) > 0 else "F.Cu"
        tp_mask = "B.Mask" if tp_lyr == "B.Cu" else "F.Mask"
        lines.extend(
            [
                f'\t(footprint "TestPoint:TestPoint_Pad_D{tp_d:.1f}mm" (layer "{tp_lyr}") (uuid "{_uid(f"tp-{tp_id}")}")',
                f"\t\t(at {tp_x:.4f} {tp_y:.4f} 0)",
                f'\t\t(property "Reference" "{tp_id}" (at 0 -1.0 0) (layer "{"B.SilkS" if tp_lyr == "B.Cu" else "F.SilkS"}") (uuid "{_uid(f"tp-{tp_id}-ref")}") (effects (font (size 0.75 0.75) (thickness 0.12))))',
                f'\t\t(pad "1" smd circle (at 0 0) (size {tp_d:.4f} {tp_d:.4f}) (layers "{tp_lyr}" "{tp_mask}") (net {tp_nid} "{tp_nname}"){td_pad_block} (uuid "{_uid(f"tp-{tp_id}-pad")}"))',
                "\t)",
            ]
        )

    # Routed Copper Segments
    seg_idx = 0
    for net_id in sorted(board_state.routed_traces.keys()):
        for seg in board_state.routed_traces[net_id]:
            layer_name = _kicad_layer_name(seg.layer, num_layers)
            lines.append(
                f"\t(segment (start {seg.x1_mm:.4f} {seg.y1_mm:.4f}) "
                f"(end {seg.x2_mm:.4f} {seg.y2_mm:.4f}) "
                f"(width {seg.width_mm or cfg.trace_width_mm:.4f}) "
                f'(layer "{layer_name}") (net {net_id}) '
                f'(uuid "{_uid(f"seg-{net_id}-{seg_idx}")}"))'
            )
            seg_idx += 1

    # Routed Vias (through-hole, blind/buried core, and buildup microvias)
    via_idx = 0
    for net_id in sorted(board_state.routed_vias.keys()):
        for via in board_state.routed_vias[net_id]:
            vtype = getattr(via, "via_type", "through")
            if num_layers >= 4 and vtype == "micro":
                s_lyr = _kicad_layer_name(getattr(via, "start_layer", 0), num_layers)
                e_lyr = _kicad_layer_name(getattr(via, "end_layer", 1), num_layers)
                lines.append(
                    f"\t(via micro (at {via.x_mm:.4f} {via.y_mm:.4f}) "
                    f"(size {via.diameter_mm:.4f}) (drill {via.drill_mm:.4f}) "
                    f'(layers "{s_lyr}" "{e_lyr}") (net {net_id}){td_pad_block} '
                    f'(uuid "{_uid(f"via-{net_id}-{via_idx}")}"))'
                )
            elif num_layers >= 6 and vtype == "buried":
                s_lyr = _kicad_layer_name(getattr(via, "start_layer", 2), num_layers)
                e_lyr = _kicad_layer_name(getattr(via, "end_layer", 5), num_layers)
                lines.append(
                    f"\t(via blind (at {via.x_mm:.4f} {via.y_mm:.4f}) "
                    f"(size {via.diameter_mm:.4f}) (drill {via.drill_mm:.4f}) "
                    f'(layers "{s_lyr}" "{e_lyr}") (net {net_id}){td_pad_block} '
                    f'(uuid "{_uid(f"via-{net_id}-{via_idx}")}"))'
                )
            else:
                lines.append(
                    f"\t(via (at {via.x_mm:.4f} {via.y_mm:.4f}) "
                    f"(size {via.diameter_mm:.4f}) (drill {via.drill_mm:.4f}) "
                    f'(layers "F.Cu" "B.Cu") (net {net_id}){td_pad_block} '
                    f'(uuid "{_uid(f"via-{net_id}-{via_idx}")}"))'
                )
            via_idx += 1

    # Copper Ground / Power Plane Pour Zones
    if include_gnd_pour:
        margin = max(0.3, float(cfg.edge_margin_mm))
        x0, y0 = margin, margin
        x1, y1 = max(x0 + 1.0, w - margin), max(y0 + 1.0, h - margin)
        if num_layers >= 4 and getattr(board_state, "plane_assignments", None):
            for l_idx in sorted(board_state.plane_assignments.keys()):
                pnid, pname = board_state.plane_assignments[l_idx]
                kicad_zone_layer = _kicad_layer_name(l_idx, num_layers)
                _emit_zone_block(
                    lines=lines,
                    net_id=pnid,
                    net_name=pname,
                    kicad_layer=kicad_zone_layer,
                    board_name=netlist.board_name,
                    x0=x0,
                    y0=y0,
                    x1=x1,
                    y1=y1,
                    clearance_mm=cfg.clearance_mm,
                )
        else:
            gnd_info = _find_gnd_net(board_state)
            if gnd_info is not None:
                gnd_nid, gnd_name = gnd_info
                _emit_zone_block(
                    lines=lines,
                    net_id=gnd_nid,
                    net_name=gnd_name,
                    kicad_layer="B.Cu",
                    board_name=netlist.board_name,
                    x0=x0,
                    y0=y0,
                    x1=x1,
                    y1=y1,
                    clearance_mm=cfg.clearance_mm,
                )

    # RF Antenna, Isolation Keepout Zones & Thermal Isolation Copper-Pour Slots (rev13)
    for comp_id in sorted(netlist.components.keys()):
        comp = netlist.components[comp_id]
        for kz_idx, (kx0, ky0, kx1, ky1, all_layers, kind) in enumerate(comp.world_keepout_zones()):
            k_layer = "F&B.Cu" if all_layers else _kicad_layer_name(1 if comp.side == "bottom" else 0, num_layers)
            k_pts = (
                f"(xy {kx0:.4f} {ky0:.4f}) (xy {kx1:.4f} {ky0:.4f}) "
                f"(xy {kx1:.4f} {ky1:.4f}) (xy {kx0:.4f} {ky1:.4f})"
            )
            lines.extend(
                [
                    f'\t(zone (net 0) (net_name "") (layer "{k_layer}") '
                    f'(uuid "{_uid(f"{netlist.board_name}-keepout-{comp_id}-{kz_idx}-{kind}")}")',
                    "\t\t(hatch edge 0.5)",
                    "\t\t(keepout (tracks not_allowed) (vias not_allowed) (pads not_allowed) "
                    "(copperpour not_allowed) (footprints not_allowed))",
                    f"\t\t(polygon (pts {k_pts}))",
                    "\t)",
                ]
            )

    for slot_idx, slot in enumerate(getattr(board_state, "thermal_isolation_slots", [])):
        sx0 = float(slot["x0_mm"])
        sy0 = float(slot["y0_mm"])
        sx1 = float(slot["x1_mm"])
        sy1 = float(slot["y1_mm"])
        s_layer = str(slot.get("layer", "F.Cu"))
        s_pts = (
            f"(xy {sx0:.4f} {sy0:.4f}) (xy {sx1:.4f} {sy0:.4f}) "
            f"(xy {sx1:.4f} {sy1:.4f}) (xy {sx0:.4f} {sy1:.4f})"
        )
        lines.extend(
            [
                f'\t(zone (net 0) (net_name "") (layer "{s_layer}") '
                f'(uuid "{_uid(f"{netlist.board_name}-thermal-slot-{slot_idx}")}")',
                "\t\t(hatch edge 0.5)",
                "\t\t(keepout (tracks allowed) (vias allowed) (pads not_allowed) "
                "(copperpour not_allowed) (footprints allowed))",
                f"\t\t(polygon (pts {s_pts}))",
                "\t)",
            ]
        )

    # Rev 14: Cross-hatched polyimide flex reference mesh & bend-zone keepout zones
    gnd_info_flex = _find_gnd_net(board_state)
    for fz in getattr(netlist, "substrate_zones", []) or []:
        if getattr(fz, "zone_type", "rigid") != "dynamic_flex":
            continue
        fx0 = max(0.0, float(fz.x0_mm))
        fy0 = max(0.0, float(fz.y0_mm))
        fx1 = min(w, float(fz.x1_mm))
        fy1 = min(h, float(fz.y1_mm))
        f_pts = (
            f"(xy {fx0:.4f} {fy0:.4f}) (xy {fx1:.4f} {fy0:.4f}) "
            f"(xy {fx1:.4f} {fy1:.4f}) (xy {fx0:.4f} {fy1:.4f})"
        )
        if gnd_info_flex is not None and num_layers >= 2:
            gnid, gname = gnd_info_flex
            safe_gname = gname.replace('"', "")
            ref_layer = _kicad_layer_name(1, num_layers)
            fill_r = max(0.20, min(0.95, float(getattr(fz, "hatch_fill_ratio", 0.50))))
            h_ang = float(getattr(fz, "hatch_angle_deg", 45.0))
            h_thick = 0.2000
            h_gap = max(0.1000, h_thick * (1.0 - fill_r) / max(0.05, fill_r))
            lines.extend(
                [
                    f'\t(zone (net {gnid}) (net_name "{safe_gname}") (layer "{ref_layer}") '
                    f'(uuid "{_uid(f"{netlist.board_name}-flex-hatch-{fz.zone_id}")}")',
                    f"\t\t(hatch full {fill_r:.2f})",
                    f"\t\t(connect_pads (clearance {cfg.clearance_mm:.4f}))",
                    "\t\t(min_thickness 0.15)",
                    "\t\t(island_removal_mode 1)",
                    f"\t\t(fill yes (mode hatch) (hatch_thickness {h_thick:.4f}) (hatch_gap {h_gap:.4f}) "
                    f"(hatch_orientation {h_ang:.1f}) (hatch_smoothing_level 1) (thermal_gap 0.3000) (thermal_bridge_width 0.3500))",
                    f"\t\t(polygon (pts {f_pts}))",
                    f'\t\t(filled_polygon (layer "{ref_layer}") (pts {f_pts}))',
                    "\t)",
                ]
            )
        lines.extend(
            [
                f'\t(zone (net 0) (net_name "") (layer "F&B.Cu") '
                f'(uuid "{_uid(f"{netlist.board_name}-flex-keepout-{fz.zone_id}")}")',
                "\t\t(hatch edge 0.5)",
                "\t\t(keepout (tracks allowed) (vias not_allowed) (pads not_allowed) "
                "(copperpour not_allowed) (footprints not_allowed))",
                f"\t\t(polygon (pts {f_pts}))",
                "\t)",
            ]
        )

    lines.append(")")
    return "\n".join(lines) + "\n"


def export(
    board_state: BoardState,
    output_path: str | Path,
    include_gnd_pour: bool = True,
) -> Path:
    """Write a BoardState as a `.kicad_pcb` file to disk and return the resolved path."""
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        serialize_kicad_pcb(board_state, include_gnd_pour=include_gnd_pour),
        encoding="utf-8",
    )
    return out
