"""Exporter translating BoardState back into valid KiCad 8 `.kicad_pcb` S-expressions."""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from env.board_state import BoardState


def _uid(seed_str: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, seed_str))


def serialize_kicad_pcb(board_state: BoardState) -> str:
    """Serialize a BoardState into a manufacturing-ready KiCad `.kicad_pcb` string."""
    netlist = board_state.netlist
    cfg = board_state.config
    w = netlist.width_mm
    h = netlist.height_mm

    safe_board = netlist.board_name.replace('"', "")
    lines: list[str] = [
        '(kicad_pcb (version 20240108) (generator "pcb-world-model") (generator_version "rev1")',
        "\t(general",
        "\t\t(thickness 1.6)",
        "\t)",
        '\t(paper "A4")',
        "\t(title_block",
        f'\t\t(title "{safe_board}")',
        '\t\t(rev "1")',
        "\t)",
        "\t(layers",
        '\t\t(0 "F.Cu" signal)',
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
        "\t)",
        "\t(setup",
        "\t\t(pad_to_mask_clearance 0.05)",
        "\t\t(pcbplotparams",
        "\t\t\t(layerselection 0x00010fc_ffffffff)",
        "\t\t)",
        "\t)",
        '\t(net 0 "")',
    ]

    for nid in sorted(netlist.nets.keys()):
        net = netlist.nets[nid]
        safe_name = net.name.replace('"', "")
        lines.append(f'\t(net {nid} "{safe_name}")')

    # Board Edge.Cuts outline
    lines.extend(
        [
            "\t(gr_rect (start 0.0 0.0) "
            f"(end {w:.4f} {h:.4f}) "
            '(stroke (width 0.1) (type default)) (fill none) (layer "Edge.Cuts") '
            f'(uuid "{_uid(f"{netlist.board_name}-edge")}"))'
        ]
    )

    # Footprints & Pads
    for comp_id in sorted(netlist.components.keys()):
        comp = netlist.components[comp_id]
        hw = comp.width / 2.0
        hh = comp.height / 2.0
        locked_str = " (locked)" if comp.is_locked else ""
        lines.append(
            f'\t(footprint "{comp.footprint}" (layer "F.Cu") (uuid "{_uid(comp.comp_id)}")'
        )
        lines.append(f"\t\t(at {comp.x:.4f} {comp.y:.4f} {comp.theta}){locked_str}")
        lines.append(
            f'\t\t(property "Reference" "{comp.comp_id}" (at 0 {-hh - 1.2:.4f} 0) '
            '(layer "F.SilkS") (uuid "' + _uid(f"{comp.comp_id}-ref") + '") '
            "(effects (font (size 1 1) (thickness 0.15))))"
        )
        lines.append(
            f'\t\t(property "Value" "{comp.value or comp.comp_id}" (at 0 {hh + 1.2:.4f} 0) '
            '(layer "F.Fab") (uuid "' + _uid(f"{comp.comp_id}-val") + '") '
            "(effects (font (size 1 1) (thickness 0.15))))"
        )
        lines.append(
            f"\t\t(fp_rect (start {-hw:.4f} {-hh:.4f}) (end {hw:.4f} {hh:.4f}) "
            '(stroke (width 0.05) (type default)) (fill none) (layer "F.CrtYd") '
            f'(uuid "{_uid(f"{comp.comp_id}-crtyd")}"))'
        )
        for pad in comp.pads:
            if pad.layer_mask == 3:
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
                f'(net {pad.net_id} "{net_name}") '
                f'(uuid "{_uid(f"{comp.comp_id}-{pad.pad_id}")}"))'
            )
        lines.append("\t)")

    # Routed Copper Segments
    seg_idx = 0
    for net_id in sorted(board_state.routed_traces.keys()):
        for seg in board_state.routed_traces[net_id]:
            layer_name = "B.Cu" if seg.layer == 1 else "F.Cu"
            lines.append(
                f"\t(segment (start {seg.x1_mm:.4f} {seg.y1_mm:.4f}) "
                f"(end {seg.x2_mm:.4f} {seg.y2_mm:.4f}) "
                f"(width {seg.width_mm or cfg.trace_width_mm:.4f}) "
                f'(layer "{layer_name}") (net {net_id}) '
                f'(uuid "{_uid(f"seg-{net_id}-{seg_idx}")}"))'
            )
            seg_idx += 1

    # Routed Vias
    via_idx = 0
    for net_id in sorted(board_state.routed_vias.keys()):
        for via in board_state.routed_vias[net_id]:
            lines.append(
                f"\t(via (at {via.x_mm:.4f} {via.y_mm:.4f}) "
                f"(size {via.diameter_mm:.4f}) (drill {via.drill_mm:.4f}) "
                f'(layers "F.Cu" "B.Cu") (net {net_id}) '
                f'(uuid "{_uid(f"via-{net_id}-{via_idx}")}"))'
            )
            via_idx += 1

    lines.append(")")
    return "\n".join(lines) + "\n"


def export(board_state: BoardState, output_path: str | Path) -> Path:
    """Write a BoardState as a `.kicad_pcb` file to disk and return the resolved path."""
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(serialize_kicad_pcb(board_state), encoding="utf-8")
    return out
