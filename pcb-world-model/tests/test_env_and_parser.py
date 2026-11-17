"""Unit tests for configs, S-expression parser, KiCad exporter, BoardState S_t tensor, and metrics."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from env.board_state import BoardState, TraceSegment, Via
from env.config import BoardConfig, TrainConfig
from env.footprint_generator import (
    make_keyswitch_mx,
    make_pin_header,
    make_qfn,
    make_smd_passive,
    make_soic_or_sop,
)
from env.kicad_exporter import export, serialize_kicad_pcb
from env.manufacturing_exporter import (
    export_manufacturing_bundle,
    generate_bom_csv,
    generate_cpl_csv,
)
from env.metrics import (
    compute_hpwl,
    compute_mst_wirelength,
    compute_signal_integrity_telemetry,
    compute_trace_dc_resistance_ohm,
    rank_nets_by_criticality,
)
from env.sexpr_parser import (
    ALL_PRESET_DESCRIPTIONS,
    COMPLEX_PRESET_DESCRIPTIONS,
    PRESET_DESCRIPTIONS,
    SExprParser,
    build_preset_board,
)


def test_configs_load_from_yaml() -> None:
    board_cfg = BoardConfig.from_yaml(ROOT_DIR / "configs" / "default_2layer.yaml")
    train_cfg = TrainConfig.from_yaml(ROOT_DIR / "configs" / "train_config.yaml")

    assert board_cfg.board_width_mm == 50.0
    assert board_cfg.board_height_mm == 50.0
    assert board_cfg.grid_resolution_mm == 0.2
    assert board_cfg.grid_height == 250
    assert board_cfg.grid_width == 250
    assert board_cfg.clearance_mm == 0.15
    assert train_cfg.placer_num_layers == 4
    assert train_cfg.world_base_channels == 16


@pytest.mark.parametrize("preset_name", list(PRESET_DESCRIPTIONS.keys()))
def test_preset_boards_and_kicad_roundtrip(preset_name: str, tmp_path: Path) -> None:
    graph = build_preset_board(preset_name)
    cfg = BoardConfig()
    state = BoardState(netlist=graph, config=cfg)

    out_file = tmp_path / f"{preset_name}.kicad_pcb"
    export(state, out_file)
    assert out_file.exists()

    parsed = SExprParser.parse_file(out_file)
    assert parsed.width_mm == pytest.approx(50.0, abs=0.05)
    assert parsed.height_mm == pytest.approx(50.0, abs=0.05)
    assert set(parsed.components.keys()) == set(graph.components.keys())
    assert len(parsed.routable_net_ids) == len(graph.routable_net_ids)


def test_board_state_tensor_shape_and_channels() -> None:
    graph = build_preset_board("mcu_sensor_node")
    cfg = BoardConfig()
    state = BoardState(netlist=graph, config=cfg)

    tensor = state.to_tensor(active_net_id=4)
    assert isinstance(tensor, torch.Tensor)
    assert tensor.shape == (7, 250, 250)
    assert tensor.dtype == torch.float32
    assert float(tensor[0].max()) == 1.0  # board_outline
    assert float(tensor[0, 0, 0]) == 0.0  # edge margin keepout
    assert float(tensor[1].max()) == 1.0  # courtyard_mask
    assert float(tensor[2].max()) > 0.0   # copper_L1 pads
    assert float(tensor[5].max()) > 0.5   # active_net_pins Gaussian peak
    assert float(tensor[6].max()) > 0.0   # unrouted_demand rat's nest

    # Commit and rip up trace
    seg = TraceSegment(net_id=4, layer=1, x1_mm=10.0, y1_mm=10.0, x2_mm=20.0, y2_mm=10.0)
    via = Via(net_id=4, x_mm=20.0, y_mm=10.0)
    state.commit_trace(4, [seg], [via])
    assert state.is_routed(4)
    t_after = state.to_tensor(active_net_id=4)
    assert float(t_after[3].max()) > 0.2  # copper_L2 populated
    assert float(t_after[4].max()) == 1.0  # via_mask populated

    state.rip_up_net(4)
    assert not state.is_routed(4)


def test_metrics_and_net_ranking() -> None:
    graph = build_preset_board("buck_converter")
    hpwl = compute_hpwl(graph)
    mst = compute_mst_wirelength(graph)
    ranked = rank_nets_by_criticality(graph)

    assert hpwl > 20.0
    assert mst > 20.0
    assert set(ranked) == set(graph.routable_net_ids)


def test_parametric_ipc7351_footprint_generator() -> None:
    for pkg in ("0402", "0603", "0805", "1206"):
        comp = make_smd_passive("C1", "100nF", "cap", package=pkg, net1=1, net2=2)
        assert len(comp.pads) == 2
        assert pkg in comp.footprint
        assert comp.pads[0].rel_x < 0.0 and comp.pads[1].rel_x > 0.0

    with pytest.raises(ValueError):
        make_smd_passive("C_BAD", "100nF", "cap", package="9999")

    for n_pins in (8, 14, 16):
        soic = make_soic_or_sop("U1", f"IC_{n_pins}", list(range(1, n_pins + 1)))
        assert len(soic.pads) == n_pins
        assert f"SOIC-{n_pins}" in soic.footprint

    with pytest.raises(ValueError):
        make_soic_or_sop("U_ODD", "BAD", [1, 2, 3, 4, 5])

    for n_pins in (12, 16, 20):
        qfn = make_qfn("U2", f"QFN_{n_pins}", list(range(1, n_pins + 1)))
        assert len(qfn.pads) == n_pins
        assert f"QFN-{n_pins}" in qfn.footprint

    for n_pins in (4, 6):
        hdr = make_pin_header("J1", "HDR", list(range(1, n_pins + 1)))
        assert len(hdr.pads) == n_pins
        assert hdr.is_locked is True
        assert all(p.layer_mask == 3 for p in hdr.pads)

    sw = make_keyswitch_mx("SW1", "KEY_1", net1=6, net2=8, x=32.0, y=14.0)
    assert len(sw.pads) == 2
    assert sw.is_locked is True
    assert all(p.layer_mask == 3 for p in sw.pads)


def test_kicad_gnd_zone_pour_and_manufacturing_bundle(tmp_path: Path) -> None:
    import csv

    graph = build_preset_board("mcu_sensor_node")
    cfg = BoardConfig.from_yaml(ROOT_DIR / "configs" / "default_2layer.yaml")
    state = BoardState(netlist=graph, config=cfg)

    pcb_with_pour = serialize_kicad_pcb(state)
    assert '(zone (net 3) (net_name "GND") (layer "B.Cu")' in pcb_with_pour
    assert "(filled_polygon" in pcb_with_pour

    pcb_without_pour = serialize_kicad_pcb(state, include_gnd_pour=False)
    assert "(zone " not in pcb_without_pour

    summary = state.to_summary_dict()
    assert summary["gnd_pour"] is not None
    assert summary["gnd_pour"]["layer"] == "B.Cu"
    assert summary["gnd_pour"]["net_name"] == "GND"

    bom_rows = list(csv.reader(generate_bom_csv(state).splitlines()))
    assert bom_rows[0] == ["Comment", "Designator", "Footprint", "Quantity"]
    total_qty = sum(int(r[3]) for r in bom_rows[1:])
    assert total_qty == len(graph.components)

    cpl_rows = list(csv.reader(generate_cpl_csv(state).splitlines()))
    assert cpl_rows[0] == ["Designator", "Val", "Package", "Mid X", "Mid Y", "Rotation", "Layer"]
    assert len(cpl_rows) == 1 + len(graph.components)
    assert all(r[3].endswith("mm") and r[4].endswith("mm") and r[6] == "Top" for r in cpl_rows[1:])

    bundle = export_manufacturing_bundle(state, tmp_path / "fab")
    assert bundle["kicad_pcb"].exists()
    assert bundle["bom_csv"].exists()
    assert bundle["positions_csv"].exists()
    reparsed = SExprParser.parse_file(bundle["kicad_pcb"])
    assert set(reparsed.components.keys()) == set(graph.components.keys())


def test_4layer_stackup_config_and_kicad_roundtrip(tmp_path: Path) -> None:
    cfg = BoardConfig.from_yaml(ROOT_DIR / "configs" / "default_4layer.yaml")
    assert cfg.layers == 4
    assert cfg.microvia_diameter_mm == pytest.approx(0.40)
    assert cfg.copper_thickness_mm == pytest.approx(0.035)

    assert "flight_controller_50" in COMPLEX_PRESET_DESCRIPTIONS
    assert "flight_controller_50" in ALL_PRESET_DESCRIPTIONS
    assert "flight_controller_50" not in PRESET_DESCRIPTIONS

    graph = build_preset_board("flight_controller_50")
    assert len(graph.components) == 50
    assert graph.width_mm == 80.0 and graph.height_mm == 80.0
    assert graph.layers == 4
    assert len(graph.routable_net_ids) == 43

    state = BoardState(netlist=graph, config=cfg)
    assert state.num_layers == 4
    assert 1 in state.plane_assignments and 2 in state.plane_assignments
    assert state.plane_assignments[1][1] == "GND"
    assert state.plane_assignments[2][1] == "VCC_3V3"

    state.commit_trace(
        net_id=10,
        segments=[TraceSegment(10, 3, 12.0, 12.0, 22.0, 12.0, 0.25)],
        vias=[
            Via(4, 15.0, 15.0, 0.40, 0.20, start_layer=0, end_layer=1, via_type="micro"),
            Via(10, 22.0, 12.0, 0.60, 0.30, start_layer=0, end_layer=3, via_type="through"),
        ],
    )
    out_pcb = export(state, tmp_path / "fc50_roundtrip.kicad_pcb")
    text = out_pcb.read_text(encoding="utf-8")
    assert '(1 "In1.Cu" power "In1.GND")' in text
    assert '(2 "In2.Cu" power "In2.PWR")' in text
    assert "(uvias_allowed yes)" in text
    assert "(via micro " in text
    assert '(layer "In1.Cu")' in text
    assert '(layer "In2.Cu")' in text

    reparsed = SExprParser.parse_file(out_pcb)
    assert reparsed.layers == 4
    assert len(reparsed.components) == 50
    micro_vias = [v for v in reparsed.raw_vias if v.get("via_type") == "micro"]
    assert len(micro_vias) == 1
    assert micro_vias[0]["start_layer"] == 0 and micro_vias[0]["end_layer"] == 1
    assert len(reparsed.raw_segments) == 1 and reparsed.raw_segments[0]["layer"] == 3


def test_signal_integrity_and_thermal_telemetry() -> None:
    expected_r = 1.724e-5 * (10.0 / (0.40 * 0.035))
    assert compute_trace_dc_resistance_ohm(10.0, 0.40, 0.035) == pytest.approx(expected_r, rel=1e-6)

    graph = build_preset_board("buck_converter")
    cfg = BoardConfig.from_yaml(ROOT_DIR / "configs" / "default_2layer.yaml")
    state = BoardState(netlist=graph, config=cfg)

    # Route switching node cleanly on F.Cu with 0.40mm copper
    state.commit_trace(
        net_id=4,  # SW_NODE
        segments=[TraceSegment(4, 0, 20.0, 25.0, 28.0, 25.0, 0.40)],
        vias=[],
    )
    si = compute_signal_integrity_telemetry(state)
    assert si["copper_weight_oz"] == pytest.approx(1.0)
    assert si["max_dc_resistance_mohm"] == pytest.approx(expected_r * 0.8 * 1000.0, rel=0.05)
    assert si["sw_node_inductance_nh"] is not None and si["sw_node_inductance_nh"] > 0.0
    assert si["si_thermal_status"] == "PASS"

    # Undersized high-current trace triggers thermal WARN
    state.commit_trace(
        net_id=1,  # VIN_12V (1.5A design current)
        segments=[TraceSegment(1, 0, 10.0, 10.0, 20.0, 10.0, 0.08)],
        vias=[],
    )
    si_warn = compute_signal_integrity_telemetry(state)
    assert si_warn["trace_capacity_all_pass"] is False
    assert si_warn["si_thermal_status"] == "WARN"


def test_rev7_two_tier_component_library_all_12_domains() -> None:
    from env.component_library import (
        COMMODITY_CATALOG,
        INDUSTRY_DOMAINS,
        SPECIAL_LAYOUT_RULE_DOCS,
        SPECIALIZED_CATALOG,
        instantiate_commodity,
        instantiate_device,
        list_devices,
        summarize_catalog,
    )

    assert len(INDUSTRY_DOMAINS) == 12
    assert len(SPECIAL_LAYOUT_RULE_DOCS) == 8
    assert {"resistor", "capacitor_mlcc", "power_inductor", "diode"}.issubset(COMMODITY_CATALOG.keys())

    r_0402 = instantiate_commodity(
        "resistor",
        "R_CC1",
        value="5.1k",
        package="0402",
        nets=[1, 3],
        tolerance="1%",
        power_rating="0.0625W",
    )
    assert r_0402.value == "5.1k"
    assert r_0402.electrical_meta["class"] == "A"
    assert r_0402.electrical_meta["tolerance"] == "1%"

    summary = summarize_catalog()
    assert summary["counts"]["domains"] == 12
    assert summary["counts"]["class_b_devices"] >= 60

    covered_domains = {spec.domain for spec in SPECIALIZED_CATALOG.values()}
    assert covered_domains == set(INDUSTRY_DOMAINS)

    esp = instantiate_device("ESP32-C3FN4", "U1", thermal_pad_net=2, keep_back_flat=True)
    assert "thermal_via_ep" in esp.special_layout_rules
    assert len(esp.pads) == 33

    rf = instantiate_device("ESP32-C3-MINI-1", "U1")
    assert "rf_antenna_keepout" in rf.special_layout_rules
    assert len(rf.keepout_zones) >= 1

    devices_rf = list_devices(query="rf_antenna_keepout")
    assert any(d["part_id"] == "ESP32-C3-MINI-1" for d in devices_rf)


def test_rev7_class_c_kicad_mod_and_datasheet_compiler(tmp_path: Path) -> None:
    from env.kicad_mod_loader import compile_datasheet_pin_table, parse_kicad_mod

    mod_text = """(footprint "Custom_Sensors:MY_QFN16"
      (property "Value" "CUSTOM_IMU")
      (fp_rect (start -2.0 -2.0) (end 2.0 2.0) (layer "F.CrtYd"))
      (pad "1" smd rect (at -1.4 -0.75) (size 0.65 0.28) (layers "F.Cu" "F.Paste" "F.Mask"))
      (pad "2" smd rect (at -1.4 -0.25) (size 0.65 0.28) (layers "F.Cu" "F.Paste" "F.Mask"))
      (pad "3" smd rect (at 1.4 -0.25) (size 0.65 0.28) (layers "F.Cu" "F.Paste" "F.Mask"))
      (pad "4" smd rect (at 1.4 -0.75) (size 0.65 0.28) (layers "F.Cu" "F.Paste" "F.Mask"))
      (pad "EP" smd rect (at 0 0) (size 1.8 1.8) (layers "F.Cu" "F.Paste" "F.Mask"))
    )"""
    mod_file = tmp_path / "MY_QFN16.kicad_mod"
    mod_file.write_text(mod_text, encoding="utf-8")

    comp = parse_kicad_mod(mod_file, comp_id="U_MOD")
    assert comp.value == "CUSTOM_IMU"
    assert len(comp.pads) == 5
    assert "thermal_via_ep" in comp.special_layout_rules
    assert comp.width == pytest.approx(4.0)
    assert comp.height == pytest.approx(4.0)

    spec = {
        "part_number": "CUSTOM_SOC_QFN16",
        "package": "QFN-16",
        "pitch": 0.5,
        "body_size": 3.0,
        "thermal_pad": "GND",
        "pins": {
            "1": "VDD",
            "2": "GND",
            "3": "SDA",
            "4": "SCL",
            "5": "XIN",
            "6": "XOUT",
            "7": "USB_DP",
            "8": "USB_DM",
            "9": "CC1",
            "10": "CC2",
            "11": "NRST",
        },
    }
    ds_comp, support = compile_datasheet_pin_table(
        spec,
        comp_id="U_DS",
        net_name_to_id={"VDD": 1, "GND": 2, "SDA": 3, "SCL": 4},
    )
    assert ds_comp.value == "CUSTOM_SOC_QFN16"
    assert "thermal_via_ep" in ds_comp.special_layout_rules
    support_types = {s["type"] for s in support}
    assert {"decoupling_cap", "i2c_pullups", "crystal_oscillator", "usb_c_cc", "diff_pair", "thermal_via_array", "reset_rc"}.issubset(support_types)


def test_rev7_wearable_keychain_kicad_roundtrip_and_cpl_sides(tmp_path: Path) -> None:
    graph = build_preset_board("wearable_oled_keychain_26x26")
    assert graph.width_mm == pytest.approx(26.0)
    assert graph.height_mm == pytest.approx(26.0)
    assert graph.thickness_mm == pytest.approx(0.8)
    assert graph.single_sided is True
    assert len(graph.design_rationale) >= 1

    # Rear battery pads sit on bottom, while all other SMD parts obey keep_back_flat
    assert graph.components["J3"].side == "bottom"
    for cid, comp in graph.components.items():
        if cid != "J3":
            assert comp.side == "top"

    cfg = BoardConfig.from_yaml(ROOT_DIR / "configs" / "default_2layer.yaml")
    cfg = cfg.__class__(
        **{
            **cfg.__dict__,
            "board_width_mm": graph.width_mm,
            "board_height_mm": graph.height_mm,
            "board_thickness_mm": graph.thickness_mm,
            "single_sided": True,
            "keep_back_flat": True,
        }
    )
    state = BoardState(netlist=graph, config=cfg)
    out_pcb = tmp_path / "wearable_b6.kicad_pcb"
    export(state, out_pcb)
    pcb_text = out_pcb.read_text(encoding="utf-8")

    assert "(thickness 0.80)" in pcb_text
    assert "np_thru_hole circle" in pcb_text
    assert "(keepout" in pcb_text

    reparsed = SExprParser.parse_text(pcb_text)
    assert reparsed.thickness_mm == pytest.approx(0.8)
    assert reparsed.components["J3"].side == "bottom"

    cpl = generate_cpl_csv(state)
    cpl_lines = [line for line in cpl.strip().splitlines() if line]
    j3_rows = [row for row in cpl_lines if row.startswith("J3,")]
    assert len(j3_rows) == 1
    assert j3_rows[0].endswith(",Bottom")
    non_j3_rows = [row for row in cpl_lines[1:] if not row.startswith("J3,") and not row.startswith("MH")]
    assert all(row.endswith(",Top") for row in non_j3_rows)


def test_rev7_6layer_stackup_export_and_parse(tmp_path: Path) -> None:
    graph = build_preset_board("wearable_oled_keychain_26x26")
    cfg = BoardConfig.from_yaml(ROOT_DIR / "configs" / "default_4layer.yaml")
    cfg = cfg.__class__(**{**cfg.__dict__, "layers": 6, "board_width_mm": 26.0, "board_height_mm": 26.0, "board_thickness_mm": 0.8})
    state = BoardState(netlist=graph, config=cfg)
    assert state.num_layers == 6
    out_pcb = tmp_path / "stack6.kicad_pcb"
    export(state, out_pcb)
    text = out_pcb.read_text(encoding="utf-8")
    assert '(2 "In2.Cu" signal "In2.Sig1")' in text
    assert '(4 "In4.Cu" power "In4.PWR")' in text
    reparsed = SExprParser.parse_file(out_pcb)
    assert reparsed.layers == 6
