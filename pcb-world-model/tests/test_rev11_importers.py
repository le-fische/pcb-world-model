"""Revision 11 tests: External Schematic/Pin-Diagram/.STEP/.DXF Importers, 3D Z-Map, Lock-Preserving Re-Optimization & B10-B12."""

from __future__ import annotations

import base64
import importlib
from pathlib import Path
import sys

from fastapi.testclient import TestClient

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from env.board_state import BoardState
from env.config import BoardConfig
from env.footprint_generator import make_smd_passive
from env.importers import (
    apply_enclosure_constraints,
    audit_schematic_craftsmanship,
    auto_fix_schematic,
    build_legacy_high_via_kicad_pcb,
    build_sample_enclosure_step,
    build_synthetic_pinout_pdf,
    build_synthetic_pinout_png,
    import_pin_diagram,
    load_custom_library,
    optimize_imported_board,
    parse_dxf_enclosure,
    parse_schematic,
    parse_step_enclosure,
)
from env.sexpr_parser import NetInfo, NetlistGraph, build_preset_board
from env.step_exporter import generate_step_assembly, validate_step_structure
from ui.server import app
from verifier.fast_clearance import check_enclosure_collisions

_inf_mod = importlib.import_module("scripts.04_run_inference")
design_pcb_world_model = _inf_mod.design_pcb_world_model


def test_rev11_schematic_importer_formats_and_auto_erc() -> None:
    # 1. atopile (.ato) with missing decoupling & missing I2C pullups
    ato_src = """
module SensorNode:
  u_mcu = new ATtiny1614
  u_sens = new BME280_I2C
  vcc ~ u_mcu.1
  gnd ~ u_mcu.8
  vcc ~ u_sens.1
  gnd ~ u_sens.2
  sda ~ u_mcu.2
  sda ~ u_sens.3
  scl ~ u_mcu.3
  scl ~ u_sens.4
"""
    g_ato, pre_audit, _ = parse_schematic(ato_src, fmt="atopile", auto_fix=False)
    codes = {i.code for i in pre_audit.issues}
    assert "MISSING_DECOUPLING_CAP" in codes
    assert "MISSING_I2C_PULLUP" in codes

    g_fixed, fixes = auto_fix_schematic(g_ato, pre_audit)
    assert len(fixes) >= 3
    post_audit = audit_schematic_craftsmanship(g_fixed)
    assert post_audit.error_count == 0
    assert post_audit.warning_count == 0

    # 2. SPICE netlist (.cir) with un-clamped relay coil
    spice_src = """
* Relay driver snippet
K_RELAY1 COIL_P 0 RELAY_COIL
R1 VCC COIL_P 100
"""
    g_sp, sp_audit, sp_fixes = parse_schematic(spice_src, fmt="spice", auto_fix=True)
    assert "K_RELAY1" in g_sp.components
    assert any(f["code"] == "UNCLAMPED_INDUCTIVE_LOAD" for f in sp_fixes)
    assert sp_audit.error_count == 0

    # 3. KiCad .net netlist with USB-C missing CC pulldowns
    kicad_net = """
(export (version "E")
  (components
    (comp (ref "J1") (value "USB_C") (footprint "Connector_USB:USB_C_Receptacle"))
    (comp (ref "U1") (value "ATtiny1614") (footprint "Package_SO:SOIC-8"))
  )
  (nets
    (net (code "1") (name "VBUS") (node (ref "J1") (pin "1")) (node (ref "U1") (pin "1")))
    (net (code "2") (name "GND") (node (ref "J1") (pin "2")) (node (ref "U1") (pin "8")))
    (net (code "3") (name "CC1") (node (ref "J1") (pin "3")))
    (net (code "4") (name "CC2") (node (ref "J1") (pin "4")))
  )
)
"""
    g_net, net_audit, net_fixes = parse_schematic(kicad_net, fmt="kicad_net", auto_fix=True)
    assert any(f["code"] == "MISSING_USBC_CC_PULLDOWN" for f in net_fixes)
    assert any(f["code"] == "MISSING_DECOUPLING_CAP" for f in net_fixes)
    assert net_audit.error_count == 0

    # 4. KiCad .kicad_sch S-expression
    kicad_sch = """
(kicad_sch (version 20231120)
  (symbol (lib_id "Device:R") (at 100 100 0)
    (property "Reference" "R1") (property "Value" "10k") (property "Footprint" "Resistor_SMD:R_0603")
    (pin "1" (net "VCC_3V3")) (pin "2" (net "SIG_A"))
  )
  (symbol (lib_id "Device:C") (at 120 100 0)
    (property "Reference" "C1") (property "Value" "100nF") (property "Footprint" "Capacitor_SMD:C_0603")
    (pin "1" (net "VCC_3V3")) (pin "2" (net "GND"))
  )
)
"""
    g_sch, _, _ = parse_schematic(kicad_sch, fmt="kicad_sch")
    assert {"R1", "C1"}.issubset(set(g_sch.components.keys()))


def test_rev11_pin_diagram_png_pdf_csv_and_library(tmp_path: Path) -> None:
    spec = {
        "part_number": "TMC2209_TEST",
        "package": "QFN-16",
        "pitch": 0.5,
        "body_size": 4.0,
        "thermal_pad": "GND",
        "pins": {
            "1": "VM",
            "2": "GND",
            "3": "A1",
            "4": "OA1",
            "5": "BRA",
            "6": "OA2",
            "7": "OB1",
            "8": "BRB",
            "9": "OB2",
            "10": "VCC_IO",
            "11": "STEP",
            "12": "DIR",
            "13": "EN",
            "14": "UART_TX",
            "15": "ISEN_A",
            "16": "CLK",
        },
    }
    lib_file = tmp_path / "custom_lib.json"

    png_bytes = build_synthetic_pinout_png(spec)
    comp_png, ext_png = import_pin_diagram(
        png_bytes, filename="tmc2209.png", comp_id="U1", save_to_library=True, library_path=lib_file
    )
    assert ext_png.source_format == "png"
    assert "thermal_via_ep" in comp_png.special_layout_rules
    assert "kelvin_sense_pair" in comp_png.special_layout_rules
    assert ext_png.pin_roles["15"] == "SENSE"
    assert ext_png.geometry_hints.get("img_width") == 256
    assert "TMC2209_TEST" in load_custom_library(lib_file)

    pdf_bytes = build_synthetic_pinout_pdf(spec)
    comp_pdf, ext_pdf = import_pin_diagram(pdf_bytes, filename="tmc2209.pdf", comp_id="U2")
    assert ext_pdf.source_format == "pdf"
    assert len(comp_pdf.pads) >= 16

    csv_text = "Pin,Name\n1,VM\n2,GND\n3,AOUT1\n4,AISEN\n5,VCC\n6,SDA\n7,SCL\n8,GND\nEP,GND\n"
    comp_csv, ext_csv = import_pin_diagram(csv_text, filename="driver.csv", comp_id="U3")
    assert ext_csv.source_format == "csv"
    assert "high_di_dt_loop" in comp_csv.special_layout_rules


def test_rev11_step_and_dxf_enclosure_zmax_and_b10() -> None:
    # 1. Explicit 3D Z-collision unit check
    cfg = BoardConfig.from_yaml(ROOT_DIR / "configs" / "default_2layer.yaml")
    ind_tall = make_smd_passive("L1", "10uH", "inductor", "1206", 1, 2, x=12.0, y=12.0)
    r_ok = make_smd_passive("R1", "10k", "res", "0402", 1, 2, x=22.0, y=22.0)
    z_graph = NetlistGraph(
        board_name="z_test",
        width_mm=30.0,
        height_mm=30.0,
        components={"L1": ind_tall, "R1": r_ok},
        nets={1: NetInfo(1, "A", [("L1", "1"), ("R1", "1")]), 2: NetInfo(2, "GND", [("L1", "2"), ("R1", "2")], is_power_ground=True)},
        z_max_regions=[{"x0": 8.0, "y0": 8.0, "x1": 16.0, "y1": 16.0, "z_max_mm": 1.15, "label": "RIB_LOW"}],
    )
    bad_state = BoardState(netlist=z_graph, config=cfg)
    z_errs = check_enclosure_collisions(bad_state)
    assert len(z_errs) == 1
    assert z_errs[0].error_type == "enclosure_z_collision"

    # 2. DXF enclosure parse
    dxf_text = (
        "0\nSECTION\n2\nENTITIES\n"
        "0\nLWPOLYLINE\n8\nOUTLINE\n"
        "10\n0.0\n20\n0.0\n10\n30.0\n20\n0.0\n10\n30.0\n20\n30.0\n10\n0.0\n20\n30.0\n"
        "0\nCIRCLE\n8\nBOSS_M2\n10\n4.0\n20\n4.0\n40\n1.0\n"
        "0\nLWPOLYLINE\n8\nZ_MAX_1.2\n10\n10.0\n20\n10.0\n10\n20.0\n20\n20.0\n"
        "0\nENDSEC\n0\nEOF\n"
    )
    dxf_enc = parse_dxf_enclosure(dxf_text)
    assert dxf_enc.board_width_mm == 30.0
    assert len(dxf_enc.mounting_bosses) == 1
    assert dxf_enc.z_max_regions[0]["z_max_mm"] == 1.2

    # 3. B10: step_enclosure_keychain full end-to-end run
    step_str = build_sample_enclosure_step()
    enc = parse_step_enclosure(step_str)
    assert len(enc.outline_polygon) == 8
    assert len(enc.mounting_bosses) == 2
    assert len(enc.port_windows) == 3
    assert len(enc.z_max_regions) == 2

    g10 = build_preset_board("step_enclosure_keychain")
    res10 = design_pcb_world_model(g10, cfg, num_candidates=4, max_iterations=5)
    assert res10.success is True
    assert res10.drc_errors == []
    s10 = res10.board_state.to_summary_dict()
    assert s10["routed_nets"] == s10["total_nets"]
    assert s10["enclosure_collisions"] == []
    assert s10["dfm_audit"]["passed"] is True

    step_out = generate_step_assembly(res10.board_state, mask_theme="black")
    diag = validate_step_structure(step_out)
    assert diag["valid"] is True
    assert diag["dangling_refs"] == []


def test_rev11_b11_datasheet_pinout_ingest_end_to_end() -> None:
    cfg = BoardConfig.from_yaml(ROOT_DIR / "configs" / "default_2layer.yaml")
    g11 = build_preset_board("datasheet_pinout_ingest")
    erc = audit_schematic_craftsmanship(g11)
    assert erc.error_count == 0
    assert erc.warning_count == 0
    assert "thermal_via_ep" in g11.components["U1"].special_layout_rules
    assert "high_di_dt_loop" in g11.components["U1"].special_layout_rules

    res11 = design_pcb_world_model(g11, cfg, num_candidates=4, max_iterations=5)
    assert res11.success is True
    assert res11.drc_errors == []
    assert res11.board_state.dfm_audit["passed"] is True


def test_rev11_b12_lock_preserving_reopt_and_fastapi_endpoints() -> None:
    pcb_text = build_legacy_high_via_kicad_pcb()
    opt = optimize_imported_board(
        source=pcb_text,
        lock_ids=["J1", "J2", "U1", "MH1", "MH2"],
        preserve_existing_locks=True,
        num_candidates=4,
        max_iterations=5,
    )
    res = opt["inference_result"]
    diff = opt["diff"]
    assert res.success is True
    assert diff.lock_preservation_verified is True
    assert diff.via_reduction_pct >= 30.0
    assert diff.after.drc_errors == 0
    assert diff.after.dfm_violations == 0
    assert "Optimization Metric" in diff.markdown_report

    client = TestClient(app)
    home = client.get("/")
    assert home.status_code == 200
    assert "REV 11" in home.text
    presets = client.get("/api/presets?include_rev11=true")
    assert presets.status_code == 200
    ids = {p["id"] for p in presets.json()}
    assert {"step_enclosure_keychain", "datasheet_pinout_ingest", "legacy_kicad_reoptimize"}.issubset(ids)

    sch_resp = client.post(
        "/api/import-schematic",
        json={"schematic_text": "R1 VCC SDA 4.7k\nC1 VCC 0 100nF\n", "fmt": "spice", "auto_fix": True},
    )
    assert sch_resp.status_code == 200
    assert sch_resp.json()["board"]["board_name"] == "imported_schematic"

    png_b64 = base64.b64encode(
        build_synthetic_pinout_png(
            {
                "part_number": "DRV_API",
                "package": "QFN-16",
                "pitch": 0.5,
                "body_size": 4.0,
                "thermal_pad": "GND",
                "pins": {str(i): f"P{i}" for i in range(1, 17)},
            }
        )
    ).decode("ascii")
    pin_resp = client.post(
        "/api/import-pin-diagram",
        json={"content_base64": png_b64, "filename": "drv.png", "comp_id": "U9", "save_to_library": False},
    )
    assert pin_resp.status_code == 200
    assert "thermal_via_ep" in pin_resp.json()["special_layout_rules"]

    enc_resp = client.post(
        "/api/import-step-enclosure",
        json={"enclosure_text": build_sample_enclosure_step(), "filename": "case.step"},
    )
    assert enc_resp.status_code == 200
    assert len(enc_resp.json()["enclosure"]["outline_polygon"]) == 8

    reopt_resp = client.post(
        "/api/reoptimize-board",
        json={"kicad_pcb_text": pcb_text, "lock_ids": ["J1", "J2", "U1", "MH1", "MH2"], "candidates": 4},
    )
    assert reopt_resp.status_code == 200
    reopt_json = reopt_resp.json()
    assert reopt_json["success"] is True
    assert reopt_json["optimization_diff"]["lock_preservation_verified"] is True
    assert reopt_json["optimization_diff"]["improvements"]["via_reduction_pct"] >= 30.0

    diff_md = client.get("/api/export-diff-report")
    assert diff_md.status_code == 200
    assert "Before-vs-After Optimization Diff Report" in diff_md.text
