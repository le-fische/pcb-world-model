"""Revision 8 tests: Verilog RTL synthesis, iverilog/Verilator sim, .pcf pin-swap co-design, OpenLane GDSII, and B9 PCB."""

from __future__ import annotations

import base64
import importlib
from pathlib import Path
import sys

from fastapi.testclient import TestClient

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from env.config import BoardConfig
from env.manufacturing_exporter import export_manufacturing_bundle
from env.sexpr_parser import build_preset_board
from models.composer.schematic_composer import compose_netlist_from_prompt, verify_erc
from models.silicon import (
    assess_silicon_feasibility,
    compile_asic_openlane,
    compose_verilog_module,
    count_ratnest_crossings,
    optimize_fpga_pcb_codesign,
)

_inf_mod = importlib.import_module("scripts.04_run_inference")
design_pcb_world_model = _inf_mod.design_pcb_world_model
run_silicon_codesign_for_graph = _inf_mod.run_silicon_codesign_for_graph


def test_rev8_physics_boundary_digital_vs_analog() -> None:
    digital = assess_silicon_feasibility(
        "Need a custom 8-channel high-speed WS2812 + hardware quadrature encoder controller over SPI"
    )
    assert digital.synthesizable is True
    assert digital.template_id == "quad_ws2812_spi"
    assert "ws2812_8ch_nrz_driver" in digital.digital_capabilities

    analog_limit = assess_silicon_feasibility(
        "Synthesize a 5A switching buck converter and MEMS accelerometer in Verilog"
    )
    assert analog_limit.synthesizable is False
    assert analog_limit.domain == "analog_physics_limit"
    assert analog_limit.verilog_ams_stub is not None
    assert "disciplines.vams" in analog_limit.verilog_ams_stub


def test_rev8_verilog_synthesis_and_iverilog_simulation() -> None:
    for tpl, min_checks in (
        ("quad_ws2812_spi", 4),
        ("multi_pwm_controller", 1),
        ("spi_i2s_bridge", 1),
        ("safety_interlock", 3),
    ):
        bundle = compose_verilog_module(template_id=tpl, clock_hz=48_000_000, run_simulation=True)
        assert bundle.simulation.passed is True, f"{tpl} failed: {bundle.simulation.errors}"
        assert bundle.simulation.lint_passed is True, f"{tpl} lint failed: {bundle.simulation.warnings}"
        assert bundle.simulation.checks_passed >= min_checks
        assert bundle.simulation.errors == []
        assert "SIM_PASS" in bundle.simulation.stdout


def test_rev8_fpga_pcf_pinswap_reduces_crossings_and_preserves_erc() -> None:
    graph = build_preset_board("fpga_custom_protocol_bridge")
    assert verify_erc(graph) == []

    cx_raw, wl_raw = count_ratnest_crossings(graph, fpga_cid="U1")
    assert cx_raw > 20

    verilog_bundle = compose_verilog_module(template_id="quad_ws2812_spi", run_simulation=True)
    codesign = optimize_fpga_pcb_codesign(graph, fpga_cid="U1", verilog_bundle=verilog_bundle)

    assert verify_erc(graph) == []
    assert codesign.timing_passed is True
    assert codesign.achieved_fmax_mhz >= codesign.target_clock_mhz
    assert codesign.crossings_after < codesign.crossings_before
    assert codesign.crossing_reduction_pct >= 30.0
    assert len(codesign.pin_swaps) >= 5
    assert "set_io " in codesign.optimized_pcf
    assert b"\x7e\xaa\x99\x7e" in codesign.bitstream_bytes


def test_rev8_openlane_sky130_asic_and_gdsii_export() -> None:
    bundle = compose_verilog_module(template_id="quad_ws2812_spi", run_simulation=False)
    asic = compile_asic_openlane(bundle, pdk="sky130A", package="QFN-48")
    assert asic.pdk == "sky130A"
    assert asic.drc_clean is True and asic.lvs_clean is True
    assert asic.config_json["DESIGN_NAME"] == "quad_ws2812_spi"
    assert asic.package_pinout_json["package"] == "QFN-48"
    assert len(asic.gds_bytes) > 512
    # Calma GDSII HEADER record (0x0006, 0x00, 0x02, version=600 -> 0x0258)
    assert asic.gds_bytes[:6] == bytes.fromhex("000600020258")


def test_rev8_b9_end_to_end_zero_drc_and_fab_bundle(tmp_path: Path) -> None:
    b9_prompt = (
        "Need a custom 8-channel high-speed WS2812 + hardware quadrature encoder controller "
        "that talks over SPI to an ESP32-C3; no single off-the-shelf IC exists."
    )
    graph = compose_netlist_from_prompt(b9_prompt)
    assert graph.board_name == "fpga_custom_protocol_bridge"
    assert verify_erc(graph) == []

    silicon = run_silicon_codesign_for_graph(graph, prompt=b9_prompt, include_asic=True)
    assert silicon is not None
    assert silicon["simulation"]["passed"] is True
    assert silicon["crossing_reduction_pct"] >= 30.0

    cfg = BoardConfig.from_yaml(ROOT_DIR / "configs" / "default_4layer.yaml")
    res = design_pcb_world_model(
        netlist_graph=graph,
        board_config=cfg,
        output_path=tmp_path / "b9.kicad_pcb",
        num_candidates=4,
        max_iterations=5,
    )
    summary = res.board_state.to_summary_dict()
    assert res.success is True
    assert len(res.drc_errors) == 0
    assert summary["routed_nets"] == summary["total_nets"] == 46
    assert summary["signal_integrity"]["si_thermal_status"] == "PASS"

    bundle_paths = export_manufacturing_bundle(
        res.board_state, tmp_path / "fab_b9", silicon_bundle=silicon
    )
    for key in (
        "kicad_pcb",
        "bom_csv",
        "positions_csv",
        "verilog_rtl",
        "verilog_tb",
        "pcf",
        "bitstream_asc",
        "bitstream_bin",
        "asic_config",
        "asic_pinout",
        "asic_gds",
        "silicon_report",
    ):
        assert key in bundle_paths and bundle_paths[key].exists()


def test_rev8_fastapi_silicon_endpoints_and_compose() -> None:
    from ui.server import app

    client = TestClient(app)

    presets = client.get("/api/presets?include_rev8=true")
    assert presets.status_code == 200
    assert any(p["id"] == "fpga_custom_protocol_bridge" for p in presets.json())

    synth = client.post(
        "/api/silicon/synthesize",
        json={
            "prompt": "8-channel WS2812 and quadrature encoder SPI bridge",
            "target_id": "ice40up5k_sg48",
            "include_asic": True,
        },
    )
    assert synth.status_code == 200
    sdata = synth.json()
    assert sdata["synthesizable"] is True
    assert sdata["simulation"]["passed"] is True
    assert sdata["timing_passed"] is True
    gds_raw = base64.b64decode(sdata["gds_base64"])
    assert gds_raw[:6] == bytes.fromhex("000600020258")

    codesign = client.post(
        "/api/silicon/codesign",
        json={
            "board": "fpga_custom_protocol_bridge",
            "target_id": "ice40up5k_sg48",
            "include_asic": True,
            "candidates": 3,
            "max_iterations": 5,
        },
    )
    assert codesign.status_code == 200
    cdata = codesign.json()
    assert cdata["success"] is True
    assert cdata["error_count"] == 0
    assert cdata["silicon"]["crossing_reduction_pct"] >= 30.0

    exported = client.get("/api/export-silicon")
    assert exported.status_code == 200
    assert exported.json()["module_name"] == "quad_ws2812_spi"
