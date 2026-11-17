"""Revision 10 tests: DFM/DFA Craftsmanship, Edge Fillets, Screw-Head Buffers, Teardrops, Anti-Tombstone, Acid Traps & Compromise Report."""

from __future__ import annotations

import importlib
import math
import sys
from pathlib import Path

from fastapi.testclient import TestClient

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from env.board_state import BoardState, TraceSegment, Via
from env.config import BoardConfig
from env.footprint_generator import make_mounting_hole, make_smd_passive
from env.kicad_exporter import serialize_kicad_pcb
from env.manufacturing_exporter import export_manufacturing_bundle
from env.sexpr_parser import NetInfo, NetlistGraph, SExprParser, build_preset_board
from env.step_exporter import generate_step_assembly, validate_step_structure
from models.placer.relaxation import relax_and_snap_to_grid
from models.world.dfm_craftsmanship import apply_dfm_craftsmanship
from ui.server import app
from verifier.dfm_auditor import audit_board_dfm, generate_dfm_report_markdown
from verifier.fast_clearance import verify_board

_inf_mod = importlib.import_module("scripts.04_run_inference")
design_pcb_world_model = _inf_mod.design_pcb_world_model


def test_rev10_board_corner_fillets_kicad_and_step() -> None:
    cfg = BoardConfig.from_yaml(ROOT_DIR / "configs" / "default_2layer.yaml")
    graph = build_preset_board("wearable_oled_keychain_26x26")
    res = design_pcb_world_model(
        netlist_graph=graph,
        board_config=cfg,
        num_candidates=4,
        max_iterations=5,
    )
    assert res.success is True
    assert res.board_state.corner_fillet_radius_mm >= 1.5

    pcb_text = res.kicad_pcb_text
    assert '(generator_version "rev10")' in pcb_text
    assert "(gr_arc " in pcb_text
    assert '"Edge.Cuts"' in pcb_text
    assert "(teardrops " in pcb_text
    assert "(island_removal_mode 1)" in pcb_text

    reparsed = SExprParser.parse_text(pcb_text)
    assert abs(reparsed.width_mm - 26.0) < 0.05
    assert abs(reparsed.height_mm - 26.0) < 0.05
    assert len(reparsed.components) == len(graph.components)

    step_text = generate_step_assembly(res.board_state, mask_theme="black")
    diag = validate_step_structure(step_text)
    assert diag["valid"] is True
    assert diag["dangling_refs"] == []
    # 2 NPTH holes + 4 rounded corner cylindrical fillets => >= 6 cylindrical surfaces
    assert diag["cylindrical_holes"] >= 6


def test_rev10_screw_head_stress_keepout_pushes_mlcc() -> None:
    cfg = BoardConfig.from_yaml(ROOT_DIR / "configs" / "default_2layer.yaml")
    mh1 = make_mounting_hole("MH1", diameter_mm=2.0, x=10.0, y=10.0)
    # Place ceramic MLCC inside M2 torque stress zone (R = (2.0 + 1.8) / 2 = 1.9mm)
    c_bad = make_smd_passive("C1", "100nF", "cap", package="0402", net1=1, net2=2, x=11.0, y=10.0)
    r_anchor = make_smd_passive("R1", "10k", "res", package="0603", net1=1, net2=2, x=25.0, y=25.0, is_locked=True)

    graph = NetlistGraph(
        board_name="screw_stress_test",
        width_mm=40.0,
        height_mm=40.0,
        components={"MH1": mh1, "C1": c_bad, "R1": r_anchor},
        nets={
            1: NetInfo(1, "VCC", [("C1", "1"), ("R1", "1")], is_power_ground=True),
            2: NetInfo(2, "GND", [("C1", "2"), ("R1", "2")], is_power_ground=True),
        },
    )

    pre_state = BoardState(netlist=graph, config=cfg)
    pre_audit = audit_board_dfm(pre_state)
    assert pre_audit["screw_stress_mlcc_violations"] == 1

    raw_place = {cid: (c.x, c.y, c.theta) for cid, c in graph.components.items()}
    legal = relax_and_snap_to_grid(raw_place, graph, cfg, iterations=50)
    post_state = BoardState.from_placement(legal, graph, cfg)
    post_audit = audit_board_dfm(post_state)
    assert post_audit["screw_stress_mlcc_violations"] == 0
    cx, cy, _ = legal["C1"]
    assert math.hypot(cx - 10.0, cy - 10.0) >= 1.9


def test_rev10_acid_trap_symmetric_entry_silkscreen_and_teardrops() -> None:
    cfg = BoardConfig.from_yaml(ROOT_DIR / "configs" / "default_2layer.yaml")
    r1 = make_smd_passive("R1", "10k", "res", package="0603", net1=1, net2=2, x=20.0, y=20.0, is_locked=True)
    r2 = make_smd_passive("R2", "10k", "res", package="0603", net1=1, net2=2, x=32.0, y=20.0, is_locked=True)
    # R1 pads are at (19.05, 20.0) and (20.95, 20.0); R2 pads at (31.05, 20.0) and (32.95, 20.0)
    graph = NetlistGraph(
        board_name="craftsmanship_unit",
        width_mm=50.0,
        height_mm=50.0,
        components={"R1": r1, "R2": r2},
        nets={
            1: NetInfo(1, "SIG_A", [("R1", "1"), ("R2", "1")]),
            2: NetInfo(2, "GND", [("R1", "2"), ("R2", "2")], is_power_ground=True),
        },
    )
    state = BoardState(netlist=graph, config=cfg)
    # Deliberately create side-entry on R1.1 (90 deg upward) + acute 45 deg hairpin acid trap!
    state.commit_trace(
        1,
        [
            TraceSegment(1, 0, 19.05, 20.0, 19.05, 14.0, 0.25),   # 90-deg side-entry on R1.1
            TraceSegment(1, 0, 19.05, 14.0, 23.05, 18.0, 0.25),   # 45-deg acute acid trap!
            TraceSegment(1, 0, 23.05, 18.0, 31.05, 18.0, 0.25),
            TraceSegment(1, 0, 31.05, 18.0, 31.05, 20.0, 0.25),   # 90-deg side-entry on R2.1
        ],
    )
    state.commit_trace(
        2,
        [
            TraceSegment(2, 0, 20.95, 20.0, 20.95, 26.0, 0.40),
            TraceSegment(2, 0, 20.95, 26.0, 32.95, 26.0, 0.40),
            TraceSegment(2, 0, 32.95, 26.0, 32.95, 20.0, 0.40),
        ],
    )

    pre_audit = audit_board_dfm(state)
    assert pre_audit["passed"] is False
    assert pre_audit["acid_trap_count"] >= 1
    assert pre_audit["asymmetric_pad_entries"] >= 2

    apply_dfm_craftsmanship(state)
    drc = verify_board(state)
    assert drc == []

    pcb_text = serialize_kicad_pcb(state)
    post_audit = audit_board_dfm(state, kicad_pcb_text=pcb_text)
    assert post_audit["acid_trap_count"] == 0
    assert post_audit["asymmetric_pad_entries"] == 0
    assert post_audit["screw_stress_mlcc_violations"] == 0
    assert post_audit["silkscreen_pad_overlaps"] == 0
    assert post_audit["teardrop_count"] >= 4
    assert post_audit["corner_fillet_valid"] is True
    assert post_audit["passed"] is True


def test_rev10_end_to_end_boards_bundle_and_fastapi_endpoints(tmp_path: Path) -> None:
    cfg = BoardConfig.from_yaml(ROOT_DIR / "configs" / "default_2layer.yaml")
    for board_name in ("mcu_sensor_node", "buck_converter", "wearable_oled_keychain_26x26"):
        graph = build_preset_board(board_name)
        res = design_pcb_world_model(
            netlist_graph=graph,
            board_config=cfg,
            num_candidates=4,
            max_iterations=5,
        )
        assert res.success is True
        assert res.drc_errors == []
        audit = res.board_state.dfm_audit
        assert audit is not None
        assert audit["passed"] is True, f"{board_name} failed DFM audit: {audit}"
        assert audit["acid_trap_count"] == 0
        assert audit["asymmetric_pad_entries"] == 0
        assert audit["screw_stress_mlcc_violations"] == 0
        assert audit["silkscreen_pad_overlaps"] == 0
        assert audit["teardrop_count"] > 0

        if board_name == "mcu_sensor_node":
            paths = export_manufacturing_bundle(res.board_state, tmp_path / "fab")
            assert "dfm_report" in paths and paths["dfm_report"].exists()
            md = paths["dfm_report"].read_text(encoding="utf-8")
            assert "Acceptance Gate Verification" in md
            assert "Transparent Engineering Compromises" in md

    client = TestClient(app)
    home = client.get("/")
    assert home.status_code == 200
    assert "REV 10" in home.text
    assert "dfm-panel" in home.text
    assert "btn-download-dfm" in home.text

    run_resp = client.post("/api/run", json={"board": "mcu_sensor_node", "candidates": 4})
    assert run_resp.status_code == 200
    run_data = run_resp.json()
    assert run_data["success"] is True
    assert run_data["board"]["dfm_audit"]["passed"] is True
    assert run_data["board"]["corner_fillet_radius_mm"] >= 1.5

    audit_resp = client.get("/api/dfm-audit?board=mcu_sensor_node")
    assert audit_resp.status_code == 200
    audit_json = audit_resp.json()
    assert audit_json["audit"]["passed"] is True
    assert "Transparent Engineering Compromises" in audit_json["dfm_report_md"]

    dl_resp = client.get("/api/export-dfm-report?board=mcu_sensor_node&download=1")
    assert dl_resp.status_code == 200
    assert "dfm_report.md" in dl_resp.headers.get("content-disposition", "")

    bundle_resp = client.get("/api/manufacturing_bundle?board=mcu_sensor_node")
    assert bundle_resp.status_code == 200
    bundle = bundle_resp.json()
    assert "dfm_report_md" in bundle and "dfm_audit" in bundle
