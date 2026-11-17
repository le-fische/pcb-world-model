"""Revision 12 tests: 8-Layer (2+4+2) HDI Stackup, Buried Core Vias, Staggered Microvias & B13 BGA Fanout."""

from __future__ import annotations

import importlib
import math
from pathlib import Path
import sys

from fastapi.testclient import TestClient
import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from env.board_state import (
    BoardState,
    TraceSegment,
    Via,
    compute_hdi_via_telemetry,
    get_legal_via_spans,
)
from env.config import BoardConfig
from env.footprint_generator import make_bga, make_smd_passive
from env.kicad_exporter import serialize_kicad_pcb
from env.sexpr_parser import NetInfo, NetlistGraph, SExprParser, build_preset_board
from env.step_exporter import generate_step_assembly, validate_step_structure
from models.world.neural_router import _astar_multilayer_python, octilinear_astar_route
from ui.server import app
from verifier.fast_clearance import check_hdi_via_rules, verify_board

_inf_mod = importlib.import_module("scripts.04_run_inference")
design_pcb_world_model = _inf_mod.design_pcb_world_model


def test_rev12_8layer_stackup_make_bga_and_kicad_roundtrip() -> None:
    cfg = BoardConfig.from_yaml(ROOT_DIR / "configs" / "default_8layer.yaml")
    assert cfg.layers == 8
    assert cfg.max_lamination_cycles == 2
    assert cfg.enforce_staggered_microvias is True
    assert math.isclose(cfg.microvia_stagger_offset_mm, 0.25, abs_tol=1e-6)

    legal = get_legal_via_spans(8, max_lamination_cycles=2, hdi_buildup_type="2+4+2")
    assert {(0, 1), (1, 2), (5, 6), (6, 7), (2, 5), (0, 7)} == legal
    assert (0, 2) not in legal
    assert (0, 3) not in legal
    assert (1, 4) not in legal

    # Parametric 0.8mm & 0.65mm pitch BGA builders
    bga64 = make_bga("U1", "ECP5-BGA64", {"A1": 1, "D4": 2, "H8": 1}, rows=8, cols=8, pitch=0.8, x=15.0, y=20.0)
    assert len(bga64.pads) == 64
    pad_ids = {p.pad_id for p in bga64.pads}
    assert {"A1", "D4", "H8"}.issubset(pad_ids)
    assert math.isclose(bga64.pads[0].width, 0.40, abs_tol=1e-4)

    bga16_fine = make_bga("U2", "ICE40-BGA16", [1] * 16, rows=4, cols=4, pitch=0.65, x=20.0, y=20.0)
    assert len(bga16_fine.pads) == 16
    assert math.isclose(bga16_fine.pads[0].width, 0.325, abs_tol=1e-4)

    # 8-layer BoardState + KiCad 8 roundtrip with microvias, buried core vias, and PTH
    g = NetlistGraph(
        board_name="hdi_roundtrip_test",
        width_mm=30.0,
        height_mm=30.0,
        layers=8,
        components={"U1": bga64},
        nets={
            1: NetInfo(1, "GND", [("U1", "A1"), ("U1", "H8")], is_power_ground=True),
            2: NetInfo(2, "SIG_A", [("U1", "D4")]),
        },
    )
    state = BoardState(netlist=g, config=cfg)
    assert state.num_layers == 8
    assert state.layer_names() == [
        "F.Cu",
        "In1.GND",
        "In2.SIG1",
        "In3.PWR1",
        "In4.GND2",
        "In5.SIG2",
        "In6.GND3",
        "B.Cu",
    ]
    state.commit_trace(
        2,
        [
            TraceSegment(2, 0, 14.6, 19.6, 15.0, 20.0, 0.10),
            TraceSegment(2, 1, 15.0, 20.0, 15.3, 20.0, 0.10),
            TraceSegment(2, 2, 15.3, 20.0, 18.0, 20.0, 0.10),
            TraceSegment(2, 5, 18.0, 20.0, 22.0, 20.0, 0.10),
        ],
        [
            Via(2, 15.0, 20.0, 0.30, 0.15, start_layer=0, end_layer=1, via_type="micro"),
            Via(2, 15.3, 20.0, 0.30, 0.15, start_layer=1, end_layer=2, via_type="micro"),
            Via(2, 18.0, 20.0, 0.45, 0.20, start_layer=2, end_layer=5, via_type="buried"),
        ],
    )
    kicad_txt = serialize_kicad_pcb(state)
    assert "(blind_buried_vias_allowed yes)" in kicad_txt
    assert '(via micro ' in kicad_txt
    assert '(via blind ' in kicad_txt
    assert '"In5.Cu"' in kicad_txt
    assert '"In6.Cu"' in kicad_txt

    reparsed = SExprParser.parse_text(kicad_txt)
    assert reparsed.layers == 8
    assert len(reparsed.raw_segments) == 4
    vtypes = {(v["start_layer"], v["end_layer"], v["via_type"]) for v in reparsed.raw_vias}
    assert (0, 1, "micro") in vtypes
    assert (1, 2, "micro") in vtypes
    assert (2, 5, "buried") in vtypes


def test_rev12_lamination_cycle_and_staggered_microvia_drc() -> None:
    cfg = BoardConfig.from_yaml(ROOT_DIR / "configs" / "default_8layer.yaml")
    r1 = make_smd_passive("R1", "10k", "res", "0402", 1, 2, x=10.0, y=10.0, compact=True)
    r2 = make_smd_passive("R2", "10k", "res", "0402", 1, 2, x=20.0, y=20.0, compact=True)
    g = NetlistGraph(
        board_name="hdi_drc_unit",
        width_mm=30.0,
        height_mm=30.0,
        layers=8,
        components={"R1": r1, "R2": r2},
        nets={
            1: NetInfo(1, "SIG_1", [("R1", "1"), ("R2", "1")]),
            2: NetInfo(2, "GND", [("R1", "2"), ("R2", "2")], is_power_ground=True),
        },
    )

    # 1. Illegal skip-via (0<->3) violates 2-cycle HDI lamination constraint
    s_skip = BoardState(netlist=g, config=cfg)
    s_skip.commit_trace(
        1,
        [],
        [Via(1, 15.0, 15.0, 0.30, 0.15, start_layer=0, end_layer=3, via_type="micro")],
    )
    skip_errs = check_hdi_via_rules(s_skip)
    assert any(e.error_type == "illegal_skip_via" for e in skip_errs)

    # 2. Concentric stacked microvias (0<->1 on top of 1<->2 at dist=0.10mm < 0.25mm)
    s_stack = BoardState(netlist=g, config=cfg)
    s_stack.commit_trace(
        1,
        [],
        [
            Via(1, 15.0, 15.0, 0.30, 0.15, start_layer=0, end_layer=1, via_type="micro"),
            Via(1, 15.1, 15.0, 0.30, 0.15, start_layer=1, end_layer=2, via_type="micro"),
        ],
    )
    stack_errs = check_hdi_via_rules(s_stack)
    assert any(e.error_type == "stacked_microvia_reliability_risk" for e in stack_errs)

    # 3. Valid staggered microvias (dist = 0.30mm >= 0.25mm) + buried core via (2<->5) -> 0 errors
    s_ok = BoardState(netlist=g, config=cfg)
    s_ok.commit_trace(
        1,
        [TraceSegment(1, 1, 15.0, 15.0, 15.3, 15.0, 0.10)],
        [
            Via(1, 15.0, 15.0, 0.30, 0.15, start_layer=0, end_layer=1, via_type="micro"),
            Via(1, 15.3, 15.0, 0.30, 0.15, start_layer=1, end_layer=2, via_type="micro"),
            Via(1, 18.0, 15.0, 0.45, 0.20, start_layer=2, end_layer=5, via_type="buried"),
        ],
    )
    assert verify_board(s_ok) == []
    hdi = compute_hdi_via_telemetry(s_ok)
    assert hdi["stacked_microvia_count"] == 0
    assert hdi["illegal_skip_via_count"] == 0
    assert hdi["buried_core_via_count"] == 1
    assert hdi["min_observed_stagger_mm"] >= 0.25


def test_rev12_router_staggered_microvia_enforcement_and_python_parity() -> None:
    cfg = BoardConfig.from_yaml(ROOT_DIR / "configs" / "default_8layer.yaml")
    r1 = make_smd_passive("R1", "100R", "res", "0402", 1, 2, x=6.0, y=10.0, compact=True)
    r2 = make_smd_passive("R2", "100R", "res", "0402", 1, 2, x=18.0, y=10.0, compact=True)
    g = NetlistGraph(
        board_name="stagger_bridge_parity",
        width_mm=24.0,
        height_mm=20.0,
        layers=8,
        components={"R1": r1, "R2": r2},
        nets={
            1: NetInfo(1, "SIG_ESCAPE", [("R1", "1"), ("R2", "1")]),
            2: NetInfo(2, "FENCE_NET", [("R1", "2"), ("R2", "2")]),
        },
    )
    state = BoardState(netlist=g, config=cfg)
    # Fence F.Cu (layer 0) vertically across x=12.0mm so SIG_ESCAPE must drop 0->1->2 (staggered) -> 1->0
    state.commit_trace(
        2,
        [TraceSegment(2, 0, 12.0, 1.5, 12.0, 18.5, 0.25)],
        [],
    )
    cf = np.zeros((8, state.H, state.W), dtype=np.float32)
    segs, vias, ok = octilinear_astar_route(state, 1, cf, allow_soft_collision=False)
    assert ok is True
    assert any(s.layer == 2 for s in segs)
    state.commit_trace(1, segs, vias)
    assert verify_board(state) == []
    hdi = compute_hdi_via_telemetry(state)
    assert hdi["microvia_count"] >= 4
    assert hdi["stacked_microvia_count"] == 0
    assert hdi["illegal_skip_via_count"] == 0
    assert hdi["min_observed_stagger_mm"] is not None and hdi["min_observed_stagger_mm"] >= 0.25

    # Verify Python fallback atomic staggered microvia bridge parity on a small synthetic grid
    blocked = np.zeros((8, 60, 60), dtype=np.uint8)
    blocked[0, :, 30] = 1  # F.Cu fence
    via_blocked = np.zeros((8, 60, 60), dtype=np.uint8)
    soft_pen = np.zeros((8, 60, 60), dtype=np.float32)
    cost_f = np.zeros((8, 60, 60), dtype=np.float32)
    occ = np.zeros((8, 60, 60), dtype=np.int32)
    allowed = np.array([1, 0, 1, 0, 0, 1, 0, 1], dtype=np.uint8)
    trans = np.full((8, 8), -1.0, dtype=np.float32)
    for la, lb, c in [(0, 1, 3.5), (1, 0, 3.5), (1, 2, 3.5), (2, 1, 3.5)]:
        trans[la, lb] = c
    py_path = _astar_multilayer_python(
        8, 60, 60, [(0, 15, 30)], [(0, 45, 30)],
        blocked, via_blocked, soft_pen, cost_f, occ, allowed, trans,
        net_id=1, min_stagger_cells=2.5,
    )
    assert py_path is not None
    # Verify dogbone step on layer 1 is >= 2.5 cells between 0->1 and 1->2
    l1_pts = [(x, y) for l, x, y in py_path if l == 1]
    assert len(l1_pts) >= 2
    assert math.hypot(l1_pts[-1][0] - l1_pts[0][0], l1_pts[-1][1] - l1_pts[0][1]) >= 2.5 - 1e-4


def test_rev12_b13_hdi_fpga_bga_fanout_8l_end_to_end_and_api() -> None:
    cfg = BoardConfig.from_yaml(ROOT_DIR / "configs" / "default_8layer.yaml")
    g13 = build_preset_board("hdi_fpga_bga_fanout_8l")
    assert g13.layers == 8
    assert len(g13.components["U1"].pads) == 64

    res = design_pcb_world_model(g13, cfg, num_candidates=4, max_iterations=5)
    assert res.success is True
    assert res.drc_errors == []

    summary = res.board_state.to_summary_dict()
    assert summary["layers"] == 8
    assert summary["routed_nets"] == summary["total_nets"]
    hdi = summary["hdi_telemetry"]
    assert hdi["stackup_type"] == "2+4+2"
    assert hdi["max_lamination_cycles"] == 2
    assert hdi["microvia_count"] > 0
    assert hdi["stacked_microvia_count"] == 0
    assert hdi["illegal_skip_via_count"] == 0
    assert hdi["min_observed_stagger_mm"] is not None and hdi["min_observed_stagger_mm"] >= 0.25
    assert "In2.SIG1" in hdi["layers_used"]
    assert summary["dfm_audit"]["passed"] is True
    assert summary["signal_integrity"]["si_thermal_status"] == "PASS"

    step_txt = generate_step_assembly(res.board_state)
    step_diag = validate_step_structure(step_txt)
    assert step_diag["valid"] is True
    assert step_diag["dangling_refs"] == []

    # FastAPI endpoints with 8-layer HDI support
    client = TestClient(app)
    presets_resp = client.get("/api/presets?include_rev12=true")
    assert presets_resp.status_code == 200
    preset_ids = {p["id"] for p in presets_resp.json()}
    assert "hdi_fpga_bga_fanout_8l" in preset_ids

    si_resp = client.get("/api/signal-integrity?board=hdi_fpga_bga_fanout_8l&layers=8")
    assert si_resp.status_code == 200
    assert si_resp.json()["layers"] == 8

    run_resp = client.post(
        "/api/run",
        json={"board": "hdi_fpga_bga_fanout_8l", "layers": 8, "candidates": 2, "max_iterations": 5},
    )
    assert run_resp.status_code == 200
    body = run_resp.json()
    assert body["success"] is True
    assert body["error_count"] == 0
    assert body["board"]["layers"] == 8
    assert body["board"]["hdi_telemetry"]["stacked_microvia_count"] == 0
    assert body["board"]["hdi_telemetry"]["illegal_skip_via_count"] == 0
