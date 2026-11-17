"""Revision 14 tests: Multi-Zone Rigid-Flex Substrates, Cross-Hatched 85-Ohm PCIe Gen 4 Router,
Skew Matching, and PDN Decoupling Mounting Inductance Verifier + Under-BGA Migration.
"""

from __future__ import annotations

import importlib
from pathlib import Path
import sys

from fastapi.testclient import TestClient

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from env.board_state import BoardState, TraceSegment, Via
from env.config import BoardConfig
from env.kicad_exporter import serialize_kicad_pcb
from env.metrics import (
    compute_diff_pair_z_diff_ohm,
    compute_hatch_em_factors,
    compute_pdn_decoupling_inductance,
    solve_diff_pair_geometry,
)
from env.sexpr_parser import (
    build_preset_board,
    build_rev14_pdn_inductance_trap,
)
from env.step_exporter import generate_step_assembly, validate_step_structure
from models.world.neural_router import _apply_trombone_skew_match, route_differential_pair
from ui.server import app
from verifier.fast_clearance import check_pdn_decoupling, check_rigid_flex_rules

_inf_mod = importlib.import_module("scripts.04_run_inference")
design_pcb_world_model = _inf_mod.design_pcb_world_model


def test_rev14_multi_zone_rigid_flex_substrate_and_keepout_drc() -> None:
    cfg = BoardConfig.from_yaml(ROOT_DIR / "configs" / "default_8layer.yaml")
    g15 = build_preset_board("ultimate_rigid_flex_8l_hdi_pcie_gan_adc")
    state = BoardState(netlist=g15, config=cfg)

    assert state.has_rigid_flex is True
    assert state.zone_map.shape == (state.H, state.W)
    assert bool((state.zone_map == 1).any())
    assert bool((state.zone_map == 0).any())
    assert state.zone_at_mm(10.0, 37.5).zone_type == "dynamic_flex"
    assert state.zone_at_mm(10.0, 20.0).zone_type == "rigid"

    # Non-flex layers (L2..L8) are keepout-locked inside DynamicFlexZone
    gy_flex = int(round(37.5 / state.res))
    gx_flex = int(round(10.0 / state.res))
    assert bool(state.keepout_layers[0, gy_flex, gx_flex]) is False
    assert bool(state.keepout_layers[2, gy_flex, gx_flex]) is True

    # KiCad export emits cross-hatched zone + bend keepout
    pcb_txt = serialize_kicad_pcb(state)
    assert "(mode hatch)" in pcb_txt
    assert "(footprints not_allowed)" in pcb_txt

    # DRC catches illegal component, via, and inner-layer trace inside DynamicFlexZone
    bad = BoardState(netlist=g15.clone(), config=cfg)
    bad.netlist.components["U1"].x = 15.0
    bad.netlist.components["U1"].y = 37.5
    bad.routed_vias[6] = [Via(6, 12.0, 37.5, 0.30, 0.15, 0, 1, "micro")]
    bad.routed_traces[6] = [TraceSegment(6, 2, 10.0, 35.0, 10.0, 40.0, 0.13)]
    errs = {e.error_type for e in check_rigid_flex_rules(bad)}
    assert "flex_zone_component_violation" in errs
    assert "flex_zone_via_violation" in errs
    assert "flex_zone_layer_violation" in errs


def test_rev14_cross_hatched_impedance_solver_and_seam_compensation() -> None:
    c_ratio, l_ratio = compute_hatch_em_factors(0.50)
    assert c_ratio < 1.0
    assert l_ratio > 1.0

    w_rigid, s_rigid = solve_diff_pair_geometry(85.0, er=4.5, h_mm=0.10, hatch_fill=1.0)
    z_rigid = compute_diff_pair_z_diff_ohm(w_rigid, s_rigid, er=4.5, h_mm=0.10, hatch_fill=1.0)
    assert abs(z_rigid - 85.0) <= 4.25

    # Using rigid geometry uncompensated over 50% cross-hatched polyimide flex spikes > 85 Ohm + 5%
    z_uncomp_flex = compute_diff_pair_z_diff_ohm(w_rigid, s_rigid, er=3.4, h_mm=0.050, hatch_fill=0.50)
    assert z_uncomp_flex > 85.0 * 1.05

    # Solver compensates with wider flex width to hold 85 Ohm +- 5%
    w_flex, s_flex = solve_diff_pair_geometry(85.0, er=3.4, h_mm=0.050, hatch_fill=0.50)
    z_flex = compute_diff_pair_z_diff_ohm(w_flex, s_flex, er=3.4, h_mm=0.050, hatch_fill=0.50)
    assert w_flex > w_rigid
    assert abs(z_flex - 85.0) <= 2.0


def test_rev14_coupled_diff_pair_router_and_skew_matching() -> None:
    cfg = BoardConfig.from_yaml(ROOT_DIR / "configs" / "default_8layer.yaml")
    g15 = build_preset_board("ultimate_rigid_flex_8l_hdi_pcie_gan_adc")
    state = BoardState(netlist=g15, config=cfg)

    sp, vp, sn, vn, ok = route_differential_pair(state, 6, 7, z_diff_target=85.0)
    assert ok is True
    assert len(sp) >= 5 and len(sn) >= 5
    # Zero vias inside DynamicFlexZone (y in [34, 41.5])
    for v in vp + vn:
        assert not (34.0 <= v.y_mm <= 41.5)

    # Flex segments are wider than rigid segments for 85-Ohm compensation
    rigid_ws = [s.width_mm for s in sp if s.y2_mm < 33.4]
    flex_ws = [s.width_mm for s in sp if 34.1 < 0.5 * (s.y1_mm + s.y2_mm) < 41.4]
    assert rigid_ws and flex_ws
    assert flex_ws[0] > rigid_ws[0]

    # Direct trombone meander length matching test
    short_segs = [TraceSegment(1, 0, 5.0, 10.0, 5.0, 20.0, 0.13)]
    mate_segs = [TraceSegment(2, 0, 5.25, 10.0, 5.25, 20.0, 0.13)]
    matched = _apply_trombone_skew_match(short_segs, mate_segs, delta_mm=0.85, flex_zones=[])
    new_len = sum(((s.x2_mm - s.x1_mm) ** 2 + (s.y2_mm - s.y1_mm) ** 2) ** 0.5 for s in matched)
    assert abs(new_len - 10.85) <= 0.02


def test_rev14_b15a_pdn_decoupling_inductance_trap_and_under_bga_migration() -> None:
    cfg = BoardConfig.from_yaml(ROOT_DIR / "configs" / "default_8layer.yaml")
    g15a, _ = build_rev14_pdn_inductance_trap()

    raw_state = BoardState(netlist=g15a, config=cfg)
    pre_pdn = compute_pdn_decoupling_inductance(raw_state)
    pre_errs = check_pdn_decoupling(raw_state)
    assert pre_pdn["pdn_passed"] is False
    assert pre_pdn["max_loop_inductance_nh"] > 3.0
    assert any(e.error_type == "pdn_excessive_loop_inductance" for e in pre_errs)

    res = design_pcb_world_model(g15a, cfg, num_candidates=4, max_iterations=5)
    assert res.success is True
    assert res.drc_errors == []

    summary = res.board_state.to_summary_dict()
    assert summary["routed_nets"] == summary["total_nets"]
    pdn = summary["pdn_decoupling"]
    assert pdn["pdn_passed"] is True
    assert pdn["max_loop_inductance_nh"] <= 1.2
    assert all(r["under_bga"] is True and r["cap_side"] == "bottom" for r in pdn["cap_reports"])

    hdi = summary["hdi_telemetry"]
    assert hdi["stacked_microvia_count"] == 0
    assert hdi["illegal_skip_via_count"] == 0


def test_rev14_b15_ultimate_rigid_flex_8l_hdi_pcie_gan_adc_and_api() -> None:
    cfg = BoardConfig.from_yaml(ROOT_DIR / "configs" / "default_8layer.yaml")
    g15 = build_preset_board("ultimate_rigid_flex_8l_hdi_pcie_gan_adc")

    res = design_pcb_world_model(
        g15, cfg, num_candidates=4, max_iterations=5, include_seed_candidate=True
    )
    assert res.success is True
    assert res.drc_errors == []

    summary = res.board_state.to_summary_dict()
    assert summary["routed_nets"] == summary["total_nets"]
    assert summary["layers"] == 8

    iso = summary["ground_return_isolation"]
    assert iso["continuous_gnd"] is True
    assert iso["gan_to_adc_separation_mm"] >= 20.0
    assert iso["j_return_overlap_pct"] == 0.0
    assert iso["isolation_passed"] is True

    broad = summary["broadside_crosstalk"]
    assert broad["broadside_passed"] is True
    assert broad["violation_count"] == 0

    hdi = summary["hdi_telemetry"]
    assert hdi["stacked_microvia_count"] == 0
    assert hdi["illegal_skip_via_count"] == 0

    rf = summary["rigid_flex_impedance"]
    assert rf["has_rigid_flex"] is True
    assert rf["all_pairs_passed"] is True
    assert len(rf["pairs"]) >= 2
    for p in rf["pairs"]:
        assert p["crosses_flex"] is True
        assert abs(p["rigid_z_diff_ohm"] - 85.0) <= 4.25
        assert abs(p["flex_z_diff_ohm"] - 85.0) <= 4.25
        assert p["skew_mm"] <= 0.125

    pdn = summary["pdn_decoupling"]
    assert pdn["pdn_passed"] is True
    assert pdn["max_loop_inductance_nh"] <= 1.2
    assert summary["dfm_audit"]["passed"] is True
    assert len(summary["design_rationale"]) >= 4

    step_txt = generate_step_assembly(res.board_state, mask_theme="green")
    diag = validate_step_structure(step_txt)
    assert diag["valid"] is True
    assert diag["dangling_refs"] == []

    # FastAPI endpoints for REV 14
    client = TestClient(app)
    home = client.get("/")
    assert home.status_code == 200
    assert "REV 14" in home.text

    presets_resp = client.get("/api/presets?include_rev14=true")
    assert presets_resp.status_code == 200
    assert "ultimate_rigid_flex_8l_hdi_pcie_gan_adc" in {p["id"] for p in presets_resp.json()}

    rf_resp = client.get("/api/rigid-flex-audit?board=ultimate_rigid_flex_8l_hdi_pcie_gan_adc&layers=8")
    assert rf_resp.status_code == 200
    rf_body = rf_resp.json()
    assert rf_body["has_rigid_flex"] is True
    assert rf_body["ground_topology"]["continuous_gnd"] is True
