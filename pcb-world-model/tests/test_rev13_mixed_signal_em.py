"""Revision 13 tests: Mixed-Signal Domain Partitioning, Continuous GND Return Physics (J_return),
3D Z-Axis Broadside Crosstalk, Star-Ground Refusal, and 80%-Locked Thermal-vs-Signal Inpainting.
"""

from __future__ import annotations

import importlib
import math
from pathlib import Path
import sys

from fastapi.testclient import TestClient

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from env.board_state import BoardState, TraceSegment
from env.config import BoardConfig
from env.importers import load_adversarial_inpainting_state, solve_thermal_signal_inpainting
from env.metrics import compute_broadside_crosstalk_risk, compute_ground_return_isolation
from env.sexpr_parser import (
    build_preset_board,
    build_rev13_broadside_trap,
    build_rev13_star_ground_trap,
    build_rev13_thermal_inpainting_trap,
)
from env.step_exporter import generate_step_assembly, validate_step_structure
from models.placer.relaxation import relax_and_snap_to_grid
from ui.server import app
from verifier.fast_clearance import check_broadside_crosstalk, validate_ground_topology

_inf_mod = importlib.import_module("scripts.04_run_inference")
design_pcb_world_model = _inf_mod.design_pcb_world_model


def test_rev13_noise_domain_tagging_and_cross_domain_repulsion_blind() -> None:
    cfg = BoardConfig.from_yaml(ROOT_DIR / "configs" / "default_8layer.yaml")
    g14 = build_preset_board("mixed_signal_gan_adc_30x50")

    assert g14.components["U1"].noise_domain == "power_switching"
    assert g14.components["L1"].noise_domain == "power_switching"
    assert g14.components["U2"].noise_domain == "analog_precision"
    assert g14.components["U3"].noise_domain == "analog_precision"
    assert g14.components["U4"].noise_domain == "digital_hs"
    assert g14.nets[4].noise_domain == "power_switching"  # SW_NODE
    assert g14.nets[11].noise_domain == "analog_precision"  # ADC_AINP

    # Collapse all unlocked components to board center and verify cross-domain repulsion separates GaN & ADC >= 20mm
    collapsed = {
        cid: ((c.x, c.y, c.theta) if c.is_locked else (15.0, 25.0, 0))
        for cid, c in g14.components.items()
    }
    relaxed = relax_and_snap_to_grid(collapsed, g14, cfg, blind_anneal=True)
    ux, uy, _ = relaxed["U1"]
    ax, ay, _ = relaxed["U2"]
    assert math.hypot(ux - ax, uy - ay) >= 20.0


def test_rev13_b14_mixed_signal_gan_adc_30x50_end_to_end_seeded_and_blind() -> None:
    cfg = BoardConfig.from_yaml(ROOT_DIR / "configs" / "default_8layer.yaml")
    g14 = build_preset_board("mixed_signal_gan_adc_30x50")

    for include_seed in (True, False):
        res = design_pcb_world_model(
            g14,
            cfg,
            num_candidates=4,
            max_iterations=5,
            include_seed_candidate=include_seed,
        )
        assert res.success is True
        assert res.drc_errors == []

        summary = res.board_state.to_summary_dict()
        assert summary["routed_nets"] == summary["total_nets"]
        assert summary["layers"] == 8

        iso = summary["ground_return_isolation"]
        assert iso["has_mixed_signal_domains"] is True
        assert iso["continuous_gnd"] is True
        assert iso["gan_to_adc_separation_mm"] >= 20.0
        assert iso["j_return_overlap_pct"] == 0.0
        assert iso["switching_under_analog_count"] == 0
        assert iso["isolation_passed"] is True

        broad = summary["broadside_crosstalk"]
        assert broad["broadside_passed"] is True
        assert broad["violation_count"] == 0

        topo = summary["ground_topology"]
        assert topo["valid"] is True
        assert topo["continuous_gnd"] is True
        assert topo["slot_antenna_loops"] == 0

        assert summary["dfm_audit"]["passed"] is True
        assert len(summary["design_rationale"]) >= 3

    step_txt = generate_step_assembly(res.board_state, mask_theme="green")
    diag = validate_step_structure(step_txt)
    assert diag["valid"] is True
    assert diag["dangling_refs"] == []


def test_rev13_b14a_z_axis_broadside_crosstalk_detection_and_avoidance() -> None:
    cfg = BoardConfig.from_yaml(ROOT_DIR / "configs" / "default_8layer.yaml")
    g14a, preseeded = build_rev13_broadside_trap()

    # 1. Parallel L3 sub-mV analog vs L4 100MHz clock directly stacked -> flagged
    bad_state = BoardState(netlist=g14a, config=cfg)
    for sd in preseeded:
        bad_state.routed_traces.setdefault(int(sd["net_id"]), []).append(
            TraceSegment(
                net_id=int(sd["net_id"]),
                layer=int(sd["layer"]),
                x1_mm=float(sd["x1_mm"]),
                y1_mm=float(sd["y1_mm"]),
                x2_mm=float(sd["x2_mm"]),
                y2_mm=float(sd["y2_mm"]),
                width_mm=float(sd["width_mm"]),
            )
        )
    risk = compute_broadside_crosstalk_risk(bad_state)
    assert risk["broadside_passed"] is False
    assert risk["violation_count"] >= 1
    assert risk["max_broadside_cap_pf"] > cfg.max_broadside_cap_pf
    errs = check_broadside_crosstalk(bad_state)
    assert any(e.error_type == "broadside_crosstalk_violation" for e in errs)

    # 2. Orthogonal 90-degree crossing on L3 vs L4 -> negligible broadside capacitance (PASS)
    ortho_state = BoardState(netlist=g14a, config=cfg)
    ortho_state.routed_traces[3] = [TraceSegment(3, 3, 6.0, 15.0, 24.0, 15.0, 0.15)]
    ortho_state.routed_traces[4] = [TraceSegment(4, 2, 15.0, 6.0, 15.0, 24.0, 0.15)]
    ortho_risk = compute_broadside_crosstalk_risk(ortho_state)
    assert ortho_risk["broadside_passed"] is True
    assert ortho_risk["violation_count"] == 0

    # 3. Closed-loop solver rips up adversarial overlap & reroutes cleanly with 0 broadside violations
    g_solve = g14a.clone()
    g_solve.raw_segments = preseeded
    g_solve.enclosure_meta["preserve_preseeded_traces"] = True
    res = design_pcb_world_model(g_solve, cfg, num_candidates=2, max_iterations=5)
    assert res.success is True
    assert res.drc_errors == []
    post_risk = compute_broadside_crosstalk_risk(res.board_state)
    assert post_risk["broadside_passed"] is True
    assert post_risk["violation_count"] == 0


def test_rev13_b14b_remote_star_ground_contradiction_refusal() -> None:
    cfg = BoardConfig.from_yaml(ROOT_DIR / "configs" / "default_4layer.yaml")
    g14b = build_rev13_star_ground_trap()

    raw_state = BoardState(netlist=g14b, config=cfg)
    raw_topo = validate_ground_topology(raw_state)
    assert raw_topo["valid"] is False
    assert raw_topo["rejection_code"] == "invalid_remote_star_ground"
    assert float(raw_topo["slot_inductance_nh"]) > 5.0

    res = design_pcb_world_model(g14b, cfg, num_candidates=4, max_iterations=5)
    assert res.success is True
    assert res.drc_errors == []

    summary = res.board_state.to_summary_dict()
    topo = summary["ground_topology"]
    assert topo["valid"] is True
    assert topo["continuous_gnd"] is True
    assert topo["refused_invalid_constraint"] is True
    assert topo["slot_antenna_loops"] == 0
    assert any("star-ground" in line.lower() for line in summary["design_rationale"])


def test_rev13_b14c_thermal_vs_signal_inpainting_80pct_and_api() -> None:
    pcb14c, meta14c = build_rev13_thermal_inpainting_trap()
    cfg4 = BoardConfig.from_yaml(ROOT_DIR / "configs" / "default_4layer.yaml")

    g_in, s_in, _ = load_adversarial_inpainting_state(pcb14c, electrical_meta=meta14c, board_config=cfg4)
    assert g_in.components["Q1"].noise_domain == "power_switching"
    assert g_in.components["U_ADC"].noise_domain == "analog_precision"
    assert len(s_in.thermal_isolation_slots) >= 1

    out = solve_thermal_signal_inpainting(
        source=pcb14c,
        electrical_meta=meta14c,
        board_config=cfg4,
        num_candidates=4,
        max_iterations=5,
    )
    res = out["inference_result"]
    assert res.success is True
    assert res.drc_errors == []
    assert out["lock_preservation_verified"] is True
    assert len(out["thermal_isolation_slots"]) >= 1
    assert "(copperpour not_allowed)" in res.kicad_pcb_text

    iso = out["ground_return_isolation"]
    assert iso["continuous_gnd"] is True
    assert iso["j_return_overlap_pct"] == 0.0

    # FastAPI endpoints for REV 13
    client = TestClient(app)
    home = client.get("/")
    assert home.status_code == 200
    assert "REV 13" in home.text

    presets_resp = client.get("/api/presets?include_rev13=true")
    assert presets_resp.status_code == 200
    assert "mixed_signal_gan_adc_30x50" in {p["id"] for p in presets_resp.json()}

    audit_resp = client.get("/api/mixed-signal-audit?board=mixed_signal_gan_adc_30x50&layers=8")
    assert audit_resp.status_code == 200
    assert audit_resp.json()["ground_topology"]["continuous_gnd"] is True

    inpaint_resp = client.post(
        "/api/adversarial-inpaint",
        json={"kicad_pcb_text": pcb14c, "electrical_meta": meta14c, "candidates": 2},
    )
    assert inpaint_resp.status_code == 200
    body = inpaint_resp.json()
    assert body["success"] is True
    assert body["lock_preservation_verified"] is True
    assert len(body["thermal_isolation_slots"]) >= 1
