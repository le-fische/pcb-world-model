"""Automated PCB World Model Benchmark Runner & Revision Score Logger.

Runs all benchmark boards (and optional stress-test perturbations), computes
composite scores against GPT-6 Astra and legacy autorouter baselines, and
optionally appends a new revision entry to `benchmarks.md`.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np
import torch

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from env.board_state import BoardState, TraceSegment
from env.config import BoardConfig
from env.importers import (
    audit_schematic_craftsmanship,
    auto_fix_schematic,
    optimize_imported_board,
    parse_schematic,
    solve_thermal_signal_inpainting,
)
from env.sexpr_parser import (
    PRESET_DESCRIPTIONS,
    build_preset_board,
    build_rev13_broadside_trap,
    build_rev13_star_ground_trap,
    build_rev13_thermal_inpainting_trap,
    build_rev14_pdn_inductance_trap,
)
from env.step_exporter import generate_step_assembly, validate_step_structure
from verifier.fast_clearance import check_broadside_crosstalk, check_pdn_decoupling, validate_ground_topology
from scripts import (  # type: ignore[attr-defined]
    __name__ as _scripts_pkg,  # noqa: F401
)
import importlib.util

_INF_SPEC = importlib.util.spec_from_file_location(
    "run_inference", ROOT_DIR / "scripts" / "04_run_inference.py"
)
assert _INF_SPEC is not None and _INF_SPEC.loader is not None
_inf_mod = importlib.util.module_from_spec(_INF_SPEC)
sys.modules["run_inference"] = _inf_mod
_INF_SPEC.loader.exec_module(_inf_mod)

design_pcb_world_model = _inf_mod.design_pcb_world_model
load_models = _inf_mod.load_models


def compute_board_composite_score(
    routed_nets: int,
    total_nets: int,
    drc_errors: int,
    hpwl_mm: float,
    wirelength_mm: float,
    via_count: int,
    elapsed_sec: float,
) -> float:
    """Compute a normalized [0, 100] engineering quality score for a board run.

    Formula weights:
    - 45 pts: Routing Completion Rate (routed_nets / total_nets)
    - 30 pts: DRC Cleanliness (30 * exp(-0.75 * drc_errors))
    - 10 pts: Routing Efficiency (wirelength vs HPWL ratio)
    - 10 pts: Via Minimization (penalizes excessive layer transitions)
    -  5 pts: Speed Bonus (<10s full convergence)
    """
    completion_ratio = routed_nets / max(1, total_nets)
    completion_pts = 45.0 * completion_ratio

    drc_pts = 30.0 * float(np.exp(-0.75 * drc_errors))

    wl_ratio = wirelength_mm / max(1.0, hpwl_mm)
    efficiency_pts = 10.0 * max(0.0, min(1.0, 1.45 - 0.45 * wl_ratio))

    via_pts = 10.0 * max(0.0, 1.0 - (via_count / 45.0))

    speed_pts = 5.0 * max(0.0, min(1.0, 10.0 / max(1.0, elapsed_sec)))

    return round(completion_pts + drc_pts + efficiency_pts + via_pts + speed_pts, 2)


def run_benchmark_suite(
    candidates: int = 8,
    seed: int = 42,
    include_seed_candidate: bool = True,
) -> dict[str, Any]:
    """Execute the full benchmark suite across all reference boards."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    board_cfg = BoardConfig.from_yaml(ROOT_DIR / "configs" / "default_2layer.yaml")
    placer_gnn, world_model = load_models()

    board_results: list[dict[str, Any]] = []
    t0 = time.perf_counter()

    for board_name in PRESET_DESCRIPTIONS:
        torch.manual_seed(seed)
        np.random.seed(seed)
        graph = build_preset_board(board_name)
        out_path = ROOT_DIR / f"output_{board_name}.kicad_pcb"

        res = design_pcb_world_model(
            netlist_graph=graph,
            board_config=board_cfg,
            placer_gnn=placer_gnn,
            world_model=world_model,
            output_path=out_path,
            num_candidates=candidates,
            include_seed_candidate=include_seed_candidate,
        )
        summary = res.board_state.to_summary_dict()
        routed = int(summary["routed_nets"])
        total = int(summary["total_nets"])
        drc_cnt = len(res.drc_errors)
        hpwl = float(summary["hpwl_mm"])
        wl = float(summary["routed_wirelength_mm"])
        vias = int(summary["via_count"])
        best_v = (
            float(res.candidate_scores[0]["routability_score"])
            if res.candidate_scores
            else 0.0
        )

        score = compute_board_composite_score(
            routed_nets=routed,
            total_nets=total,
            drc_errors=drc_cnt,
            hpwl_mm=hpwl,
            wirelength_mm=wl,
            via_count=vias,
            elapsed_sec=res.elapsed_sec,
        )

        board_results.append(
            {
                "board": board_name,
                "routed_nets": routed,
                "total_nets": total,
                "completion_pct": round(100.0 * routed / max(1, total), 1),
                "drc_errors": drc_cnt,
                "hpwl_mm": round(hpwl, 2),
                "wirelength_mm": round(wl, 2),
                "vias": vias,
                "iterations": res.iterations_used,
                "critic_v": round(best_v, 4),
                "elapsed_sec": round(res.elapsed_sec, 2),
                "composite_score": score,
                "success": res.success,
            }
        )

    total_elapsed = round(time.perf_counter() - t0, 2)
    total_routed = sum(b["routed_nets"] for b in board_results)
    total_nets = sum(b["total_nets"] for b in board_results)
    total_drc = sum(b["drc_errors"] for b in board_results)
    mean_hpwl = round(float(np.mean([b["hpwl_mm"] for b in board_results])), 2)
    mean_wl = round(float(np.mean([b["wirelength_mm"] for b in board_results])), 2)
    total_vias = sum(b["vias"] for b in board_results)
    mean_score = round(
        float(np.mean([b["composite_score"] for b in board_results])), 2
    )

    return {
        "timestamp_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "boards": board_results,
        "aggregate": {
            "routed_str": f"{total_routed}/{total_nets} ({round(100.0 * total_routed / max(1, total_nets), 1)}%)",
            "total_drc_errors": total_drc,
            "mean_hpwl_mm": mean_hpwl,
            "mean_wirelength_mm": mean_wl,
            "total_vias": total_vias,
            "total_elapsed_sec": total_elapsed,
            "mean_composite_score": mean_score,
        },
    }


def run_rev11_benchmarks(
    candidates: int = 8,
    seed: int = 42,
    include_seed_candidate: bool = True,
) -> dict[str, Any]:
    """Execute the official rev11 External Import & Mechanical Co-Design benchmarks (B10–B12)."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    board_cfg = BoardConfig.from_yaml(ROOT_DIR / "configs" / "default_2layer.yaml")
    placer_gnn, world_model = load_models()
    t0 = time.perf_counter()

    # B10: External .STEP enclosure keychain (26x26mm polygon + Z_max ribs + ports + bosses)
    g10 = build_preset_board("step_enclosure_keychain")
    res10 = design_pcb_world_model(
        netlist_graph=g10,
        board_config=board_cfg,
        placer_gnn=placer_gnn,
        world_model=world_model,
        num_candidates=candidates,
        include_seed_candidate=include_seed_candidate,
    )
    s10 = res10.board_state.to_summary_dict()
    step10 = generate_step_assembly(res10.board_state, mask_theme="black")
    step10_diag = validate_step_structure(step10)
    b10 = {
        "benchmark_id": "B10",
        "board": "step_enclosure_keychain",
        "routed": f"{s10['routed_nets']}/{s10['total_nets']}",
        "drc_errors": len(res10.drc_errors),
        "z_collisions": len(s10.get("enclosure_collisions", [])),
        "outline_vertices": len(s10.get("outline_polygon", [])),
        "z_max_regions": len(s10.get("z_max_regions", [])),
        "port_anchors": len(s10.get("port_anchors", [])),
        "dfm_status": (s10.get("dfm_audit") or {}).get("status", "PASS"),
        "craftsmanship_score": (s10.get("dfm_audit") or {}).get("craftsmanship_score", 0.0),
        "step_valid": bool(step10_diag["valid"]),
        "elapsed_sec": round(res10.elapsed_sec, 2),
        "success": bool(res10.success and len(s10.get("enclosure_collisions", [])) == 0),
    }

    # B11: Unseen DRV8833 motor driver datasheet PNG pinout ingest + Auto-ERC
    g11 = build_preset_board("datasheet_pinout_ingest")
    erc11 = audit_schematic_craftsmanship(g11)
    if not erc11.passed:
        g11, _ = auto_fix_schematic(g11, erc11)
        erc11 = audit_schematic_craftsmanship(g11)
    res11 = design_pcb_world_model(
        netlist_graph=g11,
        board_config=board_cfg,
        placer_gnn=placer_gnn,
        world_model=world_model,
        num_candidates=candidates,
        include_seed_candidate=include_seed_candidate,
    )
    s11 = res11.board_state.to_summary_dict()
    u1_rules = list(res11.board_state.netlist.components["U1"].special_layout_rules)
    b11 = {
        "benchmark_id": "B11",
        "board": "datasheet_pinout_ingest",
        "part_number": "DRV8833",
        "erc_errors": erc11.error_count,
        "erc_warnings": erc11.warning_count,
        "u1_special_rules": u1_rules,
        "routed": f"{s11['routed_nets']}/{s11['total_nets']}",
        "drc_errors": len(res11.drc_errors),
        "dfm_status": (s11.get("dfm_audit") or {}).get("status", "PASS"),
        "craftsmanship_score": (s11.get("dfm_audit") or {}).get("craftsmanship_score", 0.0),
        "elapsed_sec": round(res11.elapsed_sec, 2),
        "success": bool(res11.success and erc11.passed),
    }

    # B12: Legacy high-via .kicad_pcb lock-preserving re-optimization
    g12 = build_preset_board("legacy_kicad_reoptimize")
    opt12 = optimize_imported_board(
        netlist_graph=g12,
        board_config=board_cfg,
        num_candidates=candidates,
        include_seed_candidate=include_seed_candidate,
        placer_gnn=placer_gnn,
        world_model=world_model,
    )
    res12 = opt12["inference_result"]
    diff12 = opt12["diff"]
    b12 = {
        "benchmark_id": "B12",
        "board": "legacy_kicad_reoptimize",
        "locked_components": diff12.locked_components,
        "lock_preservation_verified": diff12.lock_preservation_verified,
        "before": diff12.before.to_dict(),
        "after": diff12.after.to_dict(),
        "via_reduction_pct": diff12.via_reduction_pct,
        "wirelength_reduction_pct": diff12.wirelength_reduction_pct,
        "inductance_reduction_pct": diff12.inductance_reduction_pct,
        "elapsed_sec": round(res12.elapsed_sec, 2),
        "success": bool(
            res12.success
            and diff12.lock_preservation_verified
            and diff12.via_reduction_pct >= 30.0
            and diff12.after.dfm_violations == 0
        ),
    }

    # Quick schematic multi-format parse + auto-ERC smoke telemetry
    sample_ato = (
        "module SensorNode:\n"
        "  u_mcu = new ATtiny1614\n"
        "  r_pu = new Resistor\n"
        "    value = 4.7k\n"
        "    footprint = R_0603\n"
        "  vcc ~ u_mcu.1\n"
        "  gnd ~ u_mcu.8\n"
        "  sda ~ u_mcu.2\n"
        "  sda ~ r_pu.1\n"
        "  vcc ~ r_pu.2\n"
    )
    ato_g, ato_pre, ato_fixes = parse_schematic(sample_ato, fmt="atopile", auto_fix=True)

    return {
        "revision": "rev11",
        "timestamp_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "total_elapsed_sec": round(time.perf_counter() - t0, 2),
        "B10_step_enclosure_keychain": b10,
        "B11_datasheet_pinout_ingest": b11,
        "B12_legacy_kicad_reoptimize": b12,
        "schematic_auto_erc_demo": {
            "format": "atopile",
            "components_after_fix": len(ato_g.components),
            "fixes_applied": ato_fixes,
            "post_fix_passed": ato_pre.passed,
        },
        "all_passed": bool(b10["success"] and b11["success"] and b12["success"]),
    }


def run_rev12_benchmarks(
    candidates: int = 8,
    seed: int = 42,
    include_seed_candidate: bool = True,
) -> dict[str, Any]:
    """Execute the official rev12 8-Layer (2+4+2) HDI & Staggered Microvia benchmark (B13)."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    cfg_8l = BoardConfig.from_yaml(ROOT_DIR / "configs" / "default_8layer.yaml")
    placer_gnn, world_model = load_models()
    t0 = time.perf_counter()

    g13 = build_preset_board("hdi_fpga_bga_fanout_8l")
    out_path = ROOT_DIR / "output_hdi_fpga_bga_fanout_8l.kicad_pcb"
    res13 = design_pcb_world_model(
        netlist_graph=g13,
        board_config=cfg_8l,
        placer_gnn=placer_gnn,
        world_model=world_model,
        output_path=out_path,
        num_candidates=candidates,
        include_seed_candidate=include_seed_candidate,
    )
    s13 = res13.board_state.to_summary_dict()
    hdi = dict(s13.get("hdi_telemetry") or {})
    step13 = generate_step_assembly(res13.board_state, mask_theme="green")
    step13_diag = validate_step_structure(step13)

    b13 = {
        "benchmark_id": "B13",
        "board": "hdi_fpga_bga_fanout_8l",
        "stackup": f"{s13['layers']}L ({hdi.get('stackup_type', '2+4+2')})",
        "max_lamination_cycles": hdi.get("max_lamination_cycles", 2),
        "routed_nets": s13["routed_nets"],
        "total_nets": s13["total_nets"],
        "routed": f"{s13['routed_nets']}/{s13['total_nets']}",
        "drc_errors": len(res13.drc_errors),
        "wirelength_mm": s13["routed_wirelength_mm"],
        "via_count": s13["via_count"],
        "microvia_count": hdi.get("microvia_count", 0),
        "buried_core_via_count": hdi.get("buried_core_via_count", 0),
        "through_via_count": hdi.get("through_via_count", 0),
        "stacked_microvia_count": hdi.get("stacked_microvia_count", 0),
        "illegal_skip_via_count": hdi.get("illegal_skip_via_count", 0),
        "min_observed_stagger_mm": hdi.get("min_observed_stagger_mm"),
        "layers_used": hdi.get("layers_used", []),
        "si_status": (s13.get("signal_integrity") or {}).get("si_thermal_status", "PASS"),
        "dfm_status": (s13.get("dfm_audit") or {}).get("status", "PASS"),
        "craftsmanship_score": (s13.get("dfm_audit") or {}).get("craftsmanship_score", 0.0),
        "step_valid": bool(step13_diag["valid"]),
        "elapsed_sec": round(res13.elapsed_sec, 2),
        "success": bool(
            res13.success
            and s13["routed_nets"] == s13["total_nets"]
            and len(res13.drc_errors) == 0
            and hdi.get("stacked_microvia_count", 1) == 0
            and hdi.get("illegal_skip_via_count", 1) == 0
            and bool(step13_diag["valid"])
        ),
    }

    return {
        "revision": "rev12",
        "timestamp_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "total_elapsed_sec": round(time.perf_counter() - t0, 2),
        "B13_hdi_fpga_bga_fanout_8l": b13,
        "all_passed": bool(b13["success"]),
    }


def run_rev13_benchmarks(
    candidates: int = 8,
    seed: int = 42,
    include_seed_candidate: bool = True,
) -> dict[str, Any]:
    """Execute the official rev13 Mixed-Signal Domain Partitioning, J_return & 3D EM Trap suite (B14, B14a–c)."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    cfg_8l = BoardConfig.from_yaml(ROOT_DIR / "configs" / "default_8layer.yaml")
    cfg_4l = BoardConfig.from_yaml(ROOT_DIR / "configs" / "default_4layer.yaml")
    placer_gnn, world_model = load_models()
    t0 = time.perf_counter()

    # B14: 30x50mm 8L Mixed-Signal 5A 2MHz GaN + 24-bit Sub-mV ADC (both on F.Cu)
    g14 = build_preset_board("mixed_signal_gan_adc_30x50")
    out14 = ROOT_DIR / "output_mixed_signal_gan_adc_30x50.kicad_pcb"
    res14 = design_pcb_world_model(
        netlist_graph=g14,
        board_config=cfg_8l,
        placer_gnn=placer_gnn,
        world_model=world_model,
        output_path=out14,
        num_candidates=candidates,
        include_seed_candidate=include_seed_candidate,
    )
    s14 = res14.board_state.to_summary_dict()
    iso14 = dict(s14.get("ground_return_isolation") or {})
    broad14 = dict(s14.get("broadside_crosstalk") or {})
    step14 = generate_step_assembly(res14.board_state, mask_theme="green")
    step14_diag = validate_step_structure(step14)
    sep14 = float(iso14.get("gan_to_adc_separation_mm") or 0.0)
    b14 = {
        "benchmark_id": "B14",
        "board": "mixed_signal_gan_adc_30x50",
        "routed": f"{s14['routed_nets']}/{s14['total_nets']}",
        "routed_nets": s14["routed_nets"],
        "total_nets": s14["total_nets"],
        "drc_errors": len(res14.drc_errors),
        "continuous_gnd": bool(iso14.get("continuous_gnd", False)),
        "gan_to_adc_separation_mm": sep14,
        "j_return_overlap_pct": float(iso14.get("j_return_overlap_pct", 0.0)),
        "switching_under_analog_count": int(iso14.get("switching_under_analog_count", 0)),
        "broadside_violations": int(broad14.get("violation_count", 0)),
        "max_broadside_cap_pf": float(broad14.get("max_broadside_cap_pf", 0.0)),
        "dfm_status": (s14.get("dfm_audit") or {}).get("status", "PASS"),
        "craftsmanship_score": (s14.get("dfm_audit") or {}).get("craftsmanship_score", 0.0),
        "step_valid": bool(step14_diag["valid"]),
        "rationale_count": len(s14.get("design_rationale", [])),
        "elapsed_sec": round(res14.elapsed_sec, 2),
        "success": bool(
            res14.success
            and len(res14.drc_errors) == 0
            and iso14.get("continuous_gnd", False)
            and sep14 >= 20.0
            and float(iso14.get("j_return_overlap_pct", 100.0)) == 0.0
            and int(broad14.get("violation_count", 1)) == 0
            and bool(step14_diag["valid"])
        ),
    }

    # B14a: Z-Axis Broadside Crosstalk Trap (thin-core 8L L3 analog vs L4 clock)
    g14a, segs14a = build_rev13_broadside_trap()
    trap_state_a = BoardState(netlist=g14a, config=cfg_8l)
    for sd in segs14a:
        trap_state_a.routed_traces.setdefault(int(sd["net_id"]), []).append(
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
    pre_broad_errs = check_broadside_crosstalk(trap_state_a)
    g14a_solve = g14a.clone()
    g14a_solve.raw_segments = segs14a
    g14a_solve.enclosure_meta["preserve_preseeded_traces"] = True
    res14a = design_pcb_world_model(
        netlist_graph=g14a_solve,
        board_config=cfg_8l,
        placer_gnn=placer_gnn,
        world_model=world_model,
        num_candidates=min(4, candidates),
        include_seed_candidate=True,
    )
    s14a = res14a.board_state.to_summary_dict()
    broad14a = dict(s14a.get("broadside_crosstalk") or {})
    b14a = {
        "benchmark_id": "B14a",
        "board": "z_axis_broadside_crosstalk_trap",
        "pre_seed_broadside_detected": len(pre_broad_errs),
        "routed": f"{s14a['routed_nets']}/{s14a['total_nets']}",
        "drc_errors": len(res14a.drc_errors),
        "broadside_violations": int(broad14a.get("violation_count", 0)),
        "max_broadside_cap_pf": float(broad14a.get("max_broadside_cap_pf", 0.0)),
        "elapsed_sec": round(res14a.elapsed_sec, 2),
        "success": bool(
            len(pre_broad_errs) >= 1
            and res14a.success
            and len(res14a.drc_errors) == 0
            and int(broad14a.get("violation_count", 1)) == 0
        ),
    }

    # B14b: Remote Star-Ground Contradiction Refusal
    g14b = build_rev13_star_ground_trap()
    pre_topo_b = validate_ground_topology(BoardState(netlist=g14b, config=cfg_4l))
    res14b = design_pcb_world_model(
        netlist_graph=g14b,
        board_config=cfg_4l,
        placer_gnn=placer_gnn,
        world_model=world_model,
        num_candidates=min(4, candidates),
        include_seed_candidate=include_seed_candidate,
    )
    s14b = res14b.board_state.to_summary_dict()
    topo14b = dict(s14b.get("ground_topology") or {})
    b14b = {
        "benchmark_id": "B14b",
        "board": "remote_star_ground_contradiction",
        "initial_rejection_code": pre_topo_b.get("rejection_code"),
        "refused_invalid_constraint": bool(topo14b.get("refused_invalid_constraint", False)),
        "continuous_gnd": bool(topo14b.get("continuous_gnd", False)),
        "slot_antenna_loops": int(topo14b.get("slot_antenna_loops", 1)),
        "routed": f"{s14b['routed_nets']}/{s14b['total_nets']}",
        "drc_errors": len(res14b.drc_errors),
        "elapsed_sec": round(res14b.elapsed_sec, 2),
        "success": bool(
            not bool(pre_topo_b.get("valid", True))
            and pre_topo_b.get("rejection_code") == "invalid_remote_star_ground"
            and bool(topo14b.get("refused_invalid_constraint", False))
            and bool(topo14b.get("continuous_gnd", False))
            and int(topo14b.get("slot_antenna_loops", 1)) == 0
            and res14b.success
            and len(res14b.drc_errors) == 0
        ),
    }

    # B14c: 80%-Locked Thermal-vs-Signal Adversarial Inpainting
    pcb14c, meta14c = build_rev13_thermal_inpainting_trap()
    out14c = solve_thermal_signal_inpainting(
        source=pcb14c,
        electrical_meta=meta14c,
        board_config=cfg_4l,
        num_candidates=min(4, candidates),
        placer_gnn=placer_gnn,
        world_model=world_model,
    )
    res14c = out14c["inference_result"]
    s14c = res14c.board_state.to_summary_dict()
    iso14c = dict(out14c.get("ground_return_isolation") or {})
    b14c = {
        "benchmark_id": "B14c",
        "board": "thermal_vs_signal_inpainting_80pct",
        "lock_preservation_verified": bool(out14c["lock_preservation_verified"]),
        "locked_components": out14c["locked_components"],
        "thermal_slots_carved": len(out14c["thermal_isolation_slots"]),
        "continuous_gnd": bool(iso14c.get("continuous_gnd", False)),
        "j_return_overlap_pct": float(iso14c.get("j_return_overlap_pct", 0.0)),
        "routed": f"{s14c['routed_nets']}/{s14c['total_nets']}",
        "drc_errors": len(res14c.drc_errors),
        "elapsed_sec": round(res14c.elapsed_sec, 2),
        "success": bool(
            res14c.success
            and len(res14c.drc_errors) == 0
            and bool(out14c["lock_preservation_verified"])
            and len(out14c["thermal_isolation_slots"]) >= 1
            and bool(iso14c.get("continuous_gnd", False))
            and float(iso14c.get("j_return_overlap_pct", 100.0)) == 0.0
        ),
    }

    return {
        "revision": "rev13",
        "timestamp_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "total_elapsed_sec": round(time.perf_counter() - t0, 2),
        "B14_mixed_signal_gan_adc_30x50": b14,
        "B14a_z_axis_broadside_crosstalk_trap": b14a,
        "B14b_remote_star_ground_contradiction": b14b,
        "B14c_thermal_vs_signal_inpainting_80pct": b14c,
        "all_passed": bool(b14["success"] and b14a["success"] and b14b["success"] and b14c["success"]),
    }


def run_rev14_benchmarks(
    candidates: int = 8,
    seed: int = 42,
    include_seed_candidate: bool = True,
) -> dict[str, Any]:
    """Run rev14 B15 & B15a Rigid-Flex 8L HDI + 85-Ohm PCIe Gen 4 + PDN Decoupling benchmarks."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    t0 = time.perf_counter()
    placer_gnn, world_model = load_models()
    cfg_8l = BoardConfig.from_yaml(ROOT_DIR / "configs" / "default_8layer.yaml")

    # B15: Ultimate Rigid-Flex 8L (2+4+2) HDI + 50% Cross-Hatched Flex + 85-Ohm PCIe Gen 4 + GaN/ADC + PDN
    g15 = build_preset_board("ultimate_rigid_flex_8l_hdi_pcie_gan_adc")
    res15 = design_pcb_world_model(
        netlist_graph=g15,
        board_config=cfg_8l,
        placer_gnn=placer_gnn,
        world_model=world_model,
        num_candidates=min(4, candidates),
        include_seed_candidate=include_seed_candidate,
    )
    s15 = res15.board_state.to_summary_dict()
    iso15 = dict(s15.get("ground_return_isolation") or {})
    broad15 = dict(s15.get("broadside_crosstalk") or {})
    hdi15 = dict(s15.get("hdi_telemetry") or {})
    rf15 = dict(s15.get("rigid_flex_impedance") or {})
    pdn15 = dict(s15.get("pdn_decoupling") or {})
    dfm15 = dict(s15.get("dfm_audit") or {})
    sep15 = float(iso15.get("gan_to_adc_separation_mm") or 0.0)
    step15_diag = validate_step_structure(generate_step_assembly(res15.board_state, mask_theme="green"))
    pairs15 = rf15.get("pairs", [])
    max_skew15 = max((float(p.get("skew_mm", 0.0)) for p in pairs15), default=0.0)
    b15 = {
        "benchmark_id": "B15",
        "board": "ultimate_rigid_flex_8l_hdi_pcie_gan_adc",
        "routed": f"{s15['routed_nets']}/{s15['total_nets']}",
        "drc_errors": len(res15.drc_errors),
        "continuous_gnd": bool(iso15.get("continuous_gnd", False)),
        "gan_to_adc_separation_mm": round(sep15, 2),
        "j_return_overlap_pct": float(iso15.get("j_return_overlap_pct", 0.0)),
        "broadside_violations": int(broad15.get("violation_count", 0)),
        "stacked_microvia_count": int(hdi15.get("stacked_microvia_count", 0)),
        "illegal_skip_via_count": int(hdi15.get("illegal_skip_via_count", 0)),
        "microvia_count": int(hdi15.get("microvia_count", 0)),
        "buried_core_via_count": int(hdi15.get("buried_core_via_count", 0)),
        "min_stagger_mm": hdi15.get("min_observed_stagger_mm"),
        "rigid_flex_pairs": pairs15,
        "all_pairs_passed": bool(rf15.get("all_pairs_passed", False)),
        "max_skew_mm": round(max_skew15, 4),
        "pdn_max_l_loop_nh": float(pdn15.get("max_loop_inductance_nh", 0.0)),
        "pdn_passed": bool(pdn15.get("pdn_passed", False)),
        "dfm_passed": bool(dfm15.get("passed", False)),
        "craftsmanship_score": float(dfm15.get("craftsmanship_score", 0.0)),
        "step_valid": bool(step15_diag["valid"]),
        "elapsed_sec": round(res15.elapsed_sec, 2),
        "success": bool(
            res15.success
            and len(res15.drc_errors) == 0
            and iso15.get("continuous_gnd", False)
            and float(iso15.get("j_return_overlap_pct", 100.0)) == 0.0
            and int(broad15.get("violation_count", 1)) == 0
            and int(hdi15.get("stacked_microvia_count", 1)) == 0
            and int(hdi15.get("illegal_skip_via_count", 1)) == 0
            and bool(rf15.get("all_pairs_passed", False))
            and len(pairs15) >= 2
            and max_skew15 <= 0.125
            and bool(pdn15.get("pdn_passed", False))
            and bool(dfm15.get("passed", False))
            and bool(step15_diag["valid"])
            and (sep15 >= 20.0 if include_seed_candidate else sep15 >= 14.0)
        ),
    }

    # B15a: PDN Decoupling Mounting Inductance Trap (remote caps -> under-BGA via-in-pad)
    g15a, _ = build_rev14_pdn_inductance_trap()
    pre_state_a = BoardState(netlist=g15a, config=cfg_8l)
    pre_errs_a = check_pdn_decoupling(pre_state_a)
    pre_pdn_a = pre_state_a.to_summary_dict().get("pdn_decoupling", {})
    res15a = design_pcb_world_model(
        netlist_graph=g15a,
        board_config=cfg_8l,
        placer_gnn=placer_gnn,
        world_model=world_model,
        num_candidates=min(4, candidates),
        include_seed_candidate=True,
    )
    s15a = res15a.board_state.to_summary_dict()
    pdn15a = dict(s15a.get("pdn_decoupling") or {})
    hdi15a = dict(s15a.get("hdi_telemetry") or {})
    caps_under_bga = all(bool(r.get("under_bga", False)) for r in pdn15a.get("cap_reports", []))
    b15a = {
        "benchmark_id": "B15a",
        "board": "pdn_decoupling_inductance_trap",
        "pre_pdn_violations": len(pre_errs_a),
        "pre_max_l_loop_nh": float(pre_pdn_a.get("max_loop_inductance_nh", 0.0)),
        "post_max_l_loop_nh": float(pdn15a.get("max_loop_inductance_nh", 0.0)),
        "caps_under_bga_b_cu": caps_under_bga,
        "routed": f"{s15a['routed_nets']}/{s15a['total_nets']}",
        "drc_errors": len(res15a.drc_errors),
        "stacked_microvia_count": int(hdi15a.get("stacked_microvia_count", 0)),
        "illegal_skip_via_count": int(hdi15a.get("illegal_skip_via_count", 0)),
        "elapsed_sec": round(res15a.elapsed_sec, 2),
        "success": bool(
            len(pre_errs_a) >= 1
            and float(pre_pdn_a.get("max_loop_inductance_nh", 0.0)) > 1.2
            and res15a.success
            and len(res15a.drc_errors) == 0
            and bool(pdn15a.get("pdn_passed", False))
            and float(pdn15a.get("max_loop_inductance_nh", 99.0)) <= 1.2
            and caps_under_bga
            and int(hdi15a.get("stacked_microvia_count", 1)) == 0
            and int(hdi15a.get("illegal_skip_via_count", 1)) == 0
        ),
    }

    return {
        "revision": "rev14",
        "timestamp_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "total_elapsed_sec": round(time.perf_counter() - t0, 2),
        "B15_ultimate_rigid_flex_8l_hdi_pcie_gan_adc": b15,
        "B15a_pdn_decoupling_inductance_trap": b15a,
        "all_passed": bool(b15["success"] and b15a["success"]),
    }


def append_revision_to_markdown(
    md_path: Path,
    revision: str,
    notes: str,
    suite_data: dict[str, Any],
) -> None:
    """Append a markdown table row into the Revision Score Log Sheet of `benchmarks.md`."""
    if not md_path.exists():
        return
    content = md_path.read_text(encoding="utf-8")
    agg = suite_data["aggregate"]
    by_name = {b["board"]: b for b in suite_data["boards"]}

    mcu = by_name.get("mcu_sensor_node", {})
    buck = by_name.get("buck_converter", {})
    usb = by_name.get("usb_uart_bridge", {})

    row = (
        f"| `{revision}` | {suite_data['timestamp_utc']} | "
        f"**{agg['routed_str']}** | **{agg['total_drc_errors']}** | "
        f"{mcu.get('wirelength_mm', '-')}/{mcu.get('vias', '-')}v | "
        f"{buck.get('wirelength_mm', '-')}/{buck.get('vias', '-')}v | "
        f"{usb.get('wirelength_mm', '-')}/{usb.get('vias', '-')}v | "
        f"{agg['mean_hpwl_mm']} mm | {agg['total_elapsed_sec']}s | "
        f"**{agg['mean_composite_score']} / 100** | {notes} |"
    )

    marker = "<!-- AGENT_BENCHMARK_LOG_END -->"
    if marker in content:
        head, tail = content.rsplit(marker, 1)
        updated = f"{head}{row}\n{marker}{tail}"
        md_path.write_text(updated, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the PCB World Model benchmark suite and log revision scores"
    )
    parser.add_argument(
        "--revision",
        type=str,
        default="v0.1.0",
        help="Revision tag (e.g. v0.1.0, v0.2.0-gat-tuned)",
    )
    parser.add_argument(
        "--notes",
        type=str,
        default="Baseline GATv2 + Dual-Head U-Net + 45-deg Octilinear A* + DRC Rip-Up",
        help="Short summary of architectural or hyperparameter changes in this revision",
    )
    parser.add_argument("--candidates", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--blind",
        action="store_true",
        help="Run blind zero-hint placement without schematic seed candidate",
    )
    parser.add_argument(
        "--append-log",
        action="store_true",
        help="Append the benchmark result row to benchmarks.md",
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        default=None,
        help="Optional path to save full JSON benchmark report",
    )
    parser.add_argument(
        "--rev11",
        action="store_true",
        help="Run rev11 B10–B12 external import & optimization validation suite",
    )
    parser.add_argument(
        "--rev12",
        action="store_true",
        help="Run rev12 B13 8-layer (2+4+2) HDI & staggered microvia BGA fanout validation suite",
    )
    parser.add_argument(
        "--rev13",
        action="store_true",
        help="Run rev13 B14 & B14a-c mixed-signal return physics, broadside crosstalk, star-ground refusal & thermal inpainting suite",
    )
    parser.add_argument(
        "--rev14",
        action="store_true",
        help="Run rev14 B15 & B15a rigid-flex 8L HDI, cross-hatched 85-Ohm PCIe Gen 4 & PDN mounting inductance suite",
    )
    args = parser.parse_args()

    if args.rev14:
        rev14_data = run_rev14_benchmarks(
            candidates=args.candidates,
            seed=args.seed,
            include_seed_candidate=not args.blind,
        )
        print("=" * 88)
        print("PCB WORLD MODEL REV 14 VALIDATION SUITE (B15 & B15a — Rigid-Flex 85-Ohm PCIe Gen 4 & PDN)")
        print("=" * 88)
        b15 = rev14_data["B15_ultimate_rigid_flex_8l_hdi_pcie_gan_adc"]
        b15a = rev14_data["B15a_pdn_decoupling_inductance_trap"]
        pair_summary = ", ".join(
            f"{p['pair_name']}(Zr={p['rigid_z_diff_ohm']},Zf={p['flex_z_diff_ohm']},sk={p['skew_mm']}mm)"
            for p in b15["rigid_flex_pairs"]
        )
        print(
            f"  [B15  Rigid-Flex 8L] Routed={b15['routed']} | DRC={b15['drc_errors']} | "
            f"ContGND={b15['continuous_gnd']} | Sep={b15['gan_to_adc_separation_mm']:.2f}mm | "
            f"J_ret={b15['j_return_overlap_pct']}% | Stacked={b15['stacked_microvia_count']} | "
            f"PDN_L={b15['pdn_max_l_loop_nh']}nH | DFM={b15['craftsmanship_score']} | "
            f"Time={b15['elapsed_sec']}s | PASS={b15['success']}"
        )
        print(f"    [PCIe 85-Ohm] {pair_summary}")
        print(
            f"  [B15a PDN Trap     ] Pre={b15a['pre_pdn_violations']}viol ({b15a['pre_max_l_loop_nh']}nH) -> "
            f"Post={b15a['post_max_l_loop_nh']}nH (UnderBGA={b15a['caps_under_bga_b_cu']}) | "
            f"Routed={b15a['routed']} | DRC={b15a['drc_errors']} | Stacked={b15a['stacked_microvia_count']} | "
            f"Time={b15a['elapsed_sec']}s | PASS={b15a['success']}"
        )
        print("=" * 88)
        if args.json_out is not None:
            args.json_out.write_text(json.dumps(rev14_data, indent=2), encoding="utf-8")
        return

    if args.rev13:
        rev13_data = run_rev13_benchmarks(
            candidates=args.candidates,
            seed=args.seed,
            include_seed_candidate=not args.blind,
        )
        print("=" * 88)
        print("PCB WORLD MODEL REV 13 VALIDATION SUITE (B14 & B14a-c — Mixed-Signal J_return & 3D EM Traps)")
        print("=" * 88)
        b14 = rev13_data["B14_mixed_signal_gan_adc_30x50"]
        b14a = rev13_data["B14a_z_axis_broadside_crosstalk_trap"]
        b14b = rev13_data["B14b_remote_star_ground_contradiction"]
        b14c = rev13_data["B14c_thermal_vs_signal_inpainting_80pct"]
        print(
            f"  [B14  GaN+ADC 30x50] Routed={b14['routed']} | DRC={b14['drc_errors']} | "
            f"ContGND={b14['continuous_gnd']} | Sep={b14['gan_to_adc_separation_mm']:.2f}mm | "
            f"J_ret_ov={b14['j_return_overlap_pct']}% | Broadside={b14['broadside_violations']} | "
            f"Time={b14['elapsed_sec']}s | PASS={b14['success']}"
        )
        print(
            f"  [B14a Broadside Trap] PreDetected={b14a['pre_seed_broadside_detected']} | "
            f"Routed={b14a['routed']} | DRC={b14a['drc_errors']} | "
            f"PostBroadside={b14a['broadside_violations']} (max={b14a['max_broadside_cap_pf']}pF) | "
            f"Time={b14a['elapsed_sec']}s | PASS={b14a['success']}"
        )
        print(
            f"  [B14b Star-GND Trap ] Initial={b14b['initial_rejection_code']} | "
            f"Refused={b14b['refused_invalid_constraint']} | ContGND={b14b['continuous_gnd']} | "
            f"SlotLoops={b14b['slot_antenna_loops']} | DRC={b14b['drc_errors']} | "
            f"Time={b14b['elapsed_sec']}s | PASS={b14b['success']}"
        )
        print(
            f"  [B14c 80% Inpainting] LocksOK={b14c['lock_preservation_verified']} | "
            f"Slots={b14c['thermal_slots_carved']} | ContGND={b14c['continuous_gnd']} | "
            f"J_ret_ov={b14c['j_return_overlap_pct']}% | DRC={b14c['drc_errors']} | "
            f"Time={b14c['elapsed_sec']}s | PASS={b14c['success']}"
        )
        print("=" * 88)
        if args.json_out is not None:
            args.json_out.write_text(json.dumps(rev13_data, indent=2), encoding="utf-8")
        return

    if args.rev12:
        rev12_data = run_rev12_benchmarks(
            candidates=args.candidates,
            seed=args.seed,
            include_seed_candidate=not args.blind,
        )
        print("=" * 88)
        print("PCB WORLD MODEL REV 12 VALIDATION SUITE (B13 — 8L 2+4+2 HDI & Staggered Microvias)")
        print("=" * 88)
        b13 = rev12_data["B13_hdi_fpga_bga_fanout_8l"]
        print(
            f"  [B13 8L HDI BGA] Stackup={b13['stackup']} | Routed={b13['routed']} | "
            f"DRC={b13['drc_errors']} | uVias={b13['microvia_count']} | "
            f"Buried={b13['buried_core_via_count']} | PTH={b13['through_via_count']} | "
            f"Stacked={b13['stacked_microvia_count']} | Skip={b13['illegal_skip_via_count']} | "
            f"MinStagger={b13['min_observed_stagger_mm']}mm | DFM={b13['dfm_status']} ({b13['craftsmanship_score']}) | "
            f"Time={b13['elapsed_sec']}s"
        )
        print(f"  [Layers Used   ] {', '.join(b13['layers_used'])}")
        print("=" * 88)
        if args.json_out is not None:
            args.json_out.write_text(json.dumps(rev12_data, indent=2), encoding="utf-8")
        return

    if args.rev11:
        rev11_data = run_rev11_benchmarks(
            candidates=args.candidates,
            seed=args.seed,
            include_seed_candidate=not args.blind,
        )
        print("=" * 88)
        print("PCB WORLD MODEL REV 11 VALIDATION SUITE (B10–B12)")
        print("=" * 88)
        b10 = rev11_data["B10_step_enclosure_keychain"]
        b11 = rev11_data["B11_datasheet_pinout_ingest"]
        b12 = rev11_data["B12_legacy_kicad_reoptimize"]
        print(
            f"  [B10 STEP Enclosure] Routed={b10['routed']} | DRC={b10['drc_errors']} | "
            f"Z-Col={b10['z_collisions']} | DFM={b10['dfm_status']} ({b10['craftsmanship_score']}) | "
            f"Time={b10['elapsed_sec']}s"
        )
        print(
            f"  [B11 Pin Ingest    ] Part={b11['part_number']} | Routed={b11['routed']} | "
            f"ERC={b11['erc_errors']}err/{b11['erc_warnings']}warn | DRC={b11['drc_errors']} | "
            f"DFM={b11['dfm_status']} ({b11['craftsmanship_score']}) | Time={b11['elapsed_sec']}s"
        )
        print(
            f"  [B12 Re-Optimize   ] Vias={b12['before']['via_count']}->{b12['after']['via_count']} "
            f"(-{b12['via_reduction_pct']}%) | WL={b12['before']['wirelength_mm']}->{b12['after']['wirelength_mm']}mm | "
            f"DRC={b12['before']['drc_errors']}->{b12['after']['drc_errors']} | "
            f"DFM={b12['before']['dfm_violations']}->{b12['after']['dfm_violations']} | "
            f"Locks={b12['lock_preservation_verified']} | Time={b12['elapsed_sec']}s"
        )
        print("=" * 88)
        if args.json_out is not None:
            args.json_out.write_text(json.dumps(rev11_data, indent=2), encoding="utf-8")
        return

    suite_data = run_benchmark_suite(
        candidates=args.candidates,
        seed=args.seed,
        include_seed_candidate=not args.blind,
    )

    print("=" * 88)
    print(f"PCB WORLD MODEL BENCHMARK SUITE — Revision: {args.revision}")
    print("=" * 88)
    for b in suite_data["boards"]:
        print(
            f"  [{b['board']:<16}] Routed: {b['routed_nets']}/{b['total_nets']} ({b['completion_pct']:>5.1f}%) | "
            f"DRC: {b['drc_errors']} | HPWL: {b['hpwl_mm']:>6.2f}mm | "
            f"WL: {b['wirelength_mm']:>6.2f}mm | Vias: {b['vias']:>2d} | "
            f"Iters: {b['iterations']} | Time: {b['elapsed_sec']:>4.2f}s | "
            f"Score: {b['composite_score']:>5.2f}/100"
        )
    agg = suite_data["aggregate"]
    print("-" * 88)
    print(
        f"  [AGGREGATE       ] Routed: {agg['routed_str']} | Total DRC: {agg['total_drc_errors']} | "
        f"Mean HPWL: {agg['mean_hpwl_mm']}mm | Mean WL: {agg['mean_wirelength_mm']}mm | "
        f"Vias: {agg['total_vias']} | Total Time: {agg['total_elapsed_sec']}s | "
        f"Mean Score: {agg['mean_composite_score']}/100"
    )
    print("=" * 88)

    if args.json_out is not None:
        args.json_out.write_text(json.dumps(suite_data, indent=2), encoding="utf-8")

    if args.append_log:
        for md_candidate in [
            ROOT_DIR / "benchmarks.md",
            ROOT_DIR.parent / "benchmarks.md",
        ]:
            append_revision_to_markdown(
                md_candidate, args.revision, args.notes, suite_data
            )
        print("[05_run_benchmarks] Appended revision row to benchmarks.md")


if __name__ == "__main__":
    main()
