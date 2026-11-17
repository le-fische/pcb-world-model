"""End-to-End Closed-Loop Neuro-Symbolic PCB Engineering World Model Inference Pipeline."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
import sys
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import json
from env.board_state import CHANNEL_NAMES, BoardState, downsample_spatial_map
from env.config import BoardConfig
from env.importers import (
    apply_enclosure_constraints,
    optimize_imported_board,
    parse_dxf_enclosure,
    parse_schematic,
    parse_step_enclosure,
)
from env.importers.pin_diagram_importer import import_pin_diagram
from env.kicad_exporter import export, serialize_kicad_pcb
from env.kicad_mod_loader import compile_datasheet_pin_table, parse_kicad_mod
from env.manufacturing_exporter import export_manufacturing_bundle
from env.step_exporter import export_step_assembly
from env.metrics import (
    _find_diff_pairs_in_netlist,
    compute_hpwl,
    compute_mst_wirelength,
    rank_nets_by_criticality,
)
from env.sexpr_parser import (
    ALL_PRESET_DESCRIPTIONS,
    FULL_PRESET_CATALOG,
    PRESET_DESCRIPTIONS,
    NetlistGraph,
    SExprParser,
    build_preset_board,
)
from models.composer.constraint_negotiator import negotiate_constraints
from models.composer.design_rationale import generate_physics_design_rationale
from models.composer.schematic_composer import compose_netlist_from_prompt, verify_erc
from models.placer.placer_head import MacroPlacerGNN
from models.placer.relaxation import (
    compute_spectral_anchor_placement,
    migrate_pdn_caps_under_bga,
    relax_and_snap_to_grid,
)
from models.silicon import (
    compile_asic_openlane,
    compose_verilog_module,
    optimize_fpga_pcb_codesign,
)
from models.world.dfm_craftsmanship import apply_dfm_craftsmanship
from models.world.dynamics_unet import WorldModelUNet
from models.world.native_kernel import native_kernel_info
from models.world.neural_router import octilinear_astar_route, route_differential_pair
from verifier.dfm_auditor import audit_board_dfm, generate_dfm_report_markdown
from verifier.fast_clearance import (
    DRCError,
    check_courtyard_overlaps,
    validate_ground_topology,
)
from verifier.kicad_drc import verify as verify_kicad_drc


@dataclass
class InferenceResult:
    """Complete output of the closed-loop PCB World Model design pipeline."""

    board_state: BoardState
    kicad_pcb_text: str
    output_path: Path | None
    drc_errors: list[DRCError]
    candidate_scores: list[dict[str, Any]]
    iterations_used: int
    elapsed_sec: float
    success: bool


def _select_inference_device(device: str | torch.device | None = None) -> torch.device:
    """Choose best available hardware accelerator (`mps` / `cuda` / `cpu`) for UNet inference."""
    if device is not None:
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_models(
    checkpoints_dir: Path | None = None,
    device: str | torch.device | None = None,
) -> tuple[MacroPlacerGNN, WorldModelUNet]:
    """Load trained MacroPlacerGNN and WorldModelUNet checkpoints if available."""
    ckpt_dir = checkpoints_dir or (ROOT_DIR / "checkpoints")
    placer_ckpt = ckpt_dir / "placer_gnn.pt"
    world_ckpt = ckpt_dir / "world_model_unet.pt"

    if placer_ckpt.exists():
        data = torch.load(placer_ckpt, map_location="cpu", weights_only=True)
        placer = MacroPlacerGNN(
            hidden_dim=int(data.get("hidden_dim", 64)),
            num_layers=int(data.get("num_layers", 4)),
            heads=int(data.get("heads", 4)),
        )
        placer.load_state_dict(data["model_state_dict"])
    else:
        placer = MacroPlacerGNN()

    if world_ckpt.exists():
        data = torch.load(world_ckpt, map_location="cpu", weights_only=True)
        world = WorldModelUNet(
            in_channels=int(data.get("in_channels", 7)),
            base_channels=int(data.get("base_channels", 16)),
        )
        world.load_state_dict(data["model_state_dict"])
    else:
        world = WorldModelUNet()

    placer.eval()
    world.eval()
    target_device = _select_inference_device(device)
    world.to(target_device)
    if target_device.type in ("mps", "cuda"):
        with torch.inference_mode():
            world(torch.zeros((1, world.in_channels, 64, 64), device=target_device))
    return placer, world


def _add_penalty_hotspot(
    penalty_field: np.ndarray,
    layer: int,
    x_grid: int,
    y_grid: int,
    magnitude: float = 10.0,
    radius: int = 4,
) -> None:
    """Inject a smooth spatial penalty spike at `(layer, y_grid, x_grid)` for rip-up reroute."""
    num_layers, H, W = penalty_field.shape
    l = max(0, min(num_layers - 1, int(layer)))
    x0 = max(0, x_grid - radius)
    x1 = min(W - 1, x_grid + radius)
    y0 = max(0, y_grid - radius)
    y1 = min(H - 1, y_grid + radius)
    if x0 > x1 or y0 > y1:
        return
    yy, xx = np.ogrid[y0 : y1 + 1, x0 : x1 + 1]
    dist_sq = (xx - x_grid) ** 2 + (yy - y_grid) ** 2
    kernel = magnitude * np.exp(-dist_sq / (2.0 * (radius * 0.55) ** 2))
    penalty_field[l, y0 : y1 + 1, x0 : x1 + 1] += kernel.astype(np.float32)


def _tensor_channels_payload(
    state: BoardState,
    active_net_id: int | None,
    congestion_map: np.ndarray | None = None,
    penalty_field: np.ndarray | None = None,
    downsample_size: int = 50,
) -> dict[str, Any]:
    """Build downsampled `[7, size, size]` S_t channels and M_{t+1} heatmaps for live UI."""
    tensor_np = state.to_tensor(active_net_id=active_net_id).numpy()
    channels = {
        name: downsample_spatial_map(tensor_np[idx], size=downsample_size)
        for idx, name in enumerate(CHANNEL_NAMES)
    }
    payload: dict[str, Any] = {"channels": channels}
    if congestion_map is not None:
        payload["congestion_L1"] = downsample_spatial_map(
            congestion_map[0], size=downsample_size
        )
        payload["congestion_L2"] = downsample_spatial_map(
            congestion_map[-1], size=downsample_size
        )
    if penalty_field is not None:
        combined_pen = np.max(penalty_field, axis=0)
        payload["drc_penalty"] = downsample_spatial_map(
            np.clip(combined_pen / 15.0, 0.0, 1.0), size=downsample_size
        )
    return payload


def design_pcb_world_model(
    netlist_graph: NetlistGraph,
    board_config: BoardConfig,
    placer_gnn: MacroPlacerGNN | None = None,
    world_model: WorldModelUNet | None = None,
    output_path: Path | None = None,
    num_candidates: int | None = None,
    max_iterations: int | None = None,
    step_callback: Callable[[dict[str, Any]], None] | None = None,
    stream_tensors: bool = False,
    include_seed_candidate: bool = True,
) -> InferenceResult:
    """Run the closed-loop Neuro-Symbolic PCB Engineering World Model pipeline."""
    t_start = time.perf_counter()
    if placer_gnn is None or world_model is None:
        loaded_placer, loaded_world = load_models()
        placer_gnn = placer_gnn or loaded_placer
        world_model = world_model or loaded_world

    # Sync board_config dimensions & compact wearable rules with the actual netlist without mutating caller config
    eff_w = netlist_graph.width_mm if netlist_graph.width_mm > 0 else board_config.board_width_mm
    eff_h = netlist_graph.height_mm if netlist_graph.height_mm > 0 else board_config.board_height_mm
    is_compact = max(eff_w, eff_h) <= 35.0
    has_fine_pitch = any(
        len(c.pads) >= 16 and min((min(p.width, p.height) for p in c.pads), default=1.0) <= 0.30
        for c in netlist_graph.components.values()
    )
    is_single = bool(getattr(netlist_graph, "single_sided", False) or board_config.single_sided)
    is_hdi_8l = int(board_config.layers) >= 8 or int(getattr(netlist_graph, "layers", 2)) >= 8
    has_bga = any("BGA" in (c.footprint or "").upper() for c in netlist_graph.components.values())
    eff_layers = 8 if is_hdi_8l else int(board_config.layers)
    eff_res = (
        min(board_config.grid_resolution_mm, 0.10)
        if (is_hdi_8l or has_bga)
        else (min(board_config.grid_resolution_mm, 0.125) if (is_compact or has_fine_pitch) else board_config.grid_resolution_mm)
    )
    eff_sig_w = (
        min(board_config.signal_trace_width_mm, 0.10)
        if (is_hdi_8l or has_bga)
        else (min(board_config.signal_trace_width_mm, 0.20) if (is_compact or has_fine_pitch) else board_config.signal_trace_width_mm)
    )
    eff_pwr_w = (
        min(board_config.power_trace_width_mm, 0.18)
        if (is_hdi_8l or has_bga)
        else (min(board_config.power_trace_width_mm, 0.35) if (is_compact or has_fine_pitch) else board_config.power_trace_width_mm)
    )
    effective_cfg = replace(
        board_config,
        layers=eff_layers,
        board_width_mm=eff_w,
        board_height_mm=eff_h,
        board_thickness_mm=getattr(netlist_graph, "thickness_mm", board_config.board_thickness_mm),
        single_sided=is_single,
        keep_back_flat=is_single or board_config.keep_back_flat,
        grid_resolution_mm=eff_res,
        clearance_mm=min(board_config.clearance_mm, 0.10) if (is_hdi_8l or has_bga) else board_config.clearance_mm,
        signal_trace_width_mm=eff_sig_w,
        power_trace_width_mm=eff_pwr_w,
        microvia_diameter_mm=min(board_config.microvia_diameter_mm, 0.30) if (is_hdi_8l or has_bga) else board_config.microvia_diameter_mm,
        microvia_drill_mm=min(board_config.microvia_drill_mm, 0.15) if (is_hdi_8l or has_bga) else board_config.microvia_drill_mm,
        gamma_microvia_cost=min(board_config.gamma_microvia_cost, 3.5) if is_hdi_8l else board_config.gamma_microvia_cost,
        edge_margin_mm=min(board_config.edge_margin_mm, 0.8) if is_compact else (min(board_config.edge_margin_mm, 1.2) if is_hdi_8l else board_config.edge_margin_mm),
    )

    k_samples = num_candidates or effective_cfg.num_placement_candidates
    max_iters = max_iterations or effective_cfg.max_ripup_iterations

    work_graph = netlist_graph.clone()
    pre_topo = validate_ground_topology(BoardState(netlist=work_graph, config=effective_cfg))
    if not bool(pre_topo.get("valid", True)):
        work_graph.ground_constraints["mode"] = "continuous_gnd_enforced"
        work_graph.ground_constraints["rejected_remote_star_ground"] = True
        work_graph.ground_constraints["rejected_slot_inductance_nh"] = pre_topo.get("slot_inductance_nh", 0.0)
        work_graph.ground_constraints["rejection_reason"] = pre_topo.get("reason", "")
        work_graph.design_notes.append(f"[rev13 Refusal] {pre_topo.get('reason', '')}")

    # Rev 14: Pre-flight PDN mounting inductance verifier & hostile remote cap under-BGA migration
    if getattr(effective_cfg, "enable_rigid_flex_physics", True):
        migrate_pdn_caps_under_bga(work_graph, effective_cfg)

    # 1. Generate K candidate macro-placements using GATv2 GNN + stochastic noise
    candidates = placer_gnn.sample_placements(
        work_graph, effective_cfg, num_samples=max(1, k_samples - 1)
    )
    if include_seed_candidate:
        seed_placement = {
            cid: (c.x, c.y, c.theta) for cid, c in work_graph.components.items()
        }
        candidates.append(seed_placement)
    else:
        candidates.append(compute_spectral_anchor_placement(work_graph, effective_cfg))

    # 2. Legalize & batch-evaluate every placement with WorldModelUNet V(S_t)
    cand_states: list[BoardState] = []
    cand_tensors: list[torch.Tensor] = []
    for raw_place in candidates:
        legal_place = relax_and_snap_to_grid(
            raw_place,
            work_graph,
            effective_cfg,
            blind_anneal=not include_seed_candidate,
        )
        cand_state = BoardState.from_placement(legal_place, work_graph, effective_cfg)
        cand_states.append(cand_state)
        cand_tensors.append(cand_state.to_tensor())

    _, nn_values = world_model.predict_batch(cand_tensors)

    best_state: BoardState | None = None
    best_score = -1e9
    candidate_scores: list[dict[str, Any]] = []
    doms_present = {getattr(c, "noise_domain", "neutral") for c in work_graph.components.values()}
    has_mixed_signal = ("power_switching" in doms_present) and ("analog_precision" in doms_present)

    for idx, (cand_state, nn_value) in enumerate(zip(cand_states, nn_values)):
        hpwl_mm = compute_hpwl(cand_state.netlist)
        mst_mm = compute_mst_wirelength(cand_state.netlist)
        overlap_errs = check_courtyard_overlaps(cand_state)

        hpwl_scale = max(320.0, 40.0 * len(cand_state.netlist.routable_net_ids))
        norm_wire = max(0.0, 1.0 - (hpwl_mm / hpwl_scale))
        overlap_pen = 0.5 * len(overlap_errs)
        domain_bonus = 0.0
        if has_mixed_signal:
            from env.metrics import compute_ground_return_isolation

            iso_c = compute_ground_return_isolation(cand_state)
            sep_c = iso_c.get("gan_to_adc_separation_mm") or 0.0
            jret_c = float(iso_c.get("j_return_overlap_pct", 0.0))
            req_sep = float(getattr(effective_cfg, "min_domain_separation_mm", 20.0))
            if sep_c >= req_sep:
                domain_bonus = 0.30 - 0.02 * jret_c
            else:
                domain_bonus = -0.80 - 0.40 * ((req_sep - sep_c) / max(1.0, req_sep)) - 0.02 * jret_c
        score = min(1.0, 0.55 * nn_value + 0.45 * norm_wire + domain_bonus - overlap_pen)

        cand_info = {
            "candidate_id": idx + 1,
            "world_model_value": round(nn_value, 4),
            "routability_score": round(score, 4),
            "hpwl_mm": round(hpwl_mm, 2),
            "mst_mm": round(mst_mm, 2),
            "courtyard_errors": len(overlap_errs),
        }
        candidate_scores.append(cand_info)

        if step_callback is not None:
            evt: dict[str, Any] = {
                "event": "candidate_evaluated",
                "candidate": cand_info,
                "board": cand_state.to_summary_dict(),
            }
            if stream_tensors:
                evt.update(_tensor_channels_payload(cand_state, active_net_id=None))
            step_callback(evt)

        if score > best_score:
            best_score = score
            best_state = cand_state

    assert best_state is not None
    candidate_scores.sort(key=lambda c: c["routability_score"], reverse=True)

    # 3. Order nets by criticality
    ordered_nets = rank_nets_by_criticality(best_state.netlist)

    if step_callback is not None:
        init_cong, _ = world_model.predict(best_state.to_tensor())
        evt = {
            "event": "placement_selected",
            "best_score": round(best_score, 4),
            "ordered_nets": [
                {"net_id": nid, "name": best_state.netlist.nets[nid].name}
                for nid in ordered_nets
            ],
            "candidates": candidate_scores,
            "board": best_state.to_summary_dict(),
        }
        if stream_tensors:
            evt.update(
                _tensor_channels_payload(
                    best_state, active_net_id=None, congestion_map=init_cong
                )
            )
        step_callback(evt)

    # 4. Sequential World-Model Rollout with Rip-Up & Reroute
    if work_graph.enclosure_meta.get("preserve_preseeded_traces") and work_graph.raw_segments:
        from env.board_state import TraceSegment as _TraceSeg

        for seg_d in work_graph.raw_segments:
            snid = int(seg_d.get("net_id", 0))
            if snid > 0:
                best_state.routed_traces.setdefault(snid, []).append(
                    _TraceSeg(
                        net_id=snid,
                        layer=int(seg_d.get("layer", 0)),
                        x1_mm=float(seg_d["x1_mm"]),
                        y1_mm=float(seg_d["y1_mm"]),
                        x2_mm=float(seg_d["x2_mm"]),
                        y2_mm=float(seg_d["y2_mm"]),
                        width_mm=float(seg_d.get("width_mm", effective_cfg.signal_trace_width_mm)),
                    )
                )
        best_state._rebuild_occupancy()

    num_layers = best_state.num_layers
    bottom_layer = best_state.bottom_layer
    penalty_field = np.zeros((num_layers, best_state.H, best_state.W), dtype=np.float32)
    drc_errors: list[DRCError] = []
    iterations_used = 0
    plane_net_ids = {pnid for pnid, _ in best_state.plane_assignments.values()}
    prev_conflict_pairs: set[tuple[int, int]] = set()
    coupled_diff_map: dict[int, tuple[int, int]] = {}
    if getattr(best_state, "has_rigid_flex", False) or any(
        "PCIE" in n.name.upper() for n in best_state.netlist.nets.values()
    ):
        for dp_p, dp_n, _ in _find_diff_pairs_in_netlist(best_state.netlist):
            p_name = best_state.netlist.nets[dp_p].name.upper()
            if getattr(best_state, "has_rigid_flex", False) or "PCIE" in p_name:
                if "SENS_" not in p_name and "AIN" not in p_name:
                    coupled_diff_map[dp_p] = (dp_p, dp_n)
                    coupled_diff_map[dp_n] = (dp_p, dp_n)
        if coupled_diff_map:
            ordered_nets = [nid for nid in ordered_nets if nid in coupled_diff_map] + [
                nid for nid in ordered_nets if nid not in coupled_diff_map
            ]

    for attempt in range(1, max_iters + 1):
        iterations_used = attempt
        allow_soft = attempt < max_iters

        for net_id in ordered_nets:
            if best_state.is_routed(net_id):
                continue

            s_tensor = best_state.to_tensor(active_net_id=net_id)
            congestion_map, v_val = world_model.predict(s_tensor)
            if num_layers == 2:
                cost_field = congestion_map + penalty_field
            else:
                cost_field = penalty_field.copy()
                cost_field[0] += congestion_map[0]
                cost_field[bottom_layer] += congestion_map[1]

            if net_id in coupled_diff_map:
                dp_p, dp_n = coupled_diff_map[net_id]
                sp, vp, sn, vn, connected = route_differential_pair(
                    state=best_state,
                    net_p=dp_p,
                    net_n=dp_n,
                    cost_field=cost_field,
                    z_diff_target=float(getattr(effective_cfg, "pcie_diff_target_ohm", 85.0)),
                    allow_soft_collision=allow_soft,
                )
                if sp or vp:
                    best_state.commit_trace(dp_p, sp, vp, plane_connected=False)
                if sn or vn:
                    best_state.commit_trace(dp_n, sn, vn, plane_connected=False)
                segments, vias = (sp, vp) if net_id == dp_p else (sn, vn)
            else:
                segments, vias, connected = octilinear_astar_route(
                    state=best_state,
                    net_id=net_id,
                    cost_field=cost_field,
                    allow_soft_collision=allow_soft,
                )
                is_plane = net_id in plane_net_ids
                if segments or vias or (is_plane and connected):
                    best_state.commit_trace(
                        net_id,
                        segments,
                        vias,
                        plane_connected=(is_plane and connected),
                    )

            if step_callback is not None:
                evt = {
                    "event": "net_routed",
                    "iteration": attempt,
                    "net_id": net_id,
                    "net_name": best_state.netlist.nets[net_id].name,
                    "connected": connected,
                    "segment_count": len(segments),
                    "via_count": len(vias),
                    "world_model_value": round(v_val, 4),
                    "board": best_state.to_summary_dict(),
                }
                if stream_tensors:
                    evt.update(
                        _tensor_channels_payload(
                            best_state,
                            active_net_id=net_id,
                            congestion_map=congestion_map,
                            penalty_field=penalty_field,
                        )
                    )
                step_callback(evt)

        # 5. Fast Symbolic DRC Check + Headless KiCad Verification
        drc_errors = verify_kicad_drc(best_state, run_kicad_cli=True)
        unrouted_nets = [nid for nid in ordered_nets if not best_state.is_routed(nid)]

        if step_callback is not None:
            evt = {
                "event": "drc_checked",
                "iteration": attempt,
                "error_count": len(drc_errors),
                "unrouted_count": len(unrouted_nets),
                "errors": [e.to_dict() for e in drc_errors],
                "board": best_state.to_summary_dict(),
            }
            if stream_tensors:
                evt.update(
                    _tensor_channels_payload(
                        best_state,
                        active_net_id=None,
                        penalty_field=penalty_field,
                    )
                )
            step_callback(evt)

        if len(drc_errors) == 0 and best_state.all_nets_routed():
            break
        if attempt == max_iters:
            break

        # 6. Self-Correction: Rip up offending nets & spike penalty at error coords
        ripped_nets: set[int] = set()
        curr_conflict_pairs: set[tuple[int, int]] = set()
        prioritize_nets: list[int] = []
        for err in drc_errors:
            _add_penalty_hotspot(
                penalty_field, err.layer, err.x_grid, err.y_grid, magnitude=10.0
            )
            if err.error_type == "via_clearance":
                for l_idx in range(num_layers):
                    if l_idx != err.layer:
                        _add_penalty_hotspot(
                            penalty_field, l_idx, err.x_grid, err.y_grid, magnitude=6.0
                        )
            a_nid = err.offending_net_id
            b_nid = err.secondary_net_id
            if a_nid > 0:
                ripped_nets.add(a_nid)
            if b_nid > 0 and (
                a_nid == 0
                or err.error_type == "via_clearance"
                or (a_nid, b_nid) in prev_conflict_pairs
            ):
                ripped_nets.add(b_nid)
            if a_nid > 0 and b_nid > 0:
                curr_conflict_pairs.add((a_nid, b_nid))
                # If a repeated conflict involved a higher-fanout/plane net blocked by another net,
                # route the blocked offending net before the secondary net next pass.
                if (a_nid, b_nid) in prev_conflict_pairs or err.error_type == "via_clearance":
                    prioritize_nets.append(a_nid)
        prev_conflict_pairs = curr_conflict_pairs

        # If any net failed to route at all, rip up an already-routed crossing net
        if unrouted_nets:
            for unid in unrouted_nets:
                # Spike midpoint of unrouted net's MST & rip up last routed net to open corridors
                for candidate_rip in reversed(ordered_nets):
                    if candidate_rip != unid and best_state.is_routed(candidate_rip):
                        ripped_nets.add(candidate_rip)
                        if len(ripped_nets) >= 2:
                            break
            prioritize_nets.extend(unrouted_nets)

        if prioritize_nets:
            seen_p: list[int] = list(dict.fromkeys(prioritize_nets))
            ordered_nets = seen_p + [n for n in ordered_nets if n not in seen_p]

        for r_nid in list(ripped_nets):
            if r_nid in coupled_diff_map:
                ripped_nets.update(coupled_diff_map[r_nid])
        for r_nid in ripped_nets:
            best_state.rip_up_net(r_nid)

        if step_callback is not None and ripped_nets:
            step_callback(
                {
                    "event": "ripup_triggered",
                    "iteration": attempt,
                    "ripped_nets": [
                        {"net_id": nid, "name": best_state.netlist.nets[nid].name}
                        for nid in sorted(ripped_nets)
                    ],
                    "board": best_state.to_summary_dict(),
                }
            )

    # 7. Rev 10 DFM/DFA Craftsmanship Pass & Audit Gate
    if getattr(best_state.config, "enable_dfm_craftsmanship", True):
        apply_dfm_craftsmanship(best_state)
        drc_errors = verify_kicad_drc(best_state, run_kicad_cli=True)

    # 8. Rev 13 & Rev 14 Physics Design Rationale Output (Mixed-Signal / 8L HDI / Rigid-Flex / PDN / Thermal Slots)
    if (
        has_mixed_signal
        or best_state.num_layers >= 8
        or getattr(best_state, "has_rigid_flex", False)
        or bool(getattr(best_state.netlist, "enclosure_meta", {}).get("pdn_violations_remediated"))
        or bool(getattr(best_state, "thermal_isolation_slots", []))
        or bool(getattr(best_state.netlist, "ground_constraints", {}).get("rejected_remote_star_ground", False))
    ):
        generate_physics_design_rationale(best_state)

    kicad_text = serialize_kicad_pcb(best_state)
    best_state.dfm_audit = audit_board_dfm(best_state, kicad_pcb_text=kicad_text)

    saved_path: Path | None = None
    if output_path is not None:
        saved_path = export(best_state, output_path)

    success = len(drc_errors) == 0 and best_state.all_nets_routed()
    elapsed = time.perf_counter() - t_start

    if step_callback is not None:
        final_cong, _ = world_model.predict(best_state.to_tensor())
        evt = {
            "event": "completed",
            "success": success,
            "iterations_used": iterations_used,
            "elapsed_sec": round(elapsed, 3),
            "error_count": len(drc_errors),
            "errors": [e.to_dict() for e in drc_errors],
            "kernel": native_kernel_info(),
            "kicad_pcb": kicad_text,
            "board": best_state.to_summary_dict(),
        }
        if stream_tensors:
            evt.update(
                _tensor_channels_payload(
                    best_state,
                    active_net_id=None,
                    congestion_map=final_cong,
                    penalty_field=penalty_field,
                )
            )
        step_callback(evt)

    return InferenceResult(
        board_state=best_state,
        kicad_pcb_text=kicad_text,
        output_path=saved_path,
        drc_errors=drc_errors,
        candidate_scores=candidate_scores,
        iterations_used=iterations_used,
        elapsed_sec=elapsed,
        success=success,
    )


def run_silicon_codesign_for_graph(
    graph: NetlistGraph,
    prompt: str | None = None,
    target_id: str = "ice40up5k_sg48",
    include_asic: bool = False,
    enable_pin_swap: bool = True,
) -> dict[str, Any] | None:
    """Run rev8 Verilog RTL synthesis, iverilog simulation, and joint .pcf FPGA pin-swap co-design."""
    has_fpga = any(
        any(tok in (c.value or "").upper() for tok in ("ICE40", "FPGA", "ECP5", "MACHXO", "GW1N"))
        and not (target_id == "auto" and "BGA" in (c.footprint or "").upper())
        for c in graph.components.values()
    )
    if not has_fpga and target_id == "none":
        return None
    if not has_fpga and not include_asic:
        return None

    eff_prompt = prompt or "8-channel WS2812 and 8-channel quadrature encoder SPI bridge"
    verilog_bundle = compose_verilog_module(prompt=eff_prompt, run_simulation=True)
    if not verilog_bundle.physics_assessment.synthesizable:
        return {
            "synthesizable": False,
            "module_name": verilog_bundle.module_name,
            "physics_assessment": verilog_bundle.physics_assessment.to_dict(),
            "simulation": verilog_bundle.simulation.to_dict(),
        }

    codesign = (
        optimize_fpga_pcb_codesign(
            graph,
            target_id=target_id if target_id not in ("auto", "none", "sky130_asic") else "ice40up5k_sg48",
            verilog_bundle=verilog_bundle,
        )
        if (has_fpga and enable_pin_swap)
        else None
    )

    asic_res = (
        compile_asic_openlane(verilog_bundle)
        if (include_asic or target_id == "sky130_asic")
        else None
    )

    payload: dict[str, Any] = {
        "synthesizable": True,
        "module_name": verilog_bundle.module_name,
        "template_id": verilog_bundle.template_id,
        "clock_hz": verilog_bundle.clock_hz,
        "rtl_code": verilog_bundle.rtl_code,
        "testbench_code": verilog_bundle.testbench_code,
        "ports": verilog_bundle.ports,
        "register_map": verilog_bundle.register_map,
        "capability_tags": verilog_bundle.capability_tags,
        "physics_assessment": verilog_bundle.physics_assessment.to_dict(),
        "simulation": verilog_bundle.simulation.to_dict(),
    }
    if codesign is not None:
        payload.update(
            {
                "target_id": codesign.target_id,
                "device_name": codesign.device_name,
                "toolchain_mode": codesign.toolchain_mode,
                "timing_passed": codesign.timing_passed,
                "target_clock_mhz": codesign.target_clock_mhz,
                "achieved_fmax_mhz": codesign.achieved_fmax_mhz,
                "timing_slack_ns": codesign.timing_slack_ns,
                "lut4_used": codesign.lut4_used,
                "lut4_capacity": codesign.lut4_capacity,
                "dff_used": codesign.dff_used,
                "io_used": codesign.io_used,
                "io_capacity": codesign.io_capacity,
                "crossings_before": codesign.crossings_before,
                "crossings_after": codesign.crossings_after,
                "crossing_reduction_pct": codesign.crossing_reduction_pct,
                "hpwl_before_mm": codesign.hpwl_before_mm,
                "hpwl_after_mm": codesign.hpwl_after_mm,
                "optimal_fpga_theta": codesign.optimal_fpga_theta,
                "pin_swaps": codesign.pin_swaps,
                "initial_pcf": codesign.initial_pcf,
                "optimized_pcf": codesign.optimized_pcf,
                "bitstream_asc": codesign.bitstream_asc,
                "bitstream_bytes": codesign.bitstream_bytes,
                "bitstream_size_bytes": len(codesign.bitstream_bytes),
                "notes": codesign.notes,
            }
        )
        for note in codesign.notes:
            if note not in graph.design_notes:
                graph.design_notes.append(note)
    if asic_res is not None:
        payload["asic"] = asic_res.to_dict(include_binary=False)
        payload["gds_bytes"] = asic_res.gds_bytes
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run closed-loop PCB Engineering World Model inference"
    )
    parser.add_argument(
        "--board",
        type=str,
        default="mcu_sensor_node",
        help=f"Preset name ({', '.join(FULL_PRESET_CATALOG)}) or path to .kicad_pcb",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT_DIR / "configs" / "default_2layer.yaml",
    )
    parser.add_argument(
        "--layers",
        type=int,
        choices=[2, 4, 6, 8],
        default=None,
        help="Override stackup layer count (2, 4, 6, or 8)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT_DIR / "output_clean.kicad_pcb",
    )
    parser.add_argument("--candidates", type=int, default=8)
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="Synthesize circuit netlist from natural-language hardware specification",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Enable Interactive Pre-Flight Constraint & Trade-Off Negotiator gate",
    )
    parser.add_argument("--width", type=float, default=None, help="Explicit board width in mm")
    parser.add_argument("--height", type=float, default=None, help="Explicit board height in mm")
    parser.add_argument("--thickness", type=float, default=None, help="Explicit FR4 thickness in mm (0.8 / 1.6)")
    parser.add_argument(
        "--single-sided",
        action="store_true",
        default=None,
        help="Enforce keep_back_flat single-sided SMD placement",
    )
    parser.add_argument(
        "--tradeoff-choice",
        type=str,
        default=None,
        help="Apply a feasibility trade-off option (expand_board, upgrade_4layer, double_sided, shrink_passives)",
    )
    parser.add_argument(
        "--kicad-mod",
        type=Path,
        default=None,
        help="Inspect & parse a Class C .kicad_mod footprint file",
    )
    parser.add_argument(
        "--datasheet-json",
        type=Path,
        default=None,
        help="Inspect & compile a Class C datasheet pin-table JSON file",
    )
    parser.add_argument(
        "--schematic",
        type=Path,
        default=None,
        help="rev11: Import external schematic/netlist (.kicad_sch, .net, .cir/.sp, .ato)",
    )
    parser.add_argument(
        "--auto-erc-fix",
        action="store_true",
        help="rev11: Auto-inject missing decoupling caps, USB-C CC resistors, or I2C pull-ups on imported schematics",
    )
    parser.add_argument(
        "--pin-diagram",
        type=Path,
        default=None,
        help="rev11: Import datasheet pin-diagram (.png, .pdf, .csv) and compile ComponentNode",
    )
    parser.add_argument(
        "--step-enclosure",
        type=Path,
        default=None,
        help="rev11: Import external 3D .STEP or .DXF mechanical enclosure (outline, bosses, ports, Z_max map)",
    )
    parser.add_argument(
        "--preserve-locks",
        action="store_true",
        help="rev11: Preserve locked component poses from imported .kicad_pcb and emit Before-vs-After diff report",
    )
    parser.add_argument(
        "--lock",
        type=str,
        default=None,
        help="rev11: Comma-separated component IDs to lock at fixed poses (e.g. U1,J1,MH1)",
    )
    parser.add_argument(
        "--export-diff-report",
        type=Path,
        default=None,
        help="rev11: Optional output path to save Before-vs-After Optimization Diff Report (.md)",
    )
    parser.add_argument(
        "--blind",
        action="store_true",
        help="Disable schematic seed candidate and place strictly from GNN + spectral anchor",
    )
    parser.add_argument(
        "--export-fab-dir",
        type=Path,
        default=None,
        help="Optional directory to write JLCPCB/PCBWay PCBA bundle (.kicad_pcb, .step, bom.csv, positions.csv, + rev8 RTL/bitstream)",
    )
    parser.add_argument(
        "--export-step",
        type=Path,
        default=None,
        help="Optional output path to export an ISO-10303-21 AP214 .step 3D mechanical assembly",
    )
    parser.add_argument(
        "--export-dfm-report",
        type=Path,
        default=None,
        help="Optional output path to export the rev10 DFM/DFA Compromise Report (dfm_report.md)",
    )
    parser.add_argument(
        "--silicon-target",
        type=str,
        default="auto",
        choices=["auto", "none", "ice40up5k_sg48", "ice40up5k_uwg30", "machxo2_1200", "gw1nr9", "rp2040_pio", "sky130_asic"],
        help="rev8 Verilog RTL + FPGA .pcf pin-swap / OpenLane SKY130 ASIC co-design target",
    )
    parser.add_argument(
        "--include-asic",
        action="store_true",
        help="Also emit OpenLane SKY130 GDSII + custom QFN package JSON alongside FPGA bitstream",
    )
    parser.add_argument(
        "--no-pin-swap",
        action="store_true",
        help="Disable joint .pcf FPGA pin-swap optimization (baseline comparison)",
    )
    args = parser.parse_args()

    if args.kicad_mod is not None:
        comp = parse_kicad_mod(args.kicad_mod)
        print(
            f"[04_run_inference] Parsed .kicad_mod '{comp.footprint}': "
            f"{len(comp.pads)} pads, courtyard={comp.width}x{comp.height}mm, rules={comp.special_layout_rules}"
        )
        return

    if args.datasheet_json is not None:
        spec = json.loads(args.datasheet_json.read_text(encoding="utf-8"))
        comp, support = compile_datasheet_pin_table(spec)
        print(
            f"[04_run_inference] Compiled Datasheet '{comp.value}' ({comp.footprint}): "
            f"{len(comp.pads)} pads, rules={comp.special_layout_rules}, inferred_support={len(support)}"
        )
        return

    if args.pin_diagram is not None:
        comp, ext = import_pin_diagram(args.pin_diagram, save_to_library=True)
        print(
            f"[04_run_inference] Imported Pin Diagram '{ext.part_number}' ({ext.package}, fmt={ext.source_format}): "
            f"{len(comp.pads)} pads, rules={ext.special_layout_rules}, support={len(ext.inferred_support)}"
        )
        return

    board_cfg = BoardConfig.from_yaml(args.config)
    enclosure_obj = None
    if args.step_enclosure is not None:
        if args.step_enclosure.suffix.lower() == ".dxf":
            enclosure_obj = parse_dxf_enclosure(args.step_enclosure)
        else:
            enclosure_obj = parse_step_enclosure(args.step_enclosure)
        print(
            f"[04_run_inference] Loaded {enclosure_obj.source_format.upper()} Enclosure: "
            f"{enclosure_obj.board_width_mm}x{enclosure_obj.board_height_mm}mm | "
            f"Bosses={len(enclosure_obj.mounting_bosses)} | Ports={len(enclosure_obj.port_windows)} | "
            f"Z_max Zones={len(enclosure_obj.z_max_regions)}"
        )

    lock_list = [x.strip() for x in args.lock.split(",") if x.strip()] if args.lock else []
    diff_report_md: str | None = None

    if args.schematic is not None:
        graph, sch_audit, sch_fixes = parse_schematic(
            args.schematic,
            layers=args.layers or board_cfg.layers,
            auto_fix=args.auto_erc_fix,
        )
        print(
            f"[04_run_inference] Imported Schematic '{args.schematic.name}': "
            f"Comps={len(graph.components)}, Nets={len(graph.routable_net_ids)}, "
            f"ERC={sch_audit.error_count}err/{sch_audit.warning_count}warn, Fixes={len(sch_fixes)}"
        )
    elif args.prompt:
        neg = negotiate_constraints(
            prompt=args.prompt,
            width_mm=args.width,
            height_mm=args.height,
            thickness_mm=args.thickness,
            layers=args.layers,
            single_sided=args.single_sided,
            tradeoff_choice=args.tradeoff_choice,
            interactive=args.interactive,
        )
        if neg.status == "needs_input":
            print("[04_run_inference] Pre-Flight Constraint Gate requires clarification:")
            for q in neg.questions:
                print(f"  ? {q.prompt} (recommended: {q.recommended})")
            sys.exit(2)
        if neg.status == "tradeoff_required":
            print(
                f"[04_run_inference] Pre-Flight Feasibility Gate REJECTED silent cramming: "
                f"Utilization={neg.area_utilization * 100:.1f}% "
                f"({neg.courtyard_area_mm2:.1f}mm2 / {neg.usable_area_mm2:.1f}mm2)"
            )
            for opt in neg.tradeoffs:
                print(f"  - [{opt.option_id}] {opt.title}: {opt.description}")
            sys.exit(2)
        specs = neg.locked_specs
        graph = compose_netlist_from_prompt(
            args.prompt,
            board_width_mm=float(specs.get("board_width_mm") or board_cfg.board_width_mm),
            board_height_mm=float(specs.get("board_height_mm") or board_cfg.board_height_mm),
            thickness_mm=specs.get("board_thickness_mm"),
            layers=int(specs.get("layers") or args.layers or board_cfg.layers),
            single_sided=specs.get("single_sided"),
            wireless_choice=specs.get("wireless_choice"),
            passive_package=specs.get("passive_package"),
            mounting_holes=specs.get("mounting_holes"),
        )
        erc_errors = verify_erc(graph)
        print(
            f"[04_run_inference] Prompt Composed: Comps={len(graph.components)}, "
            f"Nets={len(graph.routable_net_ids)}, ERC Errors={len(erc_errors)}"
        )
        if erc_errors:
            for err_msg in erc_errors:
                print(f"  [ERC] {err_msg}")
            sys.exit(1)
    else:
        board_path = Path(args.board)
        if board_path.exists():
            graph = SExprParser.parse_file(board_path)
        else:
            graph = build_preset_board(args.board)

    if lock_list:
        for cid in lock_list:
            if cid in graph.components:
                graph.components[cid].is_locked = True
    if enclosure_obj is not None:
        graph = apply_enclosure_constraints(graph, enclosure_obj)

    if args.layers == 8 and args.config == (ROOT_DIR / "configs" / "default_2layer.yaml"):
        cfg_8l = ROOT_DIR / "configs" / "default_8layer.yaml"
        if cfg_8l.exists():
            board_cfg = BoardConfig.from_yaml(cfg_8l)
        else:
            board_cfg = replace(board_cfg, layers=8)
    elif args.layers is not None:
        board_cfg = replace(board_cfg, layers=args.layers)
    elif graph.layers >= 8 and args.config == (ROOT_DIR / "configs" / "default_2layer.yaml"):
        cfg_8l = ROOT_DIR / "configs" / "default_8layer.yaml"
        if cfg_8l.exists():
            board_cfg = BoardConfig.from_yaml(cfg_8l)
        else:
            board_cfg = replace(board_cfg, layers=8)
    elif graph.layers >= 4 and args.config == (ROOT_DIR / "configs" / "default_2layer.yaml"):
        cfg_4l = ROOT_DIR / "configs" / "default_4layer.yaml"
        if cfg_4l.exists():
            board_cfg = BoardConfig.from_yaml(cfg_4l)
        else:
            board_cfg = replace(board_cfg, layers=4)

    silicon_bundle: dict[str, Any] | None = None
    if args.silicon_target != "none":
        silicon_bundle = run_silicon_codesign_for_graph(
            graph=graph,
            prompt=args.prompt,
            target_id=args.silicon_target,
            include_asic=args.include_asic,
            enable_pin_swap=not args.no_pin_swap,
        )
        if silicon_bundle is not None and silicon_bundle.get("synthesizable"):
            sim = silicon_bundle["simulation"]
            print(
                f"[04_run_inference] Silicon Co-Design: Module={silicon_bundle['module_name']} | "
                f"Sim={sim['simulator']}:{'PASS' if sim['passed'] else 'FAIL'} ({sim['checks_passed']} checks) | "
                f"PCF Crossings={silicon_bundle.get('crossings_before', 0)}->{silicon_bundle.get('crossings_after', 0)} "
                f"(-{silicon_bundle.get('crossing_reduction_pct', 0)}%) | "
                f"Fmax={silicon_bundle.get('achieved_fmax_mhz', '--')}MHz"
            )

    if args.preserve_locks or (args.board == "legacy_kicad_reoptimize"):
        opt_out = optimize_imported_board(
            netlist_graph=graph,
            board_config=board_cfg,
            lock_ids=lock_list or None,
            preserve_existing_locks=True,
            target_layers=args.layers,
            num_candidates=args.candidates,
            include_seed_candidate=not args.blind,
        )
        result = opt_out["inference_result"]
        diff_obj = opt_out["diff"]
        diff_report_md = diff_obj.markdown_report
        if args.output is not None:
            export(result.board_state, args.output)
        print(
            f"[04_run_inference] rev11 Re-Optimize Diff: Vias {diff_obj.before.via_count}->{diff_obj.after.via_count} "
            f"({diff_obj.via_reduction_pct:+.1f}%), WL {diff_obj.before.wirelength_mm}->{diff_obj.after.wirelength_mm}mm, "
            f"DRC {diff_obj.before.drc_errors}->{diff_obj.after.drc_errors}, "
            f"DFM {diff_obj.before.dfm_violations}->{diff_obj.after.dfm_violations}, "
            f"LocksVerified={diff_obj.lock_preservation_verified}"
        )
    else:
        result = design_pcb_world_model(
            netlist_graph=graph,
            board_config=board_cfg,
            output_path=args.output,
            num_candidates=args.candidates,
            include_seed_candidate=not args.blind,
        )

    summary = result.board_state.to_summary_dict()
    si = summary["signal_integrity"]
    dfm = summary.get("dfm_audit") or {}
    kinfo = native_kernel_info()
    print(
        f"[04_run_inference] Board={summary['board_name']} ({summary['layers']}L) | "
        f"Routed={summary['routed_nets']}/{summary['total_nets']} | "
        f"DRC Errors={len(result.drc_errors)} | "
        f"DFM={dfm.get('status', '--')} ({dfm.get('craftsmanship_score', '--')}/100) | "
        f"HPWL={summary['hpwl_mm']}mm | "
        f"Wirelength={summary['routed_wirelength_mm']}mm | "
        f"Vias={summary['via_count']} | "
        f"MaxRdc={si['max_dc_resistance_mohm']}mOhm | "
        f"SI={si['si_thermal_status']} | "
        f"Kernel={kinfo['backend']} | "
        f"Time={result.elapsed_sec:.2f}s"
    )
    if result.output_path:
        print(f"[04_run_inference] Exported KiCad PCB: {result.output_path}")
    if args.export_step is not None:
        step_out = export_step_assembly(result.board_state, args.export_step)
        print(f"[04_run_inference] Exported 3D STEP Assembly: {step_out}")
    if args.export_dfm_report is not None:
        args.export_dfm_report.parent.mkdir(parents=True, exist_ok=True)
        args.export_dfm_report.write_text(
            generate_dfm_report_markdown(result.board_state, result.board_state.dfm_audit),
            encoding="utf-8",
        )
        print(f"[04_run_inference] Exported DFM Compromise Report: {args.export_dfm_report}")
    if args.export_diff_report is not None and diff_report_md is not None:
        args.export_diff_report.parent.mkdir(parents=True, exist_ok=True)
        args.export_diff_report.write_text(diff_report_md, encoding="utf-8")
        print(f"[04_run_inference] Exported Before-vs-After Diff Report: {args.export_diff_report}")
    if args.export_fab_dir is not None:
        fab_paths = export_manufacturing_bundle(
            result.board_state, args.export_fab_dir, silicon_bundle=silicon_bundle
        )
        print(
            f"[04_run_inference] Exported Manufacturing Bundle: "
            f"pcb={fab_paths['kicad_pcb']}, step={fab_paths['step_file']}, bom={fab_paths['bom_csv']}, "
            f"cpl={fab_paths['positions_csv']}, dfm={fab_paths['dfm_report']}"
            + (f", rtl={fab_paths['verilog_rtl']}, pcf={fab_paths['pcf']}" if "verilog_rtl" in fab_paths else "")
        )
    if not result.success:
        for err in result.drc_errors:
            print(f"  [DRC] {err.message}")
        sys.exit(1)


if __name__ == "__main__":
    main()
