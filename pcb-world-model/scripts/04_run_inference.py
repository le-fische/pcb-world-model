"""End-to-End Closed-Loop Neuro-Symbolic PCB Engineering World Model Inference Pipeline."""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from env.board_state import CHANNEL_NAMES, BoardState, downsample_spatial_map
from env.config import BoardConfig
from env.kicad_exporter import export, serialize_kicad_pcb
from env.metrics import compute_hpwl, compute_mst_wirelength, rank_nets_by_criticality
from env.sexpr_parser import PRESET_DESCRIPTIONS, NetlistGraph, SExprParser, build_preset_board
from models.placer.placer_head import MacroPlacerGNN
from models.placer.relaxation import relax_and_snap_to_grid
from models.world.dynamics_unet import WorldModelUNet
from models.world.neural_router import octilinear_astar_route
from verifier.fast_clearance import DRCError, check_courtyard_overlaps
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


def load_models(
    checkpoints_dir: Path | None = None,
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
    _, H, W = penalty_field.shape
    l = max(0, min(1, int(layer)))
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
            congestion_map[1], size=downsample_size
        )
    if penalty_field is not None:
        combined_pen = np.maximum(penalty_field[0], penalty_field[1])
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
) -> InferenceResult:
    """Run the closed-loop Neuro-Symbolic PCB Engineering World Model pipeline."""
    t_start = time.perf_counter()
    if placer_gnn is None or world_model is None:
        loaded_placer, loaded_world = load_models()
        placer_gnn = placer_gnn or loaded_placer
        world_model = world_model or loaded_world

    k_samples = num_candidates or board_config.num_placement_candidates
    max_iters = max_iterations or board_config.max_ripup_iterations

    # 1. Generate K candidate macro-placements using GATv2 GNN + stochastic noise
    candidates = placer_gnn.sample_placements(
        netlist_graph, board_config, num_samples=max(1, k_samples - 1)
    )
    # Include the original schematic/board-seeded topology as one candidate baseline
    seed_placement = {
        cid: (c.x, c.y, c.theta) for cid, c in netlist_graph.components.items()
    }
    candidates.append(seed_placement)

    # 2. Legalize & evaluate every placement with WorldModelUNet V(S_t)
    best_state: BoardState | None = None
    best_score = -1e9
    candidate_scores: list[dict[str, Any]] = []

    for idx, raw_place in enumerate(candidates):
        legal_place = relax_and_snap_to_grid(raw_place, netlist_graph, board_config)
        cand_state = BoardState.from_placement(legal_place, netlist_graph, board_config)
        s_tensor = cand_state.to_tensor()
        nn_value = world_model.predict_value(s_tensor)
        hpwl_mm = compute_hpwl(cand_state.netlist)
        mst_mm = compute_mst_wirelength(cand_state.netlist)
        overlap_errs = check_courtyard_overlaps(cand_state)

        # Calibrated routability combines World Model Critic V(S_t) with geometric feasibility
        norm_wire = max(0.0, 1.0 - (hpwl_mm / 320.0))
        overlap_pen = 0.5 * len(overlap_errs)
        score = max(0.0, min(1.0, 0.55 * nn_value + 0.45 * norm_wire - overlap_pen))

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
        init_cong = world_model.predict_congestion(best_state.to_tensor())
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
    penalty_field = np.zeros((2, best_state.H, best_state.W), dtype=np.float32)
    drc_errors: list[DRCError] = []
    iterations_used = 0

    for attempt in range(1, max_iters + 1):
        iterations_used = attempt
        allow_soft = attempt < max_iters

        for net_id in ordered_nets:
            if best_state.is_routed(net_id):
                continue

            s_tensor = best_state.to_tensor(active_net_id=net_id)
            congestion_map = world_model.predict_congestion(s_tensor)
            v_val = world_model.predict_value(s_tensor)

            segments, vias, connected = octilinear_astar_route(
                state=best_state,
                net_id=net_id,
                cost_field=congestion_map + penalty_field,
                allow_soft_collision=allow_soft,
            )
            if segments:
                best_state.commit_trace(net_id, segments, vias)

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

        # 6. Self-Correction: Rip up offending nets & spike penalty at error coords
        ripped_nets: set[int] = set()
        for err in drc_errors:
            _add_penalty_hotspot(
                penalty_field, err.layer, err.x_grid, err.y_grid, magnitude=10.0
            )
            if err.offending_net_id > 0:
                ripped_nets.add(err.offending_net_id)
            if err.secondary_net_id > 0 and err.offending_net_id == 0:
                ripped_nets.add(err.secondary_net_id)

        # If any net failed to route at all, rip up an already-routed crossing net
        if unrouted_nets:
            for unid in unrouted_nets:
                # Spike midpoint of unrouted net's MST & rip up last routed net to open corridors
                for candidate_rip in reversed(ordered_nets):
                    if candidate_rip != unid and best_state.is_routed(candidate_rip):
                        ripped_nets.add(candidate_rip)
                        if len(ripped_nets) >= 2:
                            break
            # Re-order unrouted nets earlier in next pass
            ordered_nets = unrouted_nets + [n for n in ordered_nets if n not in unrouted_nets]

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

    kicad_text = serialize_kicad_pcb(best_state)
    saved_path: Path | None = None
    if output_path is not None:
        saved_path = export(best_state, output_path)

    success = len(drc_errors) == 0 and best_state.all_nets_routed()
    elapsed = time.perf_counter() - t_start

    if step_callback is not None:
        final_cong = world_model.predict_congestion(best_state.to_tensor())
        evt = {
            "event": "completed",
            "success": success,
            "iterations_used": iterations_used,
            "elapsed_sec": round(elapsed, 3),
            "error_count": len(drc_errors),
            "errors": [e.to_dict() for e in drc_errors],
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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run closed-loop PCB Engineering World Model inference"
    )
    parser.add_argument(
        "--board",
        type=str,
        default="mcu_sensor_node",
        help=f"Preset name ({', '.join(PRESET_DESCRIPTIONS)}) or path to .kicad_pcb",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT_DIR / "configs" / "default_2layer.yaml",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT_DIR / "output_clean.kicad_pcb",
    )
    parser.add_argument("--candidates", type=int, default=8)
    args = parser.parse_args()

    board_cfg = BoardConfig.from_yaml(args.config)
    board_path = Path(args.board)
    if board_path.exists():
        graph = SExprParser.parse_file(board_path)
    else:
        graph = build_preset_board(args.board)

    result = design_pcb_world_model(
        netlist_graph=graph,
        board_config=board_cfg,
        output_path=args.output,
        num_candidates=args.candidates,
    )

    summary = result.board_state.to_summary_dict()
    print(
        f"[04_run_inference] Board={summary['board_name']} | "
        f"Routed={summary['routed_nets']}/{summary['total_nets']} | "
        f"DRC Errors={len(result.drc_errors)} | "
        f"HPWL={summary['hpwl_mm']}mm | "
        f"Wirelength={summary['routed_wirelength_mm']}mm | "
        f"Vias={summary['via_count']} | "
        f"Iters={result.iterations_used} | "
        f"Time={result.elapsed_sec:.2f}s"
    )
    if result.output_path:
        print(f"[04_run_inference] Exported KiCad PCB: {result.output_path}")
    if not result.success:
        for err in result.drc_errors:
            print(f"  [DRC] {err.message}")
        sys.exit(1)


if __name__ == "__main__":
    main()
