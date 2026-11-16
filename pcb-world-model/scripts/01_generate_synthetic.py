"""Generate reference `.kicad_pcb` files and synthetic World Model training triplets."""

from __future__ import annotations

import argparse
import math
import random
import sys
from pathlib import Path

import numpy as np
import torch

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from env.board_state import BoardState
from env.config import BoardConfig
from env.kicad_exporter import export
from env.metrics import compute_hpwl, compute_rudy_congestion_target, rank_nets_by_criticality
from env.sexpr_parser import PRESET_DESCRIPTIONS, build_preset_board
from models.placer.relaxation import relax_and_snap_to_grid
from models.world.neural_router import octilinear_astar_route
from verifier.fast_clearance import verify_board


def generate_synthetic_dataset(
    raw_dir: Path,
    synth_dir: Path,
    board_config: BoardConfig,
    samples_per_board: int = 8,
    seed: int = 42,
) -> Path:
    """Export reference KiCad boards and generate synthetic `(S_t, M_{t+1}, V(S_t))` triplets."""
    rng = random.Random(seed)
    raw_dir.mkdir(parents=True, exist_ok=True)
    synth_dir.mkdir(parents=True, exist_ok=True)

    states_list: list[torch.Tensor] = []
    cong_list: list[torch.Tensor] = []
    val_list: list[torch.Tensor] = []

    for preset_name in PRESET_DESCRIPTIONS:
        graph = build_preset_board(preset_name)
        base_state = BoardState(netlist=graph, config=board_config)
        raw_path = raw_dir / f"{preset_name}.kicad_pcb"
        export(base_state, raw_path)
        print(f"[01_generate_synthetic] Exported reference board: {raw_path}")

        base_hpwl = max(1.0, compute_hpwl(graph))

        for s_idx in range(samples_per_board):
            perturbed: dict[str, tuple[float, float, int]] = {}
            for cid, comp in graph.components.items():
                if comp.is_locked:
                    perturbed[cid] = (comp.x, comp.y, comp.theta)
                else:
                    dx = rng.uniform(-7.0, 7.0)
                    dy = rng.uniform(-7.0, 7.0)
                    th = rng.choice([0, 90, 180, 270]) if s_idx > 0 else comp.theta
                    perturbed[cid] = (comp.x + dx, comp.y + dy, th)

            # Mix legalized placements and deliberately crowded placements for rich critic signal
            if s_idx % 4 != 3:
                placement = relax_and_snap_to_grid(
                    perturbed, graph, board_config, iterations=35
                )
            else:
                placement = perturbed

            state = BoardState.from_placement(placement, graph, board_config)

            # Optionally route a random prefix of nets so the World Model sees partial states S_t
            ordered_nets = rank_nets_by_criticality(state.netlist)
            route_fraction = [0.0, 0.35, 0.65, 1.0][s_idx % 4]
            num_to_route = int(round(route_fraction * len(ordered_nets)))
            zero_cost = np.zeros((2, state.H, state.W), dtype=np.float32)

            routed_ok = 0
            for nid in ordered_nets[:num_to_route]:
                segs, vias, ok = octilinear_astar_route(
                    state, nid, zero_cost, allow_soft_collision=False
                )
                if segs:
                    state.commit_trace(nid, segs, vias)
                if ok:
                    routed_ok += 1

            drc_errors = verify_board(state)
            hpwl = compute_hpwl(state.netlist)
            congestion_np = compute_rudy_congestion_target(state)

            # Compute calibrated routability target V(S_t) in [0, 1]
            hpwl_ratio = hpwl / base_hpwl
            drc_penalty = min(0.7, 0.18 * len(drc_errors))
            hpwl_penalty = min(0.35, max(0.0, (hpwl_ratio - 0.9) * 0.25))
            completion_bonus = 0.1 * (routed_ok / max(1, num_to_route)) if num_to_route > 0 else 0.0
            v_target = max(0.05, min(0.98, 0.90 - drc_penalty - hpwl_penalty + completion_bonus))

            active_nid = ordered_nets[num_to_route] if num_to_route < len(ordered_nets) else None
            s_tensor = state.to_tensor(active_net_id=active_nid)

            states_list.append(s_tensor)
            cong_list.append(torch.from_numpy(congestion_np))
            val_list.append(torch.tensor([v_target], dtype=torch.float32))

    out_path = synth_dir / "world_model_samples.pt"
    torch.save(
        {
            "states": torch.stack(states_list, dim=0),
            "congestion": torch.stack(cong_list, dim=0),
            "values": torch.stack(val_list, dim=0),
        },
        out_path,
    )
    print(
        f"[01_generate_synthetic] Saved {len(states_list)} World Model samples to {out_path}"
    )
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic PCB World Model dataset")
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT_DIR / "configs" / "default_2layer.yaml",
        help="Path to board config YAML",
    )
    parser.add_argument("--samples-per-board", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    board_cfg = BoardConfig.from_yaml(args.config)
    generate_synthetic_dataset(
        raw_dir=ROOT_DIR / "data" / "raw_kicad",
        synth_dir=ROOT_DIR / "data" / "synthetic",
        board_config=board_cfg,
        samples_per_board=args.samples_per_board,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
