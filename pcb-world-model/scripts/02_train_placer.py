"""Train the GATv2 Macro-Placement Composer (MacroPlacerGNN)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from data.dataset import PlacementDataset
from env.config import BoardConfig, TrainConfig
from models.placer.gnn_encoder import netlist_to_tensors
from models.placer.placer_head import MacroPlacerGNN, compute_placement_loss


def train_placer(
    board_config: BoardConfig,
    train_config: TrainConfig,
    checkpoint_path: Path,
    epochs: int | None = None,
) -> Path:
    """Train MacroPlacerGNN using differentiable SmoothHPWL + Overlap + OutOfBounds loss."""
    torch.manual_seed(train_config.seed)
    dataset = PlacementDataset(board_config)
    model = MacroPlacerGNN(
        node_dim=7,
        edge_dim=3,
        hidden_dim=train_config.placer_hidden_dim,
        num_layers=train_config.placer_num_layers,
        heads=train_config.placer_heads,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=train_config.placer_lr, weight_decay=1e-4)
    num_epochs = epochs if epochs is not None else train_config.placer_epochs

    model.train()
    for epoch in range(1, num_epochs + 1):
        epoch_loss = 0.0
        epoch_hpwl = 0.0
        epoch_ovl = 0.0
        for graph in dataset.graphs:
            batch = netlist_to_tensors(graph, board_config)
            optimizer.zero_grad()
            pred_xy, pred_sincos = model(batch, board_config, noise_std=0.05)
            loss, stats = compute_placement_loss(
                pred_xy,
                pred_sincos,
                batch,
                graph,
                board_config,
                lambda_hpwl=train_config.lambda_hpwl,
                lambda_overlap=train_config.lambda_overlap,
                lambda_bound=train_config.lambda_bound,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
            epoch_loss += stats["total_loss"]
            epoch_hpwl += stats["smooth_hpwl"]
            epoch_ovl += stats["overlap_area"]

        n = max(1, len(dataset.graphs))
        if epoch == 1 or epoch % 5 == 0 or epoch == num_epochs:
            print(
                f"[02_train_placer] Epoch {epoch:02d}/{num_epochs} | "
                f"loss={epoch_loss/n:.3f} | smooth_hpwl={epoch_hpwl/n:.2f}mm | "
                f"overlap={epoch_ovl/n:.3f}"
            )

    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "hidden_dim": train_config.placer_hidden_dim,
            "num_layers": train_config.placer_num_layers,
            "heads": train_config.placer_heads,
        },
        checkpoint_path,
    )
    print(f"[02_train_placer] Saved checkpoint to {checkpoint_path}")
    return checkpoint_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Train MacroPlacerGNN")
    parser.add_argument(
        "--board-config",
        type=Path,
        default=ROOT_DIR / "configs" / "default_2layer.yaml",
    )
    parser.add_argument(
        "--train-config",
        type=Path,
        default=ROOT_DIR / "configs" / "train_config.yaml",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=ROOT_DIR / "checkpoints" / "placer_gnn.pt",
    )
    parser.add_argument("--epochs", type=int, default=None)
    args = parser.parse_args()

    board_cfg = BoardConfig.from_yaml(args.board_config)
    train_cfg = TrainConfig.from_yaml(args.train_config)
    train_placer(board_cfg, train_cfg, args.checkpoint, epochs=args.epochs)


if __name__ == "__main__":
    main()
