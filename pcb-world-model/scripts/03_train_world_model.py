"""Train the Dual-Head Latent World Model U-Net (Congestion + Routability Value)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from data.dataset import WorldModelDataset
from env.config import BoardConfig, TrainConfig
from models.world.dynamics_unet import WorldModelUNet


def train_world_model(
    samples_path: Path,
    train_config: TrainConfig,
    checkpoint_path: Path,
    epochs: int | None = None,
) -> Path:
    """Train WorldModelUNet on synthetic `(S_t, M_{t+1}, V(S_t))` transitions."""
    torch.manual_seed(train_config.seed)
    dataset = WorldModelDataset(samples_path)
    loader = DataLoader(
        dataset,
        batch_size=train_config.world_batch_size,
        shuffle=True,
    )

    model = WorldModelUNet(
        in_channels=7,
        base_channels=train_config.world_base_channels,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=train_config.world_lr, weight_decay=1e-4
    )
    num_epochs = epochs if epochs is not None else train_config.world_epochs

    model.train()
    for epoch in range(1, num_epochs + 1):
        total_loss = 0.0
        total_cong = 0.0
        total_val = 0.0
        batches = 0

        for s_t, cong_gt, val_gt in loader:
            optimizer.zero_grad()
            cong_pred, val_pred = model(s_t)
            loss_cong = F.smooth_l1_loss(cong_pred, cong_gt)
            loss_val = F.mse_loss(val_pred, val_gt)
            loss = loss_cong + 0.5 * loss_val
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()

            total_loss += float(loss.item())
            total_cong += float(loss_cong.item())
            total_val += float(loss_val.item())
            batches += 1

        if epoch == 1 or epoch % 5 == 0 or epoch == num_epochs:
            print(
                f"[03_train_world_model] Epoch {epoch:02d}/{num_epochs} | "
                f"loss={total_loss/batches:.4f} | cong_l1={total_cong/batches:.4f} | "
                f"val_mse={total_val/batches:.4f}"
            )

    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "base_channels": train_config.world_base_channels,
            "in_channels": 7,
        },
        checkpoint_path,
    )
    print(f"[03_train_world_model] Saved checkpoint to {checkpoint_path}")
    return checkpoint_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Train WorldModelUNet")
    parser.add_argument(
        "--samples",
        type=Path,
        default=ROOT_DIR / "data" / "synthetic" / "world_model_samples.pt",
    )
    parser.add_argument(
        "--train-config",
        type=Path,
        default=ROOT_DIR / "configs" / "train_config.yaml",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=ROOT_DIR / "checkpoints" / "world_model_unet.pt",
    )
    parser.add_argument("--epochs", type=int, default=None)
    args = parser.parse_args()

    if not args.samples.exists():
        # Auto-generate synthetic samples if user invokes 03 before 01
        from scripts import __name__ as _  # noqa: F401
        import importlib

        gen_mod = importlib.import_module("scripts.01_generate_synthetic")
        board_cfg = BoardConfig.from_yaml(ROOT_DIR / "configs" / "default_2layer.yaml")
        gen_mod.generate_synthetic_dataset(
            raw_dir=ROOT_DIR / "data" / "raw_kicad",
            synth_dir=ROOT_DIR / "data" / "synthetic",
            board_config=board_cfg,
        )

    train_cfg = TrainConfig.from_yaml(args.train_config)
    train_world_model(args.samples, train_cfg, args.checkpoint, epochs=args.epochs)


if __name__ == "__main__":
    main()
