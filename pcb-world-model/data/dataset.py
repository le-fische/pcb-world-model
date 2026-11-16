"""PyTorch Datasets for MacroPlacerGNN and WorldModelUNet training."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset

from env.config import BoardConfig
from env.sexpr_parser import PRESET_DESCRIPTIONS, NetlistGraph, build_preset_board


class WorldModelDataset(Dataset[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]):
    """Dataset of `(S_t [7, H, W], congestion_target [2, H, W], value_target [1])`."""

    def __init__(self, samples_path: str | Path) -> None:
        path = Path(samples_path)
        if not path.exists():
            raise FileNotFoundError(f"World model samples file not found: {path}")
        payload: dict[str, Any] = torch.load(path, map_location="cpu", weights_only=True)
        self.states: torch.Tensor = payload["states"].float()
        self.congestion: torch.Tensor = payload["congestion"].float()
        self.values: torch.Tensor = payload["values"].float()

    def __len__(self) -> int:
        return int(self.states.size(0))

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return (self.states[idx], self.congestion[idx], self.values[idx])


class PlacementDataset:
    """Collection of NetlistGraph instances for GATv2 macro-placement training."""

    def __init__(self, board_config: BoardConfig | None = None) -> None:
        self.config = board_config or BoardConfig()
        self.graphs: list[NetlistGraph] = [
            build_preset_board(name) for name in PRESET_DESCRIPTIONS
        ]

    def __len__(self) -> int:
        return len(self.graphs)

    def __getitem__(self, idx: int) -> NetlistGraph:
        return self.graphs[idx]
