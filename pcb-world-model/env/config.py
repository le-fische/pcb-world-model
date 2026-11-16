"""Configuration dataclasses and lightweight YAML/JSON loader for PCB World Model."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


def _parse_scalar(raw: str) -> Any:
    """Parse a scalar YAML value without requiring PyYAML."""
    val = raw.strip()
    if not val:
        return ""
    if (val.startswith('"') and val.endswith('"')) or (
        val.startswith("'") and val.endswith("'")
    ):
        return val[1:-1]
    lower = val.lower()
    if lower == "true":
        return True
    if lower == "false":
        return False
    if lower in ("null", "none", "~"):
        return None
    try:
        if "." in val or "e" in lower:
            return float(val)
        return int(val)
    except ValueError:
        return val


def load_yaml_dict(path: str | Path) -> dict[str, Any]:
    """Load a flat or simple nested YAML file using PyYAML if available or a built-in parser."""
    file_path = Path(path)
    text = file_path.read_text(encoding="utf-8")
    try:
        import yaml  # type: ignore[import-not-found]

        loaded = yaml.safe_load(text)
        return loaded if isinstance(loaded, dict) else {}
    except ImportError:
        pass

    result: dict[str, Any] = {}
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        # Strip inline comments outside quotes
        in_quote: str | None = None
        clean_chars: list[str] = []
        for ch in stripped:
            if ch in ('"', "'"):
                if in_quote == ch:
                    in_quote = None
                elif in_quote is None:
                    in_quote = ch
            elif ch == "#" and in_quote is None:
                break
            clean_chars.append(ch)
        clean = "".join(clean_chars).strip()
        if ":" not in clean:
            continue
        key, _, raw_val = clean.partition(":")
        key = key.strip()
        raw_val = raw_val.strip()
        if key:
            result[key] = _parse_scalar(raw_val)
    return result


@dataclass
class BoardConfig:
    """Physical board geometry, raster grid, DRC rules, and routing hyperparameters."""

    board_width_mm: float = 50.0
    board_height_mm: float = 50.0
    grid_resolution_mm: float = 0.2
    snap_grid_mm: float = 0.5
    layers: int = 2
    clearance_mm: float = 0.15
    trace_width_mm: float = 0.25
    via_diameter_mm: float = 0.6
    via_drill_mm: float = 0.3
    edge_margin_mm: float = 1.5
    alpha_congestion: float = 2.5
    beta_drc_penalty: float = 1.0
    gamma_via_cost: float = 6.0
    max_ripup_iterations: int = 5
    num_placement_candidates: int = 8

    @property
    def grid_height(self) -> int:
        """Number of grid rows (H) in the spatial canvas S_t."""
        return int(round(self.board_height_mm / self.grid_resolution_mm))

    @property
    def grid_width(self) -> int:
        """Number of grid columns (W) in the spatial canvas S_t."""
        return int(round(self.board_width_mm / self.grid_resolution_mm))

    @classmethod
    def from_yaml(cls, path: str | Path) -> BoardConfig:
        """Load BoardConfig from a YAML file."""
        data = load_yaml_dict(path)
        valid_fields = {f for f in cls.__dataclass_fields__}
        filtered = {k: v for k, v in data.items() if k in valid_fields}
        return cls(**filtered)

    def to_dict(self) -> dict[str, Any]:
        """Serialize configuration to a plain dictionary."""
        return asdict(self)


@dataclass
class TrainConfig:
    """Hyperparameters for training MacroPlacerGNN and WorldModelUNet."""

    placer_hidden_dim: int = 64
    placer_num_layers: int = 4
    placer_heads: int = 4
    placer_lr: float = 0.003
    placer_epochs: int = 25
    lambda_hpwl: float = 1.0
    lambda_overlap: float = 10.0
    lambda_bound: float = 20.0

    world_base_channels: int = 16
    world_lr: float = 0.001
    world_epochs: int = 20
    world_batch_size: int = 4
    seed: int = 42

    @classmethod
    def from_yaml(cls, path: str | Path) -> TrainConfig:
        """Load TrainConfig from a YAML file."""
        data = load_yaml_dict(path)
        valid_fields = {f for f in cls.__dataclass_fields__}
        filtered = {k: v for k, v in data.items() if k in valid_fields}
        return cls(**filtered)

    def to_dict(self) -> dict[str, Any]:
        """Serialize training configuration to a plain dictionary."""
        return asdict(self)
