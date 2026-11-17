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
    signal_trace_width_mm: float = 0.25
    power_trace_width_mm: float = 0.40
    via_diameter_mm: float = 0.6
    via_drill_mm: float = 0.3
    microvia_diameter_mm: float = 0.40
    microvia_drill_mm: float = 0.20
    buried_via_diameter_mm: float = 0.45
    buried_via_drill_mm: float = 0.20
    copper_thickness_mm: float = 0.035  # 1 oz/ft^2 standard copper
    board_thickness_mm: float = 1.6
    prepreg_thickness_mm: float = 0.20  # Outer-to-inner plane dielectric height in 4-layer
    edge_margin_mm: float = 1.5
    hdi_buildup_type: str = "2+4+2"
    max_lamination_cycles: int = 2
    enforce_staggered_microvias: bool = True
    microvia_stagger_offset_mm: float = 0.25
    alpha_congestion: float = 2.5
    beta_drc_penalty: float = 1.0
    gamma_via_cost: float = 28.0
    gamma_microvia_cost: float = 4.5
    gamma_buried_via_cost: float = 9.0
    enable_internal_planes: bool = True
    single_sided: bool = False
    keep_back_flat: bool = False
    enable_special_layout_rules: bool = True
    creepage_mm: float = 2.5
    diff_pair_gap_mm: float = 0.25
    enable_dfm_craftsmanship: bool = True
    corner_fillet_radius_mm: float = 2.0
    screw_stress_buffer_mm: float = 1.8
    preferred_clearance_mm: float = 0.32
    rework_courtyard_margin_mm: float = 0.8
    enable_teardrops: bool = True
    enable_test_points: bool = True
    enable_mixed_signal_physics: bool = True
    thin_core_thickness_mm: float = 0.10
    max_broadside_cap_pf: float = 0.05
    analog_guard_halo_mm: float = 3.0
    min_domain_separation_mm: float = 20.0
    enable_rigid_flex_physics: bool = True
    rigid_er: float = 4.5
    flex_er: float = 3.4
    flex_prepreg_thickness_mm: float = 0.050
    flex_copper_thickness_mm: float = 0.018
    flex_hatch_fill_ratio: float = 0.50
    pcie_diff_target_ohm: float = 85.0
    pcie_diff_tolerance_pct: float = 5.0
    pcie_max_skew_mm: float = 0.125
    pdn_max_loop_inductance_nh: float = 1.2
    pdn_default_max_droop_mv: float = 65.0
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
