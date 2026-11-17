"""Ground-truth spatial PCB BoardState maintaining S_t tensor [C=7, H, W] and primitives."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch

from env.config import BoardConfig
from env.metrics import (
    compute_broadside_crosstalk_risk,
    compute_ground_return_isolation,
    compute_hpwl,
    compute_net_mst_pairs,
    compute_pdn_decoupling_inductance,
    compute_rigid_flex_impedance_telemetry,
    compute_routed_wirelength,
    compute_signal_integrity_telemetry,
)
from env.sexpr_parser import ComponentNode, NetlistGraph, Pad, SubstrateZone


@dataclass
class TraceSegment:
    """Single copper trace segment on a discrete routing layer."""

    net_id: int
    layer: int  # 2-layer: 0=F.Cu, 1=B.Cu | 4-layer: 0=F.Cu, 1=In1.GND, 2=In2.PWR, 3=B.Cu
    x1_mm: float
    y1_mm: float
    x2_mm: float
    y2_mm: float
    width_mm: float = 0.25


@dataclass
class Via:
    """Plated through-hole, blind/buried, or micro via connecting copper layers."""

    net_id: int
    x_mm: float
    y_mm: float
    diameter_mm: float = 0.6
    drill_mm: float = 0.3
    start_layer: int = 0
    end_layer: int = 1
    via_type: str = "through"  # "through" | "micro" | "buried"


def get_legal_via_spans(
    num_layers: int,
    max_lamination_cycles: int = 2,
    hdi_buildup_type: str = "2+4+2",
) -> set[tuple[int, int]]:
    """Return legal normalized `(lo, hi)` via layer spans under HDI lamination cycle rules."""
    if num_layers <= 2:
        return {(0, 1)}
    if num_layers == 4:
        return {(0, 3), (0, 1), (2, 3), (0, 2), (1, 3)}
    if num_layers == 6:
        return {(0, 5), (0, 1), (1, 2), (3, 4), (4, 5), (1, 4), (0, 4), (1, 5)}

    # 8-layer (2+4+2) HDI sequential buildup:
    # Inner 4L core L3..L6 (indices 2..5) + optional 1 or 2 outer buildup cycles.
    spans: set[tuple[int, int]] = {(0, 7), (2, 5)}
    if max_lamination_cycles >= 1:
        # With 1 buildup cycle: outer buildup microvias only; with 2 cycles: L2<->L3 and L6<->L7 + L1<->L2 and L7<->L8.
        spans.update({(0, 1), (6, 7)})
    if max_lamination_cycles >= 2:
        spans.update({(1, 2), (5, 6)})
    if max_lamination_cycles >= 3 and hdi_buildup_type == "3+2+3":
        spans.update({(2, 3), (4, 5)})
    return spans


def classify_via_span(start_layer: int, end_layer: int, num_layers: int) -> str:
    """Classify a via span `(start_layer, end_layer)` into through, micro, buried, or skip."""
    lo = min(int(start_layer), int(end_layer))
    hi = max(int(start_layer), int(end_layer))
    bottom = max(1, num_layers - 1)
    if lo == 0 and hi == bottom:
        return "through"
    if num_layers >= 4 and hi - lo == 1:
        return "micro"
    if num_layers >= 6 and lo > 0 and hi < bottom:
        return "buried"
    return "skip"


def compute_hdi_via_telemetry(board_state: BoardState) -> dict[str, Any]:
    """Compute HDI buildup, lamination-cycle, and staggered-microvia telemetry."""
    num_layers = getattr(board_state, "num_layers", 2)
    bottom_layer = max(1, num_layers - 1)
    cfg = board_state.config
    max_cycles = int(getattr(cfg, "max_lamination_cycles", 2))
    buildup = str(getattr(cfg, "hdi_buildup_type", "2+4+2"))
    stagger_min = float(getattr(cfg, "microvia_stagger_offset_mm", 0.25))
    legal_spans = get_legal_via_spans(num_layers, max_cycles, buildup)

    micros: list[tuple[Via, int, int]] = []
    microvia_count = 0
    buried_core_via_count = 0
    through_via_count = 0
    illegal_skip_via_count = 0

    for vs in board_state.routed_vias.values():
        for v in vs:
            lo = max(0, min(int(getattr(v, "start_layer", 0)), int(getattr(v, "end_layer", bottom_layer))))
            hi = min(bottom_layer, max(int(getattr(v, "start_layer", 0)), int(getattr(v, "end_layer", bottom_layer))))
            if num_layers >= 8 and (lo, hi) not in legal_spans:
                illegal_skip_via_count += 1
            if lo == 0 and hi == bottom_layer:
                through_via_count += 1
            elif num_layers >= 4 and hi - lo == 1:
                microvia_count += 1
                micros.append((v, lo, hi))
            elif lo > 0 and hi < bottom_layer:
                buried_core_via_count += 1
            else:
                # 4L/6L plane drop
                microvia_count += 1

    stacked_count = 0
    min_stagger_mm: float | None = None
    for i in range(len(micros)):
        va, alo, ahi = micros[i]
        for j in range(i + 1, len(micros)):
            vb, blo, bhi = micros[j]
            if (alo, ahi) == (blo, bhi):
                continue
            if ahi == blo or bhi == alo:
                dist = math.hypot(va.x_mm - vb.x_mm, va.y_mm - vb.y_mm)
                if min_stagger_mm is None or dist < min_stagger_mm:
                    min_stagger_mm = dist
                if dist < stagger_min - 1e-4:
                    stacked_count += 1

    lnames = board_state.layer_names()
    used_indices = {
        int(seg.layer)
        for segs in board_state.routed_traces.values()
        for seg in segs
        if 0 <= int(seg.layer) < len(lnames)
    }
    return {
        "stackup_type": buildup if num_layers >= 8 else f"{num_layers}L",
        "num_layers": num_layers,
        "max_lamination_cycles": max_cycles,
        "enforce_staggered_microvias": bool(getattr(cfg, "enforce_staggered_microvias", True)),
        "microvia_stagger_offset_mm": stagger_min,
        "microvia_count": microvia_count,
        "buried_core_via_count": buried_core_via_count,
        "through_via_count": through_via_count,
        "stacked_microvia_count": stacked_count,
        "illegal_skip_via_count": illegal_skip_via_count,
        "min_observed_stagger_mm": round(min_stagger_mm, 4) if min_stagger_mm is not None else None,
        "layers_used": [lnames[idx] for idx in sorted(used_indices)],
    }


# Precomputed 2D Gaussian kernel (sigma=3.5, radius=8) for active_net_pins channel
_PIN_GAUSS_RADIUS: int = 8
_PIN_GAUSS_SIGMA: float = 3.5
_YY_BASE, _XX_BASE = np.mgrid[
    -_PIN_GAUSS_RADIUS : _PIN_GAUSS_RADIUS + 1,
    -_PIN_GAUSS_RADIUS : _PIN_GAUSS_RADIUS + 1,
]
_PIN_GAUSS_KERNEL: np.ndarray = np.exp(
    -(_XX_BASE**2 + _YY_BASE**2) / (2.0 * _PIN_GAUSS_SIGMA**2)
).astype(np.float32)


CHANNEL_NAMES: tuple[str, ...] = (
    "board_outline",
    "courtyard_mask",
    "copper_L1",
    "copper_L2",
    "via_mask",
    "active_net_pins",
    "unrouted_demand",
)


def _draw_filled_rect(
    grid: np.ndarray,
    gx0: int,
    gy0: int,
    gx1: int,
    gy1: int,
    value: int | float,
) -> None:
    """Fill an axis-aligned rectangle in-place with bounds clipping."""
    H, W = grid.shape
    x_lo = max(0, min(gx0, gx1))
    x_hi = min(W - 1, max(gx0, gx1))
    y_lo = max(0, min(gy0, gy1))
    y_hi = min(H - 1, max(gy0, gy1))
    if x_lo <= x_hi and y_lo <= y_hi:
        grid[y_lo : y_hi + 1, x_lo : x_hi + 1] = value


def _draw_thick_line(
    grid: np.ndarray,
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    radius_cells: int,
    value: int | float,
    accumulate: bool = False,
) -> None:
    """Rasterize a thick line segment onto a 2D numpy grid."""
    H, W = grid.shape
    steps = max(1, int(math.ceil(max(abs(x2 - x1), abs(y2 - y1)) * 2.0)))
    xs = np.linspace(x1, x2, steps + 1)
    ys = np.linspace(y1, y2, steps + 1)
    r = max(0, radius_cells)
    for cx, cy in zip(xs, ys):
        ix = int(round(cx))
        iy = int(round(cy))
        x_lo = max(0, ix - r)
        x_hi = min(W - 1, ix + r)
        y_lo = max(0, iy - r)
        y_hi = min(H - 1, iy + r)
        if x_lo <= x_hi and y_lo <= y_hi:
            if accumulate:
                grid[y_lo : y_hi + 1, x_lo : x_hi + 1] += value
            else:
                grid[y_lo : y_hi + 1, x_lo : x_hi + 1] = value


def _draw_circle(
    grid: np.ndarray,
    cx: float,
    cy: float,
    radius_cells: float,
    value: int | float,
) -> None:
    """Rasterize a filled circle onto a 2D numpy grid."""
    H, W = grid.shape
    r_int = max(1, int(math.ceil(radius_cells)))
    ix = int(round(cx))
    iy = int(round(cy))
    x_lo = max(0, ix - r_int)
    x_hi = min(W - 1, ix + r_int)
    y_lo = max(0, iy - r_int)
    y_hi = min(H - 1, iy + r_int)
    if x_lo > x_hi or y_lo > y_hi:
        return
    yy, xx = np.ogrid[y_lo : y_hi + 1, x_lo : x_hi + 1]
    mask = ((xx - cx) ** 2 + (yy - cy) ** 2) <= (radius_cells + 0.35) ** 2
    sub = grid[y_lo : y_hi + 1, x_lo : x_hi + 1]
    sub[mask] = value


@dataclass
class BoardState:
    """Ground-truth physical simulator maintaining spatial state tensor S_t [7, H, W] (2L or 4L)."""

    netlist: NetlistGraph
    config: BoardConfig
    routed_traces: dict[int, list[TraceSegment]] = field(default_factory=dict)
    routed_vias: dict[int, list[Via]] = field(default_factory=dict)
    plane_connected_nets: set[int] = field(default_factory=set)
    corner_fillet_radius_mm: float = 0.0
    teardrops: list[dict[str, Any]] = field(default_factory=list)
    test_points: list[dict[str, Any]] = field(default_factory=list)
    silkscreen_adjustments: list[dict[str, Any]] = field(default_factory=list)
    pour_cleanup: dict[str, Any] = field(default_factory=dict)
    dfm_compromises: list[dict[str, Any]] = field(default_factory=list)
    dfm_audit: dict[str, Any] | None = None
    thermal_isolation_slots: list[dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.thermal_isolation_slots and getattr(self.netlist, "thermal_isolation_slots", None):
            self.thermal_isolation_slots = [dict(s) for s in self.netlist.thermal_isolation_slots]
        self.res: float = self.config.grid_resolution_mm
        # Respect actual netlist dimensions (e.g. 50x50mm or 80x80mm complex boards)
        board_w = self.netlist.width_mm if self.netlist.width_mm > 0 else self.config.board_width_mm
        board_h = self.netlist.height_mm if self.netlist.height_mm > 0 else self.config.board_height_mm
        self.H: int = max(16, int(round(board_h / self.res)))
        self.W: int = max(16, int(round(board_w / self.res)))
        raw_layers = int(self.config.layers)
        if raw_layers >= 8:
            self.num_layers = 8
        elif raw_layers >= 6:
            self.num_layers = 6
        elif raw_layers >= 4:
            self.num_layers = 4
        else:
            self.num_layers = 2
        self.bottom_layer: int = self.num_layers - 1
        if self.corner_fillet_radius_mm <= 0.0 and getattr(self.config, "enable_dfm_craftsmanship", True):
            req_r = float(getattr(self.config, "corner_fillet_radius_mm", 2.0))
            if req_r > 0.0:
                self.corner_fillet_radius_mm = round(
                    max(1.0, min(2.5, req_r, 0.08 * min(board_w, board_h))), 3
                )

        self.occ_layers: np.ndarray = np.zeros((self.num_layers, self.H, self.W), dtype=np.int32)
        self.via_grid_layers: np.ndarray = np.zeros(
            (self.num_layers, self.H, self.W), dtype=np.float32
        )
        self.keepout_layers: np.ndarray = np.zeros((self.num_layers, self.H, self.W), dtype=bool)
        self.via_grid: np.ndarray = np.zeros((self.H, self.W), dtype=np.float32)
        self.courtyard_grid: np.ndarray = np.zeros((self.H, self.W), dtype=np.float32)
        self.outline_grid: np.ndarray = self._build_outline_grid()
        self.z_max_map: np.ndarray | None = self._build_z_max_map()
        self.zone_map: np.ndarray = self._build_zone_map()
        self.flex_mask: np.ndarray = self.zone_map == 1
        self.has_rigid_flex: bool = bool(self.flex_mask.any())
        self.plane_assignments: dict[int, tuple[int, str]] = self._compute_plane_assignments()
        self._rebuild_occupancy()

    @property
    def occ_L1(self) -> np.ndarray:
        """Top copper occupancy grid (`F.Cu`)."""
        return self.occ_layers[0]

    @property
    def occ_L2(self) -> np.ndarray:
        """Bottom copper occupancy grid (`B.Cu`)."""
        return self.occ_layers[self.bottom_layer]

    def layer_names(self) -> list[str]:
        """Return ordered copper layer names for the active stackup."""
        if self.num_layers >= 8:
            return [
                "F.Cu",
                "In1.GND",
                "In2.SIG1",
                "In3.PWR1",
                "In4.GND2",
                "In5.SIG2",
                "In6.GND3",
                "B.Cu",
            ]
        if self.num_layers >= 6:
            return ["F.Cu", "In1.GND", "In2.Sig1", "In3.Sig2", "In4.PWR", "B.Cu"]
        if self.num_layers >= 4:
            return ["F.Cu", "In1.GND", "In2.PWR", "B.Cu"]
        return ["F.Cu", "B.Cu"]

    def _compute_plane_assignments(self) -> dict[int, tuple[int, str]]:
        """Assign internal planes (In1.GND, In2.PWR / In4.PWR / 8L 2+4+2) when running in >=4L mode."""
        if self.num_layers < 4 or not getattr(self.config, "enable_internal_planes", True):
            return {}
        assignments: dict[int, tuple[int, str]] = {}
        pwr_layer_idx = 3 if self.num_layers >= 8 else (4 if self.num_layers >= 6 else 2)
        gnd_net = next(
            (n for n in self.netlist.nets.values() if n.name.strip().upper() == "GND"),
            None,
        ) or next(
            (n for n in self.netlist.nets.values() if "GND" in n.name.upper()),
            None,
        )
        if gnd_net is not None and gnd_net.net_id > 0:
            assignments[1] = (gnd_net.net_id, gnd_net.name)
            if self.num_layers >= 8:
                assignments[4] = (gnd_net.net_id, gnd_net.name)
                assignments[6] = (gnd_net.net_id, gnd_net.name)

        pwr_candidates = [
            n
            for n in self.netlist.nets.values()
            if n.net_id > 0 and n.is_power_ground and "GND" not in n.name.upper()
        ]
        if pwr_candidates:
            def _pwr_rank(n: Any) -> tuple[int, int, int]:
                up = n.name.upper()
                pref = (
                    0
                    if any(tok in up for tok in ("VCC_3V3", "3V3", "VOUT_5V", "VCC_5V", "VCC", "VDD"))
                    else 1
                )
                return (pref, -len(n.pins), n.net_id)

            pwr_net = min(pwr_candidates, key=_pwr_rank)
            assignments[pwr_layer_idx] = (pwr_net.net_id, pwr_net.name)
        return assignments

    def _build_outline_grid(self) -> np.ndarray:
        outline = np.zeros((self.H, self.W), dtype=np.float32)
        margin_cells = int(round(self.config.edge_margin_mm / self.res))
        x0 = max(0, margin_cells)
        y0 = max(0, margin_cells)
        x1 = min(self.W - 1, self.W - 1 - margin_cells)
        y1 = min(self.H - 1, self.H - 1 - margin_cells)
        if x0 <= x1 and y0 <= y1:
            outline[y0 : y1 + 1, x0 : x1 + 1] = 1.0

        outline_poly = list(getattr(self.netlist, "outline_polygon", []) or [])
        cutouts = list(getattr(self.netlist, "enclosure_cutouts", []) or [])
        if len(outline_poly) >= 3 or cutouts:
            xs_mm = np.arange(self.W, dtype=np.float32) * self.res
            ys_mm = np.arange(self.H, dtype=np.float32) * self.res
            xx, yy = np.meshgrid(xs_mm, ys_mm)

            def _poly_mask(poly: list[tuple[float, float]]) -> np.ndarray:
                inside = np.zeros((self.H, self.W), dtype=bool)
                n_v = len(poly)
                j = n_v - 1
                for i in range(n_v):
                    xi, yi = poly[i]
                    xj, yj = poly[j]
                    cond = ((yi > yy) != (yj > yy)) & (
                        xx < (xj - xi) * (yy - yi) / ((yj - yi) if abs(yj - yi) > 1e-9 else 1e-9) + xi
                    )
                    inside ^= cond
                    j = i
                return inside

            if len(outline_poly) >= 3:
                outline = np.where(_poly_mask(outline_poly), outline, 0.0)
            for cpoly in cutouts:
                if len(cpoly) >= 3:
                    outline = np.where(_poly_mask(cpoly), 0.0, outline)

        r_fillet = float(getattr(self, "corner_fillet_radius_mm", 0.0))
        if r_fillet > 0.0 and len(outline_poly) < 3:
            bw = self.netlist.width_mm if self.netlist.width_mm > 0 else self.config.board_width_mm
            bh = self.netlist.height_mm if self.netlist.height_mm > 0 else self.config.board_height_mm
            r_in = max(0.2, r_fillet - 0.5 * self.config.edge_margin_mm)
            xs_mm = np.arange(self.W, dtype=np.float32) * self.res
            ys_mm = np.arange(self.H, dtype=np.float32) * self.res
            xx, yy = np.meshgrid(xs_mm, ys_mm)
            r_edge = r_fillet
            # Mask out points outside the 4 rounded corners (inset by edge_margin)
            in_tl = (xx < r_edge) & (yy < r_edge) & (((xx - r_edge) ** 2 + (yy - r_edge) ** 2) > (r_in ** 2))
            in_tr = (xx > bw - r_edge) & (yy < r_edge) & (((xx - (bw - r_edge)) ** 2 + (yy - r_edge) ** 2) > (r_in ** 2))
            in_bl = (xx < r_edge) & (yy > bh - r_edge) & (((xx - r_edge) ** 2 + (yy - (bh - r_edge)) ** 2) > (r_in ** 2))
            in_br = (xx > bw - r_edge) & (yy > bh - r_edge) & (((xx - (bw - r_edge)) ** 2 + (yy - (bh - r_edge)) ** 2) > (r_in ** 2))
            outline[in_tl | in_tr | in_bl | in_br] = 0.0
        return outline

    def _build_z_max_map(self) -> np.ndarray | None:
        """Build 2D enclosure ceiling height map Z_max(x, y) in mm when enclosure constraints exist."""
        z_regions = list(getattr(self.netlist, "z_max_regions", []) or [])
        z_default = getattr(self.netlist, "z_max_default_mm", None)
        if not z_regions and z_default is None:
            return None
        base_z = float(z_default) if z_default is not None else 10.0
        z_map = np.full((self.H, self.W), base_z, dtype=np.float32)
        for reg in z_regions:
            gx0 = max(0, int(math.floor(float(reg["x0"]) / self.res)))
            gx1 = min(self.W - 1, int(math.ceil(float(reg["x1"]) / self.res)))
            gy0 = max(0, int(math.floor(float(reg["y0"]) / self.res)))
            gy1 = min(self.H - 1, int(math.ceil(float(reg["y1"]) / self.res)))
            if gx0 <= gx1 and gy0 <= gy1:
                z_map[gy0 : gy1 + 1, gx0 : gx1 + 1] = np.minimum(
                    z_map[gy0 : gy1 + 1, gx0 : gx1 + 1], float(reg["z_max_mm"])
                )
        return z_map

    def _build_zone_map(self) -> np.ndarray:
        """Build 2D spatial substrate zone map `[H, W]` (`0=rigid FR4`, `1=dynamic cross-hatched flex`)."""
        zmap = np.zeros((self.H, self.W), dtype=np.int32)
        for zone in getattr(self.netlist, "substrate_zones", []) or []:
            if getattr(zone, "zone_type", "rigid") != "dynamic_flex":
                continue
            gx0 = max(0, int(math.floor(float(zone.x0_mm) / self.res)))
            gx1 = min(self.W - 1, int(math.ceil(float(zone.x1_mm) / self.res)))
            gy0 = max(0, int(math.floor(float(zone.y0_mm) / self.res)))
            gy1 = min(self.H - 1, int(math.ceil(float(zone.y1_mm) / self.res)))
            if gx0 <= gx1 and gy0 <= gy1:
                zmap[gy0 : gy1 + 1, gx0 : gx1 + 1] = 1
        return zmap

    def zone_at_mm(self, x_mm: float, y_mm: float) -> SubstrateZone | None:
        """Return the SubstrateZone containing `(x_mm, y_mm)` (prioritizing `dynamic_flex`)."""
        zones = list(getattr(self.netlist, "substrate_zones", []) or [])
        if not zones:
            return None
        for z in zones:
            if z.zone_type == "dynamic_flex" and z.x0_mm <= x_mm <= z.x1_mm and z.y0_mm <= y_mm <= z.y1_mm:
                return z
        for z in zones:
            if z.x0_mm <= x_mm <= z.x1_mm and z.y0_mm <= y_mm <= z.y1_mm:
                return z
        return None

    def mm_to_grid(self, x_mm: float, y_mm: float) -> tuple[int, int]:
        """Convert continuous mm coordinates to discrete (gx, gy) grid indices."""
        gx = max(0, min(self.W - 1, int(round(x_mm / self.res))))
        gy = max(0, min(self.H - 1, int(round(y_mm / self.res))))
        return (gx, gy)

    def grid_to_mm(self, gx: int, gy: int) -> tuple[float, float]:
        """Convert discrete (gx, gy) grid indices back to board coordinates in mm."""
        return (round(gx * self.res, 4), round(gy * self.res, 4))

    @classmethod
    def from_placement(
        cls,
        placement: dict[str, tuple[float, float, int]],
        netlist_graph: NetlistGraph,
        board_config: BoardConfig,
    ) -> BoardState:
        """Instantiate a fresh BoardState with component poses applied from placement."""
        cloned = netlist_graph.clone()
        for comp_id, (x_mm, y_mm, theta) in placement.items():
            if comp_id in cloned.components:
                comp = cloned.components[comp_id]
                if not comp.is_locked:
                    comp.x = float(x_mm)
                    comp.y = float(y_mm)
                    comp.theta = int(theta) % 360
        return cls(netlist=cloned, config=board_config)

    def _rasterize_component(self, comp: ComponentNode) -> None:
        ew, eh = comp.effective_size()
        gx0 = int(math.floor((comp.x - ew / 2.0) / self.res))
        gy0 = int(math.floor((comp.y - eh / 2.0) / self.res))
        gx1 = int(math.ceil((comp.x + ew / 2.0) / self.res))
        gy1 = int(math.ceil((comp.y + eh / 2.0) / self.res))
        _draw_filled_rect(self.courtyard_grid, gx0, gy0, gx1, gy1, 1.0)

        for pad in comp.pads:
            wx, wy = comp.pad_world_pos(pad)
            pw, ph = comp.pad_effective_size(pad)
            pgx0 = int(math.floor((wx - pw / 2.0) / self.res))
            pgy0 = int(math.floor((wy - ph / 2.0) / self.res))
            pgx1 = int(math.ceil((wx + pw / 2.0) / self.res))
            pgy1 = int(math.ceil((wy + ph / 2.0) / self.res))
            net_val = pad.net_id if pad.net_id > 0 else -1
            if pad.layer_mask & 1:
                _draw_filled_rect(self.occ_layers[0], pgx0, pgy0, pgx1, pgy1, net_val)
            if pad.layer_mask & 2:
                _draw_filled_rect(self.occ_layers[self.bottom_layer], pgx0, pgy0, pgx1, pgy1, net_val)
            if pad.layer_mask == 3:
                for l_idx in range(self.num_layers):
                    _draw_circle(
                        self.via_grid_layers[l_idx],
                        wx / self.res,
                        wy / self.res,
                        max(pw, ph) / (2.0 * self.res),
                        1.0,
                    )
                    if 0 < l_idx < self.bottom_layer:
                        _draw_filled_rect(self.occ_layers[l_idx], pgx0, pgy0, pgx1, pgy1, net_val)

        for kx0, ky0, kx1, ky1, all_layers, _ in comp.world_keepout_zones():
            kgx0 = max(0, int(math.floor(kx0 / self.res)))
            kgy0 = max(0, int(math.floor(ky0 / self.res)))
            kgx1 = min(self.W - 1, int(math.ceil(kx1 / self.res)))
            kgy1 = min(self.H - 1, int(math.ceil(ky1 / self.res)))
            if kgx0 <= kgx1 and kgy0 <= kgy1:
                if all_layers:
                    self.keepout_layers[:, kgy0 : kgy1 + 1, kgx0 : kgx1 + 1] = True
                else:
                    l_idx = self.bottom_layer if comp.side == "bottom" else 0
                    self.keepout_layers[l_idx, kgy0 : kgy1 + 1, kgx0 : kgx1 + 1] = True

    def _rebuild_occupancy(self) -> None:
        """Rebuild multi-layer occupancy, courtyards, keepouts, and via masks from components and traces."""
        self.occ_layers.fill(0)
        self.courtyard_grid.fill(0.0)
        self.via_grid_layers.fill(0.0)
        self.keepout_layers.fill(False)

        for comp in self.netlist.components.values():
            self._rasterize_component(comp)

        # Rev 14: Block non-flex layers inside DynamicFlexZone bend areas
        for zone in getattr(self.netlist, "substrate_zones", []) or []:
            if getattr(zone, "zone_type", "rigid") != "dynamic_flex":
                continue
            gx0 = max(0, int(math.floor(float(zone.x0_mm) / self.res)))
            gx1 = min(self.W - 1, int(math.ceil(float(zone.x1_mm) / self.res)))
            gy0 = max(0, int(math.floor(float(zone.y0_mm) / self.res)))
            gy1 = min(self.H - 1, int(math.ceil(float(zone.y1_mm) / self.res)))
            allowed_set = set(zone.allowed_layers) if zone.allowed_layers else {0}
            if gx0 <= gx1 and gy0 <= gy1:
                for l_idx in range(self.num_layers):
                    if l_idx not in allowed_set:
                        self.keepout_layers[l_idx, gy0 : gy1 + 1, gx0 : gx1 + 1] = True

        for net_id, segs in self.routed_traces.items():
            for seg in segs:
                l_idx = max(0, min(self.bottom_layer, int(seg.layer)))
                target = self.occ_layers[l_idx]
                r_cells = max(0, int(round((seg.width_mm / 2.0) / self.res)))
                _draw_thick_line(
                    target,
                    seg.x1_mm / self.res,
                    seg.y1_mm / self.res,
                    seg.x2_mm / self.res,
                    seg.y2_mm / self.res,
                    r_cells,
                    net_id,
                )

        for net_id, vias in self.routed_vias.items():
            for via in vias:
                r_cells = max(1.0, (via.diameter_mm / 2.0) / self.res)
                vx = via.x_mm / self.res
                vy = via.y_mm / self.res
                lo = max(0, min(via.start_layer, via.end_layer))
                hi = min(self.bottom_layer, max(via.start_layer, via.end_layer))
                for l_idx in range(lo, hi + 1):
                    _draw_circle(self.via_grid_layers[l_idx], vx, vy, r_cells, 1.0)
                    _draw_circle(self.occ_layers[l_idx], vx, vy, r_cells, net_id)

        # Re-stamp pads cleanly on top so pad net ownership is exact
        for comp in self.netlist.components.values():
            for pad in comp.pads:
                if pad.net_id <= 0:
                    continue
                wx, wy = comp.pad_world_pos(pad)
                pw, ph = comp.pad_effective_size(pad)
                pgx0 = int(math.floor((wx - pw / 2.0) / self.res))
                pgy0 = int(math.floor((wy - ph / 2.0) / self.res))
                pgx1 = int(math.ceil((wx + pw / 2.0) / self.res))
                pgy1 = int(math.ceil((wy + ph / 2.0) / self.res))
                if pad.layer_mask & 1:
                    _draw_filled_rect(self.occ_layers[0], pgx0, pgy0, pgx1, pgy1, pad.net_id)
                if pad.layer_mask & 2:
                    _draw_filled_rect(
                        self.occ_layers[self.bottom_layer], pgx0, pgy0, pgx1, pgy1, pad.net_id
                    )

        for tp in getattr(self, "test_points", []):
            tp_nid = int(tp.get("net_id", 0))
            if tp_nid <= 0:
                continue
            tx = float(tp["x_mm"])
            ty = float(tp["y_mm"])
            tr = float(tp.get("diameter_mm", 1.0)) * 0.5
            tl = int(tp.get("layer", 0))
            t_gx0 = max(0, int(math.floor((tx - tr) / self.res)))
            t_gx1 = min(self.W - 1, int(math.ceil((tx + tr) / self.res)))
            t_gy0 = max(0, int(math.floor((ty - tr) / self.res)))
            t_gy1 = min(self.H - 1, int(math.ceil((ty + tr) / self.res)))
            _draw_filled_rect(self.occ_layers[max(0, min(self.bottom_layer, tl))], t_gx0, t_gy0, t_gx1, t_gy1, tp_nid)

        np.max(self.via_grid_layers, axis=0, out=self.via_grid)

    def commit_trace(
        self,
        net_id: int,
        segments: list[TraceSegment],
        vias: list[Via] | None = None,
        plane_connected: bool = False,
    ) -> None:
        """Commit routed trace segments and vias for a net and update spatial occupancy."""
        self.routed_traces[net_id] = list(segments)
        self.routed_vias[net_id] = list(vias) if vias else []
        if plane_connected:
            self.plane_connected_nets.add(net_id)
        self._rebuild_occupancy()

    def rip_up_net(self, net_id: int) -> None:
        """Remove routed traces and vias for `net_id` and rebuild occupancy grids."""
        removed = False
        if net_id in self.routed_traces:
            del self.routed_traces[net_id]
            removed = True
        if net_id in self.routed_vias:
            del self.routed_vias[net_id]
            removed = True
        if net_id in self.plane_connected_nets:
            self.plane_connected_nets.discard(net_id)
            removed = True
        if removed:
            self._rebuild_occupancy()

    def is_routed(self, net_id: int) -> bool:
        """Check whether `net_id` currently has committed traces, vias, or plane attachment."""
        return (
            bool(self.routed_traces.get(net_id))
            or bool(self.routed_vias.get(net_id))
            or (net_id in self.plane_connected_nets)
        )

    def all_nets_routed(self) -> bool:
        """Return True when every routable net has committed traces or plane attachment."""
        return all(self.is_routed(nid) for nid in self.netlist.routable_net_ids)

    def _compute_active_net_pins_channel(self, active_net_id: int | None) -> np.ndarray:
        channel = np.zeros((self.H, self.W), dtype=np.float32)
        target_nets = (
            [active_net_id]
            if active_net_id is not None and active_net_id in self.netlist.nets
            else self.netlist.routable_net_ids
        )
        radius = _PIN_GAUSS_RADIUS
        kernel = _PIN_GAUSS_KERNEL

        for nid in target_nets:
            net = self.netlist.nets.get(nid)
            if net is None:
                continue
            for comp_id, pad_id in net.pins:
                comp = self.netlist.components.get(comp_id)
                if comp is None:
                    continue
                pad = next((p for p in comp.pads if p.pad_id == pad_id), None)
                if pad is None:
                    continue
                wx, wy = comp.pad_world_pos(pad)
                gx, gy = self.mm_to_grid(wx, wy)
                x0 = max(0, gx - radius)
                x1 = min(self.W - 1, gx + radius)
                y0 = max(0, gy - radius)
                y1 = min(self.H - 1, gy + radius)
                kx0 = x0 - (gx - radius)
                ky0 = y0 - (gy - radius)
                kx1 = kx0 + (x1 - x0) + 1
                ky1 = ky0 + (y1 - y0) + 1
                channel[y0 : y1 + 1, x0 : x1 + 1] = np.maximum(
                    channel[y0 : y1 + 1, x0 : x1 + 1],
                    kernel[ky0:ky1, kx0:kx1],
                )
        return channel

    def _compute_unrouted_demand_channel(self) -> np.ndarray:
        demand = np.zeros((self.H, self.W), dtype=np.float32)
        for nid in self.netlist.routable_net_ids:
            if self.is_routed(nid):
                continue
            pairs = compute_net_mst_pairs(self.netlist, nid)
            for (x1, y1, _), (x2, y2, _) in pairs:
                _draw_thick_line(
                    demand,
                    x1 / self.res,
                    y1 / self.res,
                    x2 / self.res,
                    y2 / self.res,
                    radius_cells=2,
                    value=0.35,
                    accumulate=True,
                )
        return np.clip(demand, 0.0, 1.0)

    def to_tensor(self, active_net_id: int | None = None) -> torch.Tensor:
        """Build the 7-channel World Model Spatial State Tensor S_t of shape [7, H, W]."""
        max_net = max(1, max(self.netlist.nets.keys(), default=1))
        # Normalized net ID encoding in (0, 1] for copper occupancy
        copper_l1 = np.where(
            self.occ_L1 > 0,
            0.25 + 0.75 * (self.occ_L1.astype(np.float32) / float(max_net)),
            np.where(self.occ_L1 < 0, 0.15, 0.0),
        ).astype(np.float32)
        copper_l2 = np.where(
            self.occ_L2 > 0,
            0.25 + 0.75 * (self.occ_L2.astype(np.float32) / float(max_net)),
            np.where(self.occ_L2 < 0, 0.15, 0.0),
        ).astype(np.float32)

        active_pins = self._compute_active_net_pins_channel(active_net_id)
        unrouted_demand = self._compute_unrouted_demand_channel()

        stacked = np.stack(
            [
                self.outline_grid,
                self.courtyard_grid,
                copper_l1,
                copper_l2,
                self.via_grid,
                active_pins,
                unrouted_demand,
            ],
            axis=0,
        ).astype(np.float32)
        return torch.from_numpy(stacked)

    def to_multilayer_tensor(self, active_net_id: int | None = None) -> torch.Tensor:
        """Build multi-layer spatial state tensor of shape `[5 + num_layers, H, W]`."""
        max_net = max(1, max(self.netlist.nets.keys(), default=1))
        copper_channels = [
            np.where(
                self.occ_layers[l_idx] > 0,
                0.25 + 0.75 * (self.occ_layers[l_idx].astype(np.float32) / float(max_net)),
                np.where(self.occ_layers[l_idx] < 0, 0.15, 0.0),
            ).astype(np.float32)
            for l_idx in range(self.num_layers)
        ]
        active_pins = self._compute_active_net_pins_channel(active_net_id)
        unrouted_demand = self._compute_unrouted_demand_channel()
        stacked = np.stack(
            [
                self.outline_grid,
                self.courtyard_grid,
                *copper_channels,
                self.via_grid,
                active_pins,
                unrouted_demand,
            ],
            axis=0,
        ).astype(np.float32)
        return torch.from_numpy(stacked)

    def get_net_pin_endpoints(self, net_id: int) -> list[tuple[float, float, int]]:
        """Return world coordinates (x_mm, y_mm, layer_mask) for every pad on `net_id`."""
        net = self.netlist.nets.get(net_id)
        if net is None:
            return []
        endpoints: list[tuple[float, float, int]] = []
        for comp_id, pad_id in net.pins:
            comp = self.netlist.components.get(comp_id)
            if comp is None:
                continue
            pad = next((p for p in comp.pads if p.pad_id == pad_id), None)
            if pad is None:
                continue
            wx, wy = comp.pad_world_pos(pad)
            endpoints.append((wx, wy, pad.layer_mask))
        return endpoints

    def to_summary_dict(self) -> dict[str, Any]:
        """Serialize board geometry, placed components, routed traces, and metrics for UI streaming."""
        components_json: list[dict[str, Any]] = []
        for comp in self.netlist.components.values():
            ew, eh = comp.effective_size()
            pads_json = []
            for pad in comp.pads:
                wx, wy = comp.pad_world_pos(pad)
                pw, ph = comp.pad_effective_size(pad)
                pads_json.append(
                    {
                        "pad_id": pad.pad_id,
                        "x_mm": round(wx, 3),
                        "y_mm": round(wy, 3),
                        "width": round(pw, 3),
                        "height": round(ph, 3),
                        "net_id": pad.net_id,
                        "net_name": pad.net_name,
                        "layer_mask": pad.layer_mask,
                    }
                )
            components_json.append(
                {
                    "comp_id": comp.comp_id,
                    "x_mm": round(comp.x, 3),
                    "y_mm": round(comp.y, 3),
                    "theta": comp.theta,
                    "width": round(ew, 3),
                    "height": round(eh, 3),
                    "body_width_mm": round(comp.width, 3),
                    "body_height_mm": round(comp.height, 3),
                    "footprint": comp.footprint,
                    "is_locked": comp.is_locked,
                    "comp_type": comp.comp_type,
                    "value": comp.value,
                    "side": comp.side,
                    "special_layout_rules": list(comp.special_layout_rules),
                    "electrical_meta": dict(comp.electrical_meta),
                    "noise_domain": getattr(comp, "noise_domain", "neutral"),
                    "pads": pads_json,
                }
            )

        nets_json: list[dict[str, Any]] = []
        for nid in sorted(self.netlist.nets.keys()):
            if nid <= 0:
                continue
            net = self.netlist.nets[nid]
            nets_json.append(
                {
                    "net_id": nid,
                    "name": net.name,
                    "is_power_ground": bool(net.is_power_ground),
                    "weight": float(net.weight),
                    "noise_domain": getattr(net, "noise_domain", "neutral"),
                    "pins": [{"comp_id": cid, "pad_id": pid} for cid, pid in net.pins],
                }
            )

        traces_json: list[dict[str, Any]] = []
        for net_id, segs in self.routed_traces.items():
            net_name = self.netlist.nets[net_id].name if net_id in self.netlist.nets else f"NET_{net_id}"
            for s in segs:
                traces_json.append(
                    {
                        "net_id": net_id,
                        "net_name": net_name,
                        "layer": s.layer,
                        "x1_mm": round(s.x1_mm, 3),
                        "y1_mm": round(s.y1_mm, 3),
                        "x2_mm": round(s.x2_mm, 3),
                        "y2_mm": round(s.y2_mm, 3),
                        "width_mm": round(s.width_mm, 3),
                    }
                )

        vias_json: list[dict[str, Any]] = []
        for net_id, vias in self.routed_vias.items():
            for v in vias:
                vias_json.append(
                    {
                        "net_id": net_id,
                        "x_mm": round(v.x_mm, 3),
                        "y_mm": round(v.y_mm, 3),
                        "diameter_mm": round(v.diameter_mm, 3),
                        "drill_mm": round(v.drill_mm, 3),
                        "start_layer": int(getattr(v, "start_layer", 0)),
                        "end_layer": int(getattr(v, "end_layer", self.bottom_layer)),
                        "via_type": getattr(v, "via_type", "through"),
                    }
                )

        ratsnest_json: list[dict[str, Any]] = []
        for nid in self.netlist.routable_net_ids:
            if self.is_routed(nid):
                continue
            net_name = self.netlist.nets[nid].name
            for (x1, y1, _), (x2, y2, _) in compute_net_mst_pairs(self.netlist, nid):
                ratsnest_json.append(
                    {
                        "net_id": nid,
                        "net_name": net_name,
                        "x1_mm": round(x1, 3),
                        "y1_mm": round(y1, 3),
                        "x2_mm": round(x2, 3),
                        "y2_mm": round(y2, 3),
                    }
                )

        routable = self.netlist.routable_net_ids
        routed_count = sum(1 for nid in routable if self.is_routed(nid))

        margin = max(0.3, float(self.config.edge_margin_mm))
        pour_x0 = round(margin, 3)
        pour_y0 = round(margin, 3)
        pour_x1 = round(max(margin + 1.0, self.netlist.width_mm - margin), 3)
        pour_y1 = round(max(margin + 1.0, self.netlist.height_mm - margin), 3)
        clr_mm = round(float(self.config.clearance_mm), 3)

        gnd_pour: dict[str, Any] | None = None
        gnd_net = next(
            (n for n in self.netlist.nets.values() if n.name.strip().upper() == "GND"),
            None,
        ) or next(
            (n for n in self.netlist.nets.values() if "GND" in n.name.upper()),
            None,
        )
        if gnd_net is not None:
            gnd_pour = {
                "net_id": gnd_net.net_id,
                "net_name": gnd_net.name,
                "layer": "In1.GND" if self.num_layers >= 4 else "B.Cu",
                "x0_mm": pour_x0,
                "y0_mm": pour_y0,
                "x1_mm": pour_x1,
                "y1_mm": pour_y1,
                "clearance_mm": clr_mm,
            }

        internal_planes: list[dict[str, Any]] = []
        if self.num_layers >= 4:
            layer_names = self.layer_names()
            for l_idx in sorted(self.plane_assignments.keys()):
                pnid, pname = self.plane_assignments[l_idx]
                internal_planes.append(
                    {
                        "layer_idx": l_idx,
                        "layer": layer_names[l_idx],
                        "net_id": pnid,
                        "net_name": pname,
                        "x0_mm": pour_x0,
                        "y0_mm": pour_y0,
                        "x1_mm": pour_x1,
                        "y1_mm": pour_y1,
                        "clearance_mm": clr_mm,
                    }
                )

        si_telemetry = compute_signal_integrity_telemetry(self)

        keepouts_json: list[dict[str, Any]] = []
        for comp in self.netlist.components.values():
            for kx0, ky0, kx1, ky1, all_layers, kind in comp.world_keepout_zones():
                keepouts_json.append(
                    {
                        "comp_id": comp.comp_id,
                        "x0_mm": round(kx0, 3),
                        "y0_mm": round(ky0, 3),
                        "x1_mm": round(kx1, 3),
                        "y1_mm": round(ky1, 3),
                        "all_layers": all_layers,
                        "kind": kind,
                    }
                )

        screw_stress_buf = float(getattr(self.config, "screw_stress_buffer_mm", 1.8))
        screw_keepouts_json: list[dict[str, Any]] = []
        for comp in self.netlist.components.values():
            if comp.comp_type == "mounting_hole":
                drill_mm = float(comp.electrical_meta.get("drill_mm", min(comp.width, comp.height)))
                screw_keepouts_json.append(
                    {
                        "comp_id": comp.comp_id,
                        "x_mm": round(comp.x, 3),
                        "y_mm": round(comp.y, 3),
                        "drill_mm": round(drill_mm, 3),
                        "stress_radius_mm": round(0.5 * (drill_mm + screw_stress_buf), 3),
                    }
                )

        from verifier.fast_clearance import check_enclosure_collisions, validate_ground_topology

        enc_cols = [e.to_dict() for e in check_enclosure_collisions(self)]
        ground_topo = validate_ground_topology(self)
        ground_iso = compute_ground_return_isolation(self)
        broadside = compute_broadside_crosstalk_risk(self)
        rigid_flex_imp = compute_rigid_flex_impedance_telemetry(self)
        pdn_decoupling = compute_pdn_decoupling_inductance(self)
        zones_json = [
            {
                "zone_id": z.zone_id,
                "zone_type": z.zone_type,
                "x0_mm": round(z.x0_mm, 3),
                "y0_mm": round(z.y0_mm, 3),
                "x1_mm": round(z.x1_mm, 3),
                "y1_mm": round(z.y1_mm, 3),
                "er": z.er,
                "prepreg_h_mm": z.prepreg_h_mm,
                "layers": z.layers,
                "hatch_fill_ratio": z.hatch_fill_ratio,
                "hatch_angle_deg": z.hatch_angle_deg,
                "allow_components": z.allow_components,
                "allow_vias": z.allow_vias,
                "allowed_layers": list(z.allowed_layers),
            }
            for z in (getattr(self.netlist, "substrate_zones", []) or [])
        ]

        return {
            "board_name": self.netlist.board_name,
            "width_mm": self.netlist.width_mm,
            "height_mm": self.netlist.height_mm,
            "thickness_mm": getattr(self.netlist, "thickness_mm", getattr(self.config, "board_thickness_mm", 1.6)),
            "single_sided": getattr(self.netlist, "single_sided", getattr(self.config, "single_sided", False)),
            "grid_resolution_mm": self.res,
            "layers": self.num_layers,
            "layer_names": self.layer_names(),
            "H": self.H,
            "W": self.W,
            "corner_fillet_radius_mm": round(float(getattr(self, "corner_fillet_radius_mm", 0.0)), 3),
            "outline_polygon": [list(pt) for pt in getattr(self.netlist, "outline_polygon", [])],
            "enclosure_cutouts": [[list(pt) for pt in poly] for poly in getattr(self.netlist, "enclosure_cutouts", [])],
            "z_max_regions": list(getattr(self.netlist, "z_max_regions", [])),
            "z_max_default_mm": getattr(self.netlist, "z_max_default_mm", None),
            "port_anchors": list(getattr(self.netlist, "port_anchors", [])),
            "enclosure_collisions": enc_cols,
            "components": components_json,
            "nets": nets_json,
            "traces": traces_json,
            "vias": vias_json,
            "ratsnest": ratsnest_json,
            "keepouts": keepouts_json,
            "screw_keepouts": screw_keepouts_json,
            "thermal_isolation_slots": list(getattr(self, "thermal_isolation_slots", [])),
            "teardrops": list(getattr(self, "teardrops", [])),
            "test_points": list(getattr(self, "test_points", [])),
            "silkscreen_adjustments": list(getattr(self, "silkscreen_adjustments", [])),
            "pour_cleanup": dict(getattr(self, "pour_cleanup", {})),
            "dfm_compromises": list(getattr(self, "dfm_compromises", [])),
            "dfm_audit": getattr(self, "dfm_audit", None),
            "gnd_pour": gnd_pour,
            "internal_planes": internal_planes,
            "signal_integrity": si_telemetry,
            "hdi_telemetry": compute_hdi_via_telemetry(self),
            "ground_return_isolation": ground_iso,
            "broadside_crosstalk": broadside,
            "ground_topology": ground_topo,
            "substrate_zones": zones_json,
            "rigid_flex_impedance": rigid_flex_imp,
            "pdn_decoupling": pdn_decoupling,
            "design_notes": list(getattr(self.netlist, "design_notes", [])),
            "design_rationale": list(getattr(self.netlist, "design_rationale", [])),
            "routed_nets": routed_count,
            "total_nets": len(routable),
            "hpwl_mm": round(compute_hpwl(self.netlist), 2),
            "routed_wirelength_mm": round(compute_routed_wirelength(self), 2),
            "via_count": len(vias_json),
        }


def downsample_spatial_map(arr: np.ndarray, size: int = 50) -> list[list[float]]:
    """Downsample a 2D spatial array [H, W] to [size, size] rounded floats for WebSocket streaming."""
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D array, got shape {arr.shape}")
    tensor = torch.from_numpy(arr.astype(np.float32)).unsqueeze(0).unsqueeze(0)
    pooled = torch.nn.functional.adaptive_avg_pool2d(tensor, (size, size))
    out = pooled.squeeze(0).squeeze(0).numpy()
    return np.round(out, 3).tolist()
