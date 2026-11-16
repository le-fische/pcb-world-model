"""Ground-truth spatial PCB BoardState maintaining S_t tensor [C=7, H, W] and primitives."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch

from env.config import BoardConfig
from env.metrics import compute_hpwl, compute_net_mst_pairs, compute_routed_wirelength
from env.sexpr_parser import ComponentNode, NetlistGraph, Pad


@dataclass
class TraceSegment:
    """Single copper trace segment on a discrete routing layer."""

    net_id: int
    layer: int  # 0 = F.Cu (Top), 1 = B.Cu (Bottom)
    x1_mm: float
    y1_mm: float
    x2_mm: float
    y2_mm: float
    width_mm: float = 0.25


@dataclass
class Via:
    """Plated through-hole via connecting F.Cu and B.Cu."""

    net_id: int
    x_mm: float
    y_mm: float
    diameter_mm: float = 0.6
    drill_mm: float = 0.3


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
    """Ground-truth physical simulator maintaining spatial state tensor S_t [7, H, W]."""

    netlist: NetlistGraph
    config: BoardConfig
    routed_traces: dict[int, list[TraceSegment]] = field(default_factory=dict)
    routed_vias: dict[int, list[Via]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.H: int = self.config.grid_height
        self.W: int = self.config.grid_width
        self.res: float = self.config.grid_resolution_mm
        self.occ_L1: np.ndarray = np.zeros((self.H, self.W), dtype=np.int32)
        self.occ_L2: np.ndarray = np.zeros((self.H, self.W), dtype=np.int32)
        self.courtyard_grid: np.ndarray = np.zeros((self.H, self.W), dtype=np.float32)
        self.via_grid: np.ndarray = np.zeros((self.H, self.W), dtype=np.float32)
        self.outline_grid: np.ndarray = self._build_outline_grid()
        self._rebuild_occupancy()

    def _build_outline_grid(self) -> np.ndarray:
        outline = np.zeros((self.H, self.W), dtype=np.float32)
        margin_cells = int(round(self.config.edge_margin_mm / self.res))
        x0 = max(0, margin_cells)
        y0 = max(0, margin_cells)
        x1 = min(self.W - 1, self.W - 1 - margin_cells)
        y1 = min(self.H - 1, self.H - 1 - margin_cells)
        if x0 <= x1 and y0 <= y1:
            outline[y0 : y1 + 1, x0 : x1 + 1] = 1.0
        return outline

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
                _draw_filled_rect(self.occ_L1, pgx0, pgy0, pgx1, pgy1, net_val)
            if pad.layer_mask & 2:
                _draw_filled_rect(self.occ_L2, pgx0, pgy0, pgx1, pgy1, net_val)
            if pad.layer_mask == 3:
                _draw_circle(
                    self.via_grid,
                    wx / self.res,
                    wy / self.res,
                    max(pw, ph) / (2.0 * self.res),
                    1.0,
                )

    def _rebuild_occupancy(self) -> None:
        """Rebuild L1/L2 occupancy, courtyards, and via masks from components and traces."""
        self.occ_L1.fill(0)
        self.occ_L2.fill(0)
        self.courtyard_grid.fill(0.0)
        self.via_grid.fill(0.0)

        for comp in self.netlist.components.values():
            self._rasterize_component(comp)

        for net_id, segs in self.routed_traces.items():
            for seg in segs:
                target = self.occ_L1 if seg.layer == 0 else self.occ_L2
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
                _draw_circle(self.via_grid, vx, vy, r_cells, 1.0)
                _draw_circle(self.occ_L1, vx, vy, r_cells, net_id)
                _draw_circle(self.occ_L2, vx, vy, r_cells, net_id)

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
                    _draw_filled_rect(self.occ_L1, pgx0, pgy0, pgx1, pgy1, pad.net_id)
                if pad.layer_mask & 2:
                    _draw_filled_rect(self.occ_L2, pgx0, pgy0, pgx1, pgy1, pad.net_id)

    def commit_trace(
        self,
        net_id: int,
        segments: list[TraceSegment],
        vias: list[Via] | None = None,
    ) -> None:
        """Commit routed trace segments and vias for a net and update spatial occupancy."""
        self.routed_traces[net_id] = list(segments)
        self.routed_vias[net_id] = list(vias) if vias else []
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
        if removed:
            self._rebuild_occupancy()

    def is_routed(self, net_id: int) -> bool:
        """Check whether `net_id` currently has committed traces."""
        return bool(self.routed_traces.get(net_id))

    def all_nets_routed(self) -> bool:
        """Return True when every routable net has committed traces."""
        return all(self.is_routed(nid) for nid in self.netlist.routable_net_ids)

    def _compute_active_net_pins_channel(self, active_net_id: int | None) -> np.ndarray:
        channel = np.zeros((self.H, self.W), dtype=np.float32)
        target_nets = (
            [active_net_id]
            if active_net_id is not None and active_net_id in self.netlist.nets
            else self.netlist.routable_net_ids
        )
        sigma_cells = 3.5
        radius = 8
        yy_base, xx_base = np.mgrid[-radius : radius + 1, -radius : radius + 1]
        kernel = np.exp(-(xx_base**2 + yy_base**2) / (2.0 * sigma_cells**2)).astype(
            np.float32
        )

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
                    "is_locked": comp.is_locked,
                    "comp_type": comp.comp_type,
                    "value": comp.value,
                    "pads": pads_json,
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

        return {
            "board_name": self.netlist.board_name,
            "width_mm": self.netlist.width_mm,
            "height_mm": self.netlist.height_mm,
            "grid_resolution_mm": self.res,
            "H": self.H,
            "W": self.W,
            "components": components_json,
            "traces": traces_json,
            "vias": vias_json,
            "ratsnest": ratsnest_json,
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
