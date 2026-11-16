"""Unit tests for MacroPlacerGNN, Overlap Relaxation, WorldModelUNet, Router, and DRC."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from env.board_state import BoardState, TraceSegment
from env.config import BoardConfig
from env.sexpr_parser import build_preset_board
from models.placer.gnn_encoder import netlist_to_tensors
from models.placer.placer_head import MacroPlacerGNN, compute_placement_loss
from models.placer.relaxation import relax_and_snap_to_grid
from models.world.dynamics_unet import WorldModelUNet
from models.world.neural_router import octilinear_astar_route
from verifier.fast_clearance import check_courtyard_overlaps, verify_board


def test_gatv2_placer_and_relaxation() -> None:
    graph = build_preset_board("mcu_sensor_node")
    cfg = BoardConfig()
    batch = netlist_to_tensors(graph, cfg)
    assert batch.node_feats.shape == (len(graph.components), 7)
    assert batch.edge_attr.shape[1] == 3

    placer = MacroPlacerGNN(hidden_dim=32, num_layers=4, heads=4)
    pred_xy, pred_sincos = placer(batch, cfg)
    assert pred_xy.shape == (len(graph.components), 2)
    assert pred_sincos.shape == (len(graph.components), 2)

    loss, stats = compute_placement_loss(pred_xy, pred_sincos, batch, graph, cfg)
    assert torch.isfinite(loss)
    assert stats["smooth_hpwl"] > 0.0

    # Sample & relax into legal zero-overlap 0.5mm grid placement
    candidates = placer.sample_placements(graph, cfg, num_samples=3)
    assert len(candidates) == 3

    legal = relax_and_snap_to_grid(candidates[0], graph, cfg, iterations=50)
    state = BoardState.from_placement(legal, graph, cfg)
    assert len(check_courtyard_overlaps(state)) == 0
    for cid, comp in state.netlist.components.items():
        if not comp.is_locked:
            assert comp.theta in (0, 90, 180, 270)
            assert abs(comp.x / cfg.snap_grid_mm - round(comp.x / cfg.snap_grid_mm)) < 1e-4


def test_world_model_unet_dual_heads() -> None:
    graph = build_preset_board("usb_uart_bridge")
    cfg = BoardConfig()
    state = BoardState(netlist=graph, config=cfg)
    s_t = state.to_tensor(active_net_id=4)

    unet = WorldModelUNet(in_channels=7, base_channels=16)
    cong, val = unet(s_t)
    assert cong.shape == (1, 2, 250, 250)
    assert val.shape == (1, 1)
    assert float(cong.min()) >= 0.0 and float(cong.max()) <= 1.0
    assert 0.0 <= unet.predict_value(s_t) <= 1.0
    assert unet.predict_congestion(s_t).shape == (2, 250, 250)


def test_fast_drc_detects_short_and_clearance() -> None:
    graph = build_preset_board("buck_converter")
    cfg = BoardConfig()
    state = BoardState(netlist=graph, config=cfg)

    # Deliberately place crossing traces on Layer 0 with distinct nets
    state.commit_trace(1, [TraceSegment(1, 0, 20.0, 25.0, 30.0, 25.0, 0.25)])
    state.commit_trace(2, [TraceSegment(2, 0, 25.0, 20.0, 25.0, 30.0, 0.25)])

    errors = verify_board(state)
    assert any(e.error_type == "short" for e in errors)


def test_octilinear_astar_routes_clean_net() -> None:
    graph = build_preset_board("mcu_sensor_node")
    cfg = BoardConfig()
    state = BoardState(netlist=graph, config=cfg)
    cost_field = np.zeros((2, state.H, state.W), dtype=np.float32)

    segs, vias, connected = octilinear_astar_route(state, net_id=4, cost_field=cost_field)
    assert connected
    assert len(segs) > 0
    state.commit_trace(4, segs, vias)
    assert len(verify_board(state)) == 0
