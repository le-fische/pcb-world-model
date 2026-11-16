"""Unit tests for configs, S-expression parser, KiCad exporter, BoardState S_t tensor, and metrics."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from env.board_state import BoardState, TraceSegment, Via
from env.config import BoardConfig, TrainConfig
from env.kicad_exporter import export, serialize_kicad_pcb
from env.metrics import compute_hpwl, compute_mst_wirelength, rank_nets_by_criticality
from env.sexpr_parser import PRESET_DESCRIPTIONS, SExprParser, build_preset_board


def test_configs_load_from_yaml() -> None:
    board_cfg = BoardConfig.from_yaml(ROOT_DIR / "configs" / "default_2layer.yaml")
    train_cfg = TrainConfig.from_yaml(ROOT_DIR / "configs" / "train_config.yaml")

    assert board_cfg.board_width_mm == 50.0
    assert board_cfg.board_height_mm == 50.0
    assert board_cfg.grid_resolution_mm == 0.2
    assert board_cfg.grid_height == 250
    assert board_cfg.grid_width == 250
    assert board_cfg.clearance_mm == 0.15
    assert train_cfg.placer_num_layers == 4
    assert train_cfg.world_base_channels == 16


@pytest.mark.parametrize("preset_name", list(PRESET_DESCRIPTIONS.keys()))
def test_preset_boards_and_kicad_roundtrip(preset_name: str, tmp_path: Path) -> None:
    graph = build_preset_board(preset_name)
    cfg = BoardConfig()
    state = BoardState(netlist=graph, config=cfg)

    out_file = tmp_path / f"{preset_name}.kicad_pcb"
    export(state, out_file)
    assert out_file.exists()

    parsed = SExprParser.parse_file(out_file)
    assert parsed.width_mm == pytest.approx(50.0, abs=0.05)
    assert parsed.height_mm == pytest.approx(50.0, abs=0.05)
    assert set(parsed.components.keys()) == set(graph.components.keys())
    assert len(parsed.routable_net_ids) == len(graph.routable_net_ids)


def test_board_state_tensor_shape_and_channels() -> None:
    graph = build_preset_board("mcu_sensor_node")
    cfg = BoardConfig()
    state = BoardState(netlist=graph, config=cfg)

    tensor = state.to_tensor(active_net_id=4)
    assert isinstance(tensor, torch.Tensor)
    assert tensor.shape == (7, 250, 250)
    assert tensor.dtype == torch.float32
    assert float(tensor[0].max()) == 1.0  # board_outline
    assert float(tensor[0, 0, 0]) == 0.0  # edge margin keepout
    assert float(tensor[1].max()) == 1.0  # courtyard_mask
    assert float(tensor[2].max()) > 0.0   # copper_L1 pads
    assert float(tensor[5].max()) > 0.5   # active_net_pins Gaussian peak
    assert float(tensor[6].max()) > 0.0   # unrouted_demand rat's nest

    # Commit and rip up trace
    seg = TraceSegment(net_id=4, layer=1, x1_mm=10.0, y1_mm=10.0, x2_mm=20.0, y2_mm=10.0)
    via = Via(net_id=4, x_mm=20.0, y_mm=10.0)
    state.commit_trace(4, [seg], [via])
    assert state.is_routed(4)
    t_after = state.to_tensor(active_net_id=4)
    assert float(t_after[3].max()) > 0.2  # copper_L2 populated
    assert float(t_after[4].max()) == 1.0  # via_mask populated

    state.rip_up_net(4)
    assert not state.is_routed(4)


def test_metrics_and_net_ranking() -> None:
    graph = build_preset_board("buck_converter")
    hpwl = compute_hpwl(graph)
    mst = compute_mst_wirelength(graph)
    ranked = rank_nets_by_criticality(graph)

    assert hpwl > 20.0
    assert mst > 20.0
    assert set(ranked) == set(graph.routable_net_ids)
