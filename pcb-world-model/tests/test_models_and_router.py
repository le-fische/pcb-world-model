"""Unit tests for MacroPlacerGNN, Overlap Relaxation, WorldModelUNet, Router, and DRC."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from env.board_state import BoardState, TraceSegment, Via
from env.config import BoardConfig
from env.sexpr_parser import build_preset_board
from models.composer.schematic_composer import compose_netlist_from_prompt, verify_erc
from models.placer.gnn_encoder import netlist_to_tensors
from models.placer.placer_head import MacroPlacerGNN, compute_placement_loss
from models.placer.relaxation import (
    _placement_wire_cost,
    relax_and_snap_to_grid,
)
from models.world.dynamics_unet import WorldModelUNet
from models.world.native_kernel import native_astar_available, native_kernel_info
from models.world.neural_router import (
    _astar_2pin,
    _astar_2pin_python,
    _build_exact_clearance_masks,
    octilinear_astar_route,
)
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


def test_exact_clearance_masks_escape_and_no_via_in_pad() -> None:
    graph = build_preset_board("mcu_sensor_node")
    cfg = BoardConfig()
    state = BoardState(netlist=graph, config=cfg)

    trace_blocked, via_blocked, _, own_pads = _build_exact_clearance_masks(
        state=state,
        net_id=4,
        trace_width_mm=cfg.trace_width_mm,
    )

    # U1 pad "2" (I2C_SDA, SMD) must block via-in-pad but keep F.Cu escape corridor open
    u1 = graph.components["U1"]
    u1_p2 = next(p for p in u1.pads if p.pad_id == "2")
    wx, wy = u1.pad_world_pos(u1_p2)
    gx, gy = state.mm_to_grid(wx, wy)
    assert own_pads[0, gy, gx]
    assert via_blocked[gy, gx]
    assert not trace_blocked[0, gy, gx - 4]

    # Foreign trace on F.Cu blocks F.Cu near its centerline without blocking B.Cu
    state.commit_trace(5, [TraceSegment(5, 0, 10.0, 40.0, 20.0, 40.0, cfg.trace_width_mm)])
    trace_blocked_after, _, _, _ = _build_exact_clearance_masks(
        state=state,
        net_id=4,
        trace_width_mm=cfg.trace_width_mm,
    )
    tx, ty = state.mm_to_grid(15.0, 40.0)
    assert trace_blocked_after[0, ty, tx]
    assert not trace_blocked_after[1, ty, tx]


def test_relaxation_2opt_swap_and_orientation_reduces_cost() -> None:
    graph = build_preset_board("mcu_sensor_node")
    cfg = BoardConfig()
    swapped = {
        cid: (comp.x, comp.y, comp.theta)
        for cid, comp in graph.components.items()
    }
    c1_pose = swapped["C1"]
    r2_pose = swapped["R2"]
    swapped["C1"] = (r2_pose[0], r2_pose[1], 180)
    swapped["R2"] = (c1_pose[0], c1_pose[1], 180)
    init_cost = _placement_wire_cost(graph, swapped)

    legal = relax_and_snap_to_grid(swapped, graph, cfg, iterations=50)
    state = BoardState.from_placement(legal, graph, cfg)
    assert len(check_courtyard_overlaps(state)) == 0
    assert _placement_wire_cost(graph, legal) < init_cost


def test_dynamic_power_vs_signal_trace_widths() -> None:
    graph = build_preset_board("buck_converter")
    cfg = BoardConfig.from_yaml(ROOT_DIR / "configs" / "default_2layer.yaml")
    state = BoardState(netlist=graph, config=cfg)
    cost_field = np.zeros((2, state.H, state.W), dtype=np.float32)

    # Net 1 (VIN_12V) is a power net -> 0.40mm; Net 6 (FB_DIV) is signal -> 0.25mm
    pwr_segs, pwr_vias, pwr_ok = octilinear_astar_route(state, net_id=1, cost_field=cost_field)
    assert pwr_ok and len(pwr_segs) > 0
    assert all(s.width_mm == pytest.approx(0.40) for s in pwr_segs)
    state.commit_trace(1, pwr_segs, pwr_vias)

    sig_segs, sig_vias, sig_ok = octilinear_astar_route(state, net_id=6, cost_field=cost_field)
    assert sig_ok and len(sig_segs) > 0
    assert all(s.width_mm == pytest.approx(0.25) for s in sig_segs)
    state.commit_trace(6, sig_segs, sig_vias)

    assert len(verify_board(state)) == 0


def test_rp2040_minimal_crystal_weight_and_proximity() -> None:
    graph = build_preset_board("rp2040_minimal")
    cfg = BoardConfig.from_yaml(ROOT_DIR / "configs" / "default_2layer.yaml")

    assert len(graph.components) == 14
    assert len(graph.routable_net_ids) == 12
    assert graph.nets[6].name == "XIN" and graph.nets[6].weight == pytest.approx(3.0)
    assert graph.nets[7].name == "XOUT" and graph.nets[7].weight == pytest.approx(3.0)

    seed_pose = {cid: (c.x, c.y, c.theta) for cid, c in graph.components.items()}
    legal = relax_and_snap_to_grid(seed_pose, graph, cfg, iterations=50)
    state = BoardState.from_placement(legal, graph, cfg)
    assert len(check_courtyard_overlaps(state)) == 0

    u1 = state.netlist.components["U1"]
    y1 = state.netlist.components["Y1"]
    assert math.hypot(y1.x - u1.x, y1.y - u1.y) < 12.0


def test_hierarchical_prompt_composer_and_erc() -> None:
    stm_graph = compose_netlist_from_prompt(
        "STM32 MCU with I2C environmental sensor, 3.3V LDO regulator, 2 status LEDs, and host header"
    )
    assert verify_erc(stm_graph) == []
    assert "U1" in stm_graph.components and "STM32" in stm_graph.components["U1"].value
    assert "U2" in stm_graph.components and "U3" in stm_graph.components
    assert {"C1", "C2", "C3"}.issubset(stm_graph.components.keys())
    assert {"R1", "R2", "R3", "R4"}.issubset(stm_graph.components.keys())
    assert {"D1", "D2"}.issubset(stm_graph.components.keys())

    qfn_graph = compose_netlist_from_prompt(
        "ESP32 QFN sensor hub with I2C BME280 and 3.3V LDO"
    )
    assert verify_erc(qfn_graph) == []
    assert len(qfn_graph.components["U1"].pads) == 12
    assert "QFN-12" in qfn_graph.components["U1"].footprint

    # Verify ERC detects single-pin floating nets and missing IC GND pin
    broken = stm_graph.clone()
    broken.nets[4].pins = broken.nets[4].pins[:1]
    for pad in broken.components["U2"].pads:
        if pad.net_id == 3:
            pad.net_id = 0
    erc_errs = verify_erc(broken)
    assert len(erc_errs) >= 2
    assert any("Floating/single-pin" in err for err in erc_errs)
    assert any("U2" in err and "GND" in err for err in erc_errs)


def test_tht_barrel_reuse_and_bcu_reservation_reduces_vias() -> None:
    graph = build_preset_board("usb_uart_bridge")
    cfg = BoardConfig.from_yaml(ROOT_DIR / "configs" / "default_2layer.yaml")
    state = BoardState(netlist=graph, config=cfg)
    cost_field = np.zeros((2, state.H, state.W), dtype=np.float32)

    segs, vias, connected = octilinear_astar_route(state, net_id=1, cost_field=cost_field)
    assert connected
    assert len(segs) > 0
    assert len(vias) == 0
    assert all(s.layer == 0 for s in segs)


def test_native_cpp20_astar_kernel_active_and_parity() -> None:
    assert native_astar_available() is True
    info = native_kernel_info()
    assert info["backend"] == "cpp20_native"

    H, W = 90, 90
    blocked = np.zeros((2, H, W), dtype=bool)
    blocked[0, 20:70, 45] = True  # Force L1->L2->L1 detour across obstacle wall
    via_blocked = np.zeros((H, W), dtype=bool)
    soft = np.zeros((2, H, W), dtype=np.float32)
    cf = np.zeros((2, H, W), dtype=np.float32)
    occ = np.zeros((2, H, W), dtype=np.int32)

    p_native = _astar_2pin(
        H=H, W=W, start_nodes=[(0, 20, 45)], target_nodes=[(0, 70, 45)],
        blocked=blocked, via_blocked=via_blocked, soft_penalty=soft,
        cost_field=cf, occ_layers=occ, net_id=1, gamma_via=6.0,
    )
    p_py = _astar_2pin_python(
        H=H, W=W, start_nodes=[(0, 20, 45)], target_nodes=[(0, 70, 45)],
        blocked=blocked, via_blocked=via_blocked, soft_penalty=soft,
        cost_field=cf, occ_layers=occ, net_id=1, gamma_via=6.0,
    )
    assert p_native is not None and p_py is not None
    assert p_native[0] == (0, 20, 45) and p_native[-1] == (0, 70, 45)
    assert len(p_native) == len(p_py)


def test_4layer_microvia_plane_drop_and_span_drc() -> None:
    graph = build_preset_board("flight_controller_50")
    cfg = BoardConfig.from_yaml(ROOT_DIR / "configs" / "default_4layer.yaml")
    state = BoardState(netlist=graph, config=cfg)
    cost_field = np.zeros((4, state.H, state.W), dtype=np.float32)

    # GND (net 4) maps to In1.GND (layer 1) via direct micro-vias
    segs, vias, connected = octilinear_astar_route(state, net_id=4, cost_field=cost_field)
    assert connected
    assert len(vias) > 0
    assert all(v.via_type == "micro" and v.end_layer == 1 for v in vias)

    # Layer-span DRC check: a micro-via (0->1) allows B.Cu (layer 3) trace underneath,
    # while colliding on F.Cu (layer 0) at the same XY coordinate
    test_state = BoardState(netlist=graph, config=cfg)
    test_state.commit_trace(
        net_id=4,
        segments=[],
        vias=[Via(4, 30.0, 60.0, 0.40, 0.20, start_layer=0, end_layer=1, via_type="micro")],
    )
    test_state.commit_trace(
        net_id=10,
        segments=[TraceSegment(10, 3, 28.0, 60.0, 32.0, 60.0, 0.25)],
        vias=[],
    )
    assert verify_board(test_state) == []

    test_state.commit_trace(
        net_id=11,
        segments=[TraceSegment(11, 0, 28.0, 60.0, 32.0, 60.0, 0.25)],
        vias=[],
    )
    errs = verify_board(test_state)
    assert len(errs) >= 1
    assert any("Trace-to-Via" in e.message for e in errs)


def test_rev7_constraint_negotiator_vague_b7_and_impossible_b8() -> None:
    from models.composer.constraint_negotiator import negotiate_constraints

    # B7: Non-technical wearable brief -> asks targeted clarification questions instead of guessing
    b7_prompt = "Build me a small wireless wearable keychain with motion wake-up, battery charging, and mini display"
    q_res = negotiate_constraints(b7_prompt, interactive=True)
    assert q_res.status == "needs_input"
    assert len(q_res.questions) >= 2
    q_ids = {q.question_id for q in q_res.questions}
    assert {"board_envelope", "placement_stackup", "wireless_architecture"}.issubset(q_ids)

    # Answering B7 locks specs cleanly within <= 72% courtyard utilization
    ready_res = negotiate_constraints(
        b7_prompt,
        width_mm=26.0,
        height_mm=26.0,
        thickness_mm=0.8,
        layers=2,
        single_sided=True,
        mounting_holes=2,
        wireless_choice="esp32_c3fn4_qfn",
        interactive=True,
    )
    assert ready_res.status == "ready"
    assert ready_res.feasible is True
    assert ready_res.area_utilization <= 0.72

    b7_graph = compose_netlist_from_prompt(
        b7_prompt,
        board_width_mm=ready_res.locked_specs["board_width_mm"],
        board_height_mm=ready_res.locked_specs["board_height_mm"],
        thickness_mm=ready_res.locked_specs["board_thickness_mm"],
        layers=ready_res.locked_specs["layers"],
        single_sided=ready_res.locked_specs["single_sided"],
        wireless_choice=ready_res.locked_specs["wireless_choice"],
        mounting_holes=ready_res.locked_specs["mounting_holes"],
    )
    assert verify_erc(b7_graph) == []
    assert b7_graph.width_mm == pytest.approx(26.0)
    assert "J3" in b7_graph.components and b7_graph.components["J3"].side == "bottom"

    # B8: Physically impossible 25x25mm 4-channel relay controller -> rejects cramming & proposes trade-offs
    b8_prompt = "Fit a 4-channel 10A SPDT relay controller with optocouplers and screw terminals inside a 25x25mm 2-layer enclosure"
    fail_res = negotiate_constraints(b8_prompt, width_mm=25.0, height_mm=25.0, layers=2)
    assert fail_res.status == "tradeoff_required"
    assert fail_res.feasible is False
    assert fail_res.area_utilization > 1.0
    option_ids = {opt.option_id for opt in fail_res.tradeoffs}
    assert {"expand_board", "upgrade_4layer", "double_sided", "shrink_passives"}.issubset(option_ids)

    expanded = negotiate_constraints(
        b8_prompt,
        width_mm=25.0,
        height_mm=25.0,
        layers=2,
        tradeoff_choice="expand_board",
    )
    assert expanded.status == "ready"
    assert expanded.feasible is True
    assert expanded.area_utilization <= 0.72
    assert expanded.locked_specs["board_width_mm"] >= 80.0
    assert expanded.locked_specs["board_width_mm"] * expanded.locked_specs["board_height_mm"] >= 5900.0


def test_rev7_special_layout_rules_enforcement() -> None:
    from dataclasses import replace

    from env.component_library import instantiate_device
    from models.world.neural_router import _find_diff_pair_mate_net, _generate_thermal_ep_vias

    # 1. RF antenna keepout & keep_back_flat & edge_lock verification in fast_clearance
    graph = build_preset_board("wearable_oled_keychain_26x26")
    cfg = BoardConfig.from_yaml(ROOT_DIR / "configs" / "default_2layer.yaml")
    cfg = replace(
        cfg,
        board_width_mm=graph.width_mm,
        board_height_mm=graph.height_mm,
        single_sided=True,
        keep_back_flat=True,
        edge_margin_mm=0.8,
        grid_resolution_mm=0.125,
    )
    clean_state = BoardState(netlist=graph, config=cfg)
    assert verify_board(clean_state) == []

    # Violate keep_back_flat by moving a non-rear-pad component to bottom
    bad_graph = graph.clone()
    bad_graph.components["U3"].to_bottom_side()
    bad_state = BoardState(netlist=bad_graph, config=cfg)
    errs = verify_board(bad_state)
    assert any(e.error_type == "back_flat_violation" for e in errs)

    # Violate perimeter_edge_lock by pulling locked USB-C away from board edge
    edge_bad = graph.clone()
    edge_bad.components["J1"].x = 13.0
    edge_bad.components["J1"].y = 13.0
    edge_errs = verify_board(BoardState(netlist=edge_bad, config=cfg))
    assert any(e.error_type == "edge_lock_violation" for e in edge_errs)

    # 2. Automatic GND thermal via stitching inside exposed pad (thermal_via_ep)
    gnd_id = next(nid for nid, n in clean_state.netlist.nets.items() if n.name == "GND")
    ep_vias = _generate_thermal_ep_vias(clean_state, net_id=gnd_id)
    assert len(ep_vias) >= 4
    u1 = clean_state.netlist.components["U1"]
    for v in ep_vias:
        assert abs(v.x_mm - u1.x) < 1.5 and abs(v.y_mm - u1.y) < 1.5

    # 3. Coupled differential pair mate detection (USB_DP <-> USB_DM)
    dp_id = next(nid for nid, n in clean_state.netlist.nets.items() if n.name == "USB_DP")
    dm_id = next(nid for nid, n in clean_state.netlist.nets.items() if n.name == "USB_DM")
    assert _find_diff_pair_mate_net(clean_state, dp_id) == dm_id

    # 4. HV isolation creepage violation check between HV_LINE/AC_LOAD and LV logic trace
    relay_graph = compose_netlist_from_prompt(
        "4-channel relay controller with optocouplers",
        board_width_mm=95.0,
        board_height_mm=88.0,
    )
    hv_cfg = replace(cfg, board_width_mm=95.0, board_height_mm=88.0, single_sided=False, keep_back_flat=False, grid_resolution_mm=0.25)
    hv_state = BoardState(netlist=relay_graph, config=hv_cfg)
    hv_nid = next(nid for nid, net in relay_graph.nets.items() if "HV_COM" in net.name or "HV_NO" in net.name)
    lv_nid = next(nid for nid, net in relay_graph.nets.items() if net.name == "VCC_3V3")
    hv_state.commit_trace(hv_nid, [TraceSegment(hv_nid, 0, 20.0, 20.0, 30.0, 20.0, 0.4)], [])
    hv_state.commit_trace(lv_nid, [TraceSegment(lv_nid, 0, 20.0, 21.0, 30.0, 21.0, 0.25)], [])  # 1.0mm < 2.5mm creepage!
    creep_errs = verify_board(hv_state)
    assert any(e.error_type == "creepage_violation" for e in creep_errs)

    # Confirm RF module also carries keepout polygon
    rf_mod = instantiate_device("ESP32-C3-MINI-1", "U_RF")
    assert "rf_antenna_keepout" in rf_mod.special_layout_rules
    assert len(rf_mod.world_keepout_zones()) >= 1
