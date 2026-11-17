"""Integration tests for closed-loop World Model inference pipeline and FastAPI/WebSocket API."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from env.config import BoardConfig
from env.sexpr_parser import PRESET_DESCRIPTIONS, SExprParser, build_preset_board
from models.composer.schematic_composer import compose_netlist_from_prompt, verify_erc
from ui.server import app

_inf_mod = importlib.import_module("scripts.04_run_inference")
design_pcb_world_model = _inf_mod.design_pcb_world_model


@pytest.mark.parametrize("preset_name", list(PRESET_DESCRIPTIONS.keys()))
def test_end_to_end_closed_loop_zero_drc(preset_name: str, tmp_path: Path) -> None:
    cfg = BoardConfig.from_yaml(ROOT_DIR / "configs" / "default_2layer.yaml")
    graph = build_preset_board(preset_name)
    out_file = tmp_path / f"{preset_name}_clean.kicad_pcb"

    events: list[str] = []
    result = design_pcb_world_model(
        netlist_graph=graph,
        board_config=cfg,
        output_path=out_file,
        num_candidates=4,
        step_callback=lambda e: events.append(str(e.get("event"))),
    )

    assert result.success
    assert len(result.drc_errors) == 0
    assert result.board_state.all_nets_routed()
    assert out_file.exists()
    assert "candidate_evaluated" in events
    assert "placement_selected" in events
    assert "net_routed" in events
    assert "drc_checked" in events
    assert "completed" in events

    # Verify exported .kicad_pcb parses cleanly back into NetlistGraph with routed segments
    parsed = SExprParser.parse_file(out_file)
    assert len(parsed.raw_segments) > 0


def test_fastapi_endpoints_and_websocket() -> None:
    client = TestClient(app)

    home = client.get("/")
    assert home.status_code == 200
    assert "PCB World Model" in home.text

    presets_resp = client.get("/api/presets")
    assert presets_resp.status_code == 200
    presets = presets_resp.json()
    assert len(presets) == len(PRESET_DESCRIPTIONS)

    run_resp = client.post(
        "/api/run",
        json={"board": "buck_converter", "candidates": 4, "max_iterations": 5},
    )
    assert run_resp.status_code == 200
    run_data = run_resp.json()
    assert run_data["success"] is True
    assert run_data["error_count"] == 0
    assert "(kicad_pcb" in run_data["kicad_pcb"]

    export_resp = client.get("/api/export-kicad")
    assert export_resp.status_code == 200
    assert "(segment" in export_resp.text

    with client.websocket_connect("/ws/rollout") as ws:
        ws.send_json(
            {
                "board": "buck_converter",
                "candidates": 3,
                "max_iterations": 5,
                "step_delay_ms": 0,
            }
        )
        seen_events: list[str] = []
        while True:
            msg = ws.receive_json()
            seen_events.append(msg["event"])
            if msg["event"] == "completed":
                assert msg["success"] is True
                assert msg["error_count"] == 0
                assert "channels" in msg
                assert len(msg["channels"]) == 7
                break
        assert "placement_selected" in seen_events
        assert "net_routed" in seen_events


@pytest.mark.parametrize("preset_name", ["buck_converter", "mcu_sensor_node"])
def test_blind_zero_hint_placement_completes_zero_drc(preset_name: str) -> None:
    cfg = BoardConfig.from_yaml(ROOT_DIR / "configs" / "default_2layer.yaml")
    graph = build_preset_board(preset_name)
    for comp in graph.components.values():
        if not comp.is_locked:
            comp.x = 25.0
            comp.y = 25.0
            comp.theta = 0

    result = design_pcb_world_model(
        netlist_graph=graph,
        board_config=cfg,
        num_candidates=6,
        include_seed_candidate=False,
    )
    assert result.success
    assert len(result.drc_errors) == 0
    assert result.board_state.all_nets_routed()


def test_prompt_to_kicad_end_to_end_and_api_compose(tmp_path: Path) -> None:
    cfg = BoardConfig.from_yaml(ROOT_DIR / "configs" / "default_2layer.yaml")
    graph = compose_netlist_from_prompt(
        "ESP32 QFN sensor hub with I2C BME280, 3.3V LDO, and 2 status LEDs"
    )
    assert verify_erc(graph) == []

    out_file = tmp_path / "prompt_qfn.kicad_pcb"
    result = design_pcb_world_model(
        netlist_graph=graph,
        board_config=cfg,
        output_path=out_file,
        num_candidates=4,
    )
    assert result.success
    assert len(result.drc_errors) == 0
    assert result.board_state.all_nets_routed()
    assert out_file.exists()

    client = TestClient(app)
    resp = client.post(
        "/api/compose",
        json={
            "prompt": "STM32 MCU with I2C sensor, 3.3V LDO, and 2 status LEDs",
            "candidates": 4,
            "max_iterations": 5,
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["erc_errors"] == []
    assert data["success"] is True
    assert data["error_count"] == 0
    assert "(kicad_pcb" in data["kicad_pcb"]
    assert "(segment" in data["kicad_pcb"]
    assert '(zone (net 3) (net_name "GND") (layer "B.Cu")' in data["kicad_pcb"]


def test_manufacturing_bundle_and_live_prompt_websocket() -> None:
    client = TestClient(app)

    # /api/run with prompt parameter
    run_prompt_resp = client.post(
        "/api/run",
        json={
            "prompt": "RP2040 sensor controller with I2C IMU, 3.3V LDO, 2 LEDs, and USB-C",
            "candidates": 4,
            "max_iterations": 5,
        },
    )
    assert run_prompt_resp.status_code == 200
    run_prompt_data = run_prompt_resp.json()
    assert run_prompt_data["board_name"] == "prompt_rp2040"
    assert run_prompt_data["success"] is True
    assert run_prompt_data["error_count"] == 0
    assert '(zone (net 3) (net_name "GND") (layer "B.Cu")' in run_prompt_data["kicad_pcb"]

    # Manufacturing bundle & CSV export endpoints
    bundle_resp = client.get("/api/manufacturing_bundle")
    assert bundle_resp.status_code == 200
    bundle = bundle_resp.json()
    assert bundle["board_name"] == "prompt_rp2040"
    assert "Comment,Designator,Footprint,Quantity" in bundle["bom_csv"]
    assert "Designator,Val,Package,Mid X,Mid Y,Rotation,Layer" in bundle["positions_csv"]
    assert '(zone (net 3) (net_name "GND")' in bundle["kicad_pcb"]

    bom_dl = client.get("/api/export-bom?download=1")
    assert bom_dl.status_code == 200
    assert "bom.csv" in bom_dl.headers.get("content-disposition", "")
    assert "RP2040_CORE" in bom_dl.text

    cpl_dl = client.get("/api/export-cpl?download=1")
    assert cpl_dl.status_code == 200
    assert "positions.csv" in cpl_dl.headers.get("content-disposition", "")
    assert "U1" in cpl_dl.text

    # WebSocket live prompt synthesis rollout
    with client.websocket_connect("/ws/rollout") as ws:
        ws.send_json(
            {
                "prompt": "STM32 MCU with I2C sensor, 3.3V LDO, and 1 LED",
                "candidates": 3,
                "max_iterations": 5,
                "step_delay_ms": 0,
            }
        )
        seen_events: list[str] = []
        while True:
            msg = ws.receive_json()
            seen_events.append(msg["event"])
            if msg["event"] == "completed":
                assert msg["success"] is True
                assert msg["error_count"] == 0
                assert msg["board"]["gnd_pour"] is not None
                assert '(zone (net 3) (net_name "GND")' in msg["kicad_pcb"]
                break
        assert "prompt_composed" in seen_events
        assert "placement_selected" in seen_events
        assert "net_routed" in seen_events

    # Unknown preset returns 400
    bad_resp = client.post("/api/run", json={"board": "nonexistent_board"})
    assert bad_resp.status_code == 400


def test_api_preset_manufacturing_exports_and_error_paths() -> None:
    client = TestClient(app)

    bom_resp = client.get("/api/export-bom?board=buck_converter&download=1")
    assert bom_resp.status_code == 200
    assert "TPS54302" in bom_resp.text

    cpl_resp = client.get("/api/export-cpl?board=buck_converter&download=1")
    assert cpl_resp.status_code == 200
    assert "U1" in cpl_resp.text

    # WebSocket invalid board emits structured error event instead of crashing
    with client.websocket_connect("/ws/rollout") as ws:
        ws.send_json({"board": "unknown_board_xyz"})
        err_msg = ws.receive_json()
        assert err_msg["event"] == "error"
        assert "Unknown preset" in err_msg["message"]


def test_4layer_50_component_flight_controller_end_to_end(tmp_path: Path) -> None:
    cfg = BoardConfig.from_yaml(ROOT_DIR / "configs" / "default_4layer.yaml")
    graph = build_preset_board("flight_controller_50")
    out_file = tmp_path / "flight_controller_50_clean.kicad_pcb"

    result = design_pcb_world_model(
        netlist_graph=graph,
        board_config=cfg,
        output_path=out_file,
        num_candidates=4,
        max_iterations=5,
        include_seed_candidate=True,
    )
    assert result.success is True
    assert len(result.drc_errors) == 0
    summary = result.board_state.to_summary_dict()
    assert summary["layers"] == 4
    assert summary["routed_nets"] == 43
    assert summary["total_nets"] == 43
    assert summary["signal_integrity"]["si_thermal_status"] == "PASS"

    reparsed = SExprParser.parse_file(out_file)
    assert reparsed.layers == 4
    assert len(reparsed.components) == 50


def test_rev6_api_presets_4layer_and_signal_integrity() -> None:
    client = TestClient(app)

    all_resp = client.get("/api/presets?include_complex=true")
    assert all_resp.status_code == 200
    all_presets = all_resp.json()
    assert len(all_presets) == len(PRESET_DESCRIPTIONS) + 1
    fc50 = next(p for p in all_presets if p["id"] == "flight_controller_50")
    assert fc50["layers"] == 4
    assert fc50["components"] == 50
    assert fc50["nets"] == 43

    si_resp = client.get("/api/signal-integrity?board=buck_converter")
    assert si_resp.status_code == 200
    si_body = si_resp.json()
    assert si_body["kernel"]["available"] is True
    assert "signal_integrity" in si_body

    run_4l = client.post(
        "/api/run",
        json={"board": "buck_converter", "layers": 4, "candidates": 3, "max_iterations": 5},
    )
    assert run_4l.status_code == 200
    data_4l = run_4l.json()
    assert data_4l["success"] is True
    assert data_4l["error_count"] == 0
    assert data_4l["board"]["layers"] == 4
    assert "signal_integrity" in data_4l["board"]
    assert '(1 "In1.Cu" power "In1.GND")' in data_4l["kicad_pcb"]
    assert '(2 "In2.Cu" power "In2.PWR")' in data_4l["kicad_pcb"]

    bad_layers = client.post("/api/run", json={"board": "buck_converter", "layers": 3})
    assert bad_layers.status_code == 422


def test_rev7_b6_wearable_oled_keychain_end_to_end_zero_drc(tmp_path: Path) -> None:
    cfg = BoardConfig.from_yaml(ROOT_DIR / "configs" / "default_2layer.yaml")
    graph = build_preset_board("wearable_oled_keychain_26x26")
    out_file = tmp_path / "b6_wearable_clean.kicad_pcb"

    result = design_pcb_world_model(
        netlist_graph=graph,
        board_config=cfg,
        output_path=out_file,
        num_candidates=4,
        max_iterations=4,
        include_seed_candidate=True,
    )
    assert result.success is True
    assert len(result.drc_errors) == 0
    assert result.board_state.all_nets_routed()
    summary = result.board_state.to_summary_dict()
    assert summary["width_mm"] == pytest.approx(26.0)
    assert summary["height_mm"] == pytest.approx(26.0)
    assert summary["thickness_mm"] == pytest.approx(0.8)
    assert summary["single_sided"] is True
    assert summary["signal_integrity"]["si_thermal_status"] == "PASS"

    text = out_file.read_text(encoding="utf-8")
    assert "(thickness 0.80)" in text
    assert "np_thru_hole circle" in text
    assert "(keepout" in text

    reparsed = SExprParser.parse_file(out_file)
    assert reparsed.thickness_mm == pytest.approx(0.8)
    assert reparsed.components["J3"].side == "bottom"


def test_rev7_b7_prompt_compose_and_cpl_sides() -> None:
    client = TestClient(app)

    b7_prompt = "Build me a small wireless wearable keychain with motion wake-up, battery charging, and mini display"
    resp = client.post(
        "/api/compose",
        json={
            "prompt": b7_prompt,
            "board_width_mm": 26.0,
            "board_height_mm": 26.0,
            "board_thickness_mm": 0.8,
            "layers": 2,
            "single_sided": True,
            "wireless_choice": "esp32_c3fn4_qfn",
            "mounting_holes": 2,
            "candidates": 4,
            "max_iterations": 4,
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["erc_errors"] == []
    assert body["success"] is True
    assert body["error_count"] == 0
    assert body["board"]["thickness_mm"] == pytest.approx(0.8)
    assert len(body["board"]["design_rationale"]) >= 1

    cpl_resp = client.get("/api/export-cpl")
    assert cpl_resp.status_code == 200
    assert any(line.startswith("J3,") and line.endswith(",Bottom") for line in cpl_resp.text.splitlines())


def test_rev7_api_negotiate_b8_rejection_library_and_class_c() -> None:
    client = TestClient(app)

    # Presets catalog preserves rev6 default while exposing B6 with include_rev7=true
    presets_rev7 = client.get("/api/presets?include_complex=true&include_rev7=true")
    assert presets_rev7.status_code == 200
    ids = {p["id"] for p in presets_rev7.json()}
    assert "wearable_oled_keychain_26x26" in ids

    # B7 interactive pre-flight asks clarification questions
    neg_b7 = client.post(
        "/api/negotiate",
        json={
            "prompt": "Build me a small wireless wearable keychain with motion wake-up, battery charging, and mini display",
            "interactive": True,
        },
    )
    assert neg_b7.status_code == 200
    neg_b7_data = neg_b7.json()
    assert neg_b7_data["status"] == "needs_input"
    assert len(neg_b7_data["questions"]) >= 2

    # B8 impossible 25x25 relay box: /api/negotiate reports >100% util; /api/compose rejects w/ HTTP 409
    b8_prompt = "Fit a 4-channel 10A SPDT relay controller with optocouplers and screw terminals inside a 25x25mm 2-layer enclosure"
    neg_b8 = client.post(
        "/api/negotiate",
        json={"prompt": b8_prompt, "board_width_mm": 25.0, "board_height_mm": 25.0, "layers": 2},
    )
    assert neg_b8.status_code == 200
    assert neg_b8.json()["status"] == "tradeoff_required"
    assert neg_b8.json()["area_utilization"] > 1.0

    rejected = client.post(
        "/api/compose",
        json={"prompt": b8_prompt, "board_width_mm": 25.0, "board_height_mm": 25.0, "layers": 2},
    )
    assert rejected.status_code == 409
    assert rejected.json()["detail"]["status"] == "tradeoff_required"

    # Applying quantified trade-off expand_board unlocks HTTP 200 success with 0 DRC
    recovered = client.post(
        "/api/compose",
        json={
            "prompt": b8_prompt,
            "board_width_mm": 25.0,
            "board_height_mm": 25.0,
            "layers": 2,
            "tradeoff_choice": "expand_board",
            "candidates": 3,
            "max_iterations": 3,
        },
    )
    assert recovered.status_code == 200
    assert recovered.json()["success"] is True
    assert recovered.json()["error_count"] == 0
    assert recovered.json()["board"]["width_mm"] >= 80.0

    # 12-Domain Class A/B library endpoint
    lib_resp = client.get("/api/library?query=LIS3DH")
    assert lib_resp.status_code == 200
    lib_data = lib_resp.json()
    assert lib_data["counts"]["domains"] == 12
    assert any(d["part_id"] == "LIS3DH" for d in lib_data["filtered_devices"])

    # Class C: raw .kicad_mod ingestion + datasheet pin-table compilation
    mod_resp = client.post(
        "/api/import-kicad-mod",
        json={
            "kicad_mod_text": '(footprint "Test:QFN8" (pad "1" smd rect (at -1 0) (size 0.6 0.3) (layers "F.Cu")) (pad "EP" smd rect (at 0 0) (size 1.6 1.6) (layers "F.Cu")))',
            "comp_id": "U9",
        },
    )
    assert mod_resp.status_code == 200
    assert "thermal_via_ep" in mod_resp.json()["special_layout_rules"]

    ds_resp = client.post(
        "/api/import-datasheet",
        json={
            "comp_id": "U10",
            "spec": {
                "part_number": "MY_I2C_ACCEL",
                "package": "LGA-14",
                "thermal_pad": "GND",
                "pins": {"1": "VDD", "2": "GND", "3": "SDA", "4": "SCL"},
            },
        },
    )
    assert ds_resp.status_code == 200
    assert any(s["type"] == "i2c_pullups" for s in ds_resp.json()["inferred_support"])
