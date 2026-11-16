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
    assert len(presets) == 3

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
