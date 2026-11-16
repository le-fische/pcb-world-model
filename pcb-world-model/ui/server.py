"""FastAPI + WebSocket streaming server for the PCB Engineering World Model."""

from __future__ import annotations

import asyncio
import sys
import threading
import time
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from env.board_state import BoardState
from env.config import BoardConfig
from env.kicad_exporter import serialize_kicad_pcb
from env.sexpr_parser import PRESET_DESCRIPTIONS, build_preset_board
import importlib

_inf_mod = importlib.import_module("scripts.04_run_inference")
design_pcb_world_model = _inf_mod.design_pcb_world_model
load_models = _inf_mod.load_models
_tensor_channels_payload = _inf_mod._tensor_channels_payload

WEB_DIR = Path(__file__).resolve().parent / "web"
DEFAULT_CFG_PATH = ROOT_DIR / "configs" / "default_2layer.yaml"


class RunInferenceRequest(BaseModel):
    """Pydantic V2 request schema for synchronous inference API."""

    board: str = Field(default="mcu_sensor_node", description="Preset board name")
    candidates: int = Field(default=8, ge=1, le=16)
    max_iterations: int = Field(default=5, ge=1, le=10)


class PresetItem(BaseModel):
    """Pydantic V2 preset board metadata item."""

    id: str
    description: str
    components: int
    nets: int
    board: dict[str, Any]
    channels: dict[str, Any]


class RunInferenceResponse(BaseModel):
    """Pydantic V2 summary response from closed-loop PCB World Model inference."""

    board_name: str
    success: bool
    iterations_used: int
    elapsed_sec: float
    error_count: int
    errors: list[dict[str, Any]]
    candidates: list[dict[str, Any]]
    board: dict[str, Any]
    kicad_pcb: str


app = FastAPI(
    title="PCB Engineering World Model API",
    version="1.0.0-rev1",
    description="Closed-loop Neuro-Symbolic PCB Placement, World Model Critic, and Octilinear Router",
)

if WEB_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(WEB_DIR)), name="static")

_STATE_LOCK = threading.Lock()
_LATEST_KICAD_PCB: str = ""
_LATEST_BOARD_NAME: str = "mcu_sensor_node"
_MODELS_CACHE: tuple[Any, Any] | None = None


def _get_models() -> tuple[Any, Any]:
    global _MODELS_CACHE
    with _STATE_LOCK:
        if _MODELS_CACHE is None:
            _MODELS_CACHE = load_models(ROOT_DIR / "checkpoints")
        return _MODELS_CACHE


def _get_board_config() -> BoardConfig:
    if DEFAULT_CFG_PATH.exists():
        return BoardConfig.from_yaml(DEFAULT_CFG_PATH)
    return BoardConfig()


@app.get("/")
async def index() -> FileResponse:
    """Serve the interactive PCB Engineering World Model web console."""
    index_path = WEB_DIR / "index.html"
    if not index_path.exists():
        raise HTTPException(status_code=404, detail="Web UI index.html not found")
    return FileResponse(index_path)


@app.get("/api/presets", response_model=list[PresetItem])
async def list_presets() -> list[PresetItem]:
    """Return all available hardware board presets with initial layout and S_t channels."""
    cfg = _get_board_config()
    items: list[PresetItem] = []
    for pid, desc in PRESET_DESCRIPTIONS.items():
        graph = build_preset_board(pid)
        state = BoardState(netlist=graph, config=cfg)
        tensor_payload = _tensor_channels_payload(state, active_net_id=None)
        items.append(
            PresetItem(
                id=pid,
                description=desc,
                components=len(graph.components),
                nets=len(graph.routable_net_ids),
                board=state.to_summary_dict(),
                channels=tensor_payload["channels"],
            )
        )
    return items


@app.post("/api/run", response_model=RunInferenceResponse)
async def run_inference_endpoint(payload: RunInferenceRequest) -> RunInferenceResponse:
    """Execute the full closed-loop World Model design pipeline synchronously."""
    global _LATEST_KICAD_PCB, _LATEST_BOARD_NAME
    if payload.board not in PRESET_DESCRIPTIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown preset '{payload.board}'. Valid options: {list(PRESET_DESCRIPTIONS)}",
        )

    cfg = _get_board_config()
    graph = build_preset_board(payload.board)
    placer, world = _get_models()

    result = await asyncio.to_thread(
        design_pcb_world_model,
        graph,
        cfg,
        placer,
        world,
        ROOT_DIR / "output_clean.kicad_pcb",
        payload.candidates,
        payload.max_iterations,
        None,
        False,
    )

    with _STATE_LOCK:
        _LATEST_KICAD_PCB = result.kicad_pcb_text
        _LATEST_BOARD_NAME = payload.board

    return RunInferenceResponse(
        board_name=payload.board,
        success=result.success,
        iterations_used=result.iterations_used,
        elapsed_sec=round(result.elapsed_sec, 3),
        error_count=len(result.drc_errors),
        errors=[e.to_dict() for e in result.drc_errors],
        candidates=result.candidate_scores,
        board=result.board_state.to_summary_dict(),
        kicad_pcb=result.kicad_pcb_text,
    )


@app.get("/api/export-kicad")
async def export_kicad(board: str | None = None, download: bool = False) -> PlainTextResponse:
    """Return the latest generated `.kicad_pcb` S-expression or generate one for `board`."""
    with _STATE_LOCK:
        cached = _LATEST_KICAD_PCB
        bname = _LATEST_BOARD_NAME

    if not cached or (board and board != bname and board in PRESET_DESCRIPTIONS):
        target = board if board in PRESET_DESCRIPTIONS else "mcu_sensor_node"
        cfg = _get_board_config()
        state = BoardState(netlist=build_preset_board(target), config=cfg)
        cached = serialize_kicad_pcb(state)
        bname = target

    headers: dict[str, str] = {}
    if download:
        headers["Content-Disposition"] = f'attachment; filename="{bname}_clean.kicad_pcb"'
    return PlainTextResponse(content=cached, media_type="text/plain; charset=utf-8", headers=headers)


@app.websocket("/ws/rollout")
async def rollout_websocket(websocket: WebSocket) -> None:
    """Stream real-time S_t -> S_{t+1} neuro-symbolic transitions over WebSocket."""
    global _LATEST_KICAD_PCB, _LATEST_BOARD_NAME
    await websocket.accept()
    loop = asyncio.get_running_loop()

    try:
        while True:
            msg = await websocket.receive_json()
            board_name = str(msg.get("board", "mcu_sensor_node"))
            if board_name not in PRESET_DESCRIPTIONS:
                await websocket.send_json(
                    {"event": "error", "message": f"Unknown board preset: {board_name}"}
                )
                continue

            candidates = max(1, min(16, int(msg.get("candidates", 8))))
            max_iterations = max(1, min(10, int(msg.get("max_iterations", 5))))
            step_delay_ms = max(0, min(600, int(msg.get("step_delay_ms", 110))))

            cfg = _get_board_config()
            graph = build_preset_board(board_name)
            placer, world = _get_models()

            event_queue: asyncio.Queue[dict[str, Any] | None] = asyncio.Queue()
            cancelled = threading.Event()

            def _callback(evt: dict[str, Any]) -> None:
                if cancelled.is_set():
                    return
                loop.call_soon_threadsafe(event_queue.put_nowait, evt)
                if step_delay_ms > 0 and evt.get("event") in (
                    "candidate_evaluated",
                    "placement_selected",
                    "net_routed",
                    "ripup_triggered",
                ):
                    time.sleep(step_delay_ms / 1000.0)

            def _worker() -> None:
                try:
                    res = design_pcb_world_model(
                        netlist_graph=graph,
                        board_config=cfg,
                        placer_gnn=placer,
                        world_model=world,
                        output_path=ROOT_DIR / "output_clean.kicad_pcb",
                        num_candidates=candidates,
                        max_iterations=max_iterations,
                        step_callback=_callback,
                        stream_tensors=True,
                    )
                    with _STATE_LOCK:
                        global _LATEST_KICAD_PCB, _LATEST_BOARD_NAME
                        _LATEST_KICAD_PCB = res.kicad_pcb_text
                        _LATEST_BOARD_NAME = board_name
                except Exception as exc:  # noqa: BLE001
                    loop.call_soon_threadsafe(
                        event_queue.put_nowait,
                        {"event": "error", "message": str(exc)},
                    )
                finally:
                    loop.call_soon_threadsafe(event_queue.put_nowait, None)

            worker_task = asyncio.create_task(asyncio.to_thread(_worker))

            try:
                while True:
                    item = await event_queue.get()
                    if item is None:
                        break
                    await websocket.send_json(item)
            except WebSocketDisconnect:
                cancelled.set()
                raise
            finally:
                await worker_task

    except WebSocketDisconnect:
        return
