"""FastAPI + WebSocket streaming server for the PCB Engineering World Model."""

from __future__ import annotations

import asyncio
import base64
from dataclasses import replace
import importlib
import sys
import threading
import time
from pathlib import Path
from typing import Any, Literal

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from env.board_state import BoardState
from env.component_library import list_devices, summarize_catalog
from env.config import BoardConfig
from env.importers import (
    apply_enclosure_constraints,
    build_legacy_high_via_kicad_pcb,
    build_sample_enclosure_step,
    import_pin_diagram,
    load_custom_library,
    optimize_imported_board,
    parse_dxf_enclosure,
    parse_schematic,
    parse_step_enclosure,
    solve_thermal_signal_inpainting,
)
from env.kicad_exporter import serialize_kicad_pcb
from env.kicad_mod_loader import (
    compile_datasheet_pin_table,
    infer_support_circuits_from_pins,
    parse_kicad_mod,
)
from env.manufacturing_exporter import generate_bom_csv, generate_cpl_csv
from env.metrics import (
    compute_broadside_crosstalk_risk,
    compute_ground_return_isolation,
    compute_pdn_decoupling_inductance,
    compute_rigid_flex_impedance_telemetry,
    compute_signal_integrity_telemetry,
)
from env.step_exporter import generate_step_assembly
from env.sexpr_parser import (
    ALL_PRESET_DESCRIPTIONS,
    COMPLEX_PRESET_DESCRIPTIONS,
    FULL_PRESET_CATALOG,
    PRESET_DESCRIPTIONS,
    REV7_PRESET_DESCRIPTIONS,
    REV8_PRESET_DESCRIPTIONS,
    REV11_PRESET_DESCRIPTIONS,
    REV12_PRESET_DESCRIPTIONS,
    REV13_PRESET_DESCRIPTIONS,
    REV14_PRESET_DESCRIPTIONS,
    NetlistGraph,
    build_preset_board,
)
from models.composer.constraint_negotiator import negotiate_constraints
from models.composer.design_rationale import build_structured_rationale_report
from models.composer.schematic_composer import compose_netlist_from_prompt, verify_erc
from models.silicon import (
    compile_asic_openlane,
    compose_verilog_module,
    run_fpga_toolchain,
)
from models.world.native_kernel import native_kernel_info
from verifier.dfm_auditor import audit_board_dfm, generate_dfm_report_markdown
from verifier.fast_clearance import validate_ground_topology

_inf_mod = importlib.import_module("scripts.04_run_inference")
design_pcb_world_model = _inf_mod.design_pcb_world_model
load_models = _inf_mod.load_models
run_silicon_codesign_for_graph = _inf_mod.run_silicon_codesign_for_graph
_tensor_channels_payload = _inf_mod._tensor_channels_payload

WEB_DIR = Path(__file__).resolve().parent / "web"
DEFAULT_CFG_PATH = ROOT_DIR / "configs" / "default_2layer.yaml"
DEFAULT_4LAYER_CFG_PATH = ROOT_DIR / "configs" / "default_4layer.yaml"
DEFAULT_8LAYER_CFG_PATH = ROOT_DIR / "configs" / "default_8layer.yaml"


class RunInferenceRequest(BaseModel):
    """Pydantic V2 request schema for synchronous inference API."""

    board: str = Field(default="mcu_sensor_node", description="Preset board name")
    prompt: str | None = Field(
        default=None, description="Optional natural-language hardware brief"
    )
    layers: Literal[2, 4, 6, 8] | None = Field(
        default=None, description="Copper layer stackup count (2, 4, 6, or 8)"
    )
    board_width_mm: float | None = Field(default=None, ge=10.0, le=300.0)
    board_height_mm: float | None = Field(default=None, ge=10.0, le=300.0)
    board_thickness_mm: float | None = Field(default=None, ge=0.4, le=3.2)
    single_sided: bool | None = None
    wireless_choice: str | None = None
    passive_package: str | None = None
    mounting_holes: int | None = Field(default=None, ge=0, le=8)
    tradeoff_choice: str | None = None
    user_answers: dict[str, Any] | None = None
    interactive: bool = False
    include_seed_candidate: bool = Field(
        default=True, description="Include schematic seed candidate alongside GNN samples"
    )
    candidates: int = Field(default=8, ge=1, le=16)
    max_iterations: int = Field(default=5, ge=1, le=10)


class PresetItem(BaseModel):
    """Pydantic V2 preset board metadata item."""

    id: str
    description: str
    components: int
    nets: int
    layers: int = 2
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


class ComposeRequest(BaseModel):
    """Pydantic V2 request schema for natural-language prompt-to-PCB synthesis."""

    prompt: str = Field(..., min_length=3, description="Natural-language hardware brief")
    layers: Literal[2, 4, 6, 8] | None = Field(
        default=None, description="Copper layer stackup count (2, 4, 6, or 8)"
    )
    board_width_mm: float | None = Field(default=None, ge=10.0, le=300.0)
    board_height_mm: float | None = Field(default=None, ge=10.0, le=300.0)
    board_thickness_mm: float | None = Field(default=None, ge=0.4, le=3.2)
    single_sided: bool | None = None
    wireless_choice: str | None = None
    passive_package: str | None = None
    mounting_holes: int | None = Field(default=None, ge=0, le=8)
    tradeoff_choice: str | None = None
    user_answers: dict[str, Any] | None = None
    interactive: bool = False
    include_seed_candidate: bool = Field(
        default=True, description="Include schematic seed candidate alongside GNN samples"
    )
    candidates: int = Field(default=8, ge=1, le=16)
    max_iterations: int = Field(default=5, ge=1, le=10)


class ComposeResponse(BaseModel):
    """Pydantic V2 response from hierarchical prompt synthesis and closed-loop routing."""

    board_name: str
    prompt: str
    erc_errors: list[str]
    success: bool
    iterations_used: int
    elapsed_sec: float
    error_count: int
    errors: list[dict[str, Any]]
    candidates: list[dict[str, Any]]
    board: dict[str, Any]
    kicad_pcb: str
    negotiation: dict[str, Any] = Field(default_factory=dict)
    silicon: dict[str, Any] | None = None


class SiliconSynthesizeRequest(BaseModel):
    """Pydantic V2 request schema for rev8 Verilog RTL synthesis, simulation & FPGA/ASIC flow."""

    prompt: str = Field(
        default="Need a custom 8-channel high-speed WS2812 + hardware quadrature encoder controller that talks over SPI to an ESP32-C3",
        min_length=3,
    )
    template_id: str | None = None
    target_id: Literal["ice40up5k_sg48", "ice40up5k_uwg30", "machxo2_1200", "gw1nr9", "rp2040_pio", "sky130_asic"] = "ice40up5k_sg48"
    clock_hz: int = Field(default=48_000_000, ge=1_000_000, le=200_000_000)
    include_asic: bool = False


class SiliconCodesignRequest(BaseModel):
    """Pydantic V2 request schema for rev8 joint .pcf FPGA pin-swap + PCB layout co-design."""

    board: str = Field(default="fpga_custom_protocol_bridge")
    prompt: str | None = None
    target_id: Literal["ice40up5k_sg48", "ice40up5k_uwg30", "machxo2_1200", "gw1nr9", "rp2040_pio", "sky130_asic"] = "ice40up5k_sg48"
    include_asic: bool = True
    enable_pin_swap: bool = True
    candidates: int = Field(default=4, ge=1, le=12)
    max_iterations: int = Field(default=5, ge=1, le=10)


class NegotiateRequest(BaseModel):
    """Pydantic V2 request schema for rev7 pre-flight constraint & trade-off negotiation."""

    prompt: str = Field(..., min_length=3, description="Natural-language hardware brief")
    board_width_mm: float | None = Field(default=None, ge=10.0, le=300.0)
    board_height_mm: float | None = Field(default=None, ge=10.0, le=300.0)
    board_thickness_mm: float | None = Field(default=None, ge=0.4, le=3.2)
    layers: Literal[2, 4, 6, 8] | None = None
    single_sided: bool | None = None
    mounting_holes: int | None = Field(default=None, ge=0, le=8)
    hole_diameter_mm: float | None = Field(default=None, ge=0.8, le=6.0)
    passive_package: str | None = None
    wireless_choice: str | None = None
    user_answers: dict[str, Any] | None = None
    tradeoff_choice: str | None = None
    interactive: bool = True


class KicadModImportRequest(BaseModel):
    """Pydantic V2 request schema for Class C raw `.kicad_mod` ingestion."""

    kicad_mod_text: str = Field(..., min_length=10, description="Raw KiCad .kicad_mod S-expression")
    comp_id: str = Field(default="U_CUSTOM", min_length=1, max_length=24)
    value: str | None = None
    side: Literal["top", "bottom"] = "top"


class DatasheetImportRequest(BaseModel):
    """Pydantic V2 request schema for Class C lightweight JSON datasheet pin-table compilation."""

    spec: dict[str, Any] = Field(..., description="Lightweight JSON datasheet pin-table specification")
    comp_id: str = Field(default="U_CUSTOM", min_length=1, max_length=24)


class SchematicImportRequest(BaseModel):
    """Pydantic V2 request schema for rev11 external schematic/netlist import + Auto-ERC."""

    schematic_text: str = Field(..., min_length=5)
    filename: str | None = None
    fmt: Literal["auto", "kicad_sch", "kicad_net", "spice", "atopile"] = "auto"
    board_name: str | None = None
    width_mm: float = Field(default=50.0, ge=15.0, le=300.0)
    height_mm: float = Field(default=50.0, ge=15.0, le=300.0)
    layers: Literal[2, 4, 6, 8] = 2
    auto_fix: bool = False


class PinDiagramImportRequest(BaseModel):
    """Pydantic V2 request schema for rev11 datasheet pin-diagram (.png/.pdf/.csv/text) ingestion."""

    content_text: str | None = None
    content_base64: str | None = None
    filename: str | None = None
    ocr_text: str | None = None
    comp_id: str = Field(default="U_CUSTOM", min_length=1, max_length=24)
    save_to_library: bool = True


class StepEnclosureImportRequest(BaseModel):
    """Pydantic V2 request schema for rev11 external .STEP / .DXF enclosure parsing."""

    enclosure_text: str = Field(..., min_length=10)
    filename: str | None = None
    fmt: Literal["auto", "step", "dxf"] = "auto"


class ReoptimizeBoardRequest(BaseModel):
    """Pydantic V2 request schema for rev11 lock-preserving .kicad_pcb re-optimization."""

    kicad_pcb_text: str = Field(..., min_length=20)
    lock_ids: list[str] = Field(default_factory=list)
    unlock_ids: list[str] = Field(default_factory=list)
    preserve_existing_locks: bool = True
    step_enclosure_text: str | None = None
    target_layers: Literal[2, 4, 6, 8] | None = None
    candidates: int = Field(default=4, ge=1, le=12)
    max_iterations: int = Field(default=5, ge=1, le=10)


class AdversarialInpaintRequest(BaseModel):
    """Pydantic V2 request schema for rev13 thermal-vs-signal 80%-locked adversarial inpainting."""

    kicad_pcb_text: str = Field(..., min_length=20)
    electrical_meta: dict[str, Any] = Field(default_factory=dict)
    candidates: int = Field(default=4, ge=1, le=12)
    max_iterations: int = Field(default=5, ge=1, le=10)


app = FastAPI(
    title="PCB Engineering World Model API",
    version="1.0.0-rev14",
    description="Closed-loop Neuro-Symbolic PCB Placement, Multi-Zone Rigid-Flex & 85-Ohm PCIe Gen 4 + PDN Loop Inductance (rev14), Mixed-Signal J_return & 3D EM Physics (rev13), 8-Layer (2+4+2) HDI & Staggered Microvia BGA Fanout (rev12), External Import & Optimization Engine, DFM/DFA Craftsmanship Auditor, Pro 3D PBR & Schematic Workstation, .STEP Export, Silicon-to-Board Co-Design, and C++20 2L/4L/6L/8L Octilinear Router",
)

if WEB_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(WEB_DIR)), name="static")

_STATE_LOCK = threading.Lock()
_LATEST_KICAD_PCB: str = ""
_LATEST_BOARD_NAME: str = "mcu_sensor_node"
_LATEST_BOARD_STATE: BoardState | None = None
_LATEST_SILICON_BUNDLE: dict[str, Any] | None = None
_LATEST_DIFF_REPORT: str = ""
_MODELS_CACHE: tuple[Any, Any] | None = None


def _get_models() -> tuple[Any, Any]:
    global _MODELS_CACHE
    with _STATE_LOCK:
        if _MODELS_CACHE is None:
            _MODELS_CACHE = load_models(ROOT_DIR / "checkpoints")
        return _MODELS_CACHE


class NegotiationRequiredError(Exception):
    """Raised when pre-flight physical negotiation requires user answers or trade-offs."""

    def __init__(self, negotiation: dict[str, Any]) -> None:
        self.negotiation = negotiation
        warnings = negotiation.get("warnings") or []
        super().__init__(str(warnings[0]) if warnings else "Pre-flight constraint negotiation required.")


def _get_board_config(
    layers: int | None = None,
    board: str | None = None,
    graph: NetlistGraph | None = None,
) -> BoardConfig:
    want_8l = (
        (layers is not None and layers >= 8)
        or (layers is None and (board in REV12_PRESET_DESCRIPTIONS or board in REV13_PRESET_DESCRIPTIONS))
        or (layers is None and graph is not None and graph.layers >= 8)
    )
    want_multi = (
        (layers is not None and layers >= 4)
        or (layers is None and (board in COMPLEX_PRESET_DESCRIPTIONS or board in REV8_PRESET_DESCRIPTIONS))
        or (layers is None and graph is not None and graph.layers >= 4)
    )
    if want_8l and DEFAULT_8LAYER_CFG_PATH.exists():
        cfg = BoardConfig.from_yaml(DEFAULT_8LAYER_CFG_PATH)
    elif want_multi and DEFAULT_4LAYER_CFG_PATH.exists():
        cfg = BoardConfig.from_yaml(DEFAULT_4LAYER_CFG_PATH)
    elif DEFAULT_CFG_PATH.exists():
        cfg = BoardConfig.from_yaml(DEFAULT_CFG_PATH)
    else:
        cfg = BoardConfig()
    if layers in (2, 4, 6, 8):
        cfg = replace(cfg, layers=layers)
    elif graph is not None and graph.layers in (2, 4, 6, 8):
        cfg = replace(cfg, layers=graph.layers)
    elif want_8l:
        cfg = replace(cfg, layers=8)
    elif want_multi:
        cfg = replace(cfg, layers=4)
    if graph is not None:
        cfg = replace(
            cfg,
            board_width_mm=graph.width_mm if graph.width_mm > 0 else cfg.board_width_mm,
            board_height_mm=graph.height_mm if graph.height_mm > 0 else cfg.board_height_mm,
            board_thickness_mm=getattr(graph, "thickness_mm", cfg.board_thickness_mm) or cfg.board_thickness_mm,
            single_sided=bool(getattr(graph, "single_sided", False)),
            keep_back_flat=bool(getattr(graph, "single_sided", False)),
        )
    return cfg


def _serialize_silicon_bundle(bundle: dict[str, Any] | None) -> dict[str, Any] | None:
    if bundle is None:
        return None
    out: dict[str, Any] = {}
    for k, v in bundle.items():
        if k == "bitstream_bytes":
            raw = bytes(v or b"")
            out["bitstream_base64"] = base64.b64encode(raw).decode("ascii")
            out["bitstream_size_bytes"] = len(raw)
        elif k == "gds_bytes":
            raw = bytes(v or b"")
            out["gds_base64"] = base64.b64encode(raw).decode("ascii")
            out["gds_size_bytes"] = len(raw)
        else:
            out[k] = v
    return out


def _cache_latest_result(
    board_name: str,
    board_state: BoardState,
    kicad_pcb: str,
    silicon_bundle: dict[str, Any] | None = None,
) -> None:
    global _LATEST_KICAD_PCB, _LATEST_BOARD_NAME, _LATEST_BOARD_STATE, _LATEST_SILICON_BUNDLE
    with _STATE_LOCK:
        _LATEST_KICAD_PCB = kicad_pcb
        _LATEST_BOARD_NAME = board_name
        _LATEST_BOARD_STATE = board_state
        if silicon_bundle is not None:
            _LATEST_SILICON_BUNDLE = _serialize_silicon_bundle(silicon_bundle)


def _get_or_init_board_state(
    board: str | None = None,
    layers: int | None = None,
) -> tuple[BoardState, str]:
    with _STATE_LOCK:
        cached_state = _LATEST_BOARD_STATE
        cached_pcb = _LATEST_KICAD_PCB
        cached_name = _LATEST_BOARD_NAME

    if (
        cached_state is not None
        and (not board or board == cached_name)
        and (layers is None or cached_state.num_layers == layers)
    ):
        pcb_text = cached_pcb or serialize_kicad_pcb(cached_state)
        return cached_state, pcb_text

    target = board if board in FULL_PRESET_CATALOG else "mcu_sensor_node"
    graph = build_preset_board(target)
    cfg = _get_board_config(layers=layers, board=target, graph=graph)
    state = BoardState(netlist=graph, config=cfg)
    return state, serialize_kicad_pcb(state)


def _compose_or_load_graph(
    board: str,
    prompt: str | None,
    cfg: BoardConfig,
    *,
    width_mm: float | None = None,
    height_mm: float | None = None,
    thickness_mm: float | None = None,
    layers: int | None = None,
    single_sided: bool | None = None,
    mounting_holes: int | None = None,
    passive_package: str | None = None,
    wireless_choice: str | None = None,
    user_answers: dict[str, Any] | None = None,
    tradeoff_choice: str | None = None,
    interactive: bool = False,
) -> tuple[NetlistGraph, dict[str, Any], list[str]]:
    clean_prompt = (prompt or "").strip()
    if clean_prompt:
        neg = negotiate_constraints(
            prompt=clean_prompt,
            width_mm=width_mm,
            height_mm=height_mm,
            thickness_mm=thickness_mm,
            layers=layers,
            single_sided=single_sided,
            mounting_holes=mounting_holes,
            passive_package=passive_package,
            wireless_choice=wireless_choice,
            user_answers=user_answers,
            tradeoff_choice=tradeoff_choice,
            interactive=interactive,
        )
        neg_dict = neg.to_dict()
        if neg.status == "tradeoff_required" or (interactive and neg.status == "needs_input"):
            raise NegotiationRequiredError(neg_dict)

        specs = neg.locked_specs
        graph = compose_netlist_from_prompt(
            clean_prompt,
            board_width_mm=float(specs.get("board_width_mm") or cfg.board_width_mm),
            board_height_mm=float(specs.get("board_height_mm") or cfg.board_height_mm),
            thickness_mm=specs.get("board_thickness_mm"),
            layers=int(specs.get("layers") or layers or cfg.layers),
            single_sided=specs.get("single_sided"),
            wireless_choice=specs.get("wireless_choice"),
            passive_package=specs.get("passive_package"),
            mounting_holes=specs.get("mounting_holes"),
        )
        erc_errors = verify_erc(graph)
        if erc_errors:
            raise ValueError("; ".join(erc_errors))
        return graph, neg_dict, erc_errors

    if board not in FULL_PRESET_CATALOG:
        raise KeyError(
            f"Unknown preset '{board}'. Valid options: {list(FULL_PRESET_CATALOG)}"
        )
    return build_preset_board(board), {}, []


@app.get("/")
async def index() -> FileResponse:
    """Serve the interactive PCB Engineering World Model web console."""
    index_path = WEB_DIR / "index.html"
    if not index_path.exists():
        raise HTTPException(status_code=404, detail="Web UI index.html not found")
    return FileResponse(index_path)


@app.get("/api/presets", response_model=list[PresetItem])
async def list_presets(
    include_complex: bool = False,
    include_rev7: bool = False,
    include_rev8: bool = False,
    include_rev11: bool = False,
    include_rev12: bool = False,
    include_rev13: bool = False,
    include_rev14: bool = False,
) -> list[PresetItem]:
    """Return available hardware board presets with initial layout and S_t channels."""
    catalog: dict[str, str] = dict(PRESET_DESCRIPTIONS)
    if include_complex:
        catalog.update(COMPLEX_PRESET_DESCRIPTIONS)
    if include_rev7:
        catalog.update(REV7_PRESET_DESCRIPTIONS)
    if include_rev8:
        catalog.update(REV8_PRESET_DESCRIPTIONS)
    if include_rev11:
        catalog.update(REV11_PRESET_DESCRIPTIONS)
    if include_rev12:
        catalog.update(REV12_PRESET_DESCRIPTIONS)
    if include_rev13:
        catalog.update(REV13_PRESET_DESCRIPTIONS)
    if include_rev14:
        catalog.update(REV14_PRESET_DESCRIPTIONS)
    items: list[PresetItem] = []
    for pid, desc in catalog.items():
        graph = build_preset_board(pid)
        cfg = _get_board_config(board=pid, graph=graph)
        state = BoardState(netlist=graph, config=cfg)
        tensor_payload = _tensor_channels_payload(state, active_net_id=None)
        items.append(
            PresetItem(
                id=pid,
                description=desc,
                components=len(graph.components),
                nets=len(graph.routable_net_ids),
                layers=state.num_layers,
                board=state.to_summary_dict(),
                channels=tensor_payload["channels"],
            )
        )
    return items


@app.get("/api/signal-integrity")
async def get_signal_integrity(
    board: str | None = None,
    layers: int | None = None,
) -> dict[str, Any]:
    """Return electrical signal-integrity, mixed-signal J_return/broadside EM telemetry, and C++20 native kernel status."""
    state, _ = _get_or_init_board_state(board, layers=layers)
    return {
        "board_name": state.netlist.board_name,
        "layers": state.num_layers,
        "kernel": native_kernel_info(),
        "signal_integrity": compute_signal_integrity_telemetry(state),
        "ground_return_isolation": compute_ground_return_isolation(state),
        "broadside_crosstalk": compute_broadside_crosstalk_risk(state),
        "ground_topology": validate_ground_topology(state),
    }


@app.get("/api/mixed-signal-audit")
async def get_mixed_signal_audit(
    board: str | None = None,
    layers: int | None = None,
) -> dict[str, Any]:
    """Return rev13 mixed-signal return physics, 3D Z-axis broadside coupling, and structured rationale."""
    state, _ = _get_or_init_board_state(board, layers=layers)
    iso = compute_ground_return_isolation(state)
    broad = compute_broadside_crosstalk_risk(state)
    topo = validate_ground_topology(state)
    report = build_structured_rationale_report(
        state, ground_isolation=iso, broadside_report=broad, ground_topology=topo
    )
    return {
        "board_name": state.netlist.board_name,
        "layers": state.num_layers,
        "ground_return_isolation": iso,
        "broadside_crosstalk": broad,
        "ground_topology": topo,
        "rationale_report": report,
    }


@app.get("/api/rigid-flex-audit")
async def get_rigid_flex_audit(
    board: str | None = None,
    layers: int | None = None,
) -> dict[str, Any]:
    """Return rev14 multi-zone rigid-flex 85-Ohm PCIe impedance, skew, and PDN loop inductance audit."""
    state, _ = _get_or_init_board_state(board, layers=layers)
    rf_imp = compute_rigid_flex_impedance_telemetry(state)
    pdn = compute_pdn_decoupling_inductance(state)
    iso = compute_ground_return_isolation(state)
    broad = compute_broadside_crosstalk_risk(state)
    topo = validate_ground_topology(state)
    report = build_structured_rationale_report(
        state, ground_isolation=iso, broadside_report=broad, ground_topology=topo
    )
    return {
        "board_name": state.netlist.board_name,
        "layers": state.num_layers,
        "has_rigid_flex": bool(getattr(state, "has_rigid_flex", False)),
        "rigid_flex_impedance": rf_imp,
        "pdn_decoupling": pdn,
        "ground_return_isolation": iso,
        "broadside_crosstalk": broad,
        "ground_topology": topo,
        "rationale_report": report,
    }


@app.post("/api/adversarial-inpaint")
async def adversarial_inpaint_endpoint(payload: AdversarialInpaintRequest) -> dict[str, Any]:
    """Run rev13 80%-locked thermal-vs-signal adversarial inpainting with thermal slot carving."""
    placer, world = _get_models()
    out = await asyncio.to_thread(
        solve_thermal_signal_inpainting,
        source=payload.kicad_pcb_text,
        electrical_meta=payload.electrical_meta,
        num_candidates=payload.candidates,
        max_iterations=payload.max_iterations,
        placer_gnn=placer,
        world_model=world,
    )
    res = out["inference_result"]
    _cache_latest_result(res.board_state.netlist.board_name, res.board_state, res.kicad_pcb_text)
    return {
        "board_name": res.board_state.netlist.board_name,
        "success": res.success,
        "error_count": len(res.drc_errors),
        "errors": [e.to_dict() for e in res.drc_errors],
        "lock_preservation_verified": out["lock_preservation_verified"],
        "locked_components": out["locked_components"],
        "thermal_isolation_slots": out["thermal_isolation_slots"],
        "ground_return_isolation": out["ground_return_isolation"],
        "broadside_crosstalk": out["broadside_crosstalk"],
        "design_rationale": out["design_rationale"],
        "board": res.board_state.to_summary_dict(),
        "kicad_pcb": res.kicad_pcb_text,
    }


@app.post("/api/negotiate")
async def negotiate_constraints_endpoint(payload: NegotiateRequest) -> dict[str, Any]:
    """Run the rev7 interactive Pre-Flight Constraint & Trade-Off Gate before placement."""
    try:
        res = negotiate_constraints(
            prompt=payload.prompt,
            width_mm=payload.board_width_mm,
            height_mm=payload.board_height_mm,
            thickness_mm=payload.board_thickness_mm,
            layers=payload.layers,
            single_sided=payload.single_sided,
            mounting_holes=payload.mounting_holes,
            hole_diameter_mm=payload.hole_diameter_mm,
            passive_package=payload.passive_package,
            wireless_choice=payload.wireless_choice,
            user_answers=payload.user_answers,
            tradeoff_choice=payload.tradeoff_choice,
            interactive=payload.interactive,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return res.to_dict()


@app.get("/api/library")
async def library_catalog_endpoint(
    domain: str | None = None,
    query: str | None = None,
) -> dict[str, Any]:
    """Return the Two-Tier (Class A + 12-Domain Class B) component catalog and special layout rules."""
    summary = summarize_catalog()
    filtered_devices = list_devices(domain=domain, query=query) if (domain or query) else summary["class_b_specialized"]
    return {
        **summary,
        "filtered_devices": filtered_devices,
        "filtered_count": len(filtered_devices),
    }


def _serialize_component_node(comp: Any) -> dict[str, Any]:
    return {
        "comp_id": comp.comp_id,
        "value": comp.value,
        "footprint": comp.footprint,
        "comp_type": comp.comp_type,
        "width": comp.width,
        "height": comp.height,
        "x_mm": comp.x,
        "y_mm": comp.y,
        "theta": comp.theta,
        "side": getattr(comp, "side", "top"),
        "is_locked": comp.is_locked,
        "special_layout_rules": list(getattr(comp, "special_layout_rules", [])),
        "electrical_meta": dict(getattr(comp, "electrical_meta", {})),
        "keepout_zones": list(getattr(comp, "keepout_zones", [])),
        "pads": [
            {
                "pad_id": p.pad_id,
                "rel_x": p.rel_x,
                "rel_y": p.rel_y,
                "width": p.width,
                "height": p.height,
                "net_id": p.net_id,
                "layer_mask": p.layer_mask,
                "shape": p.shape,
            }
            for p in comp.pads
        ],
    }


@app.post("/api/import-kicad-mod")
async def import_kicad_mod_endpoint(payload: KicadModImportRequest) -> dict[str, Any]:
    """Parse a raw KiCad `.kicad_mod` footprint S-expression into a ComponentNode + layout rules."""
    try:
        comp = parse_kicad_mod(
            payload.kicad_mod_text,
            comp_id=payload.comp_id,
            value=payload.value,
            side=payload.side,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    ep_hint = {"EP": "EP"} if "thermal_via_ep" in comp.special_layout_rules else {}
    return {
        "component": _serialize_component_node(comp),
        "inferred_support": infer_support_circuits_from_pins(
            ep_hint,
            thermal_pad="GND" if ep_hint else None,
        ),
        "special_layout_rules": list(comp.special_layout_rules),
    }


@app.post("/api/import-datasheet")
async def import_datasheet_endpoint(payload: DatasheetImportRequest) -> dict[str, Any]:
    """Compile a lightweight JSON datasheet pin table into a pin-exact ComponentNode + support circuits."""
    try:
        comp, support = compile_datasheet_pin_table(payload.spec, comp_id=payload.comp_id)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return {
        "component": _serialize_component_node(comp),
        "inferred_support": support,
        "special_layout_rules": list(comp.special_layout_rules),
    }


@app.post("/api/import-schematic")
async def import_schematic_endpoint(payload: SchematicImportRequest) -> dict[str, Any]:
    """rev11: Import `.kicad_sch`, `.net`, SPICE, or `atopile` circuit files + Auto-ERC Craftsmanship Audit."""
    try:
        graph, audit, fixes = parse_schematic(
            payload.schematic_text,
            fmt=payload.fmt,
            board_name=payload.board_name or (Path(payload.filename).stem if payload.filename else "imported_schematic"),
            width_mm=payload.width_mm,
            height_mm=payload.height_mm,
            layers=payload.layers,
            auto_fix=payload.auto_fix,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    cfg = _get_board_config(layers=payload.layers, graph=graph)
    preview_state = BoardState(netlist=graph, config=cfg)
    _cache_latest_result(graph.board_name, preview_state, serialize_kicad_pcb(preview_state))
    return {
        "board_name": graph.board_name,
        "format": payload.fmt,
        "erc_audit": audit.to_dict(),
        "fixes_applied": fixes,
        "board": preview_state.to_summary_dict(),
        "kicad_pcb": serialize_kicad_pcb(preview_state),
    }


@app.post("/api/import-pin-diagram")
async def import_pin_diagram_endpoint(payload: PinDiagramImportRequest) -> dict[str, Any]:
    """rev11: Ingest datasheet pin-diagram (.png/.pdf/.csv/text) -> ComponentNode + roles + local library."""
    src: Any = payload.content_base64 if payload.content_base64 else payload.content_text
    if not src:
        raise HTTPException(status_code=422, detail="Provide either content_text or content_base64.")
    try:
        comp, ext = import_pin_diagram(
            src,
            filename=payload.filename,
            ocr_text=payload.ocr_text,
            comp_id=payload.comp_id,
            save_to_library=payload.save_to_library,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return {
        "component": _serialize_component_node(comp),
        "extraction": ext.to_dict(),
        "inferred_support": ext.inferred_support,
        "special_layout_rules": ext.special_layout_rules,
        "custom_library_count": len(load_custom_library()),
    }


@app.post("/api/import-step-enclosure")
async def import_step_enclosure_endpoint(payload: StepEnclosureImportRequest) -> dict[str, Any]:
    """rev11: Parse external `.STEP` / `.DXF` mechanical enclosure into polygon, bosses, ports & Z_max map."""
    try:
        is_dxf = payload.fmt == "dxf" or (payload.filename and payload.filename.lower().endswith(".dxf"))
        enc = parse_dxf_enclosure(payload.enclosure_text) if is_dxf else parse_step_enclosure(payload.enclosure_text)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return {
        "enclosure": enc.to_dict(),
    }


@app.get("/api/rev11-samples")
async def rev11_samples_endpoint() -> dict[str, Any]:
    """Return built-in rev11 demo artifacts (sample .STEP enclosure & legacy high-via .kicad_pcb)."""
    return {
        "sample_enclosure_step": build_sample_enclosure_step(),
        "sample_legacy_kicad_pcb": build_legacy_high_via_kicad_pcb(),
        "custom_library": load_custom_library(),
    }


@app.post("/api/reoptimize-board")
async def reoptimize_board_endpoint(payload: ReoptimizeBoardRequest) -> dict[str, Any]:
    """rev11: Lock-preserving partial optimization on imported `.kicad_pcb` + Before-vs-After Diff Report."""
    global _LATEST_DIFF_REPORT
    enclosure = None
    if payload.step_enclosure_text:
        try:
            enclosure = (
                parse_dxf_enclosure(payload.step_enclosure_text)
                if "SECTION" in payload.step_enclosure_text and "ENTITIES" in payload.step_enclosure_text
                else parse_step_enclosure(payload.step_enclosure_text)
            )
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=f"Enclosure parse failed: {exc}") from exc

    placer, world = _get_models()
    try:
        opt_out = await asyncio.to_thread(
            optimize_imported_board,
            source=payload.kicad_pcb_text,
            lock_ids=payload.lock_ids,
            unlock_ids=payload.unlock_ids,
            preserve_existing_locks=payload.preserve_existing_locks,
            enclosure=enclosure,
            target_layers=payload.target_layers,
            num_candidates=payload.candidates,
            max_iterations=payload.max_iterations,
            placer_gnn=placer,
            world_model=world,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    res = opt_out["inference_result"]
    diff = opt_out["diff"]
    _cache_latest_result(res.board_state.netlist.board_name, res.board_state, res.kicad_pcb_text)
    with _STATE_LOCK:
        _LATEST_DIFF_REPORT = diff.markdown_report

    return {
        "board_name": res.board_state.netlist.board_name,
        "success": res.success,
        "elapsed_sec": round(res.elapsed_sec, 3),
        "error_count": len(res.drc_errors),
        "errors": [e.to_dict() for e in res.drc_errors],
        "board": res.board_state.to_summary_dict(),
        "kicad_pcb": res.kicad_pcb_text,
        "optimization_diff": diff.to_dict(),
    }


@app.get("/api/export-diff-report")
async def export_diff_report(download: bool = False) -> PlainTextResponse:
    """Return the latest rev11 Before-vs-After Optimization Diff Report markdown."""
    with _STATE_LOCK:
        md = _LATEST_DIFF_REPORT or "# No rev11 Optimization Diff Report Generated Yet\nRun `/api/reoptimize-board` or B12 to generate a Before-vs-After report.\n"
    headers: dict[str, str] = {}
    if download:
        headers["Content-Disposition"] = 'attachment; filename="optimization_diff_report.md"'
    return PlainTextResponse(content=md, media_type="text/markdown; charset=utf-8", headers=headers)


@app.post("/api/run", response_model=RunInferenceResponse)
async def run_inference_endpoint(payload: RunInferenceRequest) -> RunInferenceResponse:
    """Execute the full closed-loop World Model design pipeline synchronously."""
    cfg_init = _get_board_config(layers=payload.layers, board=payload.board)
    try:
        graph, _neg, _erc = _compose_or_load_graph(
            payload.board,
            payload.prompt,
            cfg_init,
            width_mm=payload.board_width_mm,
            height_mm=payload.board_height_mm,
            thickness_mm=payload.board_thickness_mm,
            layers=payload.layers,
            single_sided=payload.single_sided,
            mounting_holes=payload.mounting_holes,
            passive_package=payload.passive_package,
            wireless_choice=payload.wireless_choice,
            user_answers=payload.user_answers,
            tradeoff_choice=payload.tradeoff_choice,
            interactive=payload.interactive,
        )
    except NegotiationRequiredError as exc:
        raise HTTPException(status_code=409, detail=exc.negotiation) from exc
    except KeyError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    cfg = _get_board_config(layers=payload.layers, board=payload.board, graph=graph)
    silicon_raw = await asyncio.to_thread(
        run_silicon_codesign_for_graph,
        graph=graph,
        prompt=payload.prompt,
    )
    placer, world = _get_models()

    result = await asyncio.to_thread(
        design_pcb_world_model,
        netlist_graph=graph,
        board_config=cfg,
        placer_gnn=placer,
        world_model=world,
        output_path=ROOT_DIR / "output_clean.kicad_pcb",
        num_candidates=payload.candidates,
        max_iterations=payload.max_iterations,
        step_callback=None,
        stream_tensors=False,
        include_seed_candidate=payload.include_seed_candidate,
    )

    _cache_latest_result(graph.board_name, result.board_state, result.kicad_pcb_text, silicon_raw)

    return RunInferenceResponse(
        board_name=graph.board_name,
        success=result.success,
        iterations_used=result.iterations_used,
        elapsed_sec=round(result.elapsed_sec, 3),
        error_count=len(result.drc_errors),
        errors=[e.to_dict() for e in result.drc_errors],
        candidates=result.candidate_scores,
        board=result.board_state.to_summary_dict(),
        kicad_pcb=result.kicad_pcb_text,
    )


@app.post("/api/compose", response_model=ComposeResponse)
async def compose_from_prompt_endpoint(payload: ComposeRequest) -> ComposeResponse:
    """Synthesize a netlist from a natural-language hardware brief and run closed-loop routing."""
    cfg_init = _get_board_config(layers=payload.layers)
    try:
        graph, neg_dict, erc_errors = _compose_or_load_graph(
            board="mcu_sensor_node",
            prompt=payload.prompt,
            cfg=cfg_init,
            width_mm=payload.board_width_mm,
            height_mm=payload.board_height_mm,
            thickness_mm=payload.board_thickness_mm,
            layers=payload.layers,
            single_sided=payload.single_sided,
            mounting_holes=payload.mounting_holes,
            passive_package=payload.passive_package,
            wireless_choice=payload.wireless_choice,
            user_answers=payload.user_answers,
            tradeoff_choice=payload.tradeoff_choice,
            interactive=payload.interactive,
        )
    except NegotiationRequiredError as exc:
        raise HTTPException(status_code=409, detail=exc.negotiation) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    cfg = _get_board_config(layers=payload.layers, graph=graph)
    silicon_raw = await asyncio.to_thread(
        run_silicon_codesign_for_graph,
        graph=graph,
        prompt=payload.prompt,
    )
    placer, world = _get_models()
    result = await asyncio.to_thread(
        design_pcb_world_model,
        netlist_graph=graph,
        board_config=cfg,
        placer_gnn=placer,
        world_model=world,
        output_path=ROOT_DIR / "output_clean.kicad_pcb",
        num_candidates=payload.candidates,
        max_iterations=payload.max_iterations,
        step_callback=None,
        stream_tensors=False,
        include_seed_candidate=payload.include_seed_candidate,
    )

    _cache_latest_result(graph.board_name, result.board_state, result.kicad_pcb_text, silicon_raw)

    return ComposeResponse(
        board_name=graph.board_name,
        prompt=payload.prompt,
        erc_errors=erc_errors,
        success=result.success,
        iterations_used=result.iterations_used,
        elapsed_sec=round(result.elapsed_sec, 3),
        error_count=len(result.drc_errors),
        errors=[e.to_dict() for e in result.drc_errors],
        candidates=result.candidate_scores,
        board=result.board_state.to_summary_dict(),
        kicad_pcb=result.kicad_pcb_text,
        negotiation=neg_dict,
        silicon=_serialize_silicon_bundle(silicon_raw),
    )


@app.post("/api/silicon/synthesize")
async def silicon_synthesize_endpoint(payload: SiliconSynthesizeRequest) -> dict[str, Any]:
    """Synthesize Verilog-2001 RTL, run iverilog/Verilator simulation, FPGA PNR, and optional SKY130 GDSII."""
    bundle = await asyncio.to_thread(
        compose_verilog_module,
        prompt=payload.prompt,
        template_id=payload.template_id,
        clock_hz=payload.clock_hz,
        run_simulation=True,
    )
    out = bundle.to_dict()
    if not bundle.physics_assessment.synthesizable:
        out["synthesizable"] = False
        return out

    out["synthesizable"] = True
    clock_mhz = round(payload.clock_hz / 1_000_000.0, 2)
    fpga_tc = await asyncio.to_thread(
        run_fpga_toolchain,
        module_name=bundle.module_name,
        rtl_code=bundle.rtl_code,
        pcf_text="",
        target_id="ice40up5k_sg48" if payload.target_id == "sky130_asic" else payload.target_id,
        target_clock_mhz=clock_mhz,
    )
    out["target_id"] = payload.target_id
    out["toolchain_mode"] = fpga_tc["toolchain_mode"]
    out["timing_passed"] = fpga_tc["timing_passed"]
    out["achieved_fmax_mhz"] = fpga_tc["achieved_fmax_mhz"]
    out["timing_slack_ns"] = fpga_tc["timing_slack_ns"]
    out["lut4_used"] = fpga_tc["lut4_used"]
    out["dff_used"] = fpga_tc["dff_used"]
    out["bitstream_asc"] = fpga_tc["bitstream_asc"]
    out["bitstream_base64"] = base64.b64encode(fpga_tc["bitstream_bytes"]).decode("ascii")
    out["bitstream_size_bytes"] = len(fpga_tc["bitstream_bytes"])

    if payload.include_asic or payload.target_id == "sky130_asic":
        asic_res = await asyncio.to_thread(compile_asic_openlane, bundle)
        out["asic"] = asic_res.to_dict(include_binary=False)
        out["gds_base64"] = base64.b64encode(asic_res.gds_bytes).decode("ascii")
        out["gds_size_bytes"] = len(asic_res.gds_bytes)

    with _STATE_LOCK:
        global _LATEST_SILICON_BUNDLE
        _LATEST_SILICON_BUNDLE = out
    return out


@app.post("/api/silicon/codesign")
async def silicon_codesign_endpoint(payload: SiliconCodesignRequest) -> dict[str, Any]:
    """Run full rev8 joint .pcf FPGA pin-swap optimization + closed-loop 4-layer PCB routing."""
    cfg_init = _get_board_config(board=payload.board)
    try:
        graph, _neg, erc_errors = _compose_or_load_graph(
            board=payload.board,
            prompt=payload.prompt,
            cfg=cfg_init,
        )
    except (KeyError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    silicon_raw = await asyncio.to_thread(
        run_silicon_codesign_for_graph,
        graph=graph,
        prompt=payload.prompt,
        target_id=payload.target_id,
        include_asic=payload.include_asic,
        enable_pin_swap=payload.enable_pin_swap,
    )
    post_erc = verify_erc(graph)
    cfg = _get_board_config(board=payload.board, graph=graph)
    placer, world = _get_models()

    result = await asyncio.to_thread(
        design_pcb_world_model,
        netlist_graph=graph,
        board_config=cfg,
        placer_gnn=placer,
        world_model=world,
        output_path=ROOT_DIR / "output_clean.kicad_pcb",
        num_candidates=payload.candidates,
        max_iterations=payload.max_iterations,
        step_callback=None,
        stream_tensors=False,
        include_seed_candidate=True,
    )

    _cache_latest_result(graph.board_name, result.board_state, result.kicad_pcb_text, silicon_raw)
    return {
        "board_name": graph.board_name,
        "erc_errors": post_erc or erc_errors,
        "success": result.success,
        "error_count": len(result.drc_errors),
        "errors": [e.to_dict() for e in result.drc_errors],
        "elapsed_sec": round(result.elapsed_sec, 3),
        "board": result.board_state.to_summary_dict(),
        "kicad_pcb": result.kicad_pcb_text,
        "silicon": _serialize_silicon_bundle(silicon_raw),
    }


@app.get("/api/export-silicon")
async def export_silicon_endpoint() -> dict[str, Any]:
    """Return the latest rev8 silicon bundle (Verilog RTL, testbench, .pcf, bitstream, GDSII)."""
    global _LATEST_SILICON_BUNDLE
    with _STATE_LOCK:
        cached = _LATEST_SILICON_BUNDLE
    if cached is not None:
        return cached
    graph = build_preset_board("fpga_custom_protocol_bridge")
    silicon_raw = await asyncio.to_thread(run_silicon_codesign_for_graph, graph=graph, include_asic=True)
    serialized = _serialize_silicon_bundle(silicon_raw) or {}
    with _STATE_LOCK:
        _LATEST_SILICON_BUNDLE = serialized
    return serialized


@app.get("/api/export-kicad")
async def export_kicad(board: str | None = None, download: bool = False) -> PlainTextResponse:
    """Return the latest generated `.kicad_pcb` S-expression or generate one for `board`."""
    state, cached = _get_or_init_board_state(board)
    bname = state.netlist.board_name

    headers: dict[str, str] = {}
    if download:
        headers["Content-Disposition"] = f'attachment; filename="{bname}_clean.kicad_pcb"'
    return PlainTextResponse(content=cached, media_type="text/plain; charset=utf-8", headers=headers)


@app.get("/api/export-bom")
async def export_bom(board: str | None = None, download: bool = False) -> PlainTextResponse:
    """Return a JLCPCB/PCBWay-compatible `bom.csv` for the latest or selected board."""
    state, _ = _get_or_init_board_state(board)
    bom_csv = generate_bom_csv(state)
    headers: dict[str, str] = {}
    if download:
        headers["Content-Disposition"] = 'attachment; filename="bom.csv"'
    return PlainTextResponse(content=bom_csv, media_type="text/csv; charset=utf-8", headers=headers)


@app.get("/api/export-cpl")
async def export_cpl(board: str | None = None, download: bool = False) -> PlainTextResponse:
    """Return a JLCPCB/PCBWay-compatible Pick-and-Place `positions.csv` for the latest or selected board."""
    state, _ = _get_or_init_board_state(board)
    cpl_csv = generate_cpl_csv(state)
    headers: dict[str, str] = {}
    if download:
        headers["Content-Disposition"] = 'attachment; filename="positions.csv"'
    return PlainTextResponse(content=cpl_csv, media_type="text/csv; charset=utf-8", headers=headers)


@app.get("/api/export-step")
async def export_step(
    board: str | None = None,
    mask_theme: str = "green",
    download: bool = False,
) -> PlainTextResponse:
    """Return an ISO-10303-21 AP214 `.step` 3D mechanical CAD assembly for enclosure co-design."""
    state, _ = _get_or_init_board_state(board)
    bname = state.netlist.board_name.replace("/", "_").replace(" ", "_")
    step_text = generate_step_assembly(state, mask_theme=mask_theme)
    headers: dict[str, str] = {}
    if download:
        headers["Content-Disposition"] = f'attachment; filename="{bname}_assembly.step"'
    return PlainTextResponse(content=step_text, media_type="application/step; charset=utf-8", headers=headers)


@app.get("/api/dfm-audit")
async def get_dfm_audit(
    board: str | None = None,
    layers: int | None = None,
) -> dict[str, Any]:
    """Return the rev10 DFM/DFA Craftsmanship Audit and transparent Compromise Report."""
    state, kicad_pcb = _get_or_init_board_state(board, layers=layers)
    audit = getattr(state, "dfm_audit", None) or audit_board_dfm(state, kicad_pcb_text=kicad_pcb)
    return {
        "board_name": state.netlist.board_name,
        "layers": state.num_layers,
        "audit": audit,
        "dfm_report_md": generate_dfm_report_markdown(state, audit),
    }


@app.get("/api/export-dfm-report")
async def export_dfm_report(
    board: str | None = None,
    download: bool = False,
) -> PlainTextResponse:
    """Return the rev10 DFM/DFA Compromise Report (`dfm_report.md`)."""
    state, kicad_pcb = _get_or_init_board_state(board)
    bname = state.netlist.board_name.replace("/", "_").replace(" ", "_")
    audit = getattr(state, "dfm_audit", None) or audit_board_dfm(state, kicad_pcb_text=kicad_pcb)
    md_text = generate_dfm_report_markdown(state, audit)
    headers: dict[str, str] = {}
    if download:
        headers["Content-Disposition"] = f'attachment; filename="{bname}_dfm_report.md"'
    return PlainTextResponse(content=md_text, media_type="text/markdown; charset=utf-8", headers=headers)


@app.get("/api/manufacturing_bundle")
async def manufacturing_bundle(board: str | None = None) -> dict[str, Any]:
    """Return the complete turnkey PCBA + 3D CAD + DFM bundle (`.kicad_pcb`, `.step`, `bom.csv`, `positions.csv`, `dfm_report.md`)."""
    state, kicad_pcb = _get_or_init_board_state(board)
    audit = getattr(state, "dfm_audit", None) or audit_board_dfm(state, kicad_pcb_text=kicad_pcb)
    return {
        "board_name": state.netlist.board_name,
        "kicad_pcb": kicad_pcb,
        "step_assembly": generate_step_assembly(state),
        "bom_csv": generate_bom_csv(state),
        "positions_csv": generate_cpl_csv(state),
        "dfm_audit": audit,
        "dfm_report_md": generate_dfm_report_markdown(state, audit),
        "component_count": len(state.netlist.components),
        "net_count": len(state.netlist.routable_net_ids),
    }


@app.websocket("/ws/rollout")
async def rollout_websocket(websocket: WebSocket) -> None:
    """Stream real-time S_t -> S_{t+1} neuro-symbolic transitions over WebSocket."""
    await websocket.accept()
    loop = asyncio.get_running_loop()

    try:
        while True:
            msg = await websocket.receive_json()
            board_name = str(msg.get("board", "mcu_sensor_node"))
            prompt = str(msg.get("prompt") or "").strip() or None
            raw_layers = msg.get("layers")
            requested_layers: int | None = None
            if raw_layers is not None:
                try:
                    requested_layers = int(raw_layers)
                except (TypeError, ValueError):
                    await websocket.send_json({"event": "error", "message": f"Invalid layer count '{raw_layers}'"})
                    continue
                if requested_layers not in (2, 4, 6, 8):
                    await websocket.send_json(
                        {"event": "error", "message": f"Unsupported layer count {requested_layers}; expected 2, 4, 6, or 8"}
                    )
                    continue

            cfg_init = _get_board_config(layers=requested_layers, board=board_name)

            def _opt_float(val: Any) -> float | None:
                return float(val) if val not in (None, "") else None

            def _opt_int(val: Any) -> int | None:
                return int(val) if val not in (None, "") else None

            try:
                graph, neg_dict, _erc = _compose_or_load_graph(
                    board_name,
                    prompt,
                    cfg_init,
                    width_mm=_opt_float(msg.get("board_width_mm")),
                    height_mm=_opt_float(msg.get("board_height_mm")),
                    thickness_mm=_opt_float(msg.get("board_thickness_mm")),
                    layers=requested_layers,
                    single_sided=msg.get("single_sided"),
                    mounting_holes=_opt_int(msg.get("mounting_holes")),
                    passive_package=msg.get("passive_package"),
                    wireless_choice=msg.get("wireless_choice"),
                    user_answers=msg.get("user_answers"),
                    tradeoff_choice=msg.get("tradeoff_choice"),
                    interactive=bool(msg.get("interactive", False)),
                )
            except NegotiationRequiredError as exc:
                await websocket.send_json({"event": "negotiation_required", "negotiation": exc.negotiation})
                continue
            except (KeyError, ValueError) as exc:
                await websocket.send_json({"event": "error", "message": str(exc)})
                continue

            cfg = _get_board_config(layers=requested_layers, board=board_name, graph=graph)
            silicon_raw = await asyncio.to_thread(
                run_silicon_codesign_for_graph,
                graph=graph,
                prompt=prompt,
            )
            silicon_serialized = _serialize_silicon_bundle(silicon_raw)

            if prompt is not None or silicon_serialized is not None:
                preview_state = BoardState(netlist=graph, config=cfg)
                await websocket.send_json(
                    {
                        "event": "prompt_composed",
                        "prompt": prompt or "",
                        "board_name": graph.board_name,
                        "negotiation": neg_dict,
                        "silicon": silicon_serialized,
                        "board": preview_state.to_summary_dict(),
                        **_tensor_channels_payload(preview_state, active_net_id=None),
                    }
                )

            candidates = max(1, min(16, int(msg.get("candidates", 8))))
            max_iterations = max(1, min(10, int(msg.get("max_iterations", 5))))
            step_delay_ms = max(0, min(600, int(msg.get("step_delay_ms", 110))))

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
                    _cache_latest_result(
                        graph.board_name, res.board_state, res.kicad_pcb_text, silicon_raw
                    )
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
