"""Revision 9 tests: ISO-10303-21 AP214 .STEP exporter, enriched BoardState, and Pro EDA Workstation UI/API."""

from __future__ import annotations

import importlib
from pathlib import Path
import sys

from fastapi.testclient import TestClient

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from env.config import BoardConfig
from env.manufacturing_exporter import export_manufacturing_bundle
from env.sexpr_parser import build_preset_board
from env.step_exporter import generate_step_assembly, validate_step_structure
from ui.server import app

_inf_mod = importlib.import_module("scripts.04_run_inference")
design_pcb_world_model = _inf_mod.design_pcb_world_model


def test_rev9_step_exporter_brep_holes_and_components() -> None:
    cfg = BoardConfig.from_yaml(ROOT_DIR / "configs" / "default_2layer.yaml")
    graph = build_preset_board("wearable_oled_keychain_26x26")
    res = design_pcb_world_model(
        netlist_graph=graph,
        board_config=cfg,
        num_candidates=4,
        max_iterations=5,
    )
    assert res.success is True

    step_text = generate_step_assembly(res.board_state, mask_theme="black")
    diag = validate_step_structure(step_text)

    assert diag["valid"] is True, f"STEP validation failed: {diag}"
    assert diag["dangling_refs"] == []
    assert diag["ap214_schema"] is True
    assert diag["cylindrical_holes"] >= 2
    non_holes = [c for c in res.board_state.netlist.components.values() if c.comp_type != "mounting_hole"]
    assert diag["manifold_solids"] > len(non_holes) + 1
    assert diag["assembly_occurrences"] == len(non_holes) + 1
    assert diag["presentation_reps"] == len(non_holes) + 1
    assert diag["fill_area_style_colors"] >= 5
    assert "SOLDER_MASK_BLACK" in step_text
    assert "ENIG_GOLD" in step_text
    assert "COPPER_TRACE" in step_text
    assert "SILVER_LEAD" in step_text or "SILVER_TIN_ENDCAP" in step_text


def test_rev9_manufacturing_bundle_step_and_summary_metadata(tmp_path: Path) -> None:
    cfg = BoardConfig.from_yaml(ROOT_DIR / "configs" / "default_2layer.yaml")
    graph = build_preset_board("mcu_sensor_node")
    res = design_pcb_world_model(
        netlist_graph=graph,
        board_config=cfg,
        num_candidates=4,
    )
    assert res.success is True

    summary = res.board_state.to_summary_dict()
    assert len(summary["nets"]) >= 8
    assert all("footprint" in c and "body_width_mm" in c for c in summary["components"])

    paths = export_manufacturing_bundle(res.board_state, tmp_path)
    assert "step_file" in paths
    assert paths["step_file"].exists()
    diag = validate_step_structure(paths["step_file"].read_text(encoding="utf-8"))
    assert diag["valid"] is True
    assert diag["dangling_refs"] == []


def test_rev9_fastapi_step_export_and_workstation_assets() -> None:
    client = TestClient(app)

    home = client.get("/")
    assert home.status_code == 200
    assert "REV 9" in home.text
    assert "btn-view-schematic" in home.text
    assert "btn-download-step" in home.text
    assert "btn-flip-board" in home.text
    assert "mask-swatch-btn" in home.text

    for asset in (
        "/static/app.js",
        "/static/pcb_pbr_3d.js",
        "/static/component_meshes_3d.js",
        "/static/pin_diagram.js",
        "/static/schematic_view.js",
        "/static/vendor/three/three.module.min.js",
        "/static/vendor/three/OrbitControls.js",
        "/static/vendor/three/RoomEnvironment.js",
    ):
        resp = client.get(asset)
        assert resp.status_code == 200, f"Asset missing: {asset}"

    step_resp = client.get(
        "/api/export-step?board=wearable_oled_keychain_26x26&mask_theme=blue&download=1"
    )
    assert step_resp.status_code == 200
    assert "wearable_oled_keychain_26x26_assembly.step" in step_resp.headers.get("content-disposition", "")
    assert "SOLDER_MASK_BLUE" in step_resp.text
    assert validate_step_structure(step_resp.text)["valid"] is True

    bundle_resp = client.get("/api/manufacturing_bundle?board=mcu_sensor_node")
    assert bundle_resp.status_code == 200
    bundle = bundle_resp.json()
    assert "step_assembly" in bundle
    assert validate_step_structure(bundle["step_assembly"])["valid"] is True
