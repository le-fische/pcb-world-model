"""Headless wrapper for `kicad-cli pcb drc --format json` with symbolic 2D DRC fallback."""

from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

from env.kicad_exporter import export
from verifier.fast_clearance import DRCError, verify_board

if TYPE_CHECKING:
    from env.board_state import BoardState


def find_kicad_cli() -> str | None:
    """Locate `kicad-cli` on PATH or in standard macOS KiCad installation."""
    on_path = shutil.which("kicad-cli")
    if on_path:
        return on_path
    mac_default = Path("/Applications/KiCad/KiCad.app/Contents/MacOS/kicad-cli")
    if mac_default.exists():
        return str(mac_default)
    return None


def _run_headless_kicad_cli(
    board_state: BoardState, kicad_cli_path: str
) -> list[DRCError]:
    """Run `kicad-cli pcb drc --format json` on an exported temporary `.kicad_pcb`."""
    errors: list[DRCError] = []
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_dir = Path(tmpdir)
        pcb_path = tmp_dir / "verify_board.kicad_pcb"
        report_path = tmp_dir / "drc_report.json"
        export(board_state, pcb_path)

        cmd = [
            kicad_cli_path,
            "pcb",
            "drc",
            "--format",
            "json",
            "--output",
            str(report_path),
            str(pcb_path),
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if proc.returncode != 0 or not report_path.exists():
            return errors

        data = json.loads(report_path.read_text(encoding="utf-8"))
        for violation in data.get("violations", []):
            severity = violation.get("severity", "error")
            if severity not in ("error", "warning"):
                continue
            v_type = str(violation.get("type", "kicad_drc"))
            desc = str(violation.get("description", v_type))
            items = violation.get("items", [])
            x_mm, y_mm = board_state.netlist.width_mm / 2.0, board_state.netlist.height_mm / 2.0
            if items and "pos" in items[0]:
                x_mm = float(items[0]["pos"].get("x", x_mm))
                y_mm = float(items[0]["pos"].get("y", y_mm))
            gx, gy = board_state.mm_to_grid(x_mm, y_mm)
            errors.append(
                DRCError(
                    error_type=v_type,
                    layer=0,
                    x_mm=x_mm,
                    y_mm=y_mm,
                    x_grid=gx,
                    y_grid=gy,
                    offending_net_id=0,
                    message=f"[kicad-cli] {desc}",
                )
            )
    return errors


def verify(board_state: BoardState, run_kicad_cli: bool = True) -> list[DRCError]:
    """Run deterministic 2D geometric DRC and optional headless `kicad-cli` verification."""
    errors = verify_board(board_state)
    if run_kicad_cli:
        cli_path = find_kicad_cli()
        if cli_path is not None:
            cli_errors = _run_headless_kicad_cli(board_state, cli_path)
            if cli_errors and not errors:
                errors.extend(cli_errors)
    return errors
