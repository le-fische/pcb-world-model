"""Zero-copy ctypes wrapper and auto-compiler for the C++20 native A* kernel."""

from __future__ import annotations

import ctypes
import platform
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

_WORLD_DIR = Path(__file__).resolve().parent
_CPP_SRC = _WORLD_DIR / "native_astar.cpp"
_NATIVE_DIR = _WORLD_DIR / "_native"


def _lib_extension() -> str:
    sys_name = platform.system()
    if sys_name == "Darwin":
        return ".dylib"
    if sys_name == "Windows":
        return ".dll"
    return ".so"


def _find_cpp_compiler() -> str | None:
    for candidate in ("clang++", "g++", "c++"):
        found = shutil.which(candidate)
        if found:
            return found
    return None


def _compile_shared_library(src: Path, dst: Path) -> bool:
    compiler = _find_cpp_compiler()
    if compiler is None or not src.exists():
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        compiler,
        "-O3",
        "-std=c++20",
        "-Wall",
        "-Wextra",
        "-Wpedantic",
        "-shared",
        "-fPIC",
        str(src),
        "-o",
        str(dst),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    return proc.returncode == 0 and dst.exists()


class _NativeAStarBinding:
    """Lazy-loaded ctypes binding to `libpcb_astar` with automatic build & thread-safe buffer reuse."""

    def __init__(self) -> None:
        self._lib: ctypes.CDLL | None = None
        self._lib_path: Path | None = None
        self._load_error: str | None = None
        self._load()

    def _load(self) -> None:
        ext = _lib_extension()
        target_lib = _NATIVE_DIR / f"libpcb_astar{ext}"

        needs_build = not target_lib.exists() or (
            _CPP_SRC.exists() and _CPP_SRC.stat().st_mtime > target_lib.stat().st_mtime
        )
        if needs_build:
            ok = _compile_shared_library(_CPP_SRC, target_lib)
            if not ok:
                # Try compiling into a writable temp directory if repo tree is read-only
                tmp_lib = Path(tempfile.gettempdir()) / f"libpcb_astar{ext}"
                if _compile_shared_library(_CPP_SRC, tmp_lib):
                    target_lib = tmp_lib

        if not target_lib.exists():
            self._load_error = "C++ compiler or compiled shared library not available"
            return

        try:
            lib = ctypes.CDLL(str(target_lib))
            lib.pcb_astar_abi_version.argtypes = []
            lib.pcb_astar_abi_version.restype = ctypes.c_int32

            if int(lib.pcb_astar_abi_version()) != 2:
                self._load_error = "ABI version mismatch"
                return

            lib.astar_multilayer.argtypes = [
                ctypes.c_int32,  # L
                ctypes.c_int32,  # H
                ctypes.c_int32,  # W
                ctypes.c_void_p,  # start_nodes
                ctypes.c_int32,  # num_starts
                ctypes.c_void_p,  # target_nodes
                ctypes.c_int32,  # num_targets
                ctypes.c_void_p,  # blocked
                ctypes.c_void_p,  # via_blocked
                ctypes.c_void_p,  # soft_penalty
                ctypes.c_void_p,  # cost_field
                ctypes.c_void_p,  # occ_layers
                ctypes.c_void_p,  # allowed_layers
                ctypes.c_void_p,  # transition_costs
                ctypes.c_int32,  # net_id
                ctypes.c_int32,  # max_expansions
                ctypes.c_void_p,  # out_path
                ctypes.c_int32,  # max_path_len
                ctypes.c_void_p,  # out_path_len
            ]
            lib.astar_multilayer.restype = ctypes.c_int32

            if hasattr(lib, "astar_multilayer_hdi"):
                lib.astar_multilayer_hdi.argtypes = [
                    ctypes.c_int32,  # L
                    ctypes.c_int32,  # H
                    ctypes.c_int32,  # W
                    ctypes.c_void_p,  # start_nodes
                    ctypes.c_int32,  # num_starts
                    ctypes.c_void_p,  # target_nodes
                    ctypes.c_int32,  # num_targets
                    ctypes.c_void_p,  # blocked
                    ctypes.c_void_p,  # via_blocked
                    ctypes.c_void_p,  # stagger_blocked
                    ctypes.c_void_p,  # soft_penalty
                    ctypes.c_void_p,  # cost_field
                    ctypes.c_void_p,  # occ_layers
                    ctypes.c_void_p,  # allowed_layers
                    ctypes.c_void_p,  # transition_costs
                    ctypes.c_int32,  # net_id
                    ctypes.c_float,  # min_stagger_cells
                    ctypes.c_int32,  # max_expansions
                    ctypes.c_void_p,  # out_path
                    ctypes.c_int32,  # max_path_len
                    ctypes.c_void_p,  # out_path_len
                ]
                lib.astar_multilayer_hdi.restype = ctypes.c_int32

            self._lib = lib
            self._lib_path = target_lib
            self._load_error = None
        except OSError as exc:
            self._lib = None
            self._load_error = str(exc)

    @property
    def available(self) -> bool:
        return self._lib is not None

    def info(self) -> dict[str, Any]:
        return {
            "available": self.available,
            "backend": "cpp20_native" if self.available else "python_fallback",
            "library_path": str(self._lib_path) if self._lib_path else None,
            "error": self._load_error,
        }

    def run_astar(
        self,
        L: int,
        H: int,
        W: int,
        start_nodes: list[tuple[int, int, int]],
        target_nodes: list[tuple[int, int, int]],
        blocked: np.ndarray,
        via_blocked: np.ndarray,
        soft_penalty: np.ndarray,
        cost_field: np.ndarray,
        occ_layers: np.ndarray,
        allowed_layers: np.ndarray,
        transition_costs: np.ndarray,
        net_id: int,
        max_expansions: int,
        stagger_blocked: np.ndarray | None = None,
        min_stagger_cells: float = 0.0,
    ) -> list[tuple[int, int, int]] | None:
        """Execute zero-copy C++20 multi-layer octilinear A* search."""
        if self._lib is None or not start_nodes or not target_nodes:
            return None

        starts_arr = np.ascontiguousarray(start_nodes, dtype=np.int32)
        targets_arr = np.ascontiguousarray(target_nodes, dtype=np.int32)
        blocked_u8 = np.ascontiguousarray(blocked, dtype=np.uint8)
        via_u8 = np.ascontiguousarray(via_blocked, dtype=np.uint8)
        soft_f32 = np.ascontiguousarray(soft_penalty, dtype=np.float32)
        cost_f32 = np.ascontiguousarray(cost_field, dtype=np.float32)
        occ_i32 = np.ascontiguousarray(occ_layers, dtype=np.int32)
        allowed_u8 = np.ascontiguousarray(allowed_layers, dtype=np.uint8)
        trans_f32 = np.ascontiguousarray(transition_costs, dtype=np.float32)

        max_path_len = max(4096, min(L * H * W, 65536))
        out_path = np.empty((max_path_len, 3), dtype=np.int32)
        out_len = ctypes.c_int32(0)

        if (min_stagger_cells > 0.0 or stagger_blocked is not None) and hasattr(
            self._lib, "astar_multilayer_hdi"
        ):
            stag_u8 = (
                np.ascontiguousarray(stagger_blocked, dtype=np.uint8)
                if stagger_blocked is not None
                else np.zeros((L, H, W), dtype=np.uint8)
            )
            rc = int(
                self._lib.astar_multilayer_hdi(
                    ctypes.c_int32(L),
                    ctypes.c_int32(H),
                    ctypes.c_int32(W),
                    starts_arr.ctypes.data,
                    ctypes.c_int32(len(start_nodes)),
                    targets_arr.ctypes.data,
                    ctypes.c_int32(len(target_nodes)),
                    blocked_u8.ctypes.data,
                    via_u8.ctypes.data,
                    stag_u8.ctypes.data,
                    soft_f32.ctypes.data,
                    cost_f32.ctypes.data,
                    occ_i32.ctypes.data,
                    allowed_u8.ctypes.data,
                    trans_f32.ctypes.data,
                    ctypes.c_int32(net_id),
                    ctypes.c_float(float(min_stagger_cells)),
                    ctypes.c_int32(max_expansions),
                    out_path.ctypes.data,
                    ctypes.c_int32(max_path_len),
                    ctypes.byref(out_len),
                )
            )
        else:
            rc = int(
                self._lib.astar_multilayer(
                    ctypes.c_int32(L),
                    ctypes.c_int32(H),
                    ctypes.c_int32(W),
                    starts_arr.ctypes.data,
                    ctypes.c_int32(len(start_nodes)),
                    targets_arr.ctypes.data,
                    ctypes.c_int32(len(target_nodes)),
                    blocked_u8.ctypes.data,
                    via_u8.ctypes.data,
                    soft_f32.ctypes.data,
                    cost_f32.ctypes.data,
                    occ_i32.ctypes.data,
                    allowed_u8.ctypes.data,
                    trans_f32.ctypes.data,
                    ctypes.c_int32(net_id),
                    ctypes.c_int32(max_expansions),
                    out_path.ctypes.data,
                    ctypes.c_int32(max_path_len),
                    ctypes.byref(out_len),
                )
            )
        if rc != 1:
            return None

        n = int(out_len.value)
        return [(int(out_path[i, 0]), int(out_path[i, 1]), int(out_path[i, 2])) for i in range(n)]


_BINDING = _NativeAStarBinding()


def native_astar_available() -> bool:
    """Return True if the compiled C++20 A* shared library is loaded."""
    return _BINDING.available


def native_kernel_info() -> dict[str, Any]:
    """Return metadata about the active A* search backend."""
    return _BINDING.info()


def run_native_astar(
    L: int,
    H: int,
    W: int,
    start_nodes: list[tuple[int, int, int]],
    target_nodes: list[tuple[int, int, int]],
    blocked: np.ndarray,
    via_blocked: np.ndarray,
    soft_penalty: np.ndarray,
    cost_field: np.ndarray,
    occ_layers: np.ndarray,
    allowed_layers: np.ndarray,
    transition_costs: np.ndarray,
    net_id: int,
    max_expansions: int = 120_000,
    stagger_blocked: np.ndarray | None = None,
    min_stagger_cells: float = 0.0,
) -> list[tuple[int, int, int]] | None:
    """Run C++20 native A* kernel if loaded; raises RuntimeError if unavailable."""
    if not _BINDING.available:
        raise RuntimeError("Native C++20 A* kernel is not loaded")
    return _BINDING.run_astar(
        L=L,
        H=H,
        W=W,
        start_nodes=start_nodes,
        target_nodes=target_nodes,
        blocked=blocked,
        via_blocked=via_blocked,
        soft_penalty=soft_penalty,
        cost_field=cost_field,
        occ_layers=occ_layers,
        allowed_layers=allowed_layers,
        transition_costs=transition_costs,
        net_id=net_id,
        max_expansions=max_expansions,
        stagger_blocked=stagger_blocked,
        min_stagger_cells=min_stagger_cells,
    )
