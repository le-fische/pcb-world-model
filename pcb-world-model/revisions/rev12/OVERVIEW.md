# Revision 12 (`rev12`) — 8-Layer+ HDI (`2+4+2`) Buildup, Buried Core Vias & Staggered Microvia Reliability Engine

* **Revision Tag:** `rev12`
* **Status:** **Active in Repo (Verified)** (`68/68` unit & integration tests passing; `99.29 / 100` `B1–B5` seeded mean in `2.85s` / `99.38 / 100` `--blind` mean in `2.68s`; **`B13: hdi_fpga_bga_fanout_8l`** `23/23 (100%)` routed, `0` DRC, `95` microvias, `7` buried core vias (`L3<->L6`), `0` stacked microvias, `0` illegal skip-vias, `min_stagger=0.2828mm >= 0.25mm`, `SI=PASS`, `DFM=PASS (98.19/100)`, valid AP214 `.STEP` in `3.12s` seeded / `2.63s` `--blind`)
* **Prerequisite:** `rev10` DFM/DFA craftsmanship + `rev11` External Import & Optimization
* **Core Mission:** Scale `BoardState`, the C++20/Python router, and the DFM verifier from `2/4/6L` rigid stackups to **8-layer+ HDI (`2+4+2`) buildups**, enforcing strict lamination cycle budgets (`max_lamination_cycles = 2`) and **staggered-only microvia thermomechanical reliability** under BGA/FPGA fanouts.

---

## 1. Objective of `rev12`

Moving from `6L` to **`8-layer (2+4+2) HDI`** for dense FPGAs and PCIe Gen 4 requires understanding **fabrication lamination physics**, not just adding more layers to a loop:
1. **Sequential Lamination Cycle Constraints:** With a limit of **2 HDI lamination cycles**, the fab builds a 4-layer inner core (`L3–L6` with buried mechanical vias `L3↔L6`), adds buildup 1 (`L2`, `L7` with microvias `L2↔L3` / `L6↔L7`), and adds buildup 2 (`L1`, `L8` with microvias `L1↔L2` / `L7↔L8` + optional full through-via `L1↔L8`). Arbitrary skip-vias (`L1↔L4`) are **physically impossible** under 2 lamination cycles and must be rejected by the router & DRC kernel.
2. **Staggered vs. Stacked Microvia Reliability Trap:** Stacking microvias concentrically (`L1→L2` directly on top of `L2→L3`) saves a tiny bit of XY space under a BGA, **but creates severe copper-interface barrel cracking under thermal cycling** (especially near high-power GaN converters). `rev12` explicitly enforces **staggered offset microvias** ($\Delta xy \ge 0.25\text{mm}$).

---

## 2. Module-by-Module Implementation Plan

### A. 8-Layer (`2+4+2`) Stackup & Lamination Span Table (`env/config.py`, `env/board_state.py`)
* Generalize `BoardState.num_layers` to support **`8`** (`F.Cu`, `In1.GND`, `In2.SIG1`, `In3.PWR1`, `In4.GND2`, `In5.SIG2`, `In6.GND3`, `B.Cu`) and `BoardConfig.max_lamination_cycles: int = 2`.
* Define legal `ViaSpan` rules:
  * **Microvias (1-layer step only):** `(0↔1)`, `(1↔2)`, `(5↔6)`, `(6↔7)`.
  * **Buried Core Vias:** `(2↔5)` (inside the `L3–L6` core).
  * **Through-Board PTH Vias:** `(0↔7)`.
  * **Forbidden under 2-cycle HDI:** Multi-layer blind skip-vias (e.g., `0↔2` or `0↔3`).

### B. Staggered Microvia Enforcer (`models/world/neural_router.py`, `verifier/fast_clearance.py`)
* Add `enforce_staggered_microvias: bool = True` and `microvia_stagger_offset_mm: float = 0.25`.
* Inside A* layer transitions and DRC verification:
  * If net $k$ drops a microvia from `L1→L2` at `(x, y)`, block `L2→L3` microvia placement within radius `0.25mm` of `(x, y)` (forcing an immediate staggered dogbone/offset step before the next microvia drop!).
  * Flag any concentric stacked microvia (`dist < 0.25mm` across adjacent buildup layers) as a **`stacked_microvia_reliability_risk`** error.

### C. KiCad 8 8-Layer Export & BGA Footprint Builder (`env/kicad_exporter.py`, `env/footprint_generator.py`)
* Add `make_bga(...)` (`0.8mm` / `0.65mm` pitch FPGA BGA array) to `footprint_generator.py` and emit full 8-layer definitions + `(via (at ...) (layers "F.Cu" "In1.Cu") micro)` S-expressions in `kicad_exporter.py`.

---

## 3. `rev12` Validation Benchmark

| Benchmark ID | Topology & Stackup | Acceptance Criteria |
| :--- | :--- | :--- |
| **`B13: hdi_fpga_bga_fanout_8l`** | 8-layer (`2+4+2`) `30×50mm` rigid board with 64-ball FPGA BGA (`0.8mm` pitch), max `2` HDI lamination cycles, `0.10mm` trace/space microvia escape | **Verified PASS:** `23/23 (100%)` BGA fanout routed, **`0` concentric stacked microvias** (`min_stagger=0.2828mm >= 0.25mm`), **`0` illegal skip-vias**, **`0` DRC**, `95` microvias, `7` buried core vias (`L3<->L6`), `SI=PASS`, `DFM=PASS (98.19/100)` in `3.12s` (`2.63s` `--blind`) |

---

## 4. Implemented `rev12` Modules, CLI, API & Verified Artifacts (`pcb-world-model/revisions/rev12/`)

* **8-Layer (`2+4+2`) Stackup & Lamination Rules:** [`configs/default_8layer.yaml`](file:///Users/le/Documents/antigravity/kind-hypatia/pcb-world-model/configs/default_8layer.yaml), [`env/config.py`](file:///Users/le/Documents/antigravity/kind-hypatia/pcb-world-model/env/config.py), [`env/board_state.py`](file:///Users/le/Documents/antigravity/kind-hypatia/pcb-world-model/env/board_state.py) (`get_legal_via_spans`, `classify_via_span`, `compute_hdi_via_telemetry`, `F.Cu / In1.GND / In2.SIG1 / In3.PWR1 / In4.GND2 / In5.SIG2 / In6.GND3 / B.Cu`).
* **C++20 & Python Staggered Microvia A* Engine:** [`models/world/native_astar.cpp`](file:///Users/le/Documents/antigravity/kind-hypatia/pcb-world-model/models/world/native_astar.cpp) (`astar_multilayer_hdi`), [`models/world/native_kernel.py`](file:///Users/le/Documents/antigravity/kind-hypatia/pcb-world-model/models/world/native_kernel.py), [`models/world/neural_router.py`](file:///Users/le/Documents/antigravity/kind-hypatia/pcb-world-model/models/world/neural_router.py) (atomic 2-via staggered microvia bridges across buildup layers + `_mark_microvia_stagger_zone`).
* **HDI Lamination & Staggered Microvia DRC Verifier:** [`verifier/fast_clearance.py`](file:///Users/le/Documents/antigravity/kind-hypatia/pcb-world-model/verifier/fast_clearance.py) (`check_hdi_via_rules`: flags `illegal_skip_via` and `stacked_microvia_reliability_risk`).
* **Parametric BGA Builder, KiCad 8 Export & `.STEP`:** [`env/footprint_generator.py`](file:///Users/le/Documents/antigravity/kind-hypatia/pcb-world-model/env/footprint_generator.py) (`make_bga`), [`env/kicad_exporter.py`](file:///Users/le/Documents/antigravity/kind-hypatia/pcb-world-model/env/kicad_exporter.py) (`(blind_buried_vias_allowed yes)`, `(via micro ...)`, `(via blind ...)`), [`env/sexpr_parser.py`](file:///Users/le/Documents/antigravity/kind-hypatia/pcb-world-model/env/sexpr_parser.py) (`REV12_PRESET_DESCRIPTIONS`, `B13`), [`env/step_exporter.py`](file:///Users/le/Documents/antigravity/kind-hypatia/pcb-world-model/env/step_exporter.py).
* **CLI, API & Web UI (`REV 12`):** [`scripts/04_run_inference.py`](file:///Users/le/Documents/antigravity/kind-hypatia/pcb-world-model/scripts/04_run_inference.py) (`--layers 8`), [`scripts/05_run_benchmarks.py`](file:///Users/le/Documents/antigravity/kind-hypatia/pcb-world-model/scripts/05_run_benchmarks.py) (`--rev12`), [`ui/server.py`](file:///Users/le/Documents/antigravity/kind-hypatia/pcb-world-model/ui/server.py) (`Literal[2, 4, 6, 8]`, `include_rev12=true`), [`ui/web/`](file:///Users/le/Documents/antigravity/kind-hypatia/pcb-world-model/ui/web/).
