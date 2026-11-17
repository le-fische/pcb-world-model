# Revision 13 (`rev13`) — Mixed-Signal Domain Partitioning, Continuous Ground Return Physics ($J_{\text{return}}$) & Anti-Split-Ground Solver

* **Revision Tag:** `rev13`
* **Status:** **Active in Repo (Verified)** (`73/73` unit & integration tests passing; `99.29 / 100` `B1–B5` seeded mean in `2.23s` / `99.38 / 100` `--blind` mean in `2.18s`; **`B14`** `15/15 (100%)` routed, `0` DRC, `ContGND=True`, `Sep=28.07mm` (`22.01mm` `--blind`), `0.0%` $J_{\text{return}}$ overlap, `0` broadside violations, `DFM=100.0`, valid AP214 `.STEP`; **`B14a–c`** all PASS in `2.08s` total)
* **Prerequisite:** `rev12` (8-layer HDI buildup & staggered microvias)
* **Core Mission:** Solve the **Ground Plane & Mixed-Signal Partitioning Trap**: keep a **continuous low-inductance ground plane** (never splitting `GND` under high-speed digital lines), while partitioning noisy high-$di/dt$ switchers (`5A, 2MHz GaN` buck) away from ultra-sensitive analog front-ends (`24-bit sub-mV ADC`) on the **same constrained outer layer (`F.Cu`)** and verifying zero return-current overlap.

---

## 1. Objective of `rev13`

A weak AI or naive tutorial splits the `GND` plane into separate `AGND` and `DGND` islands when it sees a `24-bit ADC` next to a `2MHz GaN switching buck`—which **catastrophically destroys return paths** for high-speed FPGA/PCIe signals crossing the split and turns the board into a slot antenna.

Instead, `rev13` models **actual electromagnetic return physics**:
1. **Continuous `GND` Plane Enforcement:** High-speed/mixed-signal boards strictly use a unified, unsplit reference plane (`In1.GND`) so every high-speed trace sees an uninterrupted image current directly underneath it.
2. **Spatial Noise Domain Partitioning on `F.Cu`:** When enclosure limits force both a noisy `5A, 2MHz GaN` switcher and a `24-bit sub-mV ADC` onto the Top layer (`F.Cu`) inside a tight `30×50mm` rectangle, `rev13` enforces a **physical domain partition barrier** (`noisy_pwr` zone at one end vs. `quiet_analog` zone at the opposite end, with `digital_hs` in between/isolated).
3. **Return-Current Path Vector Field ($J_{\text{return}}(x, y)$):** Computes the 2D ground return current envelope between `SW_NODE` / `VIN_CAP` / `GND` pins of the GaN converter and verifies that high-$di/dt$ switching return paths remain tightly localized inside the `noisy_pwr` quadrant and **never cross beneath the ADC's analog domain or differential input traces**.

---

## 2. Module-by-Module Implementation Plan

### A. Noise-Domain Tagging & Partition Exclusion Loss (`models/placer/relaxation.py`, `env/component_library.py`)
* Add `noise_domain: str` (`"power_switching" | "analog_precision" | "digital_hs" | "neutral"`) to `ComponentNode` and `NetInfo`.
* In `relax_and_snap_to_grid()` and `MacroPlacerGNN`:
  * Add an explicit **cross-domain repulsion force** between `"power_switching"` (`GaN FET`, `Inductor`, `SW_NODE` caps) and `"analog_precision"` (`24-bit ADC`, `VREF`, `Analog Sensor Inputs`), pushing their centroids apart toward opposite ends of the board even when both are locked to `side="top"` (`F.Cu`).
  * Simultaneously pull `"power_switching"` input/boot capacitors as tight as possible (`high_di_dt_loop`) to minimize the $di/dt$ loop radius.

### B. Return-Current Density Verifier ($J_{\text{return}}$) & Analog Corridor Guard (`env/metrics.py`, `verifier/fast_clearance.py`)
* Add `compute_ground_return_isolation(board_state)` in `env/metrics.py`:
  * Computes the bounding envelope + current-density spread of `power_switching` return loops on `In1.GND`.
  * Checks that `0%` of high-$di/dt$ switching return density intersects the `analog_precision` courtyard + `3.0mm` analog guard halo, and that **zero switching traces route under or adjacent to sub-mV analog input pairs**.
  * Rejects any split/slot in `In1.GND` beneath `digital_hs` (`PCIe`/`FPGA`) nets.

### C. Z-Axis Crosstalk (Broadside Coupling) & Thermal-vs-Signal Adversarial Inpainting (`models/world/neural_router.py`, `verifier/fast_clearance.py`, `env/metrics.py`)
* **Z-Axis Broadside Coupling Detector (`compute_broadside_crosstalk_risk`):**
  * Extends 2D per-layer clearance to **3D inter-layer coupling**: detects when a high-sensitivity net (`analog_precision`, `<1mV`) on layer $L_k$ overlaps parallel $(x, y)$ tracks of an aggressor net (`digital_hs` clock or `power_switching`) on adjacent internal layer $L_{k \pm 1}$ separated by thin prepreg/core dielectric.
  * Computes broadside parasitic capacitance ($C_{\text{broadside}} \propto \varepsilon_r \cdot A_{\text{overlap}} / h_{\text{core}}$) and penalizes parallel adjacent-layer runs inside A\* costs—forcing the router to **offset tracks laterally, cross strictly at $90^\circ$, or interleave a solid `GND` reference shield layer**.
* **"Star Ground" Contradiction Refusal (`validate_ground_topology`):**
  * When an adversarial initial state specifies an explicit remote `AGND↔DGND` single-point bridge far away from a high-speed signal crossing, `rev13` computes return-path loop inductance and **rejects the remote star-ground instruction as physically invalid** (`invalid_remote_star_ground`), insisting on continuous `In1.GND` (or direct bridging immediately beneath the signal crossing) to eliminate slot-antenna radiation.
* **80%-Locked Thermal-vs-Signal Adversarial Inpainting (`load_adversarial_inpainting_state`):**
  * Loads a partially complete `.kicad_pcb` (`is_locked=True` components such as a `5W, 2MHz` switching MOSFET pinned close to a `<1mV` ADC inside a strict enclosure) plus `electrical_meta` JSON (`power_watts`, `switching_freq_hz`, `noise_floor_mv`) and runs **physics-aware inpainting** on the remaining 20%: carving thermal-isolation copper slots/reliefs around the hot MOSFET, routing sensitive differential ADC lines clear of thermal plumes, and containing switching return currents.

### D. Structured Physics Justification Output (`models/composer/design_rationale.py`)
* Emit a quantitative, verified engineering explanation in `design_rationale` detailing:
  * Why `In1.GND` was kept **continuous** (0 splits under high-speed FPGA/PCIe traces, and explicit rejection of any remote star-ground trap),
  * Exact physical separation achieved between the `2MHz GaN` `SW_NODE` and `24-bit ADC` ($\Delta d \ge 20\text{mm}$ on `30×50mm`),
  * Verified `0%` high-$di/dt$ return-current intrusion into the analog corridor, `0` broadside $L_k/L_{k\pm 1}$ parallel analog-over-clock violations, and thermal copper isolation slots around fixed power devices.

---

## 3. `rev13` Validation Benchmarks

| Benchmark ID | Topology & Adversarial Trap | Acceptance Criteria |
| :--- | :--- | :--- |
| **`B14: mixed_signal_gan_adc_30x50`** | `30×50mm` 8L HDI board: `5A 2MHz GaN` buck converter + `24-bit sub-mV ADC` **both forced onto Top layer (`F.Cu`)** + digital FPGA core | **Verified PASS:** `15/15 (100%)` routed, Continuous `In1.GND`, GaN-to-ADC separation `28.07mm` (`22.01mm` `--blind`) $\ge 20\text{mm}$, `0.0%` $J_{\text{return}}$ overlap into ADC domain, `0` broadside violations, `DFM=100.0`, `0` DRC (`1.32s`) |
| **`B14a: z_axis_broadside_crosstalk_trap`** | Thin-core 8L pre-seeded state attempting parallel `L3` sub-mV analog sensor vs `L4` fast digital clock directly stacked in $(x,y)$ | **Verified PASS:** Pre-seed detector flags `1` overlap (`broadside_crosstalk_violation`); closed-loop router reroutes with `0` broadside overlaps (`max=0.0pF`), `3/3` routed, `0` DRC (`0.21s`) |
| **`B14b: remote_star_ground_contradiction`** | Mixed-signal MCU + ADC board constrained with an adversarial remote `AGND↔DGND` star bridge far from high-speed crossing | **Verified PASS:** Verifier rejects initial constraint (`invalid_remote_star_ground`); solver enforces continuous `In1.GND`, `0` slot-antenna loops, `7/7` routed, `0` DRC (`0.20s`) |
| **`B14c: thermal_vs_signal_inpainting_80pct`** | 80%-locked enclosure: `5W 2MHz` MOSFET pinned near `<1mV` ADC; solver must finish pours, thermal reliefs, and routing | **Verified PASS:** `LocksOK=True`, `1` auto-carved `THERMAL_SLOT_Q1` copper-pour relief slot, continuous `In1.GND`, `0.0%` $J_{\text{return}}$ intrusion, `7/7` routed, `0` DRC (`0.26s`) |

---

## 4. Implemented `rev13` Modules, CLI, API & Verified Artifacts (`pcb-world-model/revisions/rev13/`)

* **Noise-Domain Tagging & Cross-Domain Repulsion:** [`env/sexpr_parser.py`](file:///Users/le/Documents/antigravity/kind-hypatia/pcb-world-model/env/sexpr_parser.py) (`noise_domain`, `annotate_noise_domains`, `mixed_signal_gan_adc_30x50`, `build_rev13_*_trap`), [`models/placer/relaxation.py`](file:///Users/le/Documents/antigravity/kind-hypatia/pcb-world-model/models/placer/relaxation.py) (cross-domain repulsion + pole bias + attenuated cross-domain springs).
* **Return-Current Density ($J_{\text{return}}$) & Z-Axis Broadside Coupling:** [`env/metrics.py`](file:///Users/le/Documents/antigravity/kind-hypatia/pcb-world-model/env/metrics.py) (`compute_ground_return_isolation`, `compute_broadside_crosstalk_risk`), [`env/config.py`](file:///Users/le/Documents/antigravity/kind-hypatia/pcb-world-model/env/config.py).
* **Broadside & Star-Ground Topology DRC Verifier:** [`verifier/fast_clearance.py`](file:///Users/le/Documents/antigravity/kind-hypatia/pcb-world-model/verifier/fast_clearance.py) (`check_broadside_crosstalk`, `validate_ground_topology`, `check_ground_topology`).
* **Broadside & Thermal Corridor A\* Router:** [`models/world/neural_router.py`](file:///Users/le/Documents/antigravity/kind-hypatia/pcb-world-model/models/world/neural_router.py) (adjacent unshielded broadside avoidance + mixed-signal corridor & thermal plume guard).
* **80%-Locked Thermal-vs-Signal Inpainting & KiCad Slot Export:** [`env/importers/board_importer.py`](file:///Users/le/Documents/antigravity/kind-hypatia/pcb-world-model/env/importers/board_importer.py) (`load_adversarial_inpainting_state`, `solve_thermal_signal_inpainting`), [`env/kicad_exporter.py`](file:///Users/le/Documents/antigravity/kind-hypatia/pcb-world-model/env/kicad_exporter.py), [`env/board_state.py`](file:///Users/le/Documents/antigravity/kind-hypatia/pcb-world-model/env/board_state.py).
* **Quantitative Physics Design Rationale:** [`models/composer/design_rationale.py`](file:///Users/le/Documents/antigravity/kind-hypatia/pcb-world-model/models/composer/design_rationale.py) (`build_structured_rationale_report`, `generate_physics_design_rationale`).
* **Benchmarks, Tests, API & Web UI (`REV 13`):** [`scripts/04_run_inference.py`](file:///Users/le/Documents/antigravity/kind-hypatia/pcb-world-model/scripts/04_run_inference.py), [`scripts/05_run_benchmarks.py`](file:///Users/le/Documents/antigravity/kind-hypatia/pcb-world-model/scripts/05_run_benchmarks.py) (`--rev13`), [`tests/test_rev13_mixed_signal_em.py`](file:///Users/le/Documents/antigravity/kind-hypatia/pcb-world-model/tests/test_rev13_mixed_signal_em.py), [`ui/server.py`](file:///Users/le/Documents/antigravity/kind-hypatia/pcb-world-model/ui/server.py) (`/api/mixed-signal-audit`, `/api/adversarial-inpaint`, `include_rev13=true`), [`ui/web/`](file:///Users/le/Documents/antigravity/kind-hypatia/pcb-world-model/ui/web/).
