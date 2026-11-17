# Revision 3 (`rev3`) — Architecture & Benchmark Overview

* **Revision Tag:** `rev3`
* **Status:** Verified (`22/22` unit & integration tests passing)
* **Prerequisite:** `rev2` verified (**99.26 / 100** mean composite score, **23/23** nets routed, **0** DRC errors, vias reduced $27 \to 10$, wirelength reduced $187.07\text{mm} \to 154.05\text{mm}$, all boards solved in **1 iteration**)
* **Mean Composite Score:** **99.28 / 100** across all 4 boards (**35/35** nets routed, **0** DRC errors, **13** total vias, **9.28s** total runtime; **98.33 / 100** in `--blind` zero-hint mode with **35/35** routed and **0** DRC errors)
* **Benchmark Artifacts:**
  * [`benchmark_report.json`](file:///Users/le/Documents/antigravity/kind-hypatia/pcb-world-model/revisions/rev3/benchmark_report.json)
  * [`boards/output_mcu_sensor_node.kicad_pcb`](file:///Users/le/Documents/antigravity/kind-hypatia/pcb-world-model/revisions/rev3/boards/output_mcu_sensor_node.kicad_pcb)
  * [`boards/output_buck_converter.kicad_pcb`](file:///Users/le/Documents/antigravity/kind-hypatia/pcb-world-model/revisions/rev3/boards/output_buck_converter.kicad_pcb)
  * [`boards/output_usb_uart_bridge.kicad_pcb`](file:///Users/le/Documents/antigravity/kind-hypatia/pcb-world-model/revisions/rev3/boards/output_usb_uart_bridge.kicad_pcb)
  * [`boards/output_rp2040_minimal.kicad_pcb`](file:///Users/le/Documents/antigravity/kind-hypatia/pcb-world-model/revisions/rev3/boards/output_rp2040_minimal.kicad_pcb)

---

## 1. Post-`rev2` Diagnostics: Why We Need `rev3`

`rev2` solved **P0** from our diagnostics log (cutting `mcu_sensor_node` vias from $22 \to 5$ and converging in 1 pass via sub-grid Euclidean clearance fields, full Steiner T-taps, and 2-opt component swapping).

Looking at the `rev2` output and [`benchmarks.md`](file:///Users/le/Documents/antigravity/kind-hypatia/benchmarks.md#L149-L159), three high-impact engineering bottlenecks remain (**P1, P2, P3**):

1. **P1 — Uniform Trace Width Across Power & Signal Nets (`env/config.py`, `models/world/neural_router.py`):**
   * Even in `rev2`, every net is routed at a single scalar `cfg.trace_width_mm = 0.25mm`. In real hardware (and where GPT-6 Astra failed in its KiCad launch demo), power and ground rails (`NetInfo.is_power_ground == True`, e.g., `VIN_12V`, `SW_NODE`, `VOUT_5V`, `VCC_3V3`, `GND`) carry higher current and require wider copper (`0.40mm`–`0.45mm`), while signal nets use `0.25mm`.
2. **P2/P3 — Blind Zero-Hint Placement Generalization (`scripts/04_run_inference.py`, `models/placer/relaxation.py`):**
   * Currently, `design_pcb_world_model()` appends `seed_placement` (the initial schematic coordinates) as candidate `#8` alongside the $K-1$ `MacroPlacerGNN` samples. A true World Model must achieve **100% routing completion and 0 DRC errors in `--no-seed-candidate` mode** (starting strictly from the GNN predictions + multi-start random initializations).
3. **Scaling Beyond 10 Components (`B4: rp2040_minimal` High-Density Board):**
   * `B1–B3` have 9–10 components and 7–8 nets. To prove our World Model scales toward complex embedded hardware, `rev3` adds a denser multi-IC embedded board (`rp2040_minimal`: MCU + QSPI Flash + Crystal Oscillator + LDO + USB-C + Decoupling network) with crystal proximity constraints.

---

## 2. What to Build in `rev3` (Module-by-Module Checklist)

### A. Dynamic Per-Net Class Trace Width (`env/config.py`, `models/world/neural_router.py`)
* **Files to edit:**
  * [`pcb-world-model/env/config.py`](file:///Users/le/Documents/antigravity/kind-hypatia/pcb-world-model/env/config.py)
  * [`pcb-world-model/configs/default_2layer.yaml`](file:///Users/le/Documents/antigravity/kind-hypatia/pcb-world-model/configs/default_2layer.yaml)
  * [`pcb-world-model/models/world/neural_router.py`](file:///Users/le/Documents/antigravity/kind-hypatia/pcb-world-model/models/world/neural_router.py)
* **Implementation Steps:**
  1. Add `power_trace_width_mm: float = 0.40` (and `signal_trace_width_mm: float = 0.25`) to `BoardConfig` in `env/config.py` and `configs/default_2layer.yaml`.
  2. In `octilinear_astar_route()`, check `net = state.netlist.nets.get(net_id)`:
     ```python
     is_pwr = bool(net and net.is_power_ground)
     trace_width_mm = cfg.power_trace_width_mm if is_pwr else cfg.trace_width_mm
     ```
  3. Because `_build_exact_clearance_masks(state, net_id, trace_width_mm)` in `rev2` already takes `trace_width_mm` as a parameter and `fast_clearance.py` checks `0.5 * (sa.width_mm + sb.width_mm)`, both the router and DRC verifier will automatically enforce wider clearance corridors around `0.40mm` power traces!

### B. Blind Zero-Hint Placement Mode (`scripts/04_run_inference.py`, `models/placer/relaxation.py`)
* **Files to edit:**
  * [`pcb-world-model/scripts/04_run_inference.py`](file:///Users/le/Documents/antigravity/kind-hypatia/pcb-world-model/scripts/04_run_inference.py)
  * [`pcb-world-model/models/placer/relaxation.py`](file:///Users/le/Documents/antigravity/kind-hypatia/pcb-world-model/models/placer/relaxation.py)
* **Implementation Steps:**
  1. Add `include_seed_candidate: bool = True` to `design_pcb_world_model()` and `--blind` CLI flag to `04_run_inference.py` and `05_run_benchmarks.py`.
  2. When `include_seed_candidate=False`, generate all $K$ candidates purely from `placer_gnn.sample_placements()` + force-directed spectral layout initialization (`relax_and_snap_to_grid` with higher initial `hpwl_pull` annealing so components self-organize around locked edge connectors `J1`/`J2` even from blind noise).

### C. Add High-Density Benchmark Board `B4: rp2040_minimal` (`env/sexpr_parser.py`)
* **File to edit:** [`pcb-world-model/env/sexpr_parser.py`](file:///Users/le/Documents/antigravity/kind-hypatia/pcb-world-model/env/sexpr_parser.py)
* **Implementation Steps:**
  1. Add `"rp2040_minimal"` to `PRESET_DESCRIPTIONS` and `build_preset_board()`:
     * **Components (14 parts):** `J1` (USB-C Connector, locked), `J2` (GPIO Header, locked), `U1` (RP2040 MCU Core), `U2` (W25Q16 QSPI Flash SOIC-8), `U3` (3.3V LDO SOT-23-5), `Y1` (12MHz SMD Crystal), `SW1` (BOOTSEL Button), `D1` (Power LED), `C1..C4` (Decoupling Capacitors), `R1..R2` (USB/LED Resistors).
     * **Nets (12 nets):** `VBUS_5V`, `VCC_3V3`, `GND`, `USB_DP`, `USB_DM`, `XIN`, `XOUT`, `QSPI_CS`, `QSPI_CLK`, `QSPI_D0`, `QSPI_D1`, `LED_STAT`.
  2. Give crystal nets (`XIN`, `XOUT`) higher spring weight (`weight = 3.0`) so `Y1` locks adjacent to `U1`.

---

## 3. `rev3` Verified Benchmark Scorecard & Acceptance Results

Executed via:

```bash
uv run pytest pcb-world-model/tests -v
uv run python pcb-world-model/scripts/05_run_benchmarks.py \
  --revision "rev3" \
  --notes "Rev 3: Dynamic Power/Signal Net-Class Widths (0.40mm/0.25mm) + Blind GNN Placement + B4 rp2040_minimal" \
  --append-log \
  --json-out pcb-world-model/revisions/rev3/benchmark_report.json
uv run python pcb-world-model/scripts/05_run_benchmarks.py --revision "rev3-blind" --blind
```

### Per-Board Telemetry (`rev3`, Standard Mode)

| Board ID | Routed Nets | DRC Errors | HPWL ($\text{mm}$) | Routed Wirelength ($\text{mm}$) | Via Count | Critic $V(S_t)$ | Rip-Up Iterations | Runtime ($\text{s}$) | Composite Score |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`mcu_sensor_node`** | **8 / 8 (100.0%)** | **0** | $293.49$ | $191.76$ | **5** | $0.5071$ | 1 | $2.60\text{s}$ | **98.89 / 100** |
| **`buck_converter`** | **7 / 7 (100.0%)** | **0** | $224.06$ | $139.10$ | **3** | $0.6048$ | 1 | $1.70\text{s}$ | **99.33 / 100** |
| **`usb_uart_bridge`** | **8 / 8 (100.0%)** | **0** | $254.91$ | $132.55$ | **2** | $0.5614$ | 1 | $1.79\text{s}$ | **99.56 / 100** |
| **`rp2040_minimal`** | **12 / 12 (100.0%)** | **0** | $320.48$ | $252.19$ | **3** | $0.6195$ | 1 | $3.19\text{s}$ | **99.33 / 100** |
| **AGGREGATE / MEAN** | **35 / 35 (100.0%)** | **0** | **$273.24$** | **$178.90$** | **13** | **$0.5732$** | **1.00** | **$9.28\text{s}$** | **99.28 / 100** |

### Per-Board Telemetry (`rev3-blind`, Zero-Hint Mode)

| Board ID | Routed Nets | DRC Errors | HPWL ($\text{mm}$) | Routed Wirelength ($\text{mm}$) | Via Count | Rip-Up Iterations | Runtime ($\text{s}$) | Composite Score |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`mcu_sensor_node`** | **8 / 8 (100.0%)** | **0** | $277.50$ | $205.21$ | **10** | 1 | $2.65\text{s}$ | **97.78 / 100** |
| **`buck_converter`** | **7 / 7 (100.0%)** | **0** | $230.42$ | $149.89$ | **6** | 1 | $1.69\text{s}$ | **98.67 / 100** |
| **`usb_uart_bridge`** | **8 / 8 (100.0%)** | **0** | $273.52$ | $162.66$ | **5** | 1 | $1.90\text{s}$ | **98.89 / 100** |
| **`rp2040_minimal`** | **12 / 12 (100.0%)** | **0** | $364.38$ | $242.74$ | **9** | 1 | $3.18\text{s}$ | **98.00 / 100** |
| **AGGREGATE / MEAN** | **35 / 35 (100.0%)** | **0** | **$286.45$** | **$190.12$** | **30** | **1.00** | **$9.42\text{s}$** | **98.33 / 100** |

### Acceptance Comparison (`rev2` vs `rev3`)

| Metric | `rev2` Verified | `rev3` Target Threshold | `rev3` Verified Result |
| :--- | :---: | :---: | :---: |
| **Active Benchmark Boards** | 3 boards (23 nets) | **4 boards (35 nets, adding `rp2040_minimal`)** | **4 boards (35 / 35 nets routed)** |
| **Power vs. Signal Trace Widths** | Uniform `0.25mm` | **`0.40mm` Power/GND, `0.25mm` Signal (0 DRC)** | **`0.40mm` Power/GND, `0.25mm` Signal (0 DRC)** |
| **Blind (`--blind`) Completion** | Untested (used seed #8) | **100% Routed, 0 DRC Errors without seed hint** | **35 / 35 Routed (100.0%), 0 DRC Errors (`98.33 / 100`)** |
| **Total Suite DRC Violations** | 0 | **0** | **0** |
| **Mean Composite Score** | $99.26 / 100$ | **$\ge 99.00 / 100$ across all 4 boards** | **99.28 / 100** |
