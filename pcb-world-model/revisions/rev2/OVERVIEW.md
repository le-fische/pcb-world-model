# Revision 2 (`rev2`) — Architecture & Benchmark Overview

* **Revision Tag:** `rev2`
* **Status:** Verified (`16/16` unit & integration tests passing)
* **Prerequisite:** `rev1` baseline verified (`98.00 / 100` composite score, `27` total vias)
* **Mean Composite Score:** **99.26 / 100** *(target $\ge 99.20 / 100$, reducing total vias from $27 \to 10$ and solving all boards in **1 iteration** with **0 rip-ups**)*
* **Benchmark Artifacts:**
  * [`benchmark_report.json`](file:///Users/le/Documents/antigravity/kind-hypatia/pcb-world-model/revisions/rev2/benchmark_report.json)
  * [`boards/output_mcu_sensor_node.kicad_pcb`](file:///Users/le/Documents/antigravity/kind-hypatia/pcb-world-model/revisions/rev2/boards/output_mcu_sensor_node.kicad_pcb)
  * [`boards/output_buck_converter.kicad_pcb`](file:///Users/le/Documents/antigravity/kind-hypatia/pcb-world-model/revisions/rev2/boards/output_buck_converter.kicad_pcb)
  * [`boards/output_usb_uart_bridge.kicad_pcb`](file:///Users/le/Documents/antigravity/kind-hypatia/pcb-world-model/revisions/rev2/boards/output_usb_uart_bridge.kicad_pcb)

---

## 1. Objective of `rev2`

In `rev1`, `mcu_sensor_node` required **22 vias** and **2 rip-up iterations** (`95.11 / 100`) because coarse integer box dilation (`math.floor`/`math.ceil` + `max_pool2d`) artificially blocked the $0.67\text{mm}$ planar channels between $1.27\text{mm}$-pitch `SOIC-8` and `SOT-23` pads on `F.Cu`.

**`rev2` upgrades the geometry engine and placement optimizer** so the World Model perceives true sub-grid millimeter clearances and optimizes component pin orientations jointly.

---

## 2. What to Build in `rev2` (Module-by-Module Checklist)

### A. Sub-Grid Euclidean Clearance Field (`models/world/neural_router.py`)
* **Problem in `rev1`:** `_build_pad_masks_for_net()` rounds pad edges outward to grid indices and dilates them with a square $5 \times 5$ kernel (`trace_clear_cells = 2`), inflating every foreign SMD pad by $+0.60\text{mm}$ to $+0.80\text{mm}$.
* **What to implement in `rev2`:**
  1. Replace `_build_pad_masks_for_net()` with `_build_exact_clearance_masks(state, net_id, trace_width_mm)`.
  2. For every foreign pad rectangle $[x_0, x_1] \times [y_0, y_1]$, foreign trace segment $S$, and foreign via $V$, compute the exact Euclidean distance from grid cell centers $(g_x \cdot \text{res}, g_y \cdot \text{res})$:
     $$d_{\text{pad}}(x, y) = \sqrt{\max(0, x_0 - x, x - x_1)^2 + \max(0, y_0 - y, y - y_1)^2}$$
  3. Block cell $(g_x, g_y)$ for traces iff $d < c_{\text{min}} + \frac{w_{\text{trace}}}{2} + 0.36 \cdot \text{res}$, and block for vias iff $d < c_{\text{min}} + \frac{d_{\text{via}}}{2} + 0.025\text{mm}$.
  4. Explicitly block **via-in-pad** on all SMD pads (`pad.layer_mask != 3`).

### B. Full Steiner-Tree Copper Reuse & Bend Penalty (`models/world/neural_router.py`)
* **Problem in `rev1`:** Multi-pin nets (`VCC_3V3`, `GND`, `I2C_SDA`) only sample `path[::step]` (16 points) as attachment targets for subsequent MST branches, and A* has zero turn penalty (producing staircase zig-zags).
* **What to implement in `rev2`:**
  1. Append **every cell** `(pl, px_c, py_c)` of a newly routed segment into `connected_cells` and update `occ_layers[pl, py_c, px_c] = net_id` immediately so the next pin on the same net can form a true T-junction Steiner tap anywhere along existing copper.
  2. Add a planar direction-change penalty `bend_pen = 0.06` inside `_astar_2pin()` whenever `(dx, dy) != (prev_dx, prev_dy)`.

### C. 2-Opt Component Swap & Pin-Aware Orientation Search (`models/placer/relaxation.py`)
* **Problem in `rev1`:** Discrete orientation search only evaluates `compute_hpwl` one component at a time without testing 2-component position swaps or pad-to-pad MST wirelength.
* **What to implement in `rev2`:**
  1. Define a combined wire objective:
     $$J(\text{placement}) = 0.5 \cdot \text{HPWL} + 0.5 \cdot \text{MST\_Wirelength}$$
  2. Add a 2-pass **2-Opt Swap Loop** over unlocked components of compatible footprint dimensions ($|\Delta w| \le 2.5\text{mm}, |\Delta h| \le 2.5\text{mm}$), accepting swaps that reduce $J$ while keeping `0` courtyard overlaps.

---

## 3. `rev2` Benchmark Scorecard & Acceptance Criteria

Executed via:

```bash
uv run pytest pcb-world-model/tests -v
uv run python pcb-world-model/scripts/05_run_benchmarks.py \
  --revision "rev2" \
  --notes "Rev 2: Sub-Grid Euclidean Clearance + Full Steiner Reuse + 2-Opt Swap" \
  --append-log \
  --json-out pcb-world-model/revisions/rev2/benchmark_report.json
```

### Per-Board Telemetry (`rev2`)

| Board ID | Routed Nets | DRC Errors | HPWL ($\text{mm}$) | Routed Wirelength ($\text{mm}$) | Via Count | Critic $V(S_t)$ | Rip-Up Iterations | Runtime ($\text{s}$) | Composite Score |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`mcu_sensor_node`** | **8 / 8 (100.0%)** | **0** | $293.49$ | $190.61$ | **5** | $0.5071$ | 1 | $2.73\text{s}$ | **98.89 / 100** |
| **`buck_converter`** | **7 / 7 (100.0%)** | **0** | $224.06$ | $138.99$ | **3** | $0.6048$ | 1 | $1.75\text{s}$ | **99.33 / 100** |
| **`usb_uart_bridge`** | **8 / 8 (100.0%)** | **0** | $254.91$ | $132.55$ | **2** | $0.5614$ | 1 | $1.88\text{s}$ | **99.56 / 100** |
| **AGGREGATE / MEAN** | **23 / 23 (100.0%)** | **0** | **$257.49$** | **$154.05$** | **10** | **$0.5578$** | **1.00** | **$6.37\text{s}$** | **99.26 / 100** |

### Acceptance Comparison (`rev1` vs `rev2`)

| Metric | `rev1` Baseline | `rev2` Target Threshold | `rev2` Verified Result |
| :--- | :---: | :---: | :---: |
| **Total Routed Nets** | 23 / 23 (100%) | **23 / 23 (100%)** | **23 / 23 (100.0%)** |
| **Total DRC Errors** | 0 | **0** | **0** |
| **`mcu_sensor_node` Vias** | 22 vias (2 iters) | **$\le 6$ vias (1 iter)** | **5 vias (1 iter)** |
| **Total Suite Vias** | 27 vias | **$\le 12$ vias** | **10 vias** |
| **Mean Routed Wirelength** | $187.07\text{mm}$ | **$\le 165.00\text{mm}$** | **$154.05\text{mm}$** |
| **Mean Composite Score** | $98.00 / 100$ | **$\ge 99.20 / 100$** | **99.26 / 100** |
