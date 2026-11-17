# Revision 1 (`rev1`) — Architecture & Benchmark Overview

* **Revision Tag:** `rev1` (`v0.1.0-baseline`)
* **Status:** Verified & Frozen (`14/14` unit & integration tests passing)
* **Mean Composite Score:** **98.00 / 100**
* **Benchmark Artifacts:**
  * [`benchmark_report.json`](file:///Users/le/Documents/antigravity/kind-hypatia/pcb-world-model/revisions/rev1/benchmark_report.json)
  * [`boards/output_mcu_sensor_node.kicad_pcb`](file:///Users/le/Documents/antigravity/kind-hypatia/pcb-world-model/revisions/rev1/boards/output_mcu_sensor_node.kicad_pcb)
  * [`boards/output_buck_converter.kicad_pcb`](file:///Users/le/Documents/antigravity/kind-hypatia/pcb-world-model/revisions/rev1/boards/output_buck_converter.kicad_pcb)
  * [`boards/output_usb_uart_bridge.kicad_pcb`](file:///Users/le/Documents/antigravity/kind-hypatia/pcb-world-model/revisions/rev1/boards/output_usb_uart_bridge.kicad_pcb)

---

## 1. Executive Summary of `rev1`

`rev1` establishes the foundational closed-loop **Neuro-Symbolic PCB Engineering World Model**:
1. **S-Expression Parser & Exporter (`env/sexpr_parser.py`, `env/kicad_exporter.py`):** Translates KiCad `.kicad_pcb` files into heterogeneous Netlist Hypergraphs $G = (V_c, V_p, E_{\text{pin}}, E_{\text{net}})$ and serializes routed boards back into valid KiCad 8 S-expressions.
2. **Multi-Layer Spatial State Tensor (`env/board_state.py`):** Encodes the $50\text{mm} \times 50\text{mm}$ 2-layer board at $0.2\text{mm}$ resolution into a 7-channel tensor $S_t \in \mathbb{R}^{7 \times 250 \times 250}$.
3. **4-Layer GATv2 Macro-Placer (`models/placer/`):** Predicts continuous component coordinates $(x, y, \sin\theta, \cos\theta)$ followed by 50-step gradient overlap repulsion (`relaxation.py`) and $0.5\text{mm}$ grid snapping.
4. **Dual-Head World Model U-Net (`models/world/dynamics_unet.py`):** Predicts future layer-wise routing congestion $\hat{M}_{t+1} \in [0, 1]^{2 \times 250 \times 250}$ and scalar routability critic $V(S_t) \in [0, 1]$ with 2D bottleneck self-attention.
5. **Octilinear A* Router & Closed-Loop DRC Verifier (`models/world/neural_router.py`, `verifier/`):** Routes traces across `F.Cu` and `B.Cu` and uses deterministic 2D polygon clearance + headless `kicad-cli pcb drc` feedback to rip up and reroute violating nets.

---

## 2. `rev1` Benchmark Scorecard

Executed via `uv run python pcb-world-model/scripts/05_run_benchmarks.py --revision rev1`:

| Board ID | Routed Nets | DRC Errors | HPWL ($\text{mm}$) | Routed Wirelength ($\text{mm}$) | Via Count | Rip-Up Iterations | Runtime ($\text{s}$) | Composite Score |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`usb_uart_bridge`** | **8 / 8 (100%)** | **0** | $257.41$ | $139.53$ | **0** | 1 | $1.94\text{s}$ | **100.00 / 100** |
| **`buck_converter`** | **7 / 7 (100%)** | **0** | $225.06$ | $143.06$ | **5** | 1 | $1.74\text{s}$ | **98.89 / 100** |
| **`mcu_sensor_node`** | **8 / 8 (100%)** | **0** | $314.73$ | $278.63$ | **22** | 2 | $3.38\text{s}$ | **95.11 / 100** |
| **AGGREGATE / MEAN** | **23 / 23 (100%)** | **0** | **$265.73$** | **$187.07$** | **27** | **1.33** | **$7.06\text{s}$** | **98.00 / 100** |

---

## 3. Root-Cause Diagnosis: Why `rev1` Needed 22 Vias & 2 Iterations on `mcu_sensor_node`

An inspection of `rev1`'s behavior on `mcu_sensor_node` reveals three specific physical/algorithmic bottlenecks targeted for **Revision 2 (`rev2`)**:

1. **Coarse Integer Box Dilation Around Fine-Pitch SMD Pads (`neural_router.py`):**
   * In `rev1`, `_build_pad_masks_for_net()` rasterized foreign pads using `math.floor()` and `math.ceil()`, inflating a $0.60\text{mm}$ SMD pad to $1.0\text{mm}$ (5 grid cells), and then applied a square `F.max_pool2d` dilation of `trace_clear_cells = 2` ($+0.4\text{mm}$ per side).
   * On `U1` (`SOIC-8` at $1.27\text{mm}$ pin pitch) and `U2` (`SOT-23-6`), this integer box inflation sealed off the planar `F.Cu` escape corridors around SMD pins, forcing the router to immediately drop vias to `B.Cu` to escape almost every pin (**22 vias total**).
2. **Sparse Steiner Attachment Sampling on Multi-Pin Nets (`neural_router.py`):**
   * When routing multi-pin nets (`VCC_3V3`, `GND`, `I2C_SDA`, `I2C_SCL`), `rev1` only sampled `path[::step]` (16 points) as valid Steiner attachment nodes and did not update `occ_layers` or `own_pads_mask` with newly routed trace copper between MST branches of the same net.
3. **Single-Component Orientation Search Without 2-Opt Component Swapping (`relaxation.py`):**
   * `rev1` only tested rotating one component at a time while holding all other positions fixed, missing beneficial 2-component position swaps and joint orientation improvements that reduce net crossings and HPWL.
4. **Uniform Trace Width Across Power & Signal Nets:**
   * `rev1` routed all nets at a uniform $0.25\text{mm}$ width instead of automatically widening high-current power/ground nets (`is_power_ground=True`, e.g., `VIN`, `VCC`, `GND`) while keeping signal traces compact.
