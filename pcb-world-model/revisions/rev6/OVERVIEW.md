# Revision 6 (`rev6`) — C++20 Compiled A* Kernel, 4-Layer HDI Stackup & Signal-Integrity Telemetry

* **Revision Tag:** `rev6`
* **Status:** Verified (`37/37` unit & integration tests passing)
* **Prerequisite:** `rev5` verified (**99.56 / 100** across `B1–B5`, **99.02 / 100** in `--blind`, `31/31` tests)
* **2-Layer `B1–B5` Benchmark (Seeded):** **99.56 / 100** mean composite score (**44/44 (100.0%)** nets routed, **0** DRC errors, **10** vias, **$1.73\text{s}$ total runtime** / $0.25\text{s}$–$0.55\text{s}$ per board — **6.3× faster** than `rev5`'s $10.89\text{s}$)
* **2-Layer `B1–B5` Benchmark (`rev6-blind` Zero-Hint):** **99.29 / 100** mean composite score (**44/44 (100.0%)** nets routed, **0** DRC errors, **16** total vias, **$1.81\text{s}$ total runtime**)
* **4-Layer 50-Component Showcase (`flight_controller_50`):** **43/43 (100.0%)** nets routed, **0** DRC errors, **1** rollout iteration, **$3.88\text{s}$ total runtime**, `105` vias (`68` blind micro-vias `0->1`/`0->2` + `37` through-vias), `MaxRdc = 127.795 mOhm`, `SW_NODE` $0$ vias ($5.559\text{ nH}$, $28.182\text{ m}\Omega$, $1.67\text{ A}$ IPC-2221 ampacity), **`SI PASS`**
* **Benchmark & Fabrication Artifacts:**
  * [`benchmark_report.json`](file:///Users/le/Documents/antigravity/kind-hypatia/pcb-world-model/revisions/rev6/benchmark_report.json)
  * [`benchmark_report_blind.json`](file:///Users/le/Documents/antigravity/kind-hypatia/pcb-world-model/revisions/rev6/benchmark_report_blind.json)
  * [`fc50_4layer_report.json`](file:///Users/le/Documents/antigravity/kind-hypatia/pcb-world-model/revisions/rev6/fc50_4layer_report.json)
  * [`boards/output_flight_controller_50_4L.kicad_pcb`](file:///Users/le/Documents/antigravity/kind-hypatia/pcb-world-model/revisions/rev6/boards/output_flight_controller_50_4L.kicad_pcb)
  * [`fab_bundle_fc50/flight_controller_50.kicad_pcb`](file:///Users/le/Documents/antigravity/kind-hypatia/pcb-world-model/revisions/rev6/fab_bundle_fc50/flight_controller_50.kicad_pcb)
  * [`fab_bundle_fc50/bom.csv`](file:///Users/le/Documents/antigravity/kind-hypatia/pcb-world-model/revisions/rev6/fab_bundle_fc50/bom.csv)
  * [`fab_bundle_fc50/positions.csv`](file:///Users/le/Documents/antigravity/kind-hypatia/pcb-world-model/revisions/rev6/fab_bundle_fc50/positions.csv)
  * [`boards/output_mcu_sensor_node.kicad_pcb`](file:///Users/le/Documents/antigravity/kind-hypatia/pcb-world-model/revisions/rev6/boards/output_mcu_sensor_node.kicad_pcb)
  * [`boards/output_buck_converter.kicad_pcb`](file:///Users/le/Documents/antigravity/kind-hypatia/pcb-world-model/revisions/rev6/boards/output_buck_converter.kicad_pcb)
  * [`boards/output_usb_uart_bridge.kicad_pcb`](file:///Users/le/Documents/antigravity/kind-hypatia/pcb-world-model/revisions/rev6/boards/output_usb_uart_bridge.kicad_pcb)
  * [`boards/output_rp2040_minimal.kicad_pcb`](file:///Users/le/Documents/antigravity/kind-hypatia/pcb-world-model/revisions/rev6/boards/output_rp2040_minimal.kicad_pcb)
  * [`boards/output_macropad_matrix.kicad_pcb`](file:///Users/le/Documents/antigravity/kind-hypatia/pcb-world-model/revisions/rev6/boards/output_macropad_matrix.kicad_pcb)
  * [`boards/output_prompt_synthesized.kicad_pcb`](file:///Users/le/Documents/antigravity/kind-hypatia/pcb-world-model/revisions/rev6/boards/output_prompt_synthesized.kicad_pcb)

---

## 1. Objective of `rev6`

`rev6` upgrades the World Model from a pure-Python 2-layer research engine into a hardware-accelerated **2L/4L multi-layer PCB workstation** capable of routing dense 50-component boards in a single closed-loop pass while auditing electrical signal-integrity and thermal margins:

1. **Compiled C++20 Multi-Layer Octilinear A* Kernel & MPS Inference (`models/world/native_astar.cpp`, `models/world/native_kernel.py`):**
   * Auto-compiles `libpcb_astar` (`-O3 -std=c++20`, ABI v2) with zero-copy NumPy `ctypes` bindings, thread-local workspace reuse, arbitrary layer counts (`L`), per-pair `transition_costs`, layer-span via blocking, and deterministic Python fallback.
   * Vectorizes placement relaxation via analytical NumPy Adam (`models/placer/relaxation.py`), pure-Python Kruskal MST (`env/metrics.py`), pad-distance caching (`models/world/neural_router.py`), and Apple Metal `mps` / CUDA U-Net evaluation (`scripts/04_run_inference.py`, `models/world/dynamics_unet.py`).
2. **Electrical Signal-Integrity & Thermal Telemetry (`env/metrics.py`, `ui/server.py`):**
   * Computes 1 oz ($0.035\text{ mm}$) trace $R_{\text{DC}} = \rho \frac{L}{W \cdot t_{\text{Cu}}}$ ($\rho = 1.724 \times 10^{-5}\ \Omega\cdot\text{mm}$), microstrip $L_{\text{loop}}$ and $Z_0$, via barrel inductance/resistance, and IPC-2221 trace/via current capacity ($\Delta T = 20^\circ\text{C}$), surfacing `PASS`/`WARN` per net via `GET /api/signal-integrity` and `BoardState.to_summary_dict()`.
   * Prioritizes `SW_NODE` / `SWNODE` switching nets in `rank_nets_by_criticality()` and widens switching traces to `power_trace_width_mm` ($0.40\text{ mm}$), dropping `buck_converter` routed wirelength ($143.79\text{ mm} \to 140.02\text{ mm}$) and keeping `SW_NODE` via-free on both `buck_converter` and `flight_controller_50`.
3. **4-Layer Internal Power/GND Planes & 50-Component Board (`flight_controller_50`):**
   * Supports `[F.Cu, In1.GND, In2.PWR, B.Cu]` stackups across `BoardConfig` (`configs/default_4layer.yaml`), `BoardState`, `kicad_exporter.py` (`(uvias_allowed yes)`, `(via micro ...)`, multi-plane `(zone ...)`), `sexpr_parser.py`, `fast_clearance.py` (span-aware micro-via clearance), and `neural_router.py` (direct `F.Cu -> In1.GND`/`In2.PWR` micro-via drops).
   * Adds `flight_controller_50` ($80\times 80\text{ mm}$, $50$ components, $43$ nets) under `COMPLEX_PRESET_DESCRIPTIONS` and exposes interactive 4-layer 2D/3D rendering + SI telemetry in the REV 6 FastAPI web console.

---

## 2. Verification Commands & `rev6` Scorecard

```bash
# 1. Run full unit & integration test suite (37 tests)
uv run pytest pcb-world-model/tests -v

# 2. Run 5-board benchmark suite (Standard & Blind modes)
uv run python pcb-world-model/scripts/05_run_benchmarks.py --revision "rev6" \
  --json-out pcb-world-model/revisions/rev6/benchmark_report.json
uv run python pcb-world-model/scripts/05_run_benchmarks.py --revision "rev6-blind" --blind \
  --json-out pcb-world-model/revisions/rev6/benchmark_report_blind.json

# 3. Route 50-component 4-layer flight controller + export JLCPCB bundle
uv run python pcb-world-model/scripts/04_run_inference.py \
  --board flight_controller_50 --layers 4 \
  --output pcb-world-model/revisions/rev6/boards/output_flight_controller_50_4L.kicad_pcb \
  --export-fab-dir pcb-world-model/revisions/rev6/fab_bundle_fc50
```

### Per-Board Telemetry (`rev6`, Standard 2L Mode)

| Board ID | Routed Nets | DRC Errors | HPWL ($\text{mm}$) | Routed Wirelength ($\text{mm}$) | Via Count | Critic $V(S_t)$ | Rip-Up Iterations | Runtime ($\text{s}$) | Composite Score |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`mcu_sensor_node`** | **8 / 8 (100.0%)** | **0** | $293.49$ | $198.32$ | **4** | $0.5071$ | 1 | $0.38\text{s}$ | **99.11 / 100** |
| **`buck_converter`** | **7 / 7 (100.0%)** | **0** | $235.98$ | $140.02$ | **2** | $0.5880$ | 1 | $0.25\text{s}$ | **99.56 / 100** |
| **`usb_uart_bridge`** | **8 / 8 (100.0%)** | **0** | $254.91$ | $139.12$ | **0** | $0.5614$ | 1 | $0.28\text{s}$ | **100.00 / 100** |
| **`rp2040_minimal`** | **12 / 12 (100.0%)** | **0** | $320.48$ | $255.51$ | **2** | $0.6195$ | 1 | $0.55\text{s}$ | **99.56 / 100** |
| **`macropad_matrix`** | **9 / 9 (100.0%)** | **0** | $263.69$ | $225.48$ | **2** | $0.5904$ | 1 | $0.26\text{s}$ | **99.56 / 100** |
| **AGGREGATE / MEAN** | **44 / 44 (100.0%)** | **0** | **$273.71$** | **$191.69$** | **10** | **$0.5733$** | **1.00** | **$1.73\text{s}$** | **99.56 / 100** |

### Per-Board Telemetry (`rev6-blind`, Zero-Hint Mode)

| Board ID | Routed Nets | DRC Errors | HPWL ($\text{mm}$) | Routed Wirelength ($\text{mm}$) | Via Count | Critic $V(S_t)$ | Rip-Up Iterations | Runtime ($\text{s}$) | Composite Score |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`mcu_sensor_node`** | **8 / 8 (100.0%)** | **0** | $277.50$ | $211.52$ | **7** | $0.5295$ | 1 | $0.39\text{s}$ | **98.44 / 100** |
| **`buck_converter`** | **7 / 7 (100.0%)** | **0** | $232.92$ | $158.97$ | **2** | $0.5923$ | 1 | $0.28\text{s}$ | **99.56 / 100** |
| **`usb_uart_bridge`** | **8 / 8 (100.0%)** | **0** | $273.52$ | $167.31$ | **4** | $0.5352$ | 1 | $0.30\text{s}$ | **99.11 / 100** |
| **`rp2040_minimal`** | **12 / 12 (100.0%)** | **0** | $364.38$ | $266.72$ | **1** | $0.5784$ | 1 | $0.56\text{s}$ | **99.78 / 100** |
| **`macropad_matrix`** | **9 / 9 (100.0%)** | **0** | $273.22$ | $166.54$ | **2** | $0.5785$ | 1 | $0.28\text{s}$ | **99.56 / 100** |
| **AGGREGATE / MEAN** | **44 / 44 (100.0%)** | **0** | **$284.31$** | **$194.21$** | **16** | **$0.5628$** | **1.00** | **$1.81\text{s}$** | **99.29 / 100** |

### Complex 4-Layer Benchmark (`flight_controller_50`, 50 Components / $80\times 80\text{ mm}$)

| Metric | Value | Notes |
| :--- | :---: | :--- |
| **Stackup** | **4-Layer HDI** | `F.Cu`, `In1.GND`, `In2.PWR` (`VCC_3V3`), `B.Cu` (`0.20mm` prepreg) |
| **Routed Nets / DRC** | **43 / 43 (100.0%) / 0 DRC** | Single rollout iteration (`1` pass) via `cpp20_native` kernel |
| **HPWL / Wirelength** | $1513.07\text{ mm}$ / $1071.34\text{ mm}$ | `0.25mm` signal / `0.40mm` power & `SW_NODE` |
| **Total Vias** | **105** (`68` micro + `37` through) | Direct `0->1` / `0->2` blind micro-via plane drops |
| **Signal Integrity** | **`SI PASS`** | Max $R_{\text{DC}} = 127.795\text{ m}\Omega$, `SW_NODE` = $0$ vias ($5.559\text{ nH}$, $1.67\text{ A}$ $\ge 1.5\text{ A}$) |
| **End-to-End Runtime** | **$3.88\text{s}$** | Analytical NumPy relaxation + Metal MPS U-Net + C++20 A* |
