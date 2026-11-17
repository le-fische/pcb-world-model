# Revision 5 (`rev5`) — Production Manufacturing Bundle, Copper Ground Pours & Live Prompt UI

* **Revision Tag:** `rev5`
* **Status:** Verified (`31/31` unit & integration tests passing)
* **Prerequisite:** `rev4` verified (**99.16 / 100** across 5 boards / 44 nets, **98.53 / 100** in `--blind` mode, `27/27` tests passing)
* **Mean Composite Score:** **99.56 / 100** across all 5 boards (**44/44** nets routed, **0** DRC errors, **10** total vias, **10.89s** total runtime; **99.02 / 100** in `--blind` zero-hint mode with **44/44** routed, **22** total vias, and **0** DRC errors) + **End-to-End Prompt Manufacturing Bundle (`prompt_rp2040`: 11/11 nets, 0 ERC/DRC errors, `bom.csv`, `positions.csv`)**
* **Benchmark Artifacts:**
  * [`benchmark_report.json`](file:///Users/le/Documents/antigravity/kind-hypatia/pcb-world-model/revisions/rev5/benchmark_report.json)
  * [`benchmark_report_blind.json`](file:///Users/le/Documents/antigravity/kind-hypatia/pcb-world-model/revisions/rev5/benchmark_report_blind.json)
  * [`fab_bundle/prompt_rp2040.kicad_pcb`](file:///Users/le/Documents/antigravity/kind-hypatia/pcb-world-model/revisions/rev5/fab_bundle/prompt_rp2040.kicad_pcb)
  * [`fab_bundle/bom.csv`](file:///Users/le/Documents/antigravity/kind-hypatia/pcb-world-model/revisions/rev5/fab_bundle/bom.csv)
  * [`fab_bundle/positions.csv`](file:///Users/le/Documents/antigravity/kind-hypatia/pcb-world-model/revisions/rev5/fab_bundle/positions.csv)
  * [`boards/output_mcu_sensor_node.kicad_pcb`](file:///Users/le/Documents/antigravity/kind-hypatia/pcb-world-model/revisions/rev5/boards/output_mcu_sensor_node.kicad_pcb)
  * [`boards/output_buck_converter.kicad_pcb`](file:///Users/le/Documents/antigravity/kind-hypatia/pcb-world-model/revisions/rev5/boards/output_buck_converter.kicad_pcb)
  * [`boards/output_usb_uart_bridge.kicad_pcb`](file:///Users/le/Documents/antigravity/kind-hypatia/pcb-world-model/revisions/rev5/boards/output_usb_uart_bridge.kicad_pcb)
  * [`boards/output_rp2040_minimal.kicad_pcb`](file:///Users/le/Documents/antigravity/kind-hypatia/pcb-world-model/revisions/rev5/boards/output_rp2040_minimal.kicad_pcb)
  * [`boards/output_macropad_matrix.kicad_pcb`](file:///Users/le/Documents/antigravity/kind-hypatia/pcb-world-model/revisions/rev5/boards/output_macropad_matrix.kicad_pcb)
  * [`boards/output_prompt_synthesized.kicad_pcb`](file:///Users/le/Documents/antigravity/kind-hypatia/pcb-world-model/revisions/rev5/boards/output_prompt_synthesized.kicad_pcb)

---

## 1. Objective of `rev5`

With `rev4` complete, our World Model can take a natural-language hardware brief (or any of the 5 reference boards), synthesize parametric IPC-7351 footprints, validate ERC, place components via GATv2 + 2-opt swap, and route 100% of nets with 0 DRC violations.

To make this an **undeniable, founder-ready showcase** that directly outshines OpenAI's GPT-6 Astra KiCad launch demo:
1. **Copper Ground Pour Zones (`env/kicad_exporter.py`):** In the GPT-6 Astra demo, one visual highlight was pouring a copper `GND` zone on `B.Cu`. Adding native KiCad 8 `(zone (net 3) (net_name "GND") (layer "B.Cu") ...)` generation to `kicad_exporter.py` gives every exported board a true low-impedance ground plane.
2. **Production Fabrication Bundle (`env/manufacturing_exporter.py`):** Real hardware engineers don't just stop at a layout—they send **3 files** to JLCPCB or PCBWay for assembly (PCBA):
   * The routed `.kicad_pcb` with copper ground pour.
   * **`bom.csv` (Bill of Materials):** Designator, Footprint, Value, Quantity.
   * **`positions.csv` (Centroid Pick-and-Place / CPL):** Designator, Mid X ($\text{mm}$), Mid Y ($\text{mm}$), Layer (`Top`), Rotation ($\theta^\circ$).
3. **Interactive Natural-Language Prompt Bar in the 2D/3D Web Console (`ui/web/`):** Allow the founder to type *any* hardware prompt (e.g., *"STM32 drone sensor hub with I2C barometer, LDO, and 3 LEDs"*) directly into the browser UI and watch the World Model compose the schematic, synthesize footprints, place components, and route traces live in 2D and 3D!

---

## 2. What to Build in `rev5` (Module-by-Module Checklist)

### A. KiCad 8 Copper Ground Zone Pour (`env/kicad_exporter.py`)
* **File to edit:** [`pcb-world-model/env/kicad_exporter.py`](file:///Users/le/Documents/antigravity/kind-hypatia/pcb-world-model/env/kicad_exporter.py)
* **What to implement:**
  1. Locate the `GND` net ID in `board_state.netlist.nets`.
  2. Emit a valid KiCad 8 `(zone ...)` S-expression block on `"B.Cu"` bounded by `[margin, W - margin] × [margin, H - margin]` with thermal relief and `clearance_mm` isolation so opening the `.kicad_pcb` in KiCad shows a complete ground plane pour.

### B. Turnkey PCBA Manufacturing Bundle Exporter (`env/manufacturing_exporter.py`)
* **File to create:** `pcb-world-model/env/manufacturing_exporter.py`
* **What to implement:**
  1. `generate_bom_csv(board_state: BoardState) -> str`: Groups components by `(value, footprint, comp_type)` and outputs standard JLCPCB/PCBWay `Comment,Designator,Footprint,Quantity` CSV rows.
  2. `generate_cpl_csv(board_state: BoardState) -> str`: Outputs `Designator,Val,Package,Mid X,Mid Y,Rotation,Layer` with exact millimeter coordinates and discrete rotation $\theta \in \{0, 90, 180, 270\}$.
  3. Expose `/api/manufacturing_bundle` in [`ui/server.py`](file:///Users/le/Documents/antigravity/kind-hypatia/pcb-world-model/ui/server.py) and `--export-fab-dir` in [`scripts/04_run_inference.py`](file:///Users/le/Documents/antigravity/kind-hypatia/pcb-world-model/scripts/04_run_inference.py).

### C. Live Prompt-to-PCB Synthesis in Web UI (`ui/server.py`, `ui/web/index.html`, `ui/web/app.js`)
* **Files to edit:**
  * [`pcb-world-model/ui/server.py`](file:///Users/le/Documents/antigravity/kind-hypatia/pcb-world-model/ui/server.py)
  * [`pcb-world-model/ui/web/index.html`](file:///Users/le/Documents/antigravity/kind-hypatia/pcb-world-model/ui/web/index.html)
  * [`pcb-world-model/ui/web/app.js`](file:///Users/le/Documents/antigravity/kind-hypatia/pcb-world-model/ui/web/app.js)
* **What to implement:**
  1. Allow `/ws/rollout` and `/api/run` in `ui/server.py` to accept an optional `prompt: str` parameter: when provided, call `compose_netlist_from_prompt(prompt)` to synthesize the `NetlistGraph` on the fly and stream its placement + routing rollout over WebSocket!
  2. Add a prompt input box in `ui/web/index.html` and export buttons for `Download BOM.csv` and `Download CPL.csv`.

---

## 3. Verification Commands & `rev5` Scorecard

```bash
# 1. Run full unit & integration test suite (31 tests)
uv run pytest pcb-world-model/tests -v

# 2. Generate a full PCBA Manufacturing Bundle from a natural-language prompt
uv run python pcb-world-model/scripts/04_run_inference.py \
  --prompt "RP2040 sensor controller with I2C IMU, 3.3V LDO, 2 LEDs, and USB-C" \
  --output pcb-world-model/revisions/rev5/boards/output_prompt_synthesized.kicad_pcb \
  --export-fab-dir pcb-world-model/revisions/rev5/fab_bundle

# 3. Run 5-board benchmark suite & log rev5
uv run python pcb-world-model/scripts/05_run_benchmarks.py \
  --revision "rev5" \
  --notes "Rev 5: Copper Ground Pour Zones + JLCPCB BOM/CPL Manufacturing Bundle + Live Prompt Web UI + B.Cu Plane Reservation" \
  --append-log \
  --json-out pcb-world-model/revisions/rev5/benchmark_report.json
```

### Per-Board Telemetry (`rev5`, Standard Mode)

| Board ID | Routed Nets | DRC Errors | HPWL ($\text{mm}$) | Routed Wirelength ($\text{mm}$) | Via Count | Critic $V(S_t)$ | Rip-Up Iterations | Runtime ($\text{s}$) | Composite Score |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`mcu_sensor_node`** | **8 / 8 (100.0%)** | **0** | $293.49$ | $198.32$ | **4** | $0.5071$ | 1 | $2.53\text{s}$ | **99.11 / 100** |
| **`buck_converter`** | **7 / 7 (100.0%)** | **0** | $224.06$ | $143.79$ | **2** | $0.6048$ | 1 | $1.66\text{s}$ | **99.56 / 100** |
| **`usb_uart_bridge`** | **8 / 8 (100.0%)** | **0** | $254.91$ | $139.12$ | **0** | $0.5614$ | 1 | $1.82\text{s}$ | **100.00 / 100** |
| **`rp2040_minimal`** | **12 / 12 (100.0%)** | **0** | $320.48$ | $256.64$ | **2** | $0.6195$ | 1 | $3.06\text{s}$ | **99.56 / 100** |
| **`macropad_matrix`** | **9 / 9 (100.0%)** | **0** | $263.69$ | $225.48$ | **2** | $0.5904$ | 1 | $1.81\text{s}$ | **99.56 / 100** |
| **AGGREGATE / MEAN** | **44 / 44 (100.0%)** | **0** | **$271.33$** | **$192.67$** | **10** | **$0.5766$** | **1.00** | **$10.89\text{s}$** | **99.56 / 100** |

### Per-Board Telemetry (`rev5-blind`, Zero-Hint Mode)

| Board ID | Routed Nets | DRC Errors | HPWL ($\text{mm}$) | Routed Wirelength ($\text{mm}$) | Via Count | Rip-Up Iterations | Runtime ($\text{s}$) | Composite Score |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`mcu_sensor_node`** | **8 / 8 (100.0%)** | **0** | $277.50$ | $211.52$ | **7** | 1 | $2.70\text{s}$ | **98.44 / 100** |
| **`buck_converter`** | **7 / 7 (100.0%)** | **0** | $230.42$ | $158.43$ | **6** | 1 | $1.72\text{s}$ | **98.67 / 100** |
| **`usb_uart_bridge`** | **8 / 8 (100.0%)** | **0** | $273.52$ | $167.46$ | **4** | 1 | $1.92\text{s}$ | **99.11 / 100** |
| **`rp2040_minimal`** | **12 / 12 (100.0%)** | **0** | $364.38$ | $259.31$ | **3** | 1 | $3.14\text{s}$ | **99.33 / 100** |
| **`macropad_matrix`** | **9 / 9 (100.0%)** | **0** | $273.22$ | $166.38$ | **2** | 1 | $1.80\text{s}$ | **99.56 / 100** |
| **AGGREGATE / MEAN** | **44 / 44 (100.0%)** | **0** | **$283.81$** | **$192.62$** | **22** | **1.00** | **$11.28\text{s}$** | **99.02 / 100** |
