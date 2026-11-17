# Revision 4 (`rev4`) — Universal Generative Hardware World Model Architecture & Verified Report

* **Revision Tag:** `rev4`
* **Status:** Verified (`27/27` unit & integration tests passing)
* **Prerequisite:** `rev3` verified (**99.28 / 100** across 4 boards / 35 nets with `0.40mm` power / `0.25mm` signal widths, and **98.33 / 100** in `--blind` zero-hint placement mode with `22/22` tests passing)
* **Mean Composite Score:** **99.16 / 100** across all 5 boards (**44/44** nets routed, **0** DRC errors, **19** total vias, **10.67s** total runtime; **98.53 / 100** in `--blind` zero-hint mode with **44/44** routed and **0** DRC errors) + **End-to-End Prompt Synthesis (`11/11` nets, `0` ERC/DRC errors)**
* **Benchmark Artifacts:**
  * [`benchmark_report.json`](file:///Users/le/Documents/antigravity/kind-hypatia/pcb-world-model/revisions/rev4/benchmark_report.json)
  * [`boards/output_mcu_sensor_node.kicad_pcb`](file:///Users/le/Documents/antigravity/kind-hypatia/pcb-world-model/revisions/rev4/boards/output_mcu_sensor_node.kicad_pcb)
  * [`boards/output_buck_converter.kicad_pcb`](file:///Users/le/Documents/antigravity/kind-hypatia/pcb-world-model/revisions/rev4/boards/output_buck_converter.kicad_pcb)
  * [`boards/output_usb_uart_bridge.kicad_pcb`](file:///Users/le/Documents/antigravity/kind-hypatia/pcb-world-model/revisions/rev4/boards/output_usb_uart_bridge.kicad_pcb)
  * [`boards/output_rp2040_minimal.kicad_pcb`](file:///Users/le/Documents/antigravity/kind-hypatia/pcb-world-model/revisions/rev4/boards/output_rp2040_minimal.kicad_pcb)
  * [`boards/output_macropad_matrix.kicad_pcb`](file:///Users/le/Documents/antigravity/kind-hypatia/pcb-world-model/revisions/rev4/boards/output_macropad_matrix.kicad_pcb)
  * [`boards/output_prompt_synthesized.kicad_pcb`](file:///Users/le/Documents/antigravity/kind-hypatia/pcb-world-model/revisions/rev4/boards/output_prompt_synthesized.kicad_pcb)

---

## 1. Objective of `rev4` ("Build Anything")

With `rev1` $\to$ `rev3` complete, the physical layout engine (GATv2 Macro-Placer + 2-Opt Spectral Relaxation + Dual-Head Congestion U-Net + Sub-Grid Euclidean Octilinear Router + Closed-Loop DRC Verifier) is proven across power regulators, USB bridges, sensor nodes, and 14-part MCU boards—even in `--blind` zero-hint mode.

**`rev4` unlocks the final layer that turns this into a true "Build Anything" Engineering World Model:**
1. **Parametric IPC-7351 Footprint Synthesizer (`env/footprint_generator.py`):** Dynamically compiles `ComponentNode` and `Pad` geometries for *any* electronic package (`SOIC-N`, `QFN-N`, `SOT-23-N`, `PinHeader-1xN`, `SMD-Passive 0402/0603/0805/1206`, `KeySwitch-MX`) from standard numeric parameters instead of hardcoded pad lists.
2. **Hierarchical Circuit Composer (`models/composer/schematic_composer.py`):** Takes a high-level hardware specification or natural-language prompt (e.g., *"esp32 sensor hub with i2c bme280, 3.3v ldo, usb-c, and 2 status leds"*) and composes a complete, electrically valid `NetlistGraph`—automatically synthesizing required decoupling capacitors, pull-up resistors, current-limiting LED resistors, and ERC-verified power/ground nets.
3. **Locked-Grid Matrix Benchmark (`B5: macropad_4x4`):** Adds a 2×2 / 3×3 or 4×4 mechanical keyswitch matrix board with locked switch positions, column/row scan nets, and diode/MCU multiplexing.

---

## 2. What to Build in `rev4` (Module-by-Module Checklist)

### A. Parametric IPC-7351 Footprint Generator (`env/footprint_generator.py`)
* **File to create:** `pcb-world-model/env/footprint_generator.py`
* **What to implement:**
  1. `make_smd_passive(comp_id: str, value: str, comp_type: str, package: str, net1: int, net2: int, x: float = 25.0, y: float = 25.0) -> ComponentNode`:
     * Supports `"0402"`, `"0603"`, `"0805"`, `"1206"` with exact IPC-7351 pad dimensions, pitch, and courtyard widths.
  2. `make_soic_or_sop(comp_id: str, value: str, pin_nets: list[int], pitch: float = 1.27, body_width: float = 3.9, x: float = 25.0, y: float = 25.0) -> ComponentNode`:
     * Dynamically generates any even pin count $N \in \{8, 14, 16, 20\}$ dual-row gull-wing IC footprint.
  3. `make_qfn(comp_id: str, value: str, pin_nets: list[int], pitch: float = 0.8, body_size: float = 5.0, x: float = 25.0, y: float = 25.0) -> ComponentNode`:
     * Dynamically generates 4-sided quad flat no-lead packages ($N \in \{12, 16, 20, 24, 32\}$) with pins distributed equally along Left, Bottom, Right, and Top edges.
  4. `make_pin_header(comp_id: str, value: str, pin_nets: list[int], pitch: float = 2.54, x: float = 6.0, y: float = 25.0, is_locked: bool = True) -> ComponentNode`.

### B. Hierarchical Circuit Composer & ERC Validator (`models/composer/schematic_composer.py`)
* **Files to create:**
  * `pcb-world-model/models/composer/__init__.py`
  * `pcb-world-model/models/composer/schematic_composer.py`
* **What to implement:**
  1. `compose_netlist_from_prompt(prompt: str, board_width_mm: float = 50.0, board_height_mm: float = 50.0) -> NetlistGraph`:
     * Parses functional blocks requested in `prompt` (e.g., MCU type, power regulator `ldo`/`buck`, sensors `i2c`/`spi`/`uart`, `led` count, `button`/`switch` count, `usb`/`header` connectors).
     * Hierarchical expansion:
       * Automatically creates `VIN`, `VCC_3V3`, and `GND` power rails (`is_power_ground=True`).
       * For every active IC (`MCU`, `Sensor`, `LDO`, `Bridge`), automatically instantiates a dedicated `100nF` (`0805`/`0603`) decoupling capacitor tied between `VCC` and `GND`.
       * If `i2c` is present, automatically synthesizes `4.7k` pull-up resistors on `I2C_SDA` and `I2C_SCL` to `VCC_3V3`.
       * For every requested `led`, automatically synthesizes a series `1k` current-limiting resistor to `GND`.
  2. `verify_erc(netlist: NetlistGraph) -> list[str]`:
     * Deterministic Electrical Rules Check (ERC) verifying:
       * Every routable net has $\ge 2$ pins.
       * `GND` and power nets are not shorted directly inside a 1-pin net.
       * Every IC has both a power pin and a `GND` pin connected.
  3. Expose `--prompt "..."` in [`scripts/04_run_inference.py`](file:///Users/le/Documents/antigravity/kind-hypatia/pcb-world-model/scripts/04_run_inference.py) and `/api/compose` in [`ui/server.py`](file:///Users/le/Documents/antigravity/kind-hypatia/pcb-world-model/ui/server.py) so a user or agent can run:
     ```bash
     uv run python pcb-world-model/scripts/04_run_inference.py \
       --prompt "Custom STM32 I2C sensor board with LDO, 2 LEDs, and USB header" \
       --output pcb-world-model/output_custom_prompt.kicad_pcb
     ```

### C. Add Locked-Grid Matrix Benchmark Board `B5: macropad_2x2` / `macropad_matrix` (`env/sexpr_parser.py`)
* **File to edit:** [`pcb-world-model/env/sexpr_parser.py`](file:///Users/le/Documents/antigravity/kind-hypatia/pcb-world-model/env/sexpr_parser.py)
* **What to implement:**
  1. Add a 5th reference benchmark board (`macropad_matrix`: locked mechanical keyswitches on an orthogonal grid + row/column scan matrix + controller IC + decoupling + USB header) to `PRESET_DESCRIPTIONS` and `build_preset_board()`.

---

## 3. Target Acceptance Criteria & Commands for `rev4`

Once Sections 2A–2C are implemented, run:

```bash
# 1. Run full unit & integration test suite (26+ tests including IPC footprints, Prompt Composer, ERC, and B5)
uv run pytest pcb-world-model/tests -v

# 2. Test end-to-end Natural Language Prompt -> Schematic Netlist -> Placed & Routed 0-DRC KiCad PCB
uv run python pcb-world-model/scripts/04_run_inference.py \
  --prompt "STM32 MCU with I2C environmental sensor, 3.3V LDO regulator, 2 status LEDs, and host header" \
  --output pcb-world-model/output_prompt_synthesized.kicad_pcb

# 3. Run 5-board benchmark suite & log rev4
mkdir -p pcb-world-model/revisions/rev4/boards
uv run python pcb-world-model/scripts/05_run_benchmarks.py \
  --revision "rev4" \
  --notes "Rev 4: Parametric IPC-7351 Footprints + Hierarchical Prompt-to-Netlist Composer + B5 macropad_matrix" \
  --append-log \
  --json-out pcb-world-model/revisions/rev4/benchmark_report.json
cp pcb-world-model/output_*.kicad_pcb pcb-world-model/revisions/rev4/boards/
```

| Metric | `rev3` Verified | `rev4` Target Threshold | `rev4` Verified Result |
| :--- | :---: | :---: | :---: |
| **Active Benchmark Boards** | 4 boards (35 nets) | **5 boards (adding locked-grid `macropad_matrix`)** | **5 boards (44 / 44 nets routed)** |
| **Parametric IPC-7351 Footprints** | Fixed helper functions | **Dynamic `0402–1206`, `SOIC-N`, `QFN-N`, `PinHeader-1xN`** | **`env/footprint_generator.py` (`0402–1206`, `SOIC-N`, `QFN-N`, `SOT-23`, `PinHeader`, `KeySwitch-MX`)** |
| **Prompt-to-PCB Synthesis (`--prompt`)** | None (preset/file only) | **Arbitrary natural-language prompt $\to$ 0 ERC & 0 DRC `.kicad_pcb`** | **`prompt_stm32`: 13 comps, 11 / 11 nets, 0 ERC, 0 DRC (`3.18s`) + `/api/compose`** |
| **Blind (`--blind`) Completion** | 35 / 35 (98.33 / 100) | **100% Routed, 0 DRC Errors across all 5 boards** | **44 / 44 Routed (100.0%), 0 DRC Errors (`98.53 / 100`)** |
| **Total Suite DRC Violations** | 0 | **0** | **0** |
| **Mean Composite Score** | $99.28 / 100$ | **$\ge 99.00 / 100$ across all 5 boards** | **99.16 / 100** |

---

## 4. `rev4` Verified Benchmark Scorecard & Telemetry

### Per-Board Telemetry (`rev4`, Standard Mode)

| Board ID | Routed Nets | DRC Errors | HPWL ($\text{mm}$) | Routed Wirelength ($\text{mm}$) | Via Count | Critic $V(S_t)$ | Rip-Up Iterations | Runtime ($\text{s}$) | Composite Score |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`mcu_sensor_node`** | **8 / 8 (100.0%)** | **0** | $293.49$ | $191.76$ | **5** | $0.5071$ | 1 | $2.46\text{s}$ | **98.89 / 100** |
| **`buck_converter`** | **7 / 7 (100.0%)** | **0** | $224.06$ | $139.10$ | **3** | $0.6048$ | 1 | $1.62\text{s}$ | **99.33 / 100** |
| **`usb_uart_bridge`** | **8 / 8 (100.0%)** | **0** | $254.91$ | $132.55$ | **2** | $0.5614$ | 1 | $1.76\text{s}$ | **99.56 / 100** |
| **`rp2040_minimal`** | **12 / 12 (100.0%)** | **0** | $320.48$ | $252.19$ | **3** | $0.6195$ | 1 | $3.00\text{s}$ | **99.33 / 100** |
| **`macropad_matrix`** | **9 / 9 (100.0%)** | **0** | $263.69$ | $216.59$ | **6** | $0.5904$ | 1 | $1.82\text{s}$ | **98.67 / 100** |
| **AGGREGATE / MEAN** | **44 / 44 (100.0%)** | **0** | **$271.33$** | **$186.44$** | **19** | **$0.5766$** | **1.00** | **$10.67\text{s}$** | **99.16 / 100** |

### Per-Board Telemetry (`rev4-blind`, Zero-Hint Mode)

| Board ID | Routed Nets | DRC Errors | HPWL ($\text{mm}$) | Routed Wirelength ($\text{mm}$) | Via Count | Rip-Up Iterations | Runtime ($\text{s}$) | Composite Score |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`mcu_sensor_node`** | **8 / 8 (100.0%)** | **0** | $277.50$ | $205.21$ | **10** | 1 | $2.66\text{s}$ | **97.78 / 100** |
| **`buck_converter`** | **7 / 7 (100.0%)** | **0** | $230.42$ | $149.89$ | **6** | 1 | $1.67\text{s}$ | **98.67 / 100** |
| **`usb_uart_bridge`** | **8 / 8 (100.0%)** | **0** | $273.52$ | $162.66$ | **5** | 1 | $1.81\text{s}$ | **98.89 / 100** |
| **`rp2040_minimal`** | **12 / 12 (100.0%)** | **0** | $364.38$ | $242.74$ | **9** | 1 | $2.98\text{s}$ | **98.00 / 100** |
| **`macropad_matrix`** | **9 / 9 (100.0%)** | **0** | $273.22$ | $162.98$ | **3** | 1 | $1.73\text{s}$ | **99.33 / 100** |
| **AGGREGATE / MEAN** | **44 / 44 (100.0%)** | **0** | **$283.81$** | **$184.70$** | **33** | **1.00** | **$10.85\text{s}$** | **98.53 / 100** |
