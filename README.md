# PCB Engineering World Model (`rev1` $\to$ `rev15`)

**Closed-Loop Neuro-Symbolic Generative EDA, C++20 Octilinear Router, 8-Layer HDI, Multi-Zone Rigid-Flex $85\Omega$ PCIe Gen 4, Mixed-Signal $J_{\text{return}}$ & 3D PBR Workstation**

- **Live Interactive Multi-Revision Workstation (Vercel Production):** [https://pcb-world-model.vercel.app/](https://pcb-world-model.vercel.app/)
- **Live Interactive Multi-Revision Workstation (GitHub Pages):** [https://le-fische.github.io/pcb-world-model/](https://le-fische.github.io/pcb-world-model/)
- **Deep-Link Any Revision Directly:** [`#rev1`](https://le-fische.github.io/pcb-world-model/#rev1) · [`#rev2`](https://le-fische.github.io/pcb-world-model/#rev2) · [`#rev3`](https://le-fische.github.io/pcb-world-model/#rev3) · [`#rev4`](https://le-fische.github.io/pcb-world-model/#rev4) · [`#rev5`](https://le-fische.github.io/pcb-world-model/#rev5) · [`#rev6`](https://le-fische.github.io/pcb-world-model/#rev6) · [`#rev7`](https://le-fische.github.io/pcb-world-model/#rev7) · [`#rev8`](https://le-fische.github.io/pcb-world-model/#rev8) · [`#rev9`](https://le-fische.github.io/pcb-world-model/#rev9) · [`#rev10`](https://le-fische.github.io/pcb-world-model/#rev10) · [`#rev11`](https://le-fische.github.io/pcb-world-model/#rev11) · [`#rev12`](https://le-fische.github.io/pcb-world-model/#rev12) · [`#rev13`](https://le-fische.github.io/pcb-world-model/#rev13) · [`#rev14`](https://le-fische.github.io/pcb-world-model/#rev14) · [`#rev15`](https://le-fische.github.io/pcb-world-model/#rev15)

---

## Interactive Web Workstation Capabilities

The deployed workstation (`pcb-world-model/ui/web/`) includes pre-compiled geometry, traces, vias, 3D meshes, schematics, pinouts, and benchmark telemetry for every revision (`rev1` through `rev15`):

1. **Revision Ribbon & Deep-Linking (`REV 1` $\to$ `REV 15`):** Switch between any revision to inspect its exact routed boards, composite scores, via counts, and architectural unlocks.
2. **2D CAD Canvas:** 60 FPS pan/zoom, per-layer copper toggles (`F.Cu`, `B.Cu`, `In1`, `In2`), rigid-flex zone overlays, thermal isolation slots, and click-to-cross-probe net highlighting.
3. **Real-Time 3D PBR Viewport (Three.js):** Drilled FR4 substrate, 5 solder mask finishes (KiCad Green, Matte Black, Royal Blue, Crimson Red, Arctic White), ENIG immersion gold pads, procedural STEP-quality 3D components, and 180° board flip.
4. **Cross-Probed Schematic & Vector SVG Pin Diagrams:** Orthogonal functional-block schematic and package pinout inspector synced with 2D and 3D views.
5. **Tech Dossier, Benchmark Matrix, Side-by-Side Revision Diff & Rev 15 LLM Studio:** Read full `OVERVIEW.md` specs, compare any two revisions side-by-side, inspect frontier baselines, or test the `PromptPhysicsSpec` gatekeeper.
6. **1-Click `.kicad_pcb` Download:** Export the manufacturing-ready KiCad 8 S-expression file for any board in any revision directly from the browser.

---

## Revision Progression Matrix

| Revision | Status | Core Capability Unlocked | Target Boards | Composite Score | Overview Doc |
| :--- | :---: | :--- | :--- | :---: | :--- |
| **`rev1`** | **Verified & Frozen** | Closed-loop GATv2 Macro-Placer + Dual-Head U-Net World Model + Box-Dilated 45° Octilinear A* + DRC Rip-Up | `usb_uart_bridge`, `buck_converter`, `mcu_sensor_node` (2L) | **98.00 / 100** *(23/23 nets, 0 DRC, 27 vias, 6.88s)* | [`rev1/OVERVIEW.md`](pcb-world-model/revisions/rev1/OVERVIEW.md) |
| **`rev2`** | **Verified & Frozen** | Sub-Grid Euclidean Pad/Trace/Via Clearance Field + Full Steiner Copper Reuse + 2-Opt Component Swap | `usb_uart_bridge`, `buck_converter`, `mcu_sensor_node` (2L) | **99.26 / 100** *(23/23 nets, 0 DRC, 10 vias, 6.37s)* | [`rev2/OVERVIEW.md`](pcb-world-model/revisions/rev2/OVERVIEW.md) |
| **`rev3`** | **Verified & Frozen** | Per-Net Width Classes (`0.40mm` Power vs `0.25mm` Signal) + Blind Zero-Hint Placement (`--blind`) + `rp2040_minimal` | `B1–B3` + `B4: rp2040_minimal` (35 nets) | **99.28 / 100** *(98.33 `--blind`)* | [`rev3/OVERVIEW.md`](pcb-world-model/revisions/rev3/OVERVIEW.md) |
| **`rev4`** | **Verified & Frozen** | Parametric IPC-7351 Footprint Generator + Hierarchical Prompt-to-Netlist Composer & ERC + `B5: macropad_matrix` | Any prompt (`--prompt`) + `B1–B5` (44 nets) | **99.16 / 100** *(98.53 `--blind`)* | [`rev4/OVERVIEW.md`](pcb-world-model/revisions/rev4/OVERVIEW.md) |
| **`rev5`** | **Verified & Frozen** | Copper Ground Pour Zones + JLCPCB/PCBWay `bom.csv` & `positions.csv` Bundle + Barrel Reuse | Any prompt + `B1–B5` (44 nets) | **99.56 / 100** *(10 vias across 5 boards)* | [`rev5/OVERVIEW.md`](pcb-world-model/revisions/rev5/OVERVIEW.md) |
| **`rev6`** | **Verified & Frozen** | Native Compiled C++20 A* Kernel (`<0.5s`/board) + 4-Layer Internal Planes + SI/IPC-2221 Thermal Telemetry | `B1–B5` + `flight_controller_50` (50 comps, 4L) | **99.56 / 100** *(`1.73s` total; `43/43` FC in `3.88s`)* | [`rev6/OVERVIEW.md`](pcb-world-model/revisions/rev6/OVERVIEW.md) |
| **`rev7`** | **Verified & Frozen** | Interactive Pre-Flight Constraint Negotiator (`HTTP 409`) + Class A/B 12-Domain Library + Class C Ingest | `B1–B8` (`wearable_oled_keychain_26x26`) | **99.56 / 100** *(46/46 tests)* | [`rev7/OVERVIEW.md`](pcb-world-model/revisions/rev7/OVERVIEW.md) |
| **`rev8`** | **Verified & Frozen** | Verilog-2001 RTL Synthesis + `iverilog`/`Verilator` + Joint `.pcf` FPGA Pin-Swap (`iCE40UP5K`) + OpenLane `SKY130` | `B1–B5` + `B9: fpga_custom_protocol_bridge` (4L) | **99.56 / 100** *(`-92.5%` crossings `147->11`)* | [`rev8/OVERVIEW.md`](pcb-world-model/revisions/rev8/OVERVIEW.md) |
| **`rev9`** | **Verified & Frozen** | Real-Time PBR 3D Rendering + Procedural STEP 3D ICs + Interactive Vector SVG Pinouts + Cross-Probed Schematic | All Production Boards (`B1–B9`, `FC-50`) | **99.56 / 100** *(AP214 `.STEP` export)* | [`rev9/OVERVIEW.md`](pcb-world-model/revisions/rev9/OVERVIEW.md) |
| **`rev10`** | **Verified & Frozen** | 6-Rule DFM/DFA Craftsmanship Auditor + Automated Gerber/Excellon Drill CAM Bundler | All Production Boards (`B1–B9`, `FC-50`) | **99.29 / 100** *(100% DFM/DFA PASS; 59/59 tests)* | [`rev10/OVERVIEW.md`](pcb-world-model/revisions/rev10/OVERVIEW.md) |
| **`rev11`** | **Verified & Frozen** | Import `.kicad_sch`/Netlists + Datasheet Pinout Ingest + External `.STEP`/`.DXF` Enclosures + Lock-Preserving Re-Opt | `B10: step_enclosure_keychain`, `B11`, `B12` | **B10–B12 PASS** *(`-87.5%` vias on `B12`; 64/64 tests)* | [`rev11/OVERVIEW.md`](pcb-world-model/revisions/rev11/OVERVIEW.md) |
| **`rev12`** | **Verified & Frozen** | **8-Layer (`2+4+2`) HDI** + Buried Core Vias (`L3↔L6`) + Staggered-Only Microvia Lockout ($\Delta xy \ge 0.25\text{mm}$) | `B13: hdi_fpga_bga_fanout_8l` (64-ball `0.8mm` BGA) | **99.29 / 100** *(`0` stacked, `0` skip-vias; 68/68 tests)* | [`rev12/OVERVIEW.md`](pcb-world-model/revisions/rev12/OVERVIEW.md) |
| **`rev13`** | **Verified & Frozen** | Noisy (`5A 2MHz GaN`) vs. Quiet (`24-bit ADC`) Partitioning + Continuous `GND` + $J_{\text{return}}$ & Broadside Crosstalk | `B14: mixed_signal_gan_adc_30x50` + `B14a–c` | **99.29 / 100** *(`0.0%` $J_{\text{return}}$ overlap; 73/73 tests)* | [`rev13/OVERVIEW.md`](pcb-world-model/revisions/rev13/OVERVIEW.md) |
| **`rev14`** | **Active (Verified)** | Multi-Zone **`Rigid FR4` + `Cross-Hatched Flex`** + Coupled **$85\Omega$ PCIe Gen 4** + Skew Match + Under-BGA PDN | `B15: ultimate_rigid_flex_8l_hdi_pcie_gan_adc` | **99.29 / 100** *(`99.51` `--blind`; `78/78` tests)* | [`rev14/OVERVIEW.md`](pcb-world-model/revisions/rev14/OVERVIEW.md) |
| **`rev15`** | **Planned Spec** | Local **`qwen3-1.7b`** via **LM Studio** (`localhost:1234/v1` `strict: true` JSON Schema) $\to$ Gatekeeper | `B16: local_llm_physics_translator` | **Target: 100% Valid `PromptPhysicsSpec` JSON** | [`rev15/OVERVIEW.md`](pcb-world-model/revisions/rev15/OVERVIEW.md) |

---

## Local Quickstart

```bash
# 1. Run the full 78-test verification suite (rev1 -> rev14)
uv run pytest pcb-world-model/tests -v

# 2. Re-compile the static multi-revision dataset for Vercel / GitHub Pages
uv run python pcb-world-model/scripts/compile_revisions_dataset.py

# 3. Launch the live FastAPI + WebSocket server locally
uv run uvicorn ui.server:app --app-dir pcb-world-model --port 8050
```
