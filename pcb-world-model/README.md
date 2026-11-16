# PCB Engineering World Model (`pcb-world-model` — Rev 1)

A closed-loop **Neuro-Symbolic PCB Engineering World Model** combining a **4-layer GATv2 Macro-Placement Composer**, a **Dual-Head Spatial World Model U-Net** (predicting future routing congestion $\hat{M}_{t+1}$ and scalar routability $V(S_t)$), a **45° Octilinear 2-Layer A* Router**, a **Deterministic 2D Geometric / Headless KiCad DRC Verifier**, and a live **FastAPI + WebSocket 2D/3D CAD Visualizer**.

## Architecture

1. **Data & Parser Layer (`env/`)**:
   - `sexpr_parser.py`: Recursive-descent `.kicad_pcb` S-expression parser translating footprints, pads, nets, segments, and vias into `NetlistGraph` (`Pad`, `ComponentNode`, `NetInfo`).
   - `board_state.py`: Maintains the `[C=7, H=250, W=250]` spatial state tensor $S_t$ (`board_outline`, `courtyard_mask`, `copper_L1`, `copper_L2`, `via_mask`, `active_net_pins`, `unrouted_demand`).
   - `kicad_exporter.py`: Exports manufacturing-ready KiCad 8 `.kicad_pcb` S-expressions.
2. **Macro-Placement Composer (`models/placer/`)**:
   - `gnn_encoder.py` & `placer_head.py`: 4-layer `GATv2Layer` encoder predicting continuous component poses $(x, y, \sin\theta, \cos\theta)$ with differentiable `SmoothHPWL`, courtyard `OverlapArea`, and `OutOfBounds` loss.
   - `relaxation.py`: 50-step gradient overlap repulsion, discrete 90° orientation optimization, and 0.5mm grid snapping.
3. **Latent Spatial World Model & Router (`models/world/`)**:
   - `dynamics_unet.py`: U-Net with bottleneck 2D spatial self-attention and dual heads for layer-wise congestion $\hat{M}_{t+1} \in [0, 1]^{2 \times H \times W}$ and scalar routability critic $V(S_t) \in [0, 1]$.
   - `neural_router.py`: 2-layer 45° octilinear MST-Steiner A* router guided by $\hat{M}_{t+1} + \text{Penalty}_{\text{DRC}}$.
4. **Deterministic Verifier (`verifier/`)**:
   - `fast_clearance.py` & `kicad_drc.py`: Millisecond 2D segment/pad/via/courtyard DRC engine plus headless `kicad-cli pcb drc --format json` integration with closed-loop rip-up & reroute penalty injection.
5. **Interactive Web Visualizer (`ui/`)**:
   - `server.py` & `web/`: FastAPI + WebSocket server streaming live $S_t \to S_{t+1}$ rollouts, 2D multi-layer CAD view, 3D isometric stackup view, all 7 spatial tensor channels, and `.kicad_pcb` inspection.

## Quickstart

```bash
# 1. Generate reference .kicad_pcb boards and synthetic World Model training triplets
uv run python pcb-world-model/scripts/01_generate_synthetic.py

# 2. Train GATv2 Macro-Placer
uv run python pcb-world-model/scripts/02_train_placer.py --epochs 25

# 3. Train Dual-Head Spatial World Model U-Net
uv run python pcb-world-model/scripts/03_train_world_model.py --epochs 20

# 4. Run closed-loop inference and export clean .kicad_pcb
uv run python pcb-world-model/scripts/04_run_inference.py --board mcu_sensor_node --output pcb-world-model/output_clean.kicad_pcb

# 5. Launch Interactive 2D/3D Web Console
uv run uvicorn ui.server:app --app-dir pcb-world-model --port 8050

# 6. Run Test Suite
uv run pytest pcb-world-model/tests -v
```
