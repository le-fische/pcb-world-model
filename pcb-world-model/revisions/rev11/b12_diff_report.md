# `rev11` Before-vs-After Optimization Diff Report — `legacy_kicad_reoptimize`

* **Locked Components Preserved (`5`):** `J1, J2, MH1, MH2, U1` (Pose Verified: **PASS**)
* **Re-Optimized Unlocked Components (`7`):** `C1, C2, C3, D1, L1, R1, R2`

| Optimization Metric | Imported Baseline | `rev11` Optimized Output | Improvement |
| :--- | :---: | :---: | :---: |
| **Total DRC & DFM Violations** | 97 DRC / 27 DFM (124 total) | **0 DRC / 0 DFM** | **100% Fab-Ready** |
| **Via Count & Wirelength** | 40 vias / 511.78 mm | **5 vias / 128.80 mm** | **-87.5% vias, -74.8% WL** |
| **Enclosure 3D $Z$-Height Collisions** | 0 collisions | **0 Mechanical Interferences** | **Verified against 3D cavity** |
| **Critical Loop Inductance (`SW_NODE`)** | 73.97 nH | **9.75 nH** | **-86.8% loop inductance** |
| **DFM Craftsmanship Score** | 0.00 / 100 | **99.40 / 100** | **+99.40 pts** |
