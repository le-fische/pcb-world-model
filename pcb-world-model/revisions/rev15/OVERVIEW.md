# Revision 15 (`rev15`) — Lightweight Local LLM Physics-Translator Frontend (`Text -> Strict Schema -> Gatekeeper`)

* **Revision Tag:** `rev15`
* **Status:** Planned Specification (Natural-Language Frontend Decoupled from Physics World Model)
* **Prerequisite:** `rev10`–`rev14` (Deterministic Physics World Model, Multi-Zone Rigid-Flex, Mixed-Signal $J_{\text{return}}$, PDN & Adversarial Verifiers)
* **Core Mission:** Decouple messy natural language cleanly from the spatial tensor engine by hooking an **ultra-lightweight local LLM (`Qwen3-1.7B` via LM Studio's `http://localhost:1234/v1` API, or embedded GGUF/Ollama)** strictly as a **Text $\to$ Physics-JSON Translator** using grammar-constrained JSON Schema decoding, validated by the deterministic `constraint_negotiator.py` gatekeeper before any geometry/tensor solver runs.

---

## 1. Architectural Principle: Untrusted LLM Translator, Trusted Physics Engine

The spatial world model (`BoardState`, `StateEncoder`, C++20/Python A\* router, PDN/$J_{\text{return}}$/impedance solvers) **never consumes English prose**. It consumes only numeric geometry tensors, locked component poses, netlist connectivity, and typed physical boundary conditions:

```text
User English Brief ──► [1. Local LLM Translator]                  (LM Studio `localhost:1234/v1` -> `qwen3-1.7b`)
                                 │                                (or Ollama / Embedded GGUF, `strict: true`, `temp=0`)
                                 ▼ `PromptPhysicsSpec` (JSON)
                       [2. Deterministic Physics Gatekeeper]      (`constraint_negotiator.py` + `verify_erc()`)
                                 │  (rejects bad math, courtyard > 0.72, star-ground traps, missing specs)
                                 ▼ Verified `BoardConfig` + `NetlistGraph` + `electrical_meta`
                       [3. Spatial Physics World Model Engine]    (`BoardState`, Placer, C++20/Python Router)
```

---

## 2. Smallest Viable Local Models & Desktop App Delivery

If packaging `pcb-world-model` as a desktop app demo, small quantized models work *precisely because* the LLM only extracts structured parameters rather than solving 3D electromagnetics:

| Tier | Model (LM Studio / Ollama) | Approx. Disk Size | Runtime RAM/VRAM | Best Role |
| :--- | :--- | :---: | :---: | :--- |
| **Absolute Minimum** | `functiongemma` / `functiongemma:270m` | **~300 MB** | ~1.5–2 GB | Ultra-lightweight flat function-call / CLI parameter extraction (`dims`, `layers`, `current`, `target_z0`). |
| **Smallest General Dense** | `qwen3-0.6b` *(or `gemma3:1b`)* | **~523 MB** / ~810 MB | ~2–3 GB | Basic edge schema extraction when paired with `llama.cpp` grammar-constrained JSON. |
| **Primary App Target** | **`qwen3-1.7b`** *(Q4_K_M GGUF)* | **~1.1–1.4 GB** | ~3–4 GB | **Default LM Studio & Desktop App model — minimum footprint that reliably extracts nested multi-component arrays, thermal watts, edge rates, and contradictions.** |
| **Desktop Precision Tier** | `qwen3-4b` | **~2.5 GB** | ~5–6 GB | Multi-paragraph hardware specifications with complex coupled physics constraints. |

### Desktop App Delivery Model (Two Modes, One Codebase)
1. **Live Dev / Demo Hookup (Current Mac Setup):** App connects directly to **LM Studio**'s OpenAI-compatible local server (`http://localhost:1234/v1`, started via Developer tab or `lms server start`) using your downloaded `qwen3-1.7b`.
2. **Packaged Standalone Desktop App:** On first launch, checks local cache for the quantized `Qwen3-1.7B` GGUF (`~1.1–1.4 GB`); if missing, downloads it once into user app data and runs an embedded local OpenAI-compatible server (`llama.cpp` / `lms`)—using the offline deterministic regex parser only if no local runtime is active.

---

## 3. Module-by-Module Implementation Plan

### A. LM Studio / OpenAI-Compatible Schema Translator (`models/composer/local_llm_translator.py`)
* Define JSON Schema / dataclass contract **`PromptPhysicsSpec`**:
  * `board`: `width_mm`, `height_mm`, `thickness_mm`, `num_layers`, `max_lamination_cycles`, `zones` (`rigid_fr4` / `dynamic_flex`).
  * `components`: list of `{role, part_hint, locked_side, is_locked, thermal_watts, noise_domain, max_noise_mv}`.
  * `nets`: list of `{net_name, i_max_amps, max_temp_rise_c, edge_rate_ns, z0_ohm, z_diff_ohm, is_clock}`.
  * `ambiguities` / `conflicting_constraints`: list of missing or contradictory user requests.
* **LM Studio Hookup (`provider="lmstudio"` default):**
  * Calls `POST http://localhost:1234/v1/chat/completions` (zero external cloud calls, standard library `urllib.request` or `openai` client w/ `base_url="http://localhost:1234/v1"`).
  * Queries `GET http://localhost:1234/v1/models` to auto-detect the loaded `qwen3-1.7b` identifier (configurable via `PCB_LLM_BASE_URL` / `PCB_LLM_MODEL`).
  * Passes LM Studio's strict `llama.cpp` grammar constraint:
    ```json
    "response_format": {
      "type": "json_schema",
      "json_schema": {
        "name": "PromptPhysicsSpec",
        "strict": true,
        "schema": { "...PromptPhysicsSpec JSON Schema..." }
      }
    }
    ```
  * Uses `temperature: 0.0` + `/no_think` system/user suffix on Qwen 3 so grammar-constrained JSON returns immediately without `<think>` latency.
* Also retains secondary `provider="ollama"` (`http://localhost:11434/api/chat`) and `--translator auto|lmstudio|ollama|regex` so pytest CI runs deterministically offline without requiring LM Studio open.

### B. Deterministic Gatekeeper & Workstation UI (`models/composer/constraint_negotiator.py`, `ui/server.py`, `ui/web/app.js`)
* Pipe `PromptPhysicsSpec` through deterministic unit checks, courtyard utilization checks (`<= 0.72`), HDI lamination rules, and physical contradiction checks (`invalid_remote_star_ground`, conflicting enclosure limits) — returning `needs_input` / HTTP `409` whenever specs are under-determined rather than letting the small LLM guess dimensions.
* Add UI endpoint & status badge in `ui/server.py` / `ui/web/app.js` showing live local engine connection (`LM Studio :1234 • qwen3-1.7b Ready` vs `Regex Fallback`).

---

## 4. `rev15` Validation Benchmark (`B16`)

| Benchmark ID | Input Prompt & Translator Mode | Acceptance Criteria |
| :--- | :--- | :--- |
| **`B16: local_llm_physics_translator`** | Unstructured natural-language prompts across wearable, motor (`20A`, `ΔT ≤ 15°C`), and 8L rigid-flex specs (`85Ω PCIe` + `5W GaN` + `24-bit ADC` on Top) routed through LM Studio `qwen3-1.7b` (`localhost:1234/v1`) and offline fallback | **100% valid `PromptPhysicsSpec` JSON via `strict: true` schema (`0` parse failures), exact numeric constraint fidelity, proper `needs_input` on vague prompts, `0` regression in `--translator regex` offline mode** |
