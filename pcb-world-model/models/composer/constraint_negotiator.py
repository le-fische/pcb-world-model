"""Interactive Pre-Flight Constraint & Physical Trade-Off Negotiator for `rev7`.

Prevents silent guessing of missing mechanical/stackup constraints and rejects
physically impossible enclosures (Area Utilization > 0.72) with quantified
engineering trade-off options before placement or routing begins.
"""

from __future__ import annotations

import math
import re
from dataclasses import asdict, dataclass, field
from typing import Any


MAX_COMFORTABLE_UTILIZATION: float = 0.72


@dataclass
class ParsedPromptConstraints:
    """Explicit physical & architectural constraints extracted from a user prompt."""

    explicit_width_mm: float | None = None
    explicit_height_mm: float | None = None
    explicit_thickness_mm: float | None = None
    explicit_layers: int | None = None
    explicit_single_sided: bool | None = None
    explicit_keep_back_flat: bool | None = None
    explicit_mounting_holes: int | None = None
    explicit_hole_diameter_mm: float | None = None
    explicit_passive_pkg: str | None = None
    explicit_wireless_choice: str | None = None  # "esp32_c3fn4_qfn" | "esp32_c3_mini" | None
    vague_size_terms: list[str] = field(default_factory=list)
    detected_intent: str = "generic"  # "wearable_keychain" | "relay_controller" | "motor_driver" | "sensor_hub" | "generic"
    has_wireless_intent: bool = False
    relay_channels: int = 0


@dataclass
class ConstraintQuestion:
    """Targeted clarification question when physical specs are underspecified."""

    question_id: str
    field: str
    prompt: str
    options: list[str]
    recommended: str
    rationale: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class TradeOffOption:
    """Quantified engineering remediation when courtyard utilization exceeds 0.72."""

    option_id: str
    title: str
    description: str
    projected_utilization: float
    resolves_feasibility: bool
    patch: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ConstraintNegotiationResult:
    """Structured outcome of pre-flight physical & electrical constraint negotiation."""

    status: str  # "ready" | "needs_input" | "tradeoff_required"
    feasible: bool
    prompt: str
    intent: str
    locked_specs: dict[str, Any]
    courtyard_area_mm2: float
    usable_area_mm2: float
    area_utilization: float
    questions: list[ConstraintQuestion] = field(default_factory=list)
    tradeoffs: list[TradeOffOption] = field(default_factory=list)
    planned_components: list[dict[str, Any]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    rationale: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "feasible": self.feasible,
            "prompt": self.prompt,
            "intent": self.intent,
            "locked_specs": self.locked_specs,
            "courtyard_area_mm2": round(self.courtyard_area_mm2, 2),
            "usable_area_mm2": round(self.usable_area_mm2, 2),
            "area_utilization": round(self.area_utilization, 4),
            "utilization_pct": round(self.area_utilization * 100.0, 1),
            "questions": [q.to_dict() for q in self.questions],
            "tradeoffs": [t.to_dict() for t in self.tradeoffs],
            "planned_components": self.planned_components,
            "warnings": self.warnings,
            "rationale": self.rationale,
        }


def parse_prompt_constraints(prompt: str) -> ParsedPromptConstraints:
    """Parse explicit physical dimensions, stackup, thickness, and functional cues from prompt."""
    low = prompt.lower()
    parsed = ParsedPromptConstraints()

    # 1. Board dimensions (e.g. "26x26mm", "20×20 mm", "80 mm x 80 mm")
    dim_match = re.search(
        r"(\d+(?:\.\d+)?)\s*(?:mm)?\s*[x×*]\s*(\d+(?:\.\d+)?)\s*(?:mm)?",
        low,
    )
    if dim_match:
        parsed.explicit_width_mm = float(dim_match.group(1))
        parsed.explicit_height_mm = float(dim_match.group(2))

    # 2. Board thickness (e.g. "0.8mm", "1.6mm thickness", "0.8mm FR4")
    thk_match = re.search(r"\b(0\.8|1\.0|1\.2|1\.6|2\.0)\s*mm\b(?:\s*(?:fr4|thick|thickness|pcb))?", low)
    if thk_match:
        val = float(thk_match.group(1))
        # Distinguish 2.0mm mounting hole from thickness
        context_slice = low[max(0, thk_match.start() - 12) : min(len(low), thk_match.end() + 18)]
        if "hole" not in context_slice and "npth" not in context_slice and "pth" not in context_slice:
            parsed.explicit_thickness_mm = val

    # 3. Copper stackup layers
    lyr_match = re.search(r"\b([246])\s*[-\s]*(?:layer|layers|l)\b", low)
    if lyr_match:
        parsed.explicit_layers = int(lyr_match.group(1))

    # 4. Single-sided / keep_back_flat vs double-sided
    if any(tok in low for tok in ("single-sided", "single sided", "keep_back_flat", "flat rear", "flat back")):
        parsed.explicit_single_sided = True
        parsed.explicit_keep_back_flat = True
    elif any(tok in low for tok in ("double-sided", "double sided", "two-sided", "both sides")):
        parsed.explicit_single_sided = False
        parsed.explicit_keep_back_flat = False

    # 5. Mounting holes
    hole_diam = re.search(r"\b(?:m(1\.6|2\.5|2|3)|(\d+(?:\.\d+)?)\s*mm\s*(?:corner\s*)?(?:npth|pth|mounting))\b", low)
    if hole_diam:
        d_str = hole_diam.group(1) or hole_diam.group(2)
        parsed.explicit_hole_diameter_mm = float(d_str)
        if any(tok in low for tok in ("two ", "2x", "2 ")):
            parsed.explicit_mounting_holes = 2
        elif any(tok in low for tok in ("four ", "4x", "4 ")):
            parsed.explicit_mounting_holes = 4
        else:
            parsed.explicit_mounting_holes = 2
    elif "mounting hole" in low or "npth" in low:
        parsed.explicit_hole_diameter_mm = 2.0
        parsed.explicit_mounting_holes = 2

    # 6. Passive package size
    for pkg in ("0201", "0402", "0603", "0805", "1206"):
        if pkg in low:
            parsed.explicit_passive_pkg = pkg
            break

    # 7. Wireless choice
    if "esp32-c3fn4" in low or "c3fn4" in low or ("qfn-32" in low and "esp32" in low):
        parsed.explicit_wireless_choice = "esp32_c3fn4_qfn"
        parsed.has_wireless_intent = True
    elif "esp32-c3-mini" in low or "esp32-s3-wroom" in low or "module" in low and "esp32" in low:
        parsed.explicit_wireless_choice = "esp32_c3_mini"
        parsed.has_wireless_intent = True
    elif any(tok in low for tok in ("wi-fi", "wifi", "ble", "bluetooth", "wireless", "esp32", "nrf52", "lora")):
        parsed.has_wireless_intent = True

    # 8. Vague physical terms
    for term in ("tiny", "miniature", "compact", "wearable", "keychain", "small", "pocket"):
        if term in low:
            parsed.vague_size_terms.append(term)

    # 9. Functional intent classification
    relay_match = re.search(r"(\d+)\s*[-\s]*(?:ch|channel)s?\b.*relay", low)
    if "relay" in low:
        parsed.detected_intent = "relay_controller"
        parsed.relay_channels = int(relay_match.group(1)) if relay_match else (6 if "6" in low else 4)
    elif any(
        tok in low
        for tok in ("fpga", "ice40", "verilog", "rtl", "ws2812", "quadrature", "no single off-the-shelf ic")
    ):
        parsed.detected_intent = "fpga_protocol_bridge"
    elif any(tok in low for tok in ("keychain", "wearable", "wakes up when shaken", "lis3dh", "ssd1306", "mcp73831")):
        parsed.detected_intent = "wearable_keychain"
    elif any(tok in low for tok in ("stepper", "motor", "tmc2209", "drv8833", "bldc")):
        parsed.detected_intent = "motor_driver"
    elif any(tok in low for tok in ("sensor", "bme280", "imu", "i2c", "stm32", "rp2040", "esp32")):
        parsed.detected_intent = "sensor_hub"

    return parsed


def _estimate_component_budget(
    prompt: str,
    parsed: ParsedPromptConstraints,
    passive_pkg: str,
    wireless_choice: str,
    mounting_holes: int,
    hole_diam: float,
    single_sided: bool,
) -> tuple[list[dict[str, Any]], list[str]]:
    """Build exact courtyard budget for all planned Class A & Class B components."""
    low = prompt.lower()
    compact = passive_pkg in ("0201", "0402")
    passive_area = {
        "0201": 1.3 * 0.8,
        "0402": 2.0 * 1.2 if compact else 3.6 * 2.2,
        "0603": 2.6 * 1.5 if compact else 4.2 * 2.6,
        "0805": 3.2 * 1.9 if compact else 4.4 * 2.8,
        "1206": 4.1 * 2.3 if compact else 4.8 * 3.0,
    }.get(passive_pkg, 2.4)

    items: list[dict[str, Any]] = []
    rationale: list[str] = []

    def add(name: str, part: str, area: float, side: str = "top", reason: str = "", movable: bool = True) -> None:
        items.append(
            {
                "name": name,
                "part_id": part,
                "courtyard_area_mm2": round(area, 2),
                "side": side,
                "movable_to_bottom": movable,
                "reason": reason,
            }
        )

    if parsed.detected_intent == "relay_controller":
        n_ch = max(1, parsed.relay_channels or 4)
        for ch in range(1, n_ch + 1):
            add(f"K{ch}", "Relay_SPDT_10A", 19.6 * 16.0, "top", f"10A isolated SPDT relay ch{ch}", movable=False)
            add(f"U_ISO{ch}", "PC817", 10.2 * 5.6, "top", f"Primary/secondary opto-isolator ch{ch}", movable=False)
            add(f"J_OUT{ch}", "ScrewTerminal_5.08mm", 8.2 * 17.44, "top", f"10A load screw terminal ch{ch}", movable=False)
            add(f"D_FLY{ch}", "SOD-323", 3.6 * 2.1, "top", "Coil flyback diode", movable=True)
            add(f"Q{ch}", "SOT-23", 6.0 * 5.8, "top", "Relay coil MOSFET driver", movable=True)
            add(f"R_OPTO{ch}", passive_pkg, passive_area, "top", "Optocoupler LED series resistor", movable=True)
        add("U1", "ESP32-C3FN4", 6.8 * 6.8, "top", "Main Wi-Fi/BLE MCU", movable=True)
        add("U_REG", "AP2112K-3.3", 6.0 * 5.6, "top", "3.3V logic regulator", movable=True)
        add("J_PWR", "ScrewTerminal_5.08mm", 8.2 * 17.44, "top", "5V/12V logic supply terminal", movable=False)
        for idx in range(6):
            add(f"P{idx+1}", passive_pkg, passive_area, "top", "Decoupling & pull-up passives", movable=True)
        rationale.append(
            f"{n_ch}-channel 10A relay design requires {n_ch}x SPDT relays (313.6 mm2 each), optocouplers with >=2.5mm creepage, and 5.08mm terminals."
        )
        return items, rationale

    if parsed.detected_intent == "fpga_protocol_bridge":
        add("U1", "Lattice_iCE40UP5K", 8.6 * 8.6, "top", "iCE40UP5K-SG48 QFN-48 FPGA synthesizing custom RTL + joint .pcf pin swap", movable=True)
        add("U2", "ESP32-C3FN4", 6.8 * 6.8, "top", "Host Wi-Fi/BLE MCU talking over SPI to FPGA", movable=True)
        add("U3", "W25Q16", 6.2 * 6.0, "top", "16Mbit SPI configuration flash for iCE40 bitstream", movable=True)
        add("U4", "AP2112K-3.3", 3.8 * 3.4, "top", "3.3V I/O & MCU LDO regulator", movable=True)
        add("U5", "XC6206", 3.6 * 3.2, "top", "1.2V FPGA core regulator", movable=True)
        add("Y1", "OSC_48MHz", 3.8 * 3.2, "top", "48MHz active reference oscillator for WS2812B 800kHz timing", movable=True)
        add("Y2", "XTAL_40MHz", 3.2 * 2.8, "top", "40MHz reference crystal for ESP32-C3FN4", movable=True)
        add("J1", "USB_C_6P", 8.2 * 4.8, "top", "USB-C 5V power/debug connector", movable=False)
        add("J2", "WS2812_CH0_3", 17.5 * 4.6, "top", "4-channel WS2812B output header (ch0..3)", movable=False)
        add("J3", "WS2812_CH4_7", 17.5 * 4.6, "top", "4-channel WS2812B output header (ch4..7)", movable=False)
        add("J4", "ENC_CH0_3", 27.6 * 4.6, "top", "4-channel quadrature encoder A/B header (ch0..3)", movable=False)
        add("J5", "ENC_CH4_7", 27.6 * 4.6, "top", "4-channel quadrature encoder A/B header (ch4..7)", movable=False)
        for idx in range(10):
            add(f"P{idx+1}", passive_pkg, passive_area, "top", "Decoupling, CC pulldowns, CRESET pull-up", movable=True)
        rationale.append(
            "Synthesizes missing multi-channel WS2812 + quadrature SPI controller in Verilog onto Lattice iCE40UP5K-SG48 with joint .pcf pin-swap PCB co-design."
        )
        return items, rationale

    if parsed.detected_intent == "wearable_keychain":
        if wireless_choice == "esp32_c3_mini":
            add("U1", "ESP32-C3-MINI-1", 13.8 * 17.2, "top", "Pre-certified Wi-Fi/BLE module w/ PCB antenna keepout", movable=False)
            rationale.append("Selected pre-certified ESP32-C3-MINI-1 module (larger footprint, zero external crystal/antenna tuning).")
        else:
            add("U1", "ESP32-C3FN4", 6.8 * 6.8, "top", "QFN-32 Wi-Fi/BLE SoC with 4MB flash & thermal EP", movable=True)
            add("Y1", "XTAL_40MHz_2016", 3.0 * 2.4, "top", "Required 40MHz reference crystal for ESP32-C3FN4", movable=True)
            add("ANT1", "ChipAntenna_2.4GHz", 3.6 * 2.4, "top", "2.4GHz ceramic chip antenna with perimeter keepout", movable=False)
            rationale.append("Selected compact ESP32-C3FN4 QFN-32 + 40MHz crystal + 2.4GHz chip antenna to fit inside miniature wearable envelope.")

        add("J1", "USB_C_6P", 8.2 * 4.8, "top", "USB-C power/programming port (with 5.1k CC1/CC2 pulldowns)", movable=False)
        add("U2", "MCP73831", 3.8 * 3.4, "top", "100mA LiPo battery charger (10k PROG resistor + status LED)", movable=True)
        add("U3", "AP2112K-3.3", 3.8 * 3.4, "top", "Low-dropout 3.3V regulator", movable=True)
        add("U4", "LIS3DH", 3.6 * 5.2, "top", "Ultra-low-power MEMS accelerometer wired to RTC wake GPIO", movable=True)
        add("J2", "SSD1306_OLED_4Pin", 8.6 * 3.0, "top", "4-pin I2C OLED display header + 4.7k SDA/SCL pull-ups", movable=False)
        add("J3", "Rear_LiPo_SolderPads", 6.4 * 2.6, "bottom", "Flat rear B.Cu gold solder pads for LiPo battery (keep_back_flat exempt)", movable=False)
        add("SW1", "SPDT_SlideSwitch_PCM12", 2.8 * 4.4, "top", "Edge power slide switch", movable=False)
        if "boot" in low or "reset" in low or "button" in low or parsed.explicit_width_mm == 26.0:
            add("SW2", "Tactile_Button_PTS810", 2.6 * 3.2, "top", "Edge BOOT tactile button", movable=False)
            add("SW3", "Tactile_Button_PTS810", 2.6 * 3.2, "top", "Edge RESET tactile button", movable=False)
        for idx in range(mounting_holes):
            crt = round(hole_diam + 0.8, 2)
            add(f"MH{idx+1}", f"NPTH_M{hole_diam:.1f}", crt * crt, "top", f"{hole_diam:.1f}mm corner NPTH mounting hole", movable=False)
        # 13 passives (CC1, CC2, PROG, LED_R, LED, SDA_PU, SCL_PU, EN_PU, 5x caps)
        for idx in range(13):
            add(f"P{idx+1}", passive_pkg, passive_area, "top", "Support passive (CC, PROG, I2C pull-up, decoupling)", movable=True)
        rationale.append("Added LIS3DH shake-wake interrupt -> RTC GPIO, SSD1306 I2C w/ 4.7k pull-ups, MCP73831 100mA charger, and USB-C 5.1k CC resistors.")
        return items, rationale

    # Generic / Sensor / Motor hub
    add("J1", "USB_C_6P" if "usb-c" in low or "usb" in low else "PinHeader_1x06", 8.2 * 4.8 if "usb" in low else 4.6 * 17.6, "top", "Host/USB edge connector", movable=False)
    if parsed.detected_intent == "motor_driver":
        add("U1", "ESP32-C3FN4", 6.8 * 6.8, "top", "Controller MCU", movable=True)
        add("U_DRV", "TMC2209" if "tmc" in low or "stepper" in low else "DRV8833", 7.0 * 7.0, "top", "Motor driver w/ thermal EP", movable=True)
    else:
        add("U1", "MCU_CORE", 8.6 * 7.0, "top", "Primary microcontroller", movable=True)
    if any(tok in low for tok in ("sensor", "bme280", "imu", "i2c", "env")):
        add("U2", "BME280", 6.0 * 5.8, "top", "I2C sensor", movable=True)
    if any(tok in low for tok in ("ldo", "3.3v", "buck", "reg", "power")) or True:
        add("U3", "AP2112K-3.3", 6.0 * 5.6, "top", "3.3V regulator", movable=True)
    for idx in range(7):
        add(f"P{idx+1}", passive_pkg, passive_area, "top", "Decoupling, pull-ups, status LEDs", movable=True)
    for idx in range(mounting_holes):
        crt = round(hole_diam + 0.8, 2)
        add(f"MH{idx+1}", f"NPTH_M{hole_diam:.1f}", crt * crt, "top", "Mounting hole", movable=False)
    return items, rationale


def negotiate_constraints(
    prompt: str,
    width_mm: float | None = None,
    height_mm: float | None = None,
    thickness_mm: float | None = None,
    layers: int | None = None,
    single_sided: bool | None = None,
    mounting_holes: int | None = None,
    hole_diameter_mm: float | None = None,
    passive_package: str | None = None,
    wireless_choice: str | None = None,
    user_answers: dict[str, Any] | None = None,
    tradeoff_choice: str | None = None,
    interactive: bool = False,
) -> ConstraintNegotiationResult:
    """Run the rev7 Interactive Pre-Flight Constraint & Trade-Off Gate.

    1. Checks whether critical physical envelope/stackup specs are missing when `interactive=True`
       or when a vague wearable brief lacks dimensions; asks targeted questions instead of guessing.
    2. Computes exact courtyard area utilization `Util = sum(Courtyard) / UsableArea`.
    3. Rejects silent overcrowding (`Util > 0.72` or individual component > board envelope) and
       returns quantified engineering trade-offs (stackup 4L/6L, double-sided, expand WxH, shrink passives).
    """
    if not prompt or not prompt.strip():
        raise ValueError("Prompt must be a non-empty hardware specification")

    parsed = parse_prompt_constraints(prompt)
    ans = dict(user_answers or {})

    eff_w = width_mm or ans.get("board_width_mm") or parsed.explicit_width_mm
    eff_h = height_mm or ans.get("board_height_mm") or parsed.explicit_height_mm
    eff_thk = thickness_mm or ans.get("board_thickness_mm") or parsed.explicit_thickness_mm
    eff_layers = layers or ans.get("layers") or parsed.explicit_layers
    eff_single = (
        single_sided
        if single_sided is not None
        else ans.get("single_sided", parsed.explicit_single_sided)
    )
    eff_holes = (
        mounting_holes
        if mounting_holes is not None
        else ans.get("mounting_holes", parsed.explicit_mounting_holes)
    )
    eff_hole_d = float(hole_diameter_mm or ans.get("hole_diameter_mm") or parsed.explicit_hole_diameter_mm or 2.0)
    eff_wireless = wireless_choice or ans.get("wireless_choice") or parsed.explicit_wireless_choice
    eff_pkg = str(passive_package or ans.get("passive_package") or parsed.explicit_passive_pkg or (
        "0402" if parsed.detected_intent == "wearable_keychain" else "0603"
    ))
    chosen_tradeoff = (tradeoff_choice or ans.get("tradeoff_choice") or "").strip().lower()

    # Check early whether an explicitly supplied board envelope already overflows physical limits
    pre_infeasible = False
    if eff_w is not None and eff_h is not None and not chosen_tradeoff:
        pre_margin = 1.0 if max(float(eff_w), float(eff_h)) <= 32.0 else 1.5
        pre_usable = max(2.0, float(eff_w) - 2.0 * pre_margin) * max(2.0, float(eff_h) - 2.0 * pre_margin)
        pre_single = bool(eff_single) if eff_single is not None else (parsed.detected_intent == "wearable_keychain")
        pre_holes = int(eff_holes) if eff_holes is not None else (2 if parsed.detected_intent == "wearable_keychain" else 0)
        pre_planned, _ = _estimate_component_budget(
            prompt, parsed, eff_pkg, eff_wireless or "esp32_c3fn4_qfn", pre_holes, eff_hole_d, pre_single
        )
        pre_top = sum(it["courtyard_area_mm2"] for it in pre_planned if it["side"] == "top")
        pre_util = (pre_top / pre_usable) if pre_single else (0.65 * pre_top / pre_usable)
        pre_max_item = max((it["courtyard_area_mm2"] for it in pre_planned), default=0.0)
        pre_infeasible = pre_util > MAX_COMFORTABLE_UTILIZATION or pre_max_item > pre_usable

    # Gate 1: Ask targeted questions when critical physical specs are missing under interactive mode
    # (Skip Gate 1 when explicit dimensions already fail Gate 2 so user immediately sees trade-offs.)
    questions: list[ConstraintQuestion] = []
    if interactive and not pre_infeasible:
        if eff_w is None or eff_h is None or eff_thk is None:
            questions.append(
                ConstraintQuestion(
                    question_id="board_envelope",
                    field="board_width_mm,board_height_mm,board_thickness_mm,mounting_holes",
                    prompt=(
                        "What are your maximum board dimensions (W x H in mm), thickness (0.8mm vs 1.6mm), "
                        "and do you need corner mounting holes (M2 / M2.5 / M3)?"
                    ),
                    options=[
                        "26x26mm, 0.8mm FR4, 2x 2.0mm corner NPTH holes",
                        "32x32mm, 1.6mm FR4, 4x 2.0mm corner NPTH holes",
                        "50x50mm, 1.6mm FR4, no mounting holes",
                    ],
                    recommended="26x26mm, 0.8mm FR4, 2x 2.0mm corner NPTH holes",
                    rationale="Physical enclosure dimensions and PCB thickness cannot be safely assumed.",
                )
            )
        if eff_single is None or eff_layers is None:
            questions.append(
                ConstraintQuestion(
                    question_id="placement_stackup",
                    field="single_sided,layers",
                    prompt=(
                        "Do you require single-sided SMD placement (keep_back_flat for flush rear battery/enclosure) "
                        "or double-sided SMD, and a 2-layer vs 4-layer stackup?"
                    ),
                    options=[
                        "Single-sided (keep_back_flat + rear LiPo pads), 2-Layer",
                        "Double-sided SMD, 2-Layer",
                        "Single-sided (keep_back_flat), 4-Layer HDI",
                    ],
                    recommended="Single-sided (keep_back_flat + rear LiPo pads), 2-Layer",
                    rationale="Rear battery mounting requires keeping the bottom surface flat.",
                )
            )
        if parsed.has_wireless_intent and eff_wireless is None:
            questions.append(
                ConstraintQuestion(
                    question_id="wireless_architecture",
                    field="wireless_choice",
                    prompt=(
                        "For wireless Wi-Fi/BLE, do you want the compact ESP32-C3FN4 QFN-32 + 40MHz crystal "
                        "(saves space, tighter RF layout) or the pre-certified ESP32-C3-MINI-1 module "
                        "(easier compliance, larger footprint)?"
                    ),
                    options=[
                        "esp32_c3fn4_qfn (ESP32-C3FN4 QFN-32 + 40MHz XTAL + Chip Antenna)",
                        "esp32_c3_mini (ESP32-C3-MINI-1 Pre-Certified Module)",
                    ],
                    recommended="esp32_c3fn4_qfn (ESP32-C3FN4 QFN-32 + 40MHz XTAL + Chip Antenna)",
                    rationale="Module vs bare QFN SoC strongly impacts board area and RF keepouts.",
                )
            )

    if questions:
        preview_w = float(eff_w or 26.0)
        preview_h = float(eff_h or 26.0)
        preview_items, rationale = _estimate_component_budget(
            prompt, parsed, eff_pkg, eff_wireless or "esp32_c3fn4_qfn",
            int(eff_holes if eff_holes is not None else 2), eff_hole_d,
            bool(eff_single if eff_single is not None else True),
        )
        return ConstraintNegotiationResult(
            status="needs_input",
            feasible=False,
            prompt=prompt,
            intent=parsed.detected_intent,
            locked_specs={
                "board_width_mm": eff_w,
                "board_height_mm": eff_h,
                "board_thickness_mm": eff_thk,
                "layers": eff_layers,
                "single_sided": eff_single,
                "mounting_holes": eff_holes,
                "wireless_choice": eff_wireless,
                "passive_package": eff_pkg,
            },
            courtyard_area_mm2=sum(it["courtyard_area_mm2"] for it in preview_items if it["side"] == "top"),
            usable_area_mm2=max(1.0, (preview_w - 2.0) * (preview_h - 2.0)),
            area_utilization=0.0,
            questions=questions,
            planned_components=preview_items,
            rationale=rationale,
        )

    # Fill resolved defaults once non-interactive or user-confirmed
    default_w = 80.0 if parsed.detected_intent == "fpga_protocol_bridge" else 50.0
    default_h = 64.0 if parsed.detected_intent == "fpga_protocol_bridge" else 50.0
    default_layers = 4 if parsed.detected_intent == "fpga_protocol_bridge" else 2
    final_w = float(eff_w or default_w)
    final_h = float(eff_h or default_h)
    final_thk = float(eff_thk or (0.8 if final_w <= 32.0 and final_h <= 32.0 else 1.6))
    final_layers = int(eff_layers or default_layers)
    final_single = bool(eff_single) if eff_single is not None else (parsed.detected_intent == "wearable_keychain")
    final_holes = int(eff_holes) if eff_holes is not None else (2 if parsed.detected_intent == "wearable_keychain" else 0)
    final_wireless = eff_wireless or "esp32_c3fn4_qfn"

    # Apply explicit trade-off selection if supplied
    chosen_tradeoff = (tradeoff_choice or ans.get("tradeoff_choice") or "").strip().lower()
    planned, rationale = _estimate_component_budget(
        prompt, parsed, eff_pkg, final_wireless, final_holes, eff_hole_d, final_single
    )

    margin = 1.0 if max(final_w, final_h) <= 32.0 else 1.5
    usable_w = max(2.0, final_w - 2.0 * margin)
    usable_h = max(2.0, final_h - 2.0 * margin)
    usable_area = usable_w * usable_h

    def _compute_util(
        items: list[dict[str, Any]], u_area: float, is_single: bool
    ) -> tuple[float, float]:
        top_area = sum(it["courtyard_area_mm2"] for it in items if it["side"] == "top")
        if is_single:
            return top_area, top_area / max(1.0, u_area)
        # Double-sided SMD splits movable passives/ICs cleanly onto B.Cu
        movable_area = sum(it["courtyard_area_mm2"] for it in items if it["side"] == "top" and it["movable_to_bottom"])
        fixed_top = top_area - movable_area
        effective_top = fixed_top + 0.55 * movable_area
        return top_area, max(effective_top, 0.55 * movable_area) / max(1.0, u_area)

    courtyard_area, util = _compute_util(planned, usable_area, final_single)

    if chosen_tradeoff == "expand_board":
        needed_usable = courtyard_area / 0.55
        aspect = max(1.15, final_w / max(1.0, final_h))
        rec_h = math.ceil((math.sqrt(needed_usable / max(0.5, aspect)) + 2.0 * 1.5) / 5.0) * 5.0
        rec_w = math.ceil((rec_h * aspect) / 5.0) * 5.0
        if parsed.detected_intent == "relay_controller":
            n_ch = max(1, parsed.relay_channels or 4)
            rec_w = max(rec_w, 95.0)
            rec_h = max(rec_h, float(math.ceil((n_ch * 19.5 + 10.0) / 5.0) * 5.0))
        final_w = max(final_w, float(rec_w))
        final_h = max(final_h, float(rec_h))
        margin = 1.0 if max(final_w, final_h) <= 32.0 else 1.5
        usable_area = max(2.0, final_w - 2.0 * margin) * max(2.0, final_h - 2.0 * margin)
        courtyard_area, util = _compute_util(planned, usable_area, final_single)
        rationale.append(f"Applied trade-off: expanded board envelope to {final_w:.0f}x{final_h:.0f}mm (utilization {util * 100:.1f}%).")
    elif chosen_tradeoff in ("upgrade_4layer", "4l", "4-layer"):
        final_layers = 4
        rationale.append("Applied trade-off: upgraded stackup from 2-Layer to 4-Layer HDI (In1.GND / In2.PWR).")
    elif chosen_tradeoff in ("upgrade_6layer", "6l", "6-layer"):
        final_layers = 6
        rationale.append("Applied trade-off: upgraded stackup to 6-Layer HDI.")
    elif chosen_tradeoff in ("double_sided", "double-sided"):
        final_single = False
        courtyard_area, util = _compute_util(planned, usable_area, False)
        rationale.append("Applied trade-off: enabled double-sided SMD placement across F.Cu and B.Cu.")
    elif chosen_tradeoff in ("shrink_passives", "0402", "0201"):
        eff_pkg = "0201" if eff_pkg == "0402" else "0402"
        planned, rationale = _estimate_component_budget(
            prompt, parsed, eff_pkg, final_wireless, final_holes, eff_hole_d, final_single
        )
        courtyard_area, util = _compute_util(planned, usable_area, final_single)
        rationale.append(f"Applied trade-off: downsized passives to {eff_pkg}.")

    warnings: list[str] = []
    if parsed.detected_intent == "relay_controller":
        warnings.append(
            "High-voltage isolation_creepage active: enforcing >=2.5mm primary-to-secondary air gap and stripping pours beneath relays/optocouplers."
        )

    # Gate 2: Feasibility / Congestion Check (Util > 0.72 triggers trade-offs!)
    max_single_item = max((it["courtyard_area_mm2"] for it in planned), default=0.0)
    if util > MAX_COMFORTABLE_UTILIZATION or max_single_item > usable_area:
        # Compute quantified options
        needed_usable = courtyard_area / 0.55
        aspect = max(1.15, final_w / max(1.0, final_h))
        rec_h = max(int(math.ceil(final_h / 5.0) * 5 + 5), int(math.ceil((math.sqrt(needed_usable / aspect) + 3.0) / 5.0) * 5))
        rec_w = max(int(math.ceil(final_w / 5.0) * 5 + 5), int(math.ceil((rec_h * aspect) / 5.0) * 5))
        if parsed.detected_intent == "relay_controller":
            n_ch = max(1, parsed.relay_channels or 4)
            rec_w = max(rec_w, 95)
            rec_h = max(rec_h, int(math.ceil((n_ch * 19.5 + 10.0) / 5.0) * 5))
        exp_usable = max(1.0, (rec_w - 3.0) * (rec_h - 3.0))
        exp_util = courtyard_area / exp_usable

        _, ds_util = _compute_util(planned, usable_area, False)
        shrunk_pkg = "0201" if eff_pkg == "0402" else "0402"
        shrunk_items, _ = _estimate_component_budget(
            prompt, parsed, shrunk_pkg, final_wireless, final_holes, eff_hole_d, final_single
        )
        shrunk_area, shrunk_util = _compute_util(shrunk_items, usable_area, final_single)
        saved_mm2 = max(0.0, courtyard_area - shrunk_area)

        # 4L/6L frees routing channels (~15% congestion relief) without shrinking physical component bodies
        l4_eff_util = util * 0.85

        tradeoffs = [
            TradeOffOption(
                option_id="expand_board",
                title=f"Expand Board Dimensions ({final_w:.0f}x{final_h:.0f}mm -> {rec_w}x{rec_h}mm)",
                description=(
                    f"Increases usable area from {usable_area:.0f} mm2 to {exp_usable:.0f} mm2, "
                    f"bringing component utilization from {util * 100:.1f}% down to {exp_util * 100:.1f}%."
                ),
                projected_utilization=round(exp_util, 4),
                resolves_feasibility=(exp_util <= MAX_COMFORTABLE_UTILIZATION),
                patch={"board_width_mm": float(rec_w), "board_height_mm": float(rec_h)},
            ),
            TradeOffOption(
                option_id="upgrade_4layer",
                title="Upgrade 2-Layer -> 4-Layer / 6-Layer Stackup",
                description=(
                    "Moves GND and PWR to internal planes (In1.GND / In2.PWR) for higher routing density"
                    + (" (note: physical relay bodies still require expanding board dimensions)." if util > 1.0 else ".")
                ),
                projected_utilization=round(l4_eff_util, 4),
                resolves_feasibility=(l4_eff_util <= MAX_COMFORTABLE_UTILIZATION and max_single_item <= usable_area),
                patch={"layers": 4 if final_layers < 4 else 6},
            ),
            TradeOffOption(
                option_id="double_sided",
                title="Enable Double-Sided SMD Placement (Top + Bottom)",
                description=(
                    f"Places decoupling caps and low-profile ICs on B.Cu; projected F.Cu utilization {ds_util * 100:.1f}%."
                ),
                projected_utilization=round(ds_util, 4),
                resolves_feasibility=(ds_util <= MAX_COMFORTABLE_UTILIZATION and max_single_item <= usable_area),
                patch={"single_sided": False},
            ),
            TradeOffOption(
                option_id="shrink_passives",
                title=f"Downsize Passives ({eff_pkg} -> {shrunk_pkg})",
                description=(
                    f"Saves {saved_mm2:.1f} mm2 of passive courtyard area (projected utilization {shrunk_util * 100:.1f}%)."
                ),
                projected_utilization=round(shrunk_util, 4),
                resolves_feasibility=(shrunk_util <= MAX_COMFORTABLE_UTILIZATION and max_single_item <= usable_area),
                patch={"passive_package": shrunk_pkg},
            ),
        ]

        warnings.append(
            f"Overcrowded enclosure: requested parts require {courtyard_area:.1f} mm2 inside {usable_area:.1f} mm2 usable area "
            f"({util * 100:.1f}% utilization > {MAX_COMFORTABLE_UTILIZATION * 100:.0f}% limit)."
        )
        return ConstraintNegotiationResult(
            status="tradeoff_required",
            feasible=False,
            prompt=prompt,
            intent=parsed.detected_intent,
            locked_specs={
                "board_width_mm": final_w,
                "board_height_mm": final_h,
                "board_thickness_mm": final_thk,
                "layers": final_layers,
                "single_sided": final_single,
                "keep_back_flat": final_single,
                "mounting_holes": final_holes,
                "hole_diameter_mm": eff_hole_d,
                "wireless_choice": final_wireless,
                "passive_package": eff_pkg,
            },
            courtyard_area_mm2=courtyard_area,
            usable_area_mm2=usable_area,
            area_utilization=util,
            tradeoffs=tradeoffs,
            planned_components=planned,
            warnings=warnings,
            rationale=rationale,
        )

    return ConstraintNegotiationResult(
        status="ready",
        feasible=True,
        prompt=prompt,
        intent=parsed.detected_intent,
        locked_specs={
            "board_width_mm": final_w,
            "board_height_mm": final_h,
            "board_thickness_mm": final_thk,
            "layers": final_layers,
            "single_sided": final_single,
            "keep_back_flat": final_single,
            "mounting_holes": final_holes,
            "hole_diameter_mm": eff_hole_d,
            "wireless_choice": final_wireless,
            "passive_package": eff_pkg,
        },
        courtyard_area_mm2=courtyard_area,
        usable_area_mm2=usable_area,
        area_utilization=util,
        planned_components=planned,
        warnings=warnings,
        rationale=rationale,
    )
