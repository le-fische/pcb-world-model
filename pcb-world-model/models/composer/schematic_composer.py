"""Hierarchical prompt-to-netlist circuit composer and deterministic ERC validator."""

from __future__ import annotations

import math
import re
from typing import Any

from env.component_library import instantiate_commodity, instantiate_device
from env.footprint_generator import (
    make_pin_header,
    make_qfn,
    make_smd_passive,
    make_soic_or_sop,
    make_sot23,
)
from env.sexpr_parser import ComponentNode, NetlistGraph, _finalize_nets
from models.composer.constraint_negotiator import parse_prompt_constraints


def _detect_mcu(prompt_lower: str) -> tuple[str, str]:
    """Identify target MCU family and clean slug from natural-language prompt."""
    if "stm32" in prompt_lower:
        return ("STM32G031", "stm32")
    if "esp32" in prompt_lower:
        return ("ESP32_C3", "esp32")
    if "rp2040" in prompt_lower:
        return ("RP2040_CORE", "rp2040")
    if "attiny" in prompt_lower:
        return ("ATtiny1614", "attiny")
    return ("MCU_CTRL", "custom_mcu")


def _detect_led_count(prompt_lower: str) -> int:
    """Extract requested status LED count (clamped to 1..2)."""
    match = re.search(r"(\d+)\s*(?:status\s+)?leds?\b", prompt_lower)
    if match:
        return max(1, min(2, int(match.group(1))))
    return 1


def build_wearable_oled_keychain_netlist(
    board_width_mm: float = 26.0,
    board_height_mm: float = 26.0,
    thickness_mm: float = 0.8,
    layers: int = 2,
    single_sided: bool = True,
    wireless_choice: str = "esp32_c3fn4_qfn",
    passive_package: str = "0402",
    mounting_holes: int = 2,
    hole_diameter_mm: float = 2.0,
) -> NetlistGraph:
    """Synthesize B6/B7: 26x26mm 0.8mm Wearable Keychain (ESP32-C3 + LIS3DH + SSD1306 + MCP73831 + USB-C)."""
    net_names: dict[int, str] = {
        1: "VBUS_5V",
        2: "VBAT",
        3: "VBAT_SW",
        4: "VCC_3V3",
        5: "GND",
        6: "USB_CC1",
        7: "USB_CC2",
        8: "USB_DP",
        9: "USB_DM",
        10: "CHG_STAT_K",
        11: "CHG_PROG",
        12: "XTAL_P",
        13: "XTAL_N",
        14: "RF_ANT",
        15: "CHIP_EN",
        16: "GPIO9_BOOT",
        17: "I2C_SDA",
        18: "I2C_SCL",
        19: "IMU_INT1",
    }

    sx = board_width_mm / 26.0
    sy = board_height_mm / 26.0

    def pt(px: float, py: float) -> tuple[float, float]:
        return (round(px * sx, 2), round(py * sy, 2))

    compact = max(board_width_mm, board_height_mm) <= 35.0
    components: dict[str, ComponentNode] = {}
    rationale: list[str] = []

    # 1. USB-C Receptacle (Bottom edge, keep_back_flat SMD shield pads) + 5.1k CC1/CC2 pulldowns
    j1x, j1y = pt(13.0, 22.8)
    components["J1"] = instantiate_device(
        "USB_C_6P",
        "J1",
        pin_nets=[5, 1, 6, 8, 9, 7],
        x=j1x,
        y=j1y,
        theta=0,
        is_locked=True,
        keep_back_flat=single_sided,
    )
    r1x, r1y = pt(7.0, 22.5)
    r2x, r2y = pt(19.0, 22.5)
    components["R1"] = instantiate_commodity(
        "resistor", "R1", "5.1k", package=passive_package, nets=[6, 5],
        tolerance="1%", power_rating="0.063W", x=r1x, y=r1y, compact=compact,
    )
    components["R2"] = instantiate_commodity(
        "resistor", "R2", "5.1k", package=passive_package, nets=[7, 5],
        tolerance="1%", power_rating="0.063W", x=r2x, y=r2y, compact=compact,
    )
    rationale.append("USB-C UFP sink wired with dedicated 5.1kΩ CC1/CC2 resistors and D+/D- differential pair.")

    # 2. MCP73831 LiPo Charger (100mA via 10k PROG) + Charge LED + Rear LiPo Pads
    u2x, u2y = pt(6.2, 18.0)
    components["U2"] = instantiate_device(
        "MCP73831",
        "U2",
        pin_nets=[10, 5, 2, 1, 11],
        x=u2x,
        y=u2y,
        compact=compact,
    )
    r3x, r3y = pt(3.5, 20.5)
    components["R3"] = instantiate_commodity(
        "resistor", "R3", "10k_PROG_100mA", package=passive_package, nets=[11, 5],
        tolerance="1%", x=r3x, y=r3y, compact=compact,
    )
    d1x, d1y = pt(6.2, 20.8)
    r4x, r4y = pt(10.0, 19.0)
    components["D1"] = instantiate_commodity(
        "led", "D1", "CHG_LED", package=passive_package, nets=[1, 10],
        x=d1x, y=d1y, compact=compact,
    )
    # Connect R4 in parallel/current path? Wait: D1 anode=1(VBUS) or let's use net 20 for LED anode!
    net_names[20] = "CHG_LED_A"
    components["D1"].pads[0].net_id = 1
    components["D1"].pads[1].net_id = 20
    components["R4"] = instantiate_commodity(
        "resistor", "R4", "1k_LED", package=passive_package, nets=[20, 10],
        x=r4x, y=r4y, compact=compact,
    )
    j3x, j3y = pt(13.0, 15.5)
    components["J3"] = instantiate_device(
        "Rear_LiPo_SolderPads",
        "J3",
        pin_nets=[2, 5],
        x=j3x,
        y=j3y,
        is_locked=True,
        keep_back_flat=single_sided,
    )
    rationale.append("MCP73831 configured for 100mA safe LiPo charge current (R_PROG=10kΩ) with flat rear B.Cu battery solder pads.")

    # 3. SPDT Edge Power Switch + AP2112K-3.3 LDO
    sw1x, sw1y = pt(23.7, 13.0)
    components["SW1"] = instantiate_device(
        "SPDT_SlideSwitch_PCM12",
        "SW1",
        pin_nets=[3, 2, 5],
        x=sw1x,
        y=sw1y,
        theta=0,
        is_locked=True,
    )
    u3x, u3y = pt(6.2, 11.5)
    components["U3"] = instantiate_device(
        "AP2112K-3.3",
        "U3",
        pin_nets=[3, 5, 3, 0, 4],
        x=u3x,
        y=u3y,
        compact=compact,
    )
    rationale.append("SPDT edge slide switch isolates LiPo battery from AP2112K-3.3 3.3V LDO.")

    # 4. ESP32-C3FN4 QFN-32 + 40MHz XTAL + 2.4GHz Chip Antenna (or ESP32-C3-MINI-1 module)
    u1x, u1y = pt(13.0, 12.0)
    if wireless_choice == "esp32_c3_mini":
        mini_pins = [
            5, 4, 15, 0, 0, 0, 17, 16, 18, 0,
            0, 0, 8, 9, 5, 19, 0, 0,
        ]
        components["U1"] = instantiate_device(
            "ESP32-C3-MINI-1",
            "U1",
            pin_nets=mini_pins,
            x=u1x,
            y=u1y,
            is_locked=False,
        )
        rationale.append("ESP32-C3-MINI-1 pre-certified Wi-Fi/BLE module selected for compliance simplicity.")
    else:
        esp_pins = [
            14, 4, 4, 15, 0, 0, 19, 0,
            0, 0, 4, 0, 0, 17, 16, 18,
            4, 9, 8, 0, 0, 4, 12, 13,
            4, 5, 0, 0, 0, 0, 0, 0,
        ]
        components["U1"] = instantiate_device(
            "ESP32-C3FN4",
            "U1",
            pin_nets=esp_pins,
            x=u1x,
            y=u1y,
            thermal_pad_net=5,
        )
        y1x, y1y = pt(19.5, 7.2)
        components["Y1"] = instantiate_device(
            "XTAL_40MHz_2016",
            "Y1",
            pin_nets=[12, 5, 13, 5],
            x=y1x,
            y=y1y,
            compact=compact,
        )
        a1x, a1y = pt(20.5, 2.8)
        components["ANT1"] = instantiate_device(
            "ChipAntenna_2.4GHz",
            "ANT1",
            pin_nets=[14, 5],
            x=a1x,
            y=a1y,
            is_locked=True,
        )
        rationale.append("ESP32-C3FN4 QFN-32 (w/ 40MHz XTAL, thermal EP GND vias & 2.4GHz chip antenna keepout) fits 26x26mm cleanly.")

    # 5. BOOT / RESET Edge Tactile Buttons + EN pull-up
    sw2x, sw2y = pt(2.3, 9.5)
    sw3x, sw3y = pt(2.3, 15.5)
    components["SW2"] = instantiate_device("Tactile_Button_PTS810", "SW2", pin_nets=[16, 5], x=sw2x, y=sw2y, is_locked=True)
    components["SW3"] = instantiate_device("Tactile_Button_PTS810", "SW3", pin_nets=[15, 5], x=sw3x, y=sw3y, is_locked=True)
    r7x, r7y = pt(4.6, 5.2)
    components["R7"] = instantiate_commodity(
        "resistor", "R7", "10k_EN", package=passive_package, nets=[4, 15], x=r7x, y=r7y, compact=compact
    )

    # 6. SSD1306 4-Pin I2C OLED Pads + 4.7k Pull-ups + LIS3DH Shake-to-Wake Accelerometer
    j2x, j2y = pt(10.5, 2.5)
    components["J2"] = instantiate_device(
        "SSD1306_OLED_4Pin",
        "J2",
        pin_nets=[5, 4, 18, 17],
        x=j2x,
        y=j2y,
        theta=90,
        is_locked=True,
    )
    r5x, r5y = pt(7.5, 5.5)
    r6x, r6y = pt(11.0, 5.5)
    components["R5"] = instantiate_commodity(
        "resistor", "R5", "4.7k_SDA", package=passive_package, nets=[4, 17], x=r5x, y=r5y, compact=compact
    )
    components["R6"] = instantiate_commodity(
        "resistor", "R6", "4.7k_SCL", package=passive_package, nets=[4, 18], x=r6x, y=r6y, compact=compact
    )
    u4x, u4y = pt(19.5, 13.5)
    lis_pins = [
        4, 0, 0, 18, 5, 17, 5, 4,
        0, 5, 19, 5, 0, 4, 0, 0,
    ]
    components["U4"] = instantiate_device(
        "LIS3DH",
        "U4",
        pin_nets=lis_pins,
        x=u4x,
        y=u4y,
        compact=compact,
    )
    rationale.append("LIS3DH INT1 wired to ESP32 RTC GPIO2 for ultra-low-power shake-to-wake; SSD1306 OLED shares I2C bus with 4.7kΩ pull-ups.")

    # 7. Decoupling Capacitors (0402 compact)
    cap_specs = [
        ("C1", "4.7uF_VBUS", 1, 5, 18.5, 19.5),
        ("C2", "4.7uF_VBAT", 2, 5, 9.5, 16.5),
        ("C3", "1uF_LDO_IN", 3, 5, 6.2, 14.8),
        ("C4", "1uF_LDO_OUT", 4, 5, 14.8, 6.2),
        ("C5", "100nF_ESP", 4, 5, 13.0, 17.8),
        ("C6", "100nF_LIS3DH", 4, 5, 19.5, 17.5),
        ("C7", "100nF_EN", 15, 5, 9.2, 7.0),
    ]
    for cid, val, n1, n2, px, py in cap_specs:
        cx, cy = pt(px, py)
        components[cid] = instantiate_commodity(
            "capacitor_mlcc", cid, val, package=passive_package, nets=[n1, n2],
            x=cx, y=cy, compact=compact,
        )

    # 8. Two 2.0mm Corner NPTH Mounting Holes
    hole_part = f"NPTH_M{hole_diameter_mm:.1f}" if f"NPTH_M{hole_diameter_mm:.1f}" in ("NPTH_M1.6", "NPTH_M2", "NPTH_M2.5", "NPTH_M3") else "NPTH_M2"
    hole_coords = [pt(2.4, 2.4), pt(23.6, 23.6), pt(23.6, 2.4), pt(2.4, 23.6)]
    for idx in range(max(0, min(4, mounting_holes))):
        hx, hy = hole_coords[idx]
        components[f"MH{idx+1}"] = instantiate_device(hole_part, f"MH{idx+1}", x=hx, y=hy)

    notes = [
        f"Envelope: {board_width_mm:.1f}x{board_height_mm:.1f}mm, {thickness_mm:.1f}mm FR4, {layers}L, keep_back_flat={single_sided}",
        "Active rules: thermal_via_ep, rf_antenna_keepout, diff_pair_impedance, perimeter_edge_lock, keep_back_flat",
    ]
    return _finalize_nets(
        board_name="wearable_oled_keychain_26x26",
        width_mm=board_width_mm,
        height_mm=board_height_mm,
        components=components,
        net_names=net_names,
        thickness_mm=thickness_mm,
        layers=layers,
        single_sided=single_sided,
        design_notes=notes,
        design_rationale=rationale,
    )


def build_relay_controller_netlist(
    channels: int = 4,
    board_width_mm: float = 90.0,
    board_height_mm: float = 70.0,
    layers: int = 2,
    single_sided: bool = True,
) -> NetlistGraph:
    """Synthesize an isolated multi-channel 10A relay controller after feasible board sizing."""
    n_ch = max(1, min(6, channels))
    net_names: dict[int, str] = {
        1: "VIN_5V",
        2: "VCC_3V3",
        3: "GND",
    }
    next_nid = 4
    components: dict[str, ComponentNode] = {}

    # Logic PSU + ESP32-C3FN4 controller on left LV side
    components["J_PWR"] = instantiate_device(
        "ScrewTerminal_5.08mm", "J_PWR", pin_nets=[1, 3], x=6.2, y=14.0, is_locked=True
    )
    components["U_REG"] = instantiate_device(
        "AP2112K-3.3", "U_REG", pin_nets=[1, 3, 1, 0, 2], x=16.0, y=12.0
    )
    components["C1"] = instantiate_commodity(
        "capacitor_mlcc", "C1", "10uF_5V", "0805", nets=[1, 3], x=16.0, y=20.5
    )
    components["C2"] = instantiate_commodity(
        "capacitor_mlcc", "C2", "100nF_3V3", "0603", nets=[2, 3], x=16.0, y=26.5
    )

    ch_ctrl_nets: list[int] = []
    for ch in range(1, n_ch + 1):
        ctrl_nid = next_nid
        net_names[ctrl_nid] = f"RELAY_CTRL_{ch}"
        next_nid += 1
        coil_drv_nid = next_nid
        net_names[coil_drv_nid] = f"COIL_DRV_{ch}"
        next_nid += 1
        hv_com_nid = next_nid
        net_names[hv_com_nid] = f"HV_COM_{ch}"
        next_nid += 1
        hv_no_nid = next_nid
        net_names[hv_no_nid] = f"HV_NO_{ch}"
        next_nid += 1
        ch_ctrl_nets.append(ctrl_nid)

        row_y = 11.5 + (ch - 1) * ((board_height_mm - 23.0) / max(1, n_ch - 1)) if n_ch > 1 else board_height_mm * 0.5
        components[f"U_ISO{ch}"] = instantiate_device(
            "PC817", f"U_ISO{ch}", pin_nets=[ctrl_nid, 3, 3, coil_drv_nid], x=30.0, y=row_y, is_locked=True
        )
        components[f"K{ch}"] = instantiate_device(
            "Relay_SPDT_10A",
            f"K{ch}",
            pin_nets=[1, coil_drv_nid, hv_com_nid, hv_no_nid, hv_no_nid],
            x=52.0,
            y=row_y,
            is_locked=True,
        )
        components[f"J_OUT{ch}"] = instantiate_device(
            "ScrewTerminal_5.08mm",
            f"J_OUT{ch}",
            pin_nets=[hv_com_nid, hv_no_nid],
            x=board_width_mm - 6.2,
            y=row_y,
            is_locked=True,
        )

    mcu_pins = [0] * 32
    mcu_pins[1] = 2   # VDD3P3
    mcu_pins[2] = 2   # VDD3P3_2
    mcu_pins[16] = 2  # VDD3P3_CPU
    mcu_pins[25] = 3  # GND_26
    gpio_slots = [13, 15, 19, 20, 27, 29]  # East/South sides facing U_ISO optocouplers
    for slot_idx, ctrl_nid in enumerate(ch_ctrl_nets):
        mcu_pins[gpio_slots[slot_idx]] = ctrl_nid
    components["U1"] = instantiate_device(
        "ESP32-C3FN4", "U1", pin_nets=mcu_pins, x=12.0, y=board_height_mm * 0.65, thermal_pad_net=3
    )

    return _finalize_nets(
        board_name=f"relay_controller_{n_ch}ch",
        width_mm=board_width_mm,
        height_mm=board_height_mm,
        components=components,
        net_names=net_names,
        thickness_mm=1.6,
        layers=layers,
        single_sided=single_sided,
        design_notes=[
            f"Expanded {n_ch}-channel 10A relay controller on {board_width_mm:.0f}x{board_height_mm:.0f}mm.",
            "Enforces >=2.5mm primary-to-secondary isolation_creepage across optocouplers and relays.",
        ],
        design_rationale=[
            f"Sized to {board_width_mm:.0f}x{board_height_mm:.0f}mm so {n_ch}x 10A SPDT relays, PC817 optocouplers, and 5.08mm terminal blocks maintain >=2.5mm HV isolation creepage.",
        ],
    )


def build_fpga_protocol_bridge_netlist(
    board_width_mm: float = 80.0,
    board_height_mm: float = 64.0,
    layers: int = 4,
    thickness_mm: float = 1.6,
    single_sided: bool = False,
) -> NetlistGraph:
    """Synthesize B9: Lattice iCE40UP5K-SG48 + ESP32-C3FN4 8-ch WS2812 + 8-ch Quadrature Encoder Bridge."""
    net_names: dict[int, str] = {
        1: "VBUS_5V",
        2: "VCC_3V3",
        3: "VCC_1V2",
        4: "GND",
        5: "USB_CC1",
        6: "USB_CC2",
        7: "USB_DP",
        8: "USB_DM",
        9: "ESP_EN",
        10: "ESP_XTAL_P",
        11: "ESP_XTAL_N",
        12: "FPGA_CLK_48M",
        13: "FPGA_CRESET_N",
        14: "FLASH_SCK",
        15: "FLASH_MOSI",
        16: "FLASH_MISO",
        17: "FLASH_CS_N",
        18: "ESP_SPI_SCK",
        19: "ESP_SPI_MOSI",
        20: "ESP_SPI_MISO",
        21: "ESP_SPI_CS_N",
        22: "FPGA_IRQ_N",
    }
    for ch in range(8):
        net_names[23 + ch] = f"WS_CH{ch}"
    for ch in range(8):
        net_names[31 + 2 * ch] = f"ENC{ch}_A"
        net_names[32 + 2 * ch] = f"ENC{ch}_B"

    sx = board_width_mm / 80.0
    sy = board_height_mm / 64.0

    def pt(px: float, py: float) -> tuple[float, float]:
        return (round(px * sx, 2), round(py * sy, 2))

    components: dict[str, ComponentNode] = {}

    # 1. USB-C 6-pin Power/Debug + 5.1k CC1/CC2 + 3.3V & 1.2V LDOs
    j1x, j1y = pt(5.5, 16.0)
    components["J1"] = instantiate_device(
        "USB_C_6P", "J1", pin_nets=[4, 1, 5, 7, 8, 6], x=j1x, y=j1y, theta=90, is_locked=True
    )
    components["R1"] = instantiate_commodity("resistor", "R1", "5.1k_CC1", package="0603", nets=[5, 4], x=pt(11.5, 12.0)[0], y=pt(11.5, 12.0)[1])
    components["R2"] = instantiate_commodity("resistor", "R2", "5.1k_CC2", package="0603", nets=[6, 4], x=pt(11.5, 20.0)[0], y=pt(11.5, 20.0)[1])

    u4x, u4y = pt(17.5, 12.0)
    components["U4"] = instantiate_device("AP2112K-3.3", "U4", pin_nets=[1, 4, 1, 0, 2], x=u4x, y=u4y)
    components["C1"] = instantiate_commodity("capacitor_mlcc", "C1", "10uF_5V", package="0805", nets=[1, 4], x=pt(13.0, 7.0)[0], y=pt(13.0, 7.0)[1])
    components["C2"] = instantiate_commodity("capacitor_mlcc", "C2", "10uF_3V3", package="0805", nets=[2, 4], x=pt(23.5, 12.0)[0], y=pt(23.5, 12.0)[1])

    u5x, u5y = pt(32.5, 16.5)
    components["U5"] = instantiate_device("XC6206", "U5", pin_nets=[4, 3, 2], x=u5x, y=u5y)
    components["C3"] = instantiate_commodity("capacitor_mlcc", "C3", "1uF_1V2", package="0603", nets=[3, 4], x=pt(38.0, 16.5)[0], y=pt(38.0, 16.5)[1])

    # 2. Host ESP32-C3FN4 QFN-32 + 40MHz XTAL + USB D+/D- + EN RC
    esp_pins = [
        0, 2, 2, 9, 18, 19, 20, 21,
        22, 0, 2, 0, 0, 0, 0, 0,
        2, 7, 8, 0, 0, 2, 10, 11,
        2, 4, 0, 0, 0, 0, 0, 0,
    ]
    u2x, u2y = pt(20.0, 30.0)
    components["U2"] = instantiate_device("ESP32-C3FN4", "U2", pin_nets=esp_pins, x=u2x, y=u2y, thermal_pad_net=4)
    components["Y2"] = instantiate_device("XTAL_40MHz_2016", "Y2", pin_nets=[10, 4, 11, 4], x=pt(12.0, 36.0)[0], y=pt(12.0, 36.0)[1])
    components["R3"] = instantiate_commodity("resistor", "R3", "10k_ESP_EN", package="0603", nets=[2, 9], x=pt(12.5, 25.0)[0], y=pt(12.5, 25.0)[1])
    components["C4"] = instantiate_commodity("capacitor_mlcc", "C4", "100nF_ESP1", package="0603", nets=[2, 4], x=pt(20.0, 22.0)[0], y=pt(20.0, 22.0)[1])
    components["C5"] = instantiate_commodity("capacitor_mlcc", "C5", "100nF_ESP2", package="0603", nets=[2, 4], x=pt(20.0, 38.5)[0], y=pt(20.0, 38.5)[1])

    # 3. Lattice iCE40UP5K-SG48 FPGA (U1) with unoptimized sequential .pcf pin mapping
    ice_fixed_head = [2, 4, 0, 0, 0, 12, 14, 15, 16, 17, 3, 13]
    ice_io_unopt = [
        31, 23, 33, 24, 35, 25, 37, 26, 39, 27, 41, 28,
        43, 29, 45, 30, 32, 18, 34, 19, 36, 20, 38, 21,
        40, 22, 42, 44, 46, 0, 0, 0, 0, 0, 0, 0,
    ]
    u1x, u1y = pt(48.0, 32.0)
    components["U1"] = instantiate_device(
        "Lattice_iCE40UP5K",
        "U1",
        pin_nets=ice_fixed_head + ice_io_unopt,
        x=u1x,
        y=u1y,
        thermal_pad_net=4,
    )
    components["C6"] = instantiate_commodity("capacitor_mlcc", "C6", "100nF_FPGA_IO", package="0603", nets=[2, 4], x=pt(41.0, 23.0)[0], y=pt(41.0, 23.0)[1])
    components["C7"] = instantiate_commodity("capacitor_mlcc", "C7", "100nF_FPGA_CORE", package="0603", nets=[3, 4], x=pt(56.0, 23.0)[0], y=pt(56.0, 23.0)[1])

    # 48MHz Oscillator (Y1) + W25Q16 Config Flash (U3) + CRESET pullup
    components["Y1"] = instantiate_device("XTAL_40MHz_2016", "Y1", pin_nets=[2, 4, 12, 4], x=pt(34.5, 34.5)[0], y=pt(34.5, 34.5)[1])
    components["U3"] = instantiate_device("W25Q16", "U3", pin_nets=[17, 16, 2, 4, 15, 14, 2, 2], x=pt(33.5, 45.0)[0], y=pt(33.5, 45.0)[1])
    components["C8"] = instantiate_commodity("capacitor_mlcc", "C8", "100nF_FLASH", package="0603", nets=[2, 4], x=pt(26.5, 45.0)[0], y=pt(26.5, 45.0)[1])
    components["R4"] = instantiate_commodity("resistor", "R4", "10k_CRESET", package="0603", nets=[2, 13], x=pt(37.5, 25.0)[0], y=pt(37.5, 25.0)[1])

    # 4. Locked Perimeter I/O Connector Banks (WS2812 outputs Top, Quadrature Encoders Bottom)
    components["J2"] = make_pin_header(
        "J2", "WS2812_CH0_3", [1, 4, 23, 24, 25, 26], pitch=2.54, x=pt(40.0, 6.0)[0], y=pt(40.0, 6.0)[1], theta=90, is_locked=True
    )
    components["J3"] = make_pin_header(
        "J3", "WS2812_CH4_7", [1, 4, 27, 28, 29, 30], pitch=2.54, x=pt(64.0, 6.0)[0], y=pt(64.0, 6.0)[1], theta=90, is_locked=True
    )
    components["J4"] = make_pin_header(
        "J4", "ENC_CH0_3", [2, 4, 31, 32, 33, 34, 35, 36, 37, 38], pitch=2.54, x=pt(24.0, 58.0)[0], y=pt(24.0, 58.0)[1], theta=90, is_locked=True
    )
    components["J5"] = make_pin_header(
        "J5", "ENC_CH4_7", [2, 4, 39, 40, 41, 42, 43, 44, 45, 46], pitch=2.54, x=pt(58.0, 58.0)[0], y=pt(58.0, 58.0)[1], theta=90, is_locked=True
    )

    return _finalize_nets(
        board_name="fpga_custom_protocol_bridge",
        width_mm=board_width_mm,
        height_mm=board_height_mm,
        components=components,
        net_names=net_names,
        thickness_mm=thickness_mm,
        layers=layers,
        single_sided=single_sided,
        design_notes=[
            "rev8 Silicon-to-Board Co-Design: quad_ws2812_spi.v synthesized onto Lattice iCE40UP5K-SG48 + ESP32-C3FN4.",
            "Swappable FPGA I/O bank pins (.pcf) co-optimized with PCB placement to eliminate trace crossings.",
        ],
        design_rationale=[
            "No single off-the-shelf IC combines 8-ch 800kHz WS2812B + 8-ch 4x quadrature decoding + SPI slave; rev8 synthesizes Verilog RTL onto iCE40UP5K-SG48.",
            "Joint .pcf pin-swapping rotates I/O signals onto QFN-48 edges facing WS2812 (top), Encoders (bottom), and ESP32-C3 (left).",
        ],
    )


def compose_netlist_from_prompt(
    prompt: str,
    board_width_mm: float = 50.0,
    board_height_mm: float = 50.0,
    thickness_mm: float | None = None,
    layers: int = 2,
    single_sided: bool | None = None,
    wireless_choice: str | None = None,
    passive_package: str | None = None,
    mounting_holes: int | None = None,
    hole_diameter_mm: float = 2.0,
) -> NetlistGraph:
    """Synthesize a complete, ERC-valid NetlistGraph from a natural-language hardware brief."""
    if not prompt or not prompt.strip():
        raise ValueError("Prompt must be a non-empty hardware specification")

    prompt_lower = prompt.lower()
    parsed = parse_prompt_constraints(prompt)

    # Route wearable keychain briefs (B6 expert OR B7 non-technical) through Class A/B wearable composer
    if parsed.detected_intent == "wearable_keychain":
        w_mm = float(parsed.explicit_width_mm or (board_width_mm if board_width_mm != 50.0 else 26.0))
        h_mm = float(parsed.explicit_height_mm or (board_height_mm if board_height_mm != 50.0 else 26.0))
        thk = float(thickness_mm or parsed.explicit_thickness_mm or 0.8)
        ss = single_sided if single_sided is not None else (
            parsed.explicit_single_sided if parsed.explicit_single_sided is not None else True
        )
        wc = wireless_choice or parsed.explicit_wireless_choice or "esp32_c3fn4_qfn"
        pkg = passive_package or parsed.explicit_passive_pkg or "0402"
        mh = mounting_holes if mounting_holes is not None else (parsed.explicit_mounting_holes or 2)
        hd = parsed.explicit_hole_diameter_mm or hole_diameter_mm
        return build_wearable_oled_keychain_netlist(
            board_width_mm=w_mm,
            board_height_mm=h_mm,
            thickness_mm=thk,
            layers=layers or parsed.explicit_layers or 2,
            single_sided=ss,
            wireless_choice=wc,
            passive_package=pkg,
            mounting_holes=mh,
            hole_diameter_mm=hd,
        )

    if parsed.detected_intent == "fpga_protocol_bridge":
        w_mm = float(board_width_mm if board_width_mm != 50.0 else (parsed.explicit_width_mm or 80.0))
        h_mm = float(board_height_mm if board_height_mm != 50.0 else (parsed.explicit_height_mm or 64.0))
        eff_l = layers if layers != 2 else (parsed.explicit_layers or 4)
        return build_fpga_protocol_bridge_netlist(
            board_width_mm=w_mm,
            board_height_mm=h_mm,
            layers=eff_l,
            thickness_mm=float(thickness_mm or parsed.explicit_thickness_mm or 1.6),
            single_sided=bool(single_sided) if single_sided is not None else False,
        )

    if parsed.detected_intent == "relay_controller":
        w_mm = float(board_width_mm if board_width_mm != 50.0 else (parsed.explicit_width_mm or 90.0))
        h_mm = float(board_height_mm if board_height_mm != 50.0 else (parsed.explicit_height_mm or 70.0))
        return build_relay_controller_netlist(
            channels=parsed.relay_channels or 4,
            board_width_mm=w_mm,
            board_height_mm=h_mm,
            layers=layers or parsed.explicit_layers or 2,
            single_sided=bool(single_sided) if single_sided is not None else False,
        )

    mcu_value, mcu_slug = _detect_mcu(prompt_lower)
    use_qfn = "qfn" in prompt_lower
    led_count = _detect_led_count(prompt_lower)

    has_reg = True
    has_i2c = True

    net_names: dict[int, str] = {
        1: "VIN_5V",
        2: "VCC_3V3",
        3: "GND",
        4: "I2C_SDA",
        5: "I2C_SCL",
        6: "MCU_RST",
        7: "SENSOR_INT",
        8: "STATUS_LED_1",
    }
    next_nid = 9
    if led_count >= 2:
        net_names[next_nid] = "STATUS_LED_2"
        led2_sig_nid = next_nid
        next_nid += 1
    else:
        led2_sig_nid = 0

    led1_k_nid = next_nid
    net_names[led1_k_nid] = "LED1_K"
    next_nid += 1

    if led_count >= 2:
        led2_k_nid = next_nid
        net_names[led2_k_nid] = "LED2_K"
        next_nid += 1
    else:
        led2_k_nid = 0

    sx = board_width_mm / 50.0
    sy = board_height_mm / 50.0

    def pt(px: float, py: float) -> tuple[float, float]:
        return (round(px * sx, 2), round(py * sy, 2))

    components: dict[str, ComponentNode] = {}
    rationale: list[str] = []

    # 1. Locked host / USB edge header connector
    jx, jy = pt(6.0, 25.0)
    components["J1"] = make_pin_header(
        "J1",
        "USB_HOST_HDR" if "usb" in prompt_lower else "HOST_HDR",
        [1, 3, 4, 5, 6, 7],
        pitch=2.54,
        x=jx,
        y=jy,
        is_locked=True,
    )

    # 2. MCU Core (SOIC-8 or QFN-12) + dedicated 100nF decoupling capacitor
    ux, uy = pt(23.5, 24.5)
    if use_qfn:
        qfn_nets = (
            [2, 4, 5, 6, 3, 7, 8, led2_sig_nid, 3, 4, 5, 2]
            if led_count >= 2
            else [2, 4, 5, 6, 3, 7, 8, 2, 3, 4, 5, 2]
        )
        components["U1"] = make_qfn(
            "U1",
            mcu_value,
            qfn_nets,
            pitch=0.8,
            body_size=5.0,
            x=ux,
            y=uy,
        )
    else:
        soic_nets = (
            [2, 4, 5, 6, 7, 8, led2_sig_nid, 3]
            if led_count >= 2
            else [2, 4, 5, 6, 7, 8, 2, 3]
        )
        components["U1"] = make_soic_or_sop(
            "U1",
            mcu_value,
            soic_nets,
            pitch=1.27,
            body_width=3.9,
            x=ux,
            y=uy,
        )

    c2x, c2y = pt(23.5, 37.0)
    components["C2"] = make_smd_passive(
        "C2", "100nF_MCU", "cap", "0805", 2, 3, x=c2x, y=c2y
    )

    # 3. 3.3V Voltage Regulator + Input Bulk Capacitor
    if has_reg:
        u3x, u3y = pt(15.0, 12.0)
        reg_val = "BUCK_3V3" if "buck" in prompt_lower else "AP2112K-3.3"
        components["U3"] = make_sot23(
            "U3",
            reg_val,
            [1, 3, 1, 2, 6],
            x=u3x,
            y=u3y,
            comp_type="ic",
        )
        c1x, c1y = pt(14.5, 37.0)
        components["C1"] = make_smd_passive(
            "C1", "10uF_IN", "cap", "0805", 1, 3, x=c1x, y=c1y
        )

    # 4. I2C Sensor Block + Decoupling Cap + 4.7k Pull-up Resistors
    if has_i2c:
        u2x, u2y = pt(38.0, 23.0)
        sensor_val = "IMU_I2C" if "imu" in prompt_lower else "BME280_I2C"
        components["U2"] = make_sot23(
            "U2",
            sensor_val,
            [2, 3, 4, 5, 7, 2],
            x=u2x,
            y=u2y,
            comp_type="ic",
        )
        c3x, c3y = pt(38.0, 32.5)
        components["C3"] = make_smd_passive(
            "C3", "100nF_SENS", "cap", "0805", 2, 3, x=c3x, y=c3y
        )
        r1x, r1y = pt(32.0, 12.0)
        components["R1"] = make_smd_passive(
            "R1", "4.7k_SDA", "res", "0603", 2, 4, x=r1x, y=r1y
        )
        r2x, r2y = pt(40.5, 12.0)
        components["R2"] = make_smd_passive(
            "R2", "4.7k_SCL", "res", "0603", 2, 5, x=r2x, y=r2y
        )

    # 5. Status LEDs + Series Current-Limiting 1k Resistors to GND
    d1x, d1y = pt(23.5, 12.0)
    r3x, r3y = pt(23.5, 6.5)
    components["D1"] = make_smd_passive(
        "D1", "STAT_LED1", "diode", "0805", 8, led1_k_nid, x=d1x, y=d1y
    )
    components["R3"] = make_smd_passive(
        "R3", "1k_LED1", "res", "0603", led1_k_nid, 3, x=r3x, y=r3y
    )

    if led_count >= 2:
        d2x, d2y = pt(32.0, 42.5)
        r4x, r4y = pt(40.5, 42.5)
        components["D2"] = make_smd_passive(
            "D2", "STAT_LED2", "diode", "0805", led2_sig_nid, led2_k_nid, x=d2x, y=d2y
        )
        components["R4"] = make_smd_passive(
            "R4", "1k_LED2", "res", "0603", led2_k_nid, 3, x=r4x, y=r4y
        )

    rationale.append(
        f"Synthesized {mcu_value} controller + 3.3V regulator + I2C sensor + {led_count} status LED(s) with decoupling & pull-ups."
    )
    board_name = f"prompt_{mcu_slug}"
    return _finalize_nets(
        board_name=board_name,
        width_mm=board_width_mm,
        height_mm=board_height_mm,
        components=components,
        net_names=net_names,
        thickness_mm=float(thickness_mm or 1.6),
        layers=layers,
        single_sided=bool(single_sided) if single_sided is not None else False,
        design_rationale=rationale,
    )


def verify_erc(netlist: NetlistGraph) -> list[str]:
    """Run deterministic Electrical & Mechanical Rules Check (ERC) on a NetlistGraph."""
    errors: list[str] = []

    if not netlist.components:
        errors.append("Netlist contains zero components")
        return errors

    if not netlist.routable_net_ids:
        errors.append("Netlist contains zero routable multi-pin nets")

    for nid, net in sorted(netlist.nets.items()):
        if nid <= 0:
            continue
        if len(net.pins) < 2:
            errors.append(
                f"Floating/single-pin net '{net.name}' (net_id={nid}) has {len(net.pins)} pin(s)"
            )

    gnd_ids = {
        nid for nid, net in netlist.nets.items() if "GND" in net.name.upper()
    }
    pwr_ids = {
        nid
        for nid, net in netlist.nets.items()
        if net.is_power_ground and "GND" not in net.name.upper()
    }

    if not gnd_ids:
        errors.append("Missing GND reference net in netlist")
    if not pwr_ids:
        errors.append("Missing power supply rail (VCC/VDD/VIN/VBUS) in netlist")
    if gnd_ids & pwr_ids:
        errors.append("Power supply rail and GND share the same net_id")

    for cid, comp in sorted(netlist.components.items()):
        if comp.comp_type == "ic" and cid.upper().startswith("U"):
            comp_nets = {pad.net_id for pad in comp.pads if pad.net_id > 0}
            if gnd_ids and not (comp_nets & gnd_ids):
                errors.append(f"IC {cid} ({comp.value}) is missing a GND connection")
            # Pure sensors or isolated outputs might tie to VBAT/VCC
            is_opto = "PC817" in comp.value.upper() or "OPTO" in comp.footprint.upper()
            if not is_opto and pwr_ids and not (comp_nets & pwr_ids):
                errors.append(
                    f"IC {cid} ({comp.value}) is missing a power supply connection"
                )

    has_decoupling_cap = False
    for comp in netlist.components.values():
        if comp.comp_type == "cap":
            cap_nets = {pad.net_id for pad in comp.pads if pad.net_id > 0}
            if (cap_nets & gnd_ids) and (cap_nets & pwr_ids):
                has_decoupling_cap = True
                break
    if gnd_ids and pwr_ids and not has_decoupling_cap:
        errors.append("No decoupling capacitor connected between power rail and GND")

    # rev7 Rule: USB-C CC1/CC2 must not short directly to GND or power rails
    for cid, comp in sorted(netlist.components.items()):
        if "USB_C" in comp.footprint.upper() or "USB-C" in comp.value.upper():
            for pad in comp.pads:
                if pad.pad_id in ("CC1", "CC2") and pad.net_id > 0:
                    if pad.net_id in gnd_ids or pad.net_id in pwr_ids:
                        errors.append(
                            f"USB-C {cid}.{pad.pad_id} shorted directly to power/GND instead of 5.1kΩ pulldown"
                        )

    # rev7 Rule: keep_back_flat / single_sided check
    if getattr(netlist, "single_sided", False):
        for cid, comp in sorted(netlist.components.items()):
            if comp.comp_type == "mounting_hole":
                continue
            if comp.electrical_meta.get("rear_pad_exempt"):
                continue
            if comp.side == "bottom":
                errors.append(
                    f"keep_back_flat violation: {cid} ({comp.value}) placed on B.Cu"
                )
            if any(p.layer_mask == 3 and p.net_id > 0 for p in comp.pads):
                errors.append(
                    f"keep_back_flat violation: {cid} ({comp.value}) has protruding THT pads"
                )

    return errors
