"""S-Expression AST parser and Netlist Hypergraph data contracts for KiCad PCB files."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class Pad:
    """Physical copper pad attached to a component footprint."""

    pad_id: str
    rel_x: float
    rel_y: float
    width: float
    height: float
    net_id: int
    layer_mask: int  # Bitmask: 1=Top (F.Cu), 2=Bottom (B.Cu), 3=Through-hole (Both)
    net_name: str = ""
    shape: str = "rect"


@dataclass
class ComponentNode:
    """Placed or unplaced circuit component node in the Netlist Hypergraph."""

    comp_id: str
    width: float
    height: float
    pads: list[Pad]
    is_locked: bool = False
    x: float = 0.0
    y: float = 0.0
    theta: int = 0  # Discrete rotation: {0, 90, 180, 270} degrees
    footprint: str = "Package_SMD:SOIC-8"
    value: str = ""
    comp_type: str = "ic"  # "ic" | "cap" | "res" | "connector" | "inductor" | "diode"

    def effective_size(self) -> tuple[float, float]:
        """Return courtyard (width, height) in mm accounting for discrete 90-degree rotation."""
        norm_theta = int(self.theta) % 360
        if norm_theta in (90, 270):
            return (self.height, self.width)
        return (self.width, self.height)

    def pad_world_pos(self, pad: Pad) -> tuple[float, float]:
        """Compute absolute board (x, y) in mm for a pad given component pose (x, y, theta)."""
        rad = math.radians(self.theta % 360)
        cos_t = math.cos(rad)
        sin_t = math.sin(rad)
        # Rotate pad offset by theta around component center
        wx = self.x + (pad.rel_x * cos_t - pad.rel_y * sin_t)
        wy = self.y + (pad.rel_x * sin_t + pad.rel_y * cos_t)
        return (wx, wy)

    def pad_effective_size(self, pad: Pad) -> tuple[float, float]:
        """Return pad (width, height) in world axes after discrete rotation."""
        norm_theta = int(self.theta) % 360
        if norm_theta in (90, 270):
            return (pad.height, pad.width)
        return (pad.width, pad.height)


@dataclass
class NetInfo:
    """Electrical hyperedge connecting two or more component pads."""

    net_id: int
    name: str
    pins: list[tuple[str, str]] = field(default_factory=list)  # (comp_id, pad_id)
    is_power_ground: bool = False
    weight: float = 1.0


@dataclass
class NetlistGraph:
    """Heterogeneous circuit hypergraph G = (V_c, V_p, E_pin, E_net)."""

    board_name: str
    width_mm: float
    height_mm: float
    components: dict[str, ComponentNode]
    nets: dict[int, NetInfo]
    origin_x_mm: float = 0.0
    origin_y_mm: float = 0.0
    raw_segments: list[dict[str, Any]] = field(default_factory=list)
    raw_vias: list[dict[str, Any]] = field(default_factory=list)

    @property
    def routable_net_ids(self) -> list[int]:
        """Return sorted net IDs that connect at least 2 distinct pins."""
        return sorted(
            nid for nid, net in self.nets.items() if nid > 0 and len(net.pins) >= 2
        )

    def clone(self) -> NetlistGraph:
        """Create an independent deep copy of the netlist graph."""
        comps = {
            cid: ComponentNode(
                comp_id=c.comp_id,
                width=c.width,
                height=c.height,
                pads=[
                    Pad(
                        pad_id=p.pad_id,
                        rel_x=p.rel_x,
                        rel_y=p.rel_y,
                        width=p.width,
                        height=p.height,
                        net_id=p.net_id,
                        layer_mask=p.layer_mask,
                        net_name=p.net_name,
                        shape=p.shape,
                    )
                    for p in c.pads
                ],
                is_locked=c.is_locked,
                x=c.x,
                y=c.y,
                theta=c.theta,
                footprint=c.footprint,
                value=c.value,
                comp_type=c.comp_type,
            )
            for cid, c in self.components.items()
        }
        nets = {
            nid: NetInfo(
                net_id=n.net_id,
                name=n.name,
                pins=list(n.pins),
                is_power_ground=n.is_power_ground,
                weight=n.weight,
            )
            for nid, n in self.nets.items()
        }
        return NetlistGraph(
            board_name=self.board_name,
            width_mm=self.width_mm,
            height_mm=self.height_mm,
            components=comps,
            nets=nets,
            origin_x_mm=self.origin_x_mm,
            origin_y_mm=self.origin_y_mm,
            raw_segments=[dict(s) for s in self.raw_segments],
            raw_vias=[dict(v) for v in self.raw_vias],
        )


def _infer_comp_type(comp_id: str, footprint: str) -> str:
    """Infer component role for GNN node feature encoding."""
    uid = comp_id.upper()
    fp = footprint.lower()
    if uid.startswith("J") or "usb" in fp or "header" in fp or "conn" in fp:
        return "connector"
    if uid.startswith("C") or "cap" in fp:
        return "cap"
    if uid.startswith("R") or "res" in fp:
        return "res"
    if uid.startswith("L") or "inductor" in fp:
        return "inductor"
    if uid.startswith("D") or "diode" in fp or "led" in fp:
        return "diode"
    return "ic"


def _is_power_net_name(name: str) -> bool:
    """Check whether a net name represents power or ground."""
    upper = name.upper()
    return any(
        tok in upper for tok in ("GND", "VCC", "VDD", "VIN", "VOUT", "3V3", "5V", "PWR")
    )


class SExprParser:
    """Lightweight recursive-descent KiCad S-expression parser."""

    @staticmethod
    def tokenize(text: str) -> list[str]:
        """Tokenize an S-expression string into parentheses, quoted strings, and atoms."""
        tokens: list[str] = []
        i = 0
        n = len(text)
        while i < n:
            ch = text[i]
            if ch.isspace():
                i += 1
            elif ch in ("(", ")"):
                tokens.append(ch)
                i += 1
            elif ch == '"':
                j = i + 1
                chars: list[str] = []
                while j < n:
                    c = text[j]
                    if c == "\\" and j + 1 < n:
                        chars.append(text[j + 1])
                        j += 2
                    elif c == '"':
                        break
                    else:
                        chars.append(c)
                        j += 1
                tokens.append(f'"{"".join(chars)}"')
                i = j + 1
            else:
                j = i
                while j < n and not text[j].isspace() and text[j] not in ("(", ")"):
                    j += 1
                tokens.append(text[i:j])
                i = j
        return tokens

    @classmethod
    def parse_ast(cls, text: str) -> list[Any]:
        """Parse S-expression text into nested Python lists."""
        tokens = cls.tokenize(text)
        if not tokens:
            raise ValueError("Empty S-expression input")

        stack: list[list[Any]] = []
        root: list[Any] | None = None
        for tok in tokens:
            if tok == "(":
                node: list[Any] = []
                if stack:
                    stack[-1].append(node)
                stack.append(node)
            elif tok == ")":
                if not stack:
                    raise ValueError("Unbalanced ')' in S-expression")
                finished = stack.pop()
                if not stack:
                    root = finished
                    break
            else:
                val: Any
                if tok.startswith('"') and tok.endswith('"'):
                    val = tok[1:-1]
                else:
                    val = tok
                if not stack:
                    raise ValueError(f"Unexpected atom outside S-expression: {tok}")
                stack[-1].append(val)

        if stack or root is None:
            raise ValueError("Unclosed '(' in S-expression")
        return root

    @staticmethod
    def _find_child(node: list[Any], tag: str) -> list[Any] | None:
        for child in node:
            if isinstance(child, list) and child and child[0] == tag:
                return child
        return None

    @staticmethod
    def _find_children(node: list[Any], tag: str) -> list[list[Any]]:
        return [
            child for child in node if isinstance(child, list) and child and child[0] == tag
        ]

    @classmethod
    def parse_file(cls, path: str | Path) -> NetlistGraph:
        """Read and parse a `.kicad_pcb` file into a NetlistGraph."""
        p = Path(path)
        return cls.parse_text(p.read_text(encoding="utf-8"), board_name=p.stem)

    @classmethod
    def parse_text(cls, text: str, board_name: str = "parsed_board") -> NetlistGraph:
        """Parse `.kicad_pcb` S-expression content into a NetlistGraph."""
        ast = cls.parse_ast(text)
        if not ast or ast[0] != "kicad_pcb":
            raise ValueError("Expected top-level (kicad_pcb ...) S-expression")

        nets: dict[int, NetInfo] = {}
        for net_node in cls._find_children(ast, "net"):
            if len(net_node) >= 3:
                nid = int(float(net_node[1]))
                nname = str(net_node[2])
                if nid > 0:
                    nets[nid] = NetInfo(
                        net_id=nid,
                        name=nname,
                        pins=[],
                        is_power_ground=_is_power_net_name(nname),
                        weight=1.5 if _is_power_net_name(nname) else 1.0,
                    )

        # Parse Edge.Cuts outline bounds
        edge_xs: list[float] = []
        edge_ys: list[float] = []
        for rect_node in cls._find_children(ast, "gr_rect"):
            layer_node = cls._find_child(rect_node, "layer")
            if layer_node and len(layer_node) >= 2 and layer_node[1] == "Edge.Cuts":
                start_node = cls._find_child(rect_node, "start")
                end_node = cls._find_child(rect_node, "end")
                if start_node and end_node:
                    edge_xs.extend([float(start_node[1]), float(end_node[1])])
                    edge_ys.extend([float(start_node[2]), float(end_node[2])])
        for line_node in cls._find_children(ast, "gr_line"):
            layer_node = cls._find_child(line_node, "layer")
            if layer_node and len(layer_node) >= 2 and layer_node[1] == "Edge.Cuts":
                start_node = cls._find_child(line_node, "start")
                end_node = cls._find_child(line_node, "end")
                if start_node and end_node:
                    edge_xs.extend([float(start_node[1]), float(end_node[1])])
                    edge_ys.extend([float(start_node[2]), float(end_node[2])])

        if edge_xs and edge_ys:
            min_x, max_x = min(edge_xs), max(edge_xs)
            min_y, max_y = min(edge_ys), max(edge_ys)
            width_mm = max(10.0, max_x - min_x)
            height_mm = max(10.0, max_y - min_y)
        else:
            min_x, min_y = 0.0, 0.0
            width_mm, height_mm = 50.0, 50.0

        components: dict[str, ComponentNode] = {}
        for idx, fp_node in enumerate(cls._find_children(ast, "footprint")):
            footprint_name = str(fp_node[1]) if len(fp_node) > 1 else "Package_SMD:Custom"
            at_node = cls._find_child(fp_node, "at")
            cx = float(at_node[1]) - min_x if at_node and len(at_node) >= 3 else 10.0
            cy = float(at_node[2]) - min_y if at_node and len(at_node) >= 3 else 10.0
            theta = int(round(float(at_node[3]))) % 360 if at_node and len(at_node) >= 4 else 0

            # Extract Reference & Value from fp_text or property nodes
            comp_id = f"U{idx + 1}"
            comp_value = ""
            for prop in cls._find_children(fp_node, "property"):
                if len(prop) >= 3:
                    if prop[1] == "Reference":
                        comp_id = str(prop[2])
                    elif prop[1] == "Value":
                        comp_value = str(prop[2])
            for fp_text in cls._find_children(fp_node, "fp_text"):
                if len(fp_text) >= 3:
                    if fp_text[1] == "reference":
                        comp_id = str(fp_text[2])
                    elif fp_text[1] == "value":
                        comp_value = str(fp_text[2])

            locked = cls._find_child(fp_node, "locked") is not None or any(
                c == "locked" for c in fp_node
            )
            if not locked and comp_id.upper().startswith("J"):
                locked = True

            # Extract Courtyard bounding box if present
            cy_xs: list[float] = []
            cy_ys: list[float] = []
            for gr_tag in ("fp_rect", "fp_line"):
                for gr in cls._find_children(fp_node, gr_tag):
                    lyr = cls._find_child(gr, "layer")
                    if lyr and len(lyr) >= 2 and "CrtYd" in str(lyr[1]):
                        s = cls._find_child(gr, "start")
                        e = cls._find_child(gr, "end")
                        if s and e:
                            cy_xs.extend([float(s[1]), float(e[1])])
                            cy_ys.extend([float(s[2]), float(e[2])])

            pads: list[Pad] = []
            for p_node in cls._find_children(fp_node, "pad"):
                if len(p_node) < 4:
                    continue
                pad_id = str(p_node[1])
                pad_type = str(p_node[2])
                pad_shape = str(p_node[3])
                pat = cls._find_child(p_node, "at")
                psize = cls._find_child(p_node, "size")
                pnet = cls._find_child(p_node, "net")
                players = cls._find_child(p_node, "layers")

                px = float(pat[1]) if pat and len(pat) >= 3 else 0.0
                py = float(pat[2]) if pat and len(pat) >= 3 else 0.0
                pw = float(psize[1]) if psize and len(psize) >= 3 else 0.8
                ph = float(psize[2]) if psize and len(psize) >= 3 else 0.8
                nid = int(float(pnet[1])) if pnet and len(pnet) >= 2 else 0
                nname = str(pnet[2]) if pnet and len(pnet) >= 3 else ""

                layer_mask = 1
                if pad_type == "thru_hole":
                    layer_mask = 3
                elif players:
                    layer_tokens = [str(x) for x in players[1:]]
                    has_top = any(t in ("F.Cu", "*.Cu") for t in layer_tokens)
                    has_bot = any(t in ("B.Cu", "*.Cu") for t in layer_tokens)
                    layer_mask = (1 if has_top else 0) | (2 if has_bot else 0)
                    if layer_mask == 0:
                        layer_mask = 1

                pads.append(
                    Pad(
                        pad_id=pad_id,
                        rel_x=px,
                        rel_y=py,
                        width=pw,
                        height=ph,
                        net_id=nid,
                        layer_mask=layer_mask,
                        net_name=nname,
                        shape=pad_shape,
                    )
                )
                if nid > 0:
                    if nid not in nets:
                        nets[nid] = NetInfo(
                            net_id=nid,
                            name=nname or f"NET_{nid}",
                            pins=[],
                            is_power_ground=_is_power_net_name(nname),
                        )
                    if (comp_id, pad_id) not in nets[nid].pins:
                        nets[nid].pins.append((comp_id, pad_id))

            if cy_xs and cy_ys:
                c_width = max(1.5, max(cy_xs) - min(cy_xs))
                c_height = max(1.5, max(cy_ys) - min(cy_ys))
            elif pads:
                min_px = min(p.rel_x - p.width / 2 for p in pads)
                max_px = max(p.rel_x + p.width / 2 for p in pads)
                min_py = min(p.rel_y - p.height / 2 for p in pads)
                max_py = max(p.rel_y + p.height / 2 for p in pads)
                c_width = max(2.0, (max_px - min_px) + 1.0)
                c_height = max(2.0, (max_py - min_py) + 1.0)
            else:
                c_width, c_height = 4.0, 4.0

            # If theta is 90/270, parsed courtyard is already in unrotated local coordinates
            components[comp_id] = ComponentNode(
                comp_id=comp_id,
                width=round(c_width, 3),
                height=round(c_height, 3),
                pads=pads,
                is_locked=locked,
                x=round(cx, 3),
                y=round(cy, 3),
                theta=theta,
                footprint=footprint_name,
                value=comp_value,
                comp_type=_infer_comp_type(comp_id, footprint_name),
            )

        raw_segments: list[dict[str, Any]] = []
        for seg in cls._find_children(ast, "segment"):
            s = cls._find_child(seg, "start")
            e = cls._find_child(seg, "end")
            w = cls._find_child(seg, "width")
            lyr = cls._find_child(seg, "layer")
            net = cls._find_child(seg, "net")
            if s and e and net:
                layer_idx = 1 if (lyr and len(lyr) >= 2 and lyr[1] == "B.Cu") else 0
                raw_segments.append(
                    {
                        "net_id": int(float(net[1])),
                        "layer": layer_idx,
                        "x1_mm": float(s[1]) - min_x,
                        "y1_mm": float(s[2]) - min_y,
                        "x2_mm": float(e[1]) - min_x,
                        "y2_mm": float(e[2]) - min_y,
                        "width_mm": float(w[1]) if w and len(w) >= 2 else 0.25,
                    }
                )

        raw_vias: list[dict[str, Any]] = []
        for via in cls._find_children(ast, "via"):
            at = cls._find_child(via, "at")
            sz = cls._find_child(via, "size")
            dr = cls._find_child(via, "drill")
            net = cls._find_child(via, "net")
            if at and net:
                raw_vias.append(
                    {
                        "net_id": int(float(net[1])),
                        "x_mm": float(at[1]) - min_x,
                        "y_mm": float(at[2]) - min_y,
                        "diameter_mm": float(sz[1]) if sz and len(sz) >= 2 else 0.6,
                        "drill_mm": float(dr[1]) if dr and len(dr) >= 2 else 0.3,
                    }
                )

        return NetlistGraph(
            board_name=board_name,
            width_mm=width_mm,
            height_mm=height_mm,
            components=components,
            nets=nets,
            origin_x_mm=min_x,
            origin_y_mm=min_y,
            raw_segments=raw_segments,
            raw_vias=raw_vias,
        )


def _finalize_nets(
    board_name: str,
    width_mm: float,
    height_mm: float,
    components: dict[str, ComponentNode],
    net_names: dict[int, str],
) -> NetlistGraph:
    """Populate NetInfo hyperedges from component pad assignments."""
    nets: dict[int, NetInfo] = {}
    for nid, name in net_names.items():
        if nid <= 0:
            continue
        is_pwr = _is_power_net_name(name)
        nets[nid] = NetInfo(
            net_id=nid,
            name=name,
            pins=[],
            is_power_ground=is_pwr,
            weight=1.4 if is_pwr else 1.0,
        )
    for cid, comp in components.items():
        for pad in comp.pads:
            if pad.net_id > 0:
                pad.net_name = net_names.get(pad.net_id, f"NET_{pad.net_id}")
                if pad.net_id not in nets:
                    nets[pad.net_id] = NetInfo(
                        net_id=pad.net_id,
                        name=pad.net_name,
                        pins=[],
                        is_power_ground=_is_power_net_name(pad.net_name),
                    )
                nets[pad.net_id].pins.append((cid, pad.pad_id))
    return NetlistGraph(
        board_name=board_name,
        width_mm=width_mm,
        height_mm=height_mm,
        components=components,
        nets=nets,
    )


def _make_soic8_pads(nets: list[int]) -> list[Pad]:
    """Create 8-pin SOIC pad geometry (1.27mm pitch, 5.4mm row spacing)."""
    pads: list[Pad] = []
    ys = [-1.905, -0.635, 0.635, 1.905]
    # Pins 1..4 on left (-2.7mm), 5..8 on right (+2.7mm)
    for i, py in enumerate(ys):
        pads.append(Pad(str(i + 1), -2.7, py, 1.5, 0.6, nets[i], 1))
    for i, py in enumerate(reversed(ys)):
        pads.append(Pad(str(5 + i), 2.7, py, 1.5, 0.6, nets[4 + i], 1))
    return pads


def _make_two_pin_smd(net1: int, net2: int, pitch: float = 2.0) -> list[Pad]:
    """Create symmetric 2-pin SMD passive footprint pads (0805 / 0603)."""
    half = pitch / 2.0
    return [
        Pad("1", -half, 0.0, 0.9, 1.1, net1, 1),
        Pad("2", half, 0.0, 0.9, 1.1, net2, 1),
    ]


def _make_header_pads(nets: list[int], pitch: float = 2.54) -> list[Pad]:
    """Create a 1xN through-hole header connector."""
    n = len(nets)
    y0 = -0.5 * (n - 1) * pitch
    return [
        Pad(str(i + 1), 0.0, y0 + i * pitch, 1.7, 1.7, nid, 3, shape="circle")
        for i, nid in enumerate(nets)
    ]


PRESET_DESCRIPTIONS: dict[str, str] = {
    "mcu_sensor_node": "STM32/ATtiny MCU + I2C Env Sensor + 3.3V LDO + USB/Header (9 comps, 8 nets)",
    "buck_converter": "Synchronous DC-DC Buck Regulator + Power Inductor + Terminal Blocks (10 comps, 7 nets)",
    "usb_uart_bridge": "USB-C to UART Bridge + ESD Protection + Status LEDs (10 comps, 8 nets)",
}


def build_preset_board(preset_name: str = "mcu_sensor_node") -> NetlistGraph:
    """Build one of the realistic 50mm x 50mm 2-layer reference hardware designs."""
    if preset_name == "mcu_sensor_node":
        net_names = {
            1: "VIN_5V",
            2: "VCC_3V3",
            3: "GND",
            4: "I2C_SDA",
            5: "I2C_SCL",
            6: "MCU_RST",
            7: "SENSOR_INT",
            8: "STATUS_LED",
        }
        components = {
            # Fixed edge header connector (locked)
            "J1": ComponentNode(
                comp_id="J1",
                width=4.6,
                height=17.6,
                pads=_make_header_pads([1, 3, 4, 5, 6, 7]),
                is_locked=True,
                x=6.0,
                y=25.0,
                theta=0,
                footprint="Connector_PinHeader_2.54mm:PinHeader_1x06_P2.54mm_Vertical",
                value="I2C_HOST_HDR",
                comp_type="connector",
            ),
            # 8-pin MCU
            "U1": ComponentNode(
                comp_id="U1",
                width=8.6,
                height=7.0,
                pads=_make_soic8_pads([2, 4, 5, 6, 7, 8, 2, 3]),
                is_locked=False,
                x=25.0,
                y=24.0,
                theta=0,
                footprint="Package_SO:SOIC-8_3.9x4.9mm_P1.27mm",
                value="ATtiny1614",
                comp_type="ic",
            ),
            # 6-pin I2C Environment Sensor
            "U2": ComponentNode(
                comp_id="U2",
                width=6.0,
                height=5.8,
                pads=[
                    Pad("1", -1.8, -1.2, 1.0, 0.6, 2, 1),
                    Pad("2", -1.8, 0.0, 1.0, 0.6, 3, 1),
                    Pad("3", -1.8, 1.2, 1.0, 0.6, 4, 1),
                    Pad("4", 1.8, 1.2, 1.0, 0.6, 5, 1),
                    Pad("5", 1.8, 0.0, 1.0, 0.6, 7, 1),
                    Pad("6", 1.8, -1.2, 1.0, 0.6, 2, 1),
                ],
                is_locked=False,
                x=37.0,
                y=20.0,
                theta=0,
                footprint="Package_TO_SOT_SMD:SOT-23-6",
                value="BME280_I2C",
                comp_type="ic",
            ),
            # 5-pin 3.3V LDO Regulator
            "U3": ComponentNode(
                comp_id="U3",
                width=6.0,
                height=5.6,
                pads=[
                    Pad("1", -1.8, -1.2, 1.0, 0.6, 1, 1),
                    Pad("2", -1.8, 0.0, 1.0, 0.6, 3, 1),
                    Pad("3", -1.8, 1.2, 1.0, 0.6, 1, 1),
                    Pad("4", 1.8, 1.0, 1.0, 0.6, 2, 1),
                    Pad("5", 1.8, -1.0, 1.0, 0.6, 6, 1),
                ],
                is_locked=False,
                x=16.0,
                y=14.0,
                theta=0,
                footprint="Package_TO_SOT_SMD:SOT-23-5",
                value="AP2112K-3.3",
                comp_type="ic",
            ),
            # Decoupling Capacitors (0805)
            "C1": ComponentNode(
                comp_id="C1",
                width=4.4,
                height=2.8,
                pads=_make_two_pin_smd(1, 3),
                x=15.0,
                y=33.0,
                footprint="Capacitor_SMD:C_0805_2012Metric",
                value="10uF",
                comp_type="cap",
            ),
            "C2": ComponentNode(
                comp_id="C2",
                width=4.4,
                height=2.8,
                pads=_make_two_pin_smd(2, 3),
                x=25.0,
                y=36.0,
                footprint="Capacitor_SMD:C_0805_2012Metric",
                value="100nF",
                comp_type="cap",
            ),
            "C3": ComponentNode(
                comp_id="C3",
                width=4.4,
                height=2.8,
                pads=_make_two_pin_smd(2, 3),
                x=37.0,
                y=32.0,
                footprint="Capacitor_SMD:C_0805_2012Metric",
                value="100nF",
                comp_type="cap",
            ),
            # I2C Pull-Up & Status LED Resistors (0603)
            "R1": ComponentNode(
                comp_id="R1",
                width=4.2,
                height=2.6,
                pads=_make_two_pin_smd(2, 4),
                x=31.0,
                y=12.0,
                footprint="Resistor_SMD:R_0603_1608Metric",
                value="4.7k",
                comp_type="res",
            ),
            "R2": ComponentNode(
                comp_id="R2",
                width=4.2,
                height=2.6,
                pads=_make_two_pin_smd(2, 5),
                x=39.0,
                y=12.0,
                footprint="Resistor_SMD:R_0603_1608Metric",
                value="4.7k",
                comp_type="res",
            ),
            "R3": ComponentNode(
                comp_id="R3",
                width=4.2,
                height=2.6,
                pads=_make_two_pin_smd(8, 3),
                x=25.0,
                y=13.0,
                footprint="Resistor_SMD:R_0603_1608Metric",
                value="1k_LED",
                comp_type="res",
            ),
        }
        return _finalize_nets("mcu_sensor_node", 50.0, 50.0, components, net_names)

    if preset_name == "buck_converter":
        net_names = {
            1: "VIN_12V",
            2: "VOUT_5V",
            3: "GND",
            4: "SW_NODE",
            5: "BOOT",
            6: "FB_DIV",
            7: "EN_CTRL",
        }
        components = {
            "J1": ComponentNode(
                comp_id="J1",
                width=4.8,
                height=9.0,
                pads=_make_header_pads([1, 3, 7], pitch=2.54),
                is_locked=True,
                x=6.0,
                y=25.0,
                footprint="TerminalBlock:TerminalBlock_bornier-3_P5.08mm",
                value="VIN_12V",
                comp_type="connector",
            ),
            "J2": ComponentNode(
                comp_id="J2",
                width=4.8,
                height=7.0,
                pads=_make_header_pads([2, 3], pitch=2.54),
                is_locked=True,
                x=44.0,
                y=25.0,
                footprint="TerminalBlock:TerminalBlock_bornier-2_P5.08mm",
                value="VOUT_5V",
                comp_type="connector",
            ),
            "U1": ComponentNode(
                comp_id="U1",
                width=8.6,
                height=7.0,
                pads=_make_soic8_pads([5, 1, 4, 3, 6, 7, 1, 4]),
                x=21.0,
                y=24.0,
                footprint="Package_SO:SOIC-8_3.9x4.9mm_P1.27mm",
                value="TPS54302",
                comp_type="ic",
            ),
            "L1": ComponentNode(
                comp_id="L1",
                width=7.4,
                height=6.8,
                pads=[
                    Pad("1", -2.2, 0.0, 1.8, 3.0, 4, 1),
                    Pad("2", 2.2, 0.0, 1.8, 3.0, 2, 1),
                ],
                x=33.0,
                y=22.0,
                footprint="Inductor_SMD:L_6.3x6.3_H3",
                value="10uH",
                comp_type="inductor",
            ),
            "D1": ComponentNode(
                comp_id="D1",
                width=5.6,
                height=3.6,
                pads=_make_two_pin_smd(3, 4, pitch=2.8),
                x=27.0,
                y=35.0,
                footprint="Diode_SMD:D_SMA",
                value="SS34",
                comp_type="diode",
            ),
            "C1": ComponentNode(
                comp_id="C1",
                width=4.6,
                height=3.0,
                pads=_make_two_pin_smd(1, 3),
                x=14.0,
                y=14.0,
                footprint="Capacitor_SMD:C_1206_3216Metric",
                value="22uF_IN",
                comp_type="cap",
            ),
            "C2": ComponentNode(
                comp_id="C2",
                width=4.6,
                height=3.0,
                pads=_make_two_pin_smd(2, 3),
                x=37.0,
                y=34.0,
                footprint="Capacitor_SMD:C_1206_3216Metric",
                value="47uF_OUT",
                comp_type="cap",
            ),
            "C3": ComponentNode(
                comp_id="C3",
                width=4.2,
                height=2.6,
                pads=_make_two_pin_smd(5, 4),
                x=18.0,
                y=35.0,
                footprint="Capacitor_SMD:C_0603_1608Metric",
                value="100nF_BST",
                comp_type="cap",
            ),
            "R1": ComponentNode(
                comp_id="R1",
                width=4.2,
                height=2.6,
                pads=_make_two_pin_smd(2, 6),
                x=32.0,
                y=12.0,
                footprint="Resistor_SMD:R_0603_1608Metric",
                value="100k",
                comp_type="res",
            ),
            "R2": ComponentNode(
                comp_id="R2",
                width=4.2,
                height=2.6,
                pads=_make_two_pin_smd(6, 3),
                x=22.0,
                y=12.0,
                footprint="Resistor_SMD:R_0603_1608Metric",
                value="18.7k",
                comp_type="res",
            ),
        }
        return _finalize_nets("buck_converter", 50.0, 50.0, components, net_names)

    if preset_name == "usb_uart_bridge":
        net_names = {
            1: "VBUS_5V",
            2: "VCC_3V3",
            3: "GND",
            4: "USB_DP",
            5: "USB_DM",
            6: "UART_TX",
            7: "UART_RX",
            8: "LED_ACT",
        }
        components = {
            "J1": ComponentNode(
                comp_id="J1",
                width=5.2,
                height=12.2,
                pads=_make_header_pads([1, 5, 4, 3], pitch=2.54),
                is_locked=True,
                x=6.0,
                y=25.0,
                footprint="Connector_USB:USB_Micro-B_Molex",
                value="USB_PORT",
                comp_type="connector",
            ),
            "J2": ComponentNode(
                comp_id="J2",
                width=4.8,
                height=12.2,
                pads=_make_header_pads([2, 6, 7, 3], pitch=2.54),
                is_locked=True,
                x=44.0,
                y=25.0,
                footprint="Connector_PinHeader_2.54mm:PinHeader_1x04_P2.54mm_Vertical",
                value="UART_HDR",
                comp_type="connector",
            ),
            "U1": ComponentNode(
                comp_id="U1",
                width=8.6,
                height=7.0,
                pads=_make_soic8_pads([1, 4, 5, 3, 2, 6, 7, 8]),
                x=25.0,
                y=25.0,
                footprint="Package_SO:SOIC-8_3.9x4.9mm_P1.27mm",
                value="CH340N_BRIDGE",
                comp_type="ic",
            ),
            "D1": ComponentNode(
                comp_id="D1",
                width=5.8,
                height=5.2,
                pads=[
                    Pad("1", -1.6, -1.0, 0.9, 0.6, 4, 1),
                    Pad("2", -1.6, 1.0, 0.9, 0.6, 5, 1),
                    Pad("3", 1.6, 1.0, 0.9, 0.6, 3, 1),
                    Pad("4", 1.6, -1.0, 0.9, 0.6, 1, 1),
                ],
                x=15.0,
                y=22.0,
                footprint="Package_TO_SOT_SMD:SOT-23-6",
                value="USBLC6-2SC6",
                comp_type="diode",
            ),
            "D2": ComponentNode(
                comp_id="D2",
                width=4.4,
                height=2.8,
                pads=_make_two_pin_smd(2, 8),
                x=33.0,
                y=14.0,
                footprint="LED_SMD:LED_0805_2012Metric",
                value="TX_LED",
                comp_type="diode",
            ),
            "R1": ComponentNode(
                comp_id="R1",
                width=4.2,
                height=2.6,
                pads=_make_two_pin_smd(8, 3),
                x=33.0,
                y=36.0,
                footprint="Resistor_SMD:R_0603_1608Metric",
                value="470R",
                comp_type="res",
            ),
            "C1": ComponentNode(
                comp_id="C1",
                width=4.4,
                height=2.8,
                pads=_make_two_pin_smd(1, 3),
                x=16.0,
                y=35.0,
                footprint="Capacitor_SMD:C_0805_2012Metric",
                value="10uF",
                comp_type="cap",
            ),
            "C2": ComponentNode(
                comp_id="C2",
                width=4.4,
                height=2.8,
                pads=_make_two_pin_smd(2, 3),
                x=25.0,
                y=14.0,
                footprint="Capacitor_SMD:C_0805_2012Metric",
                value="100nF",
                comp_type="cap",
            ),
            "R2": ComponentNode(
                comp_id="R2",
                width=4.2,
                height=2.6,
                pads=_make_two_pin_smd(6, 7),
                x=36.0,
                y=25.0,
                footprint="Resistor_SMD:R_0603_1608Metric",
                value="0R_LOOP_OPT",
                comp_type="res",
            ),
        }
        return _finalize_nets("usb_uart_bridge", 50.0, 50.0, components, net_names)

    raise ValueError(
        f"Unknown preset '{preset_name}'. Available presets: {list(PRESET_DESCRIPTIONS)}"
    )
