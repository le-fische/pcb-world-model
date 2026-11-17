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
    side: str = "top"  # "top" (F.Cu) | "bottom" (B.Cu)
    special_layout_rules: list[str] = field(default_factory=list)
    electrical_meta: dict[str, Any] = field(default_factory=dict)
    keepout_zones: list[dict[str, Any]] = field(default_factory=list)
    noise_domain: str = "neutral"  # "power_switching" | "analog_precision" | "digital_hs" | "neutral"

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

    def world_keepout_zones(self) -> list[tuple[float, float, float, float, bool, str]]:
        """Transform local keepout zones into board-frame (x0, y0, x1, y1, all_layers, kind)."""
        if not self.keepout_zones:
            return []
        norm_theta = int(self.theta) % 360
        rad = math.radians(norm_theta)
        cos_t = math.cos(rad)
        sin_t = math.sin(rad)
        out: list[tuple[float, float, float, float, bool, str]] = []
        for kz in self.keepout_zones:
            dx = float(kz.get("dx", 0.0))
            dy = float(kz.get("dy", 0.0))
            kw = float(kz.get("width", 0.0))
            kh = float(kz.get("height", 0.0))
            if norm_theta in (90, 270):
                kw, kh = kh, kw
            wx = self.x + (dx * cos_t - dy * sin_t)
            wy = self.y + (dx * sin_t + dy * cos_t)
            out.append(
                (
                    wx - kw * 0.5,
                    wy - kh * 0.5,
                    wx + kw * 0.5,
                    wy + kh * 0.5,
                    bool(kz.get("all_layers", True)),
                    str(kz.get("kind", "rf_antenna_keepout")),
                )
            )
        return out

    def to_bottom_side(self) -> None:
        """Flip a top-side SMD component onto B.Cu (mirroring X and SMD pad masks)."""
        if self.side == "bottom":
            return
        self.side = "bottom"
        for pad in self.pads:
            pad.rel_x = -pad.rel_x
            if pad.layer_mask == 1:
                pad.layer_mask = 2


@dataclass
class NetInfo:
    """Electrical hyperedge connecting two or more component pads."""

    net_id: int
    name: str
    pins: list[tuple[str, str]] = field(default_factory=list)  # (comp_id, pad_id)
    is_power_ground: bool = False
    weight: float = 1.0
    noise_domain: str = "neutral"  # "power_switching" | "analog_precision" | "digital_hs" | "neutral"


@dataclass
class SubstrateZone:
    """Spatial multi-zone substrate definition for Rigid-Flex stackups (rev14)."""

    zone_id: str
    zone_type: str  # "rigid" | "dynamic_flex"
    x0_mm: float
    y0_mm: float
    x1_mm: float
    y1_mm: float
    er: float = 4.5
    prepreg_h_mm: float = 0.10
    layers: int = 8
    hatch_fill_ratio: float = 1.0
    hatch_angle_deg: float = 45.0
    allow_components: bool = True
    allow_vias: bool = True
    allowed_layers: list[int] = field(default_factory=list)


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
    thickness_mm: float = 1.6
    layers: int = 2
    single_sided: bool = False
    design_notes: list[str] = field(default_factory=list)
    design_rationale: list[str] = field(default_factory=list)
    outline_polygon: list[tuple[float, float]] = field(default_factory=list)
    enclosure_cutouts: list[list[tuple[float, float]]] = field(default_factory=list)
    z_max_regions: list[dict[str, Any]] = field(default_factory=list)
    z_max_default_mm: float | None = None
    port_anchors: list[dict[str, Any]] = field(default_factory=list)
    enclosure_meta: dict[str, Any] = field(default_factory=dict)
    ground_constraints: dict[str, Any] = field(default_factory=dict)
    thermal_isolation_slots: list[dict[str, Any]] = field(default_factory=list)
    substrate_zones: list[SubstrateZone] = field(default_factory=list)

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
                side=c.side,
                special_layout_rules=list(c.special_layout_rules),
                electrical_meta=dict(c.electrical_meta),
                keepout_zones=[dict(kz) for kz in c.keepout_zones],
                noise_domain=getattr(c, "noise_domain", "neutral"),
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
                noise_domain=getattr(n, "noise_domain", "neutral"),
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
            thickness_mm=self.thickness_mm,
            layers=self.layers,
            single_sided=self.single_sided,
            design_notes=list(self.design_notes),
            design_rationale=list(self.design_rationale),
            outline_polygon=[tuple(pt) for pt in self.outline_polygon],
            enclosure_cutouts=[[tuple(pt) for pt in poly] for poly in self.enclosure_cutouts],
            z_max_regions=[dict(r) for r in self.z_max_regions],
            z_max_default_mm=self.z_max_default_mm,
            port_anchors=[dict(p) for p in self.port_anchors],
            enclosure_meta=dict(self.enclosure_meta),
            ground_constraints=dict(self.ground_constraints),
            thermal_isolation_slots=[dict(s) for s in self.thermal_isolation_slots],
            substrate_zones=[
                SubstrateZone(
                    zone_id=z.zone_id,
                    zone_type=z.zone_type,
                    x0_mm=z.x0_mm,
                    y0_mm=z.y0_mm,
                    x1_mm=z.x1_mm,
                    y1_mm=z.y1_mm,
                    er=z.er,
                    prepreg_h_mm=z.prepreg_h_mm,
                    layers=z.layers,
                    hatch_fill_ratio=z.hatch_fill_ratio,
                    hatch_angle_deg=z.hatch_angle_deg,
                    allow_components=z.allow_components,
                    allow_vias=z.allow_vias,
                    allowed_layers=list(z.allowed_layers),
                )
                for z in self.substrate_zones
            ],
        )


def _infer_comp_type(comp_id: str, footprint: str) -> str:
    """Infer component role for GNN node feature encoding."""
    uid = comp_id.upper()
    fp = footprint.lower()
    if uid.startswith("MH") or "mountinghole" in fp:
        return "mounting_hole"
    if uid.startswith("ANT") or "antenna" in fp:
        return "antenna"
    if uid.startswith("Y") or "crystal" in fp:
        return "crystal"
    if uid.startswith("SW") or "switch" in fp or "button" in fp:
        return "switch"
    if uid.startswith("J") or "usb" in fp or "header" in fp or "conn" in fp or "jst" in fp:
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
    """Check whether a net name represents power, ground, or high-current switching node."""
    upper = name.upper()
    return any(
        tok in upper
        for tok in (
            "GND",
            "VCC",
            "VDD",
            "VIN",
            "VOUT",
            "3V3",
            "5V",
            "PWR",
            "VBAT",
            "VBUS",
            "VSYS",
            "SW_NODE",
            "SWNODE",
        )
    )


def _is_clock_net_name(name: str) -> bool:
    """Check whether a net name represents a high-priority crystal or clock signal."""
    upper = name.upper()
    return any(tok in upper for tok in ("XIN", "XOUT", "XTAL", "CLK_12M", "OSC"))


VALID_NOISE_DOMAINS: tuple[str, ...] = (
    "power_switching",
    "analog_precision",
    "digital_hs",
    "neutral",
)


def infer_net_noise_domain(name: str) -> str:
    """Infer electrical noise domain for a net from its signal/rail name."""
    up = name.strip().upper()
    if not up or up == "GND":
        return "neutral"
    if any(
        tok in up
        for tok in (
            "SW_NODE",
            "SWNODE",
            "GAN_SW",
            "GAN_PHASE",
            "BOOT_SW",
            "VIN_SW",
            "PGND",
            "PHASE_SW",
            "LX_SW",
            "MOSFET_SW",
        )
    ) or up == "BOOT":
        return "power_switching"
    if any(
        tok in up
        for tok in (
            "PCIE",
            "CLK",
            "XIN",
            "XOUT",
            "XTAL",
            "HSE",
            "QSPI",
            "SPI",
            "USB_D",
            "HS_BUS",
            "LVDS",
            "FPGA",
            "ETH_",
            "SCLK",
            "MCLK",
            "_PWM",
            "PWM_",
            "_CS",
            "CS_N",
            "DRDY",
        )
    ):
        return "digital_hs"
    if any(
        tok in up
        for tok in (
            "AIN",
            "ADC_",
            "VREF",
            "SHUNT",
            "KELVIN",
            "ANALOG",
            "AVDD",
            "AGND",
            "SUBMV",
            "DIFF_IN",
            "SENS_P",
            "SENS_N",
            "OPAMP",
            "DAC_OUT",
        )
    ):
        return "analog_precision"
    return "neutral"


def infer_component_noise_domain(
    comp: ComponentNode,
    nets: dict[int, NetInfo] | None = None,
) -> str:
    """Infer component noise domain (`power_switching` | `analog_precision` | `digital_hs` | `neutral`)."""
    explicit = getattr(comp, "noise_domain", "neutral")
    if explicit in VALID_NOISE_DOMAINS and explicit != "neutral":
        return explicit

    meta = getattr(comp, "electrical_meta", {}) or {}
    meta_dom = str(meta.get("noise_domain", "")).strip()
    if meta_dom in VALID_NOISE_DOMAINS and meta_dom != "neutral":
        return meta_dom

    if float(meta.get("power_watts", 0.0) or 0.0) >= 2.0 or float(meta.get("switching_freq_hz", 0.0) or 0.0) >= 100_000.0:
        return "power_switching"
    noise_floor = meta.get("noise_floor_mv")
    if noise_floor is not None and float(noise_floor) <= 5.0:
        return "analog_precision"

    val_up = (comp.value or "").upper()
    rules = set(getattr(comp, "special_layout_rules", []))
    if "high_di_dt_loop" in rules or any(
        tok in val_up for tok in ("GAN", "LMG5", "TPS54", "MP23", "BUCK", "SW_FET", "PWR_FET", "SCHOTTKY")
    ):
        return "power_switching"
    if comp.comp_type == "inductor" and "diff_pair_impedance" not in rules:
        return "power_switching"
    if "kelvin_sense_pair" in rules or any(
        tok in val_up for tok in ("ADS12", "AD71", "LTC24", "ADC_24", "24B_ADC", "VREF", "REF50", "INA2", "OPAMP", "SUBMV")
    ):
        return "analog_precision"
    if comp.comp_type == "crystal" or any(
        tok in val_up for tok in ("FPGA", "ECP5", "ICE40", "PCIE", "STM32", "RP2040", "ESP32", "W25Q", "OSC")
    ):
        return "digital_hs"

    if nets:
        votes: dict[str, int] = {}
        for pad in comp.pads:
            if pad.net_id <= 0 or pad.net_id not in nets:
                continue
            net = nets[pad.net_id]
            if net.name.strip().upper() == "GND":
                continue
            nd = getattr(net, "noise_domain", "neutral")
            if nd == "neutral":
                nd = infer_net_noise_domain(net.name)
            if nd != "neutral":
                votes[nd] = votes.get(nd, 0) + 1
        if "power_switching" in votes:
            return "power_switching"
        if "analog_precision" in votes:
            return "analog_precision"
        if "digital_hs" in votes:
            return "digital_hs"
    return "neutral"


def annotate_noise_domains(graph: NetlistGraph) -> NetlistGraph:
    """Populate `noise_domain` tags across nets and components without overriding explicit non-neutral tags."""
    for net in graph.nets.values():
        if getattr(net, "noise_domain", "neutral") == "neutral":
            net.noise_domain = infer_net_noise_domain(net.name)
    for comp in graph.components.values():
        if getattr(comp, "noise_domain", "neutral") == "neutral":
            comp.noise_domain = infer_component_noise_domain(comp, graph.nets)
    # Propagate component domain to attached non-GND signal nets only when all non-neutral endpoints share 1 domain
    for net in graph.nets.values():
        if net.noise_domain != "neutral" or net.name.strip().upper() == "GND":
            continue
        comp_doms = {
            graph.components[cid].noise_domain
            for cid, _ in net.pins
            if cid in graph.components and graph.components[cid].noise_domain != "neutral"
        }
        if len(comp_doms) == 1:
            net.noise_domain = next(iter(comp_doms))
    return graph


def _build_net_info(net_id: int, name: str, power_weight: float = 1.4) -> NetInfo:
    """Construct a NetInfo with net-class flags and routing/placement spring weight."""
    is_pwr = _is_power_net_name(name)
    if _is_clock_net_name(name):
        weight = 3.0
    elif is_pwr:
        weight = power_weight
    else:
        weight = 1.0
    return NetInfo(
        net_id=net_id,
        name=name,
        pins=[],
        is_power_ground=is_pwr,
        weight=weight,
        noise_domain=infer_net_noise_domain(name),
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
                    nets[nid] = _build_net_info(nid, nname, power_weight=1.5)

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

        for arc_node in cls._find_children(ast, "gr_arc"):
            layer_node = cls._find_child(arc_node, "layer")
            if layer_node and len(layer_node) >= 2 and layer_node[1] == "Edge.Cuts":
                for tag in ("start", "mid", "end"):
                    pt_node = cls._find_child(arc_node, tag)
                    if pt_node and len(pt_node) >= 3:
                        edge_xs.append(float(pt_node[1]))
                        edge_ys.append(float(pt_node[2]))

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
            if footprint_name.startswith("TestPoint:"):
                continue
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
                if pad_type in ("thru_hole", "np_thru_hole"):
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
                        nets[nid] = _build_net_info(
                            nid, nname or f"NET_{nid}", power_weight=1.5
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

            fp_layer_node = cls._find_child(fp_node, "layer")
            fp_side = (
                "bottom"
                if (fp_layer_node and len(fp_layer_node) >= 2 and str(fp_layer_node[1]) == "B.Cu")
                else "top"
            )

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
                side=fp_side,
            )

        # Detect board stackup layer count & thickness from (layers ...) and (general ...)
        parsed_layers = 2
        layers_node = cls._find_child(ast, "layers")
        if layers_node:
            layer_names_found = {
                str(item[1])
                for item in layers_node[1:]
                if isinstance(item, list) and len(item) >= 2
            }
            if any(ln in layer_names_found for ln in ("In5.Cu", "In6.Cu", "In5.SIG2", "In6.GND3")):
                parsed_layers = 8
            elif any(ln in layer_names_found for ln in ("In3.Cu", "In4.Cu", "In3.Sig2", "In4.PWR")):
                parsed_layers = 6
            elif any(ln in layer_names_found for ln in ("In1.Cu", "In2.Cu", "In1.GND", "In2.PWR")):
                parsed_layers = 4

        parsed_thickness = 1.6
        gen_node = cls._find_child(ast, "general")
        if gen_node:
            thk_node = cls._find_child(gen_node, "thickness")
            if thk_node and len(thk_node) >= 2:
                try:
                    parsed_thickness = float(thk_node[1])
                except ValueError:
                    pass

        def _layer_name_to_idx(name: str) -> int:
            if parsed_layers >= 8:
                return {
                    "F.Cu": 0,
                    "In1.Cu": 1,
                    "In1.GND": 1,
                    "In2.Cu": 2,
                    "In2.SIG1": 2,
                    "In3.Cu": 3,
                    "In3.PWR1": 3,
                    "In4.Cu": 4,
                    "In4.GND2": 4,
                    "In5.Cu": 5,
                    "In5.SIG2": 5,
                    "In6.Cu": 6,
                    "In6.GND3": 6,
                    "B.Cu": 7,
                }.get(name, 0)
            if parsed_layers >= 6:
                return {
                    "F.Cu": 0,
                    "In1.Cu": 1,
                    "In1.GND": 1,
                    "In2.Cu": 2,
                    "In2.Sig1": 2,
                    "In3.Cu": 3,
                    "In3.Sig2": 3,
                    "In4.Cu": 4,
                    "In4.PWR": 4,
                    "B.Cu": 5,
                }.get(name, 0)
            if parsed_layers >= 4:
                if name == "F.Cu":
                    return 0
                if name in ("In1.Cu", "In1.GND"):
                    return 1
                if name in ("In2.Cu", "In2.PWR"):
                    return 2
                if name == "B.Cu":
                    return 3
            return 1 if name == "B.Cu" else 0

        raw_segments: list[dict[str, Any]] = []
        for seg in cls._find_children(ast, "segment"):
            s = cls._find_child(seg, "start")
            e = cls._find_child(seg, "end")
            w = cls._find_child(seg, "width")
            lyr = cls._find_child(seg, "layer")
            net = cls._find_child(seg, "net")
            if s and e and net:
                lname = str(lyr[1]) if (lyr and len(lyr) >= 2) else "F.Cu"
                layer_idx = _layer_name_to_idx(lname)
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
            vlayers = cls._find_child(via, "layers")
            via_tokens = [str(tok) for tok in via[1:] if not isinstance(tok, list)]
            if "micro" in via_tokens:
                via_type = "micro"
            elif "blind" in via_tokens or "buried" in via_tokens:
                via_type = "buried"
            else:
                via_type = "through"
            start_l = 0
            end_l = parsed_layers - 1
            if vlayers and len(vlayers) >= 3:
                start_l = _layer_name_to_idx(str(vlayers[1]))
                end_l = _layer_name_to_idx(str(vlayers[2]))
            if at and net:
                raw_vias.append(
                    {
                        "net_id": int(float(net[1])),
                        "x_mm": float(at[1]) - min_x,
                        "y_mm": float(at[2]) - min_y,
                        "diameter_mm": float(sz[1]) if sz and len(sz) >= 2 else 0.6,
                        "drill_mm": float(dr[1]) if dr and len(dr) >= 2 else 0.3,
                        "start_layer": start_l,
                        "end_layer": end_l,
                        "via_type": via_type,
                    }
                )

        return annotate_noise_domains(
            NetlistGraph(
                board_name=board_name,
                width_mm=width_mm,
                height_mm=height_mm,
                components=components,
                nets=nets,
                origin_x_mm=min_x,
                origin_y_mm=min_y,
                raw_segments=raw_segments,
                raw_vias=raw_vias,
                thickness_mm=parsed_thickness,
                layers=parsed_layers,
            )
        )


def _finalize_nets(
    board_name: str,
    width_mm: float,
    height_mm: float,
    components: dict[str, ComponentNode],
    net_names: dict[int, str],
    thickness_mm: float = 1.6,
    layers: int = 2,
    single_sided: bool = False,
    design_notes: list[str] | None = None,
    design_rationale: list[str] | None = None,
    ground_constraints: dict[str, Any] | None = None,
    substrate_zones: list[SubstrateZone] | None = None,
) -> NetlistGraph:
    """Populate NetInfo hyperedges from component pad assignments."""
    nets: dict[int, NetInfo] = {}
    for nid, name in net_names.items():
        if nid <= 0:
            continue
        nets[nid] = _build_net_info(nid, name, power_weight=1.4)
    for cid, comp in components.items():
        for pad in comp.pads:
            if pad.net_id > 0:
                pad.net_name = net_names.get(pad.net_id, f"NET_{pad.net_id}")
                if pad.net_id not in nets:
                    nets[pad.net_id] = _build_net_info(
                        pad.net_id, pad.net_name, power_weight=1.4
                    )
                nets[pad.net_id].pins.append((cid, pad.pad_id))
    return annotate_noise_domains(
        NetlistGraph(
            board_name=board_name,
            width_mm=width_mm,
            height_mm=height_mm,
            components=components,
            nets=nets,
            thickness_mm=thickness_mm,
            layers=layers,
            single_sided=single_sided,
            design_notes=list(design_notes) if design_notes else [],
            design_rationale=list(design_rationale) if design_rationale else [],
            ground_constraints=dict(ground_constraints) if ground_constraints else {},
            substrate_zones=list(substrate_zones) if substrate_zones else [],
        )
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
    "rp2040_minimal": "RP2040 MCU + W25Q16 QSPI Flash + 12MHz Crystal + 3.3V LDO + USB-C (14 comps, 12 nets)",
    "macropad_matrix": "2x2 Locked Mechanical Keyswitch Matrix + Controller IC + Scan Diodes + USB Header (10 comps, 9 nets)",
}

COMPLEX_PRESET_DESCRIPTIONS: dict[str, str] = {
    "flight_controller_50": "50-Component 4-Layer Drone Flight Controller (STM32F405 QFN-40 + Dual IMUs + Baro + 5V Buck + 3.3V LDO + Blackbox Flash + 6 Headers, 80x80mm)",
}

ALL_PRESET_DESCRIPTIONS: dict[str, str] = {
    **PRESET_DESCRIPTIONS,
    **COMPLEX_PRESET_DESCRIPTIONS,
}

REV7_PRESET_DESCRIPTIONS: dict[str, str] = {
    "wearable_oled_keychain_26x26": "B6: 26x26mm 0.8mm Single-Sided Wearable Keychain (ESP32-C3FN4 QFN-32 + 40MHz XTAL + USB-C + MCP73831 + AP2112K + LIS3DH + SSD1306 + Rear LiPo Pads)",
}

REV8_PRESET_DESCRIPTIONS: dict[str, str] = {
    "fpga_custom_protocol_bridge": "B9: 80x64mm 4L Silicon-to-Board FPGA Bridge (Lattice iCE40UP5K-SG48 QFN-48 + ESP32-C3FN4 + W25Q16 + 8-ch WS2812 + 8-ch Quadrature Encoders + Joint .pcf Pin-Swap)",
}

REV11_PRESET_DESCRIPTIONS: dict[str, str] = {
    "step_enclosure_keychain": "B10: 26x26mm External .STEP Enclosure Keychain (Chamfered Cavity + Z_max Rib Keepouts + Locked USB-C/Switch/Btn Port Windows + 2 M2 Bosses)",
    "datasheet_pinout_ingest": "B11: Unseen DRV8833 Dual H-Bridge Motor Driver (PNG/PDF Pin Diagram Ingest + Auto-ERC Advisor + High-di/dt Loop + Thermal EP)",
    "legacy_kicad_reoptimize": "B12: Legacy High-Via External .kicad_pcb Re-Optimization (Locked Connectors/Holes + DFM Teardrops + >=30% Via Reduction)",
}

REV12_PRESET_DESCRIPTIONS: dict[str, str] = {
    "hdi_fpga_bga_fanout_8l": "B13: 30x50mm 8-Layer (2+4+2) HDI FPGA 64-Ball BGA (0.8mm pitch, Staggered-Only Microvias >=0.25mm, Buried Core L3<->L6, Max 2 Cycles)",
}

REV13_PRESET_DESCRIPTIONS: dict[str, str] = {
    "mixed_signal_gan_adc_30x50": "B14: 30x50mm 8-Layer HDI Mixed-Signal 5A 2MHz GaN Buck + 24-bit Sub-mV ADC (Both Top F.Cu, Continuous In1.GND, J_return=0%, >=20mm Separation)",
}

REV14_PRESET_DESCRIPTIONS: dict[str, str] = {
    "ultimate_rigid_flex_8l_hdi_pcie_gan_adc": "B15: 30x50mm 8-Layer (2+4+2) HDI Rigid-Flex (FR4 + 50% Cross-Hatched Polyimide Flex, 85-Ohm PCIe Gen 4 Diff-Pairs, 5A GaN + 24-bit ADC Top F.Cu, Under-BGA PDN <=1.2nH)",
}

FULL_PRESET_CATALOG: dict[str, str] = {
    **ALL_PRESET_DESCRIPTIONS,
    **REV7_PRESET_DESCRIPTIONS,
    **REV8_PRESET_DESCRIPTIONS,
    **REV11_PRESET_DESCRIPTIONS,
    **REV12_PRESET_DESCRIPTIONS,
    **REV13_PRESET_DESCRIPTIONS,
    **REV14_PRESET_DESCRIPTIONS,
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

    if preset_name == "rp2040_minimal":
        net_names = {
            1: "VBUS_5V",
            2: "VCC_3V3",
            3: "GND",
            4: "USB_DP",
            5: "USB_DM",
            6: "XIN",
            7: "XOUT",
            8: "QSPI_CS",
            9: "QSPI_CLK",
            10: "QSPI_D0",
            11: "QSPI_D1",
            12: "LED_STAT",
        }
        u1_pads = [
            Pad("1", -3.2, -2.54, 1.4, 0.6, 2, 1),   # VCC_3V3
            Pad("2", -3.2, -1.27, 1.4, 0.6, 5, 1),   # USB_DM
            Pad("3", -3.2, 0.0, 1.4, 0.6, 4, 1),     # USB_DP
            Pad("4", -3.2, 1.27, 1.4, 0.6, 6, 1),    # XIN
            Pad("5", -3.2, 2.54, 1.4, 0.6, 7, 1),    # XOUT
            Pad("6", 3.2, -2.54, 1.4, 0.6, 12, 1),   # LED_STAT
            Pad("7", 3.2, -1.27, 1.4, 0.6, 11, 1),   # QSPI_D1
            Pad("8", 3.2, 0.0, 1.4, 0.6, 10, 1),     # QSPI_D0
            Pad("9", 3.2, 1.27, 1.4, 0.6, 9, 1),     # QSPI_CLK
            Pad("10", 3.2, 2.54, 1.4, 0.6, 8, 1),    # QSPI_CS
        ]
        components = {
            "J1": ComponentNode(
                comp_id="J1",
                width=5.2,
                height=11.6,
                pads=_make_header_pads([1, 5, 4, 3], pitch=2.54),
                is_locked=True,
                x=6.0,
                y=25.0,
                theta=0,
                footprint="Connector_USB:USB_C_Receptacle_GCT_USB4085",
                value="USB_C",
                comp_type="connector",
            ),
            "J2": ComponentNode(
                comp_id="J2",
                width=4.8,
                height=9.0,
                pads=_make_header_pads([2, 12, 3], pitch=2.54),
                is_locked=True,
                x=44.0,
                y=25.0,
                theta=0,
                footprint="Connector_PinHeader_2.54mm:PinHeader_1x03_P2.54mm_Vertical",
                value="GPIO_HDR",
                comp_type="connector",
            ),
            "U1": ComponentNode(
                comp_id="U1",
                width=8.8,
                height=7.6,
                pads=u1_pads,
                x=23.0,
                y=25.0,
                theta=0,
                footprint="Package_DFN_QFN:QFN-10_RP2040_Core",
                value="RP2040",
                comp_type="ic",
            ),
            "U2": ComponentNode(
                comp_id="U2",
                width=8.6,
                height=7.0,
                pads=_make_soic8_pads([11, 10, 9, 8, 3, 2, 2, 3]),
                x=34.5,
                y=25.0,
                theta=0,
                footprint="Package_SO:SOIC-8_3.9x4.9mm_P1.27mm",
                value="W25Q16JV",
                comp_type="ic",
            ),
            "U3": ComponentNode(
                comp_id="U3",
                width=6.0,
                height=5.6,
                pads=[
                    Pad("1", -1.8, -1.2, 1.0, 0.6, 1, 1),
                    Pad("2", -1.8, 0.0, 1.0, 0.6, 3, 1),
                    Pad("3", -1.8, 1.2, 1.0, 0.6, 1, 1),
                    Pad("4", 1.8, 1.0, 1.0, 0.6, 2, 1),
                    Pad("5", 1.8, -1.0, 1.0, 0.6, 2, 1),
                ],
                x=14.0,
                y=14.0,
                theta=0,
                footprint="Package_TO_SOT_SMD:SOT-23-5",
                value="NCP1117-3.3",
                comp_type="ic",
            ),
            "Y1": ComponentNode(
                comp_id="Y1",
                width=4.6,
                height=3.2,
                pads=[
                    Pad("1", 0.0, -1.0, 1.2, 0.8, 6, 1),
                    Pad("2", 0.0, 1.0, 1.2, 0.8, 7, 1),
                ],
                x=15.0,
                y=28.5,
                theta=0,
                footprint="Crystal:Crystal_SMD_3225",
                value="12MHz",
                comp_type="ic",
            ),
            "SW1": ComponentNode(
                comp_id="SW1",
                width=4.6,
                height=3.2,
                pads=_make_two_pin_smd(8, 3, pitch=2.2),
                x=34.5,
                y=36.0,
                theta=0,
                footprint="Button_Switch_SMD:SW_SPST_TL3342",
                value="BOOTSEL",
                comp_type="ic",
            ),
            "D1": ComponentNode(
                comp_id="D1",
                width=4.4,
                height=2.8,
                pads=_make_two_pin_smd(2, 12, pitch=2.0),
                x=42.0,
                y=14.0,
                theta=0,
                footprint="LED_SMD:LED_0805_2012Metric",
                value="PWR_LED",
                comp_type="diode",
            ),
            "C1": ComponentNode(
                comp_id="C1",
                width=4.4,
                height=2.8,
                pads=_make_two_pin_smd(1, 3, pitch=2.0),
                x=14.0,
                y=36.0,
                theta=0,
                footprint="Capacitor_SMD:C_0805_2012Metric",
                value="10uF_VBUS",
                comp_type="cap",
            ),
            "C2": ComponentNode(
                comp_id="C2",
                width=4.4,
                height=2.8,
                pads=_make_two_pin_smd(2, 3, pitch=2.0),
                x=23.0,
                y=14.0,
                theta=0,
                footprint="Capacitor_SMD:C_0805_2012Metric",
                value="100nF_VDD",
                comp_type="cap",
            ),
            "C3": ComponentNode(
                comp_id="C3",
                width=4.4,
                height=2.8,
                pads=_make_two_pin_smd(2, 3, pitch=2.0),
                x=23.0,
                y=36.0,
                theta=0,
                footprint="Capacitor_SMD:C_0805_2012Metric",
                value="100nF_CORE",
                comp_type="cap",
            ),
            "C4": ComponentNode(
                comp_id="C4",
                width=4.4,
                height=2.8,
                pads=_make_two_pin_smd(2, 3, pitch=2.0),
                x=33.0,
                y=14.0,
                theta=0,
                footprint="Capacitor_SMD:C_0805_2012Metric",
                value="100nF_FLASH",
                comp_type="cap",
            ),
            "R1": ComponentNode(
                comp_id="R1",
                width=4.2,
                height=2.6,
                pads=_make_two_pin_smd(2, 8, pitch=2.0),
                x=27.5,
                y=36.0,
                theta=0,
                footprint="Resistor_SMD:R_0603_1608Metric",
                value="10k_CS",
                comp_type="res",
            ),
            "R2": ComponentNode(
                comp_id="R2",
                width=4.2,
                height=2.6,
                pads=_make_two_pin_smd(12, 3, pitch=2.0),
                x=42.0,
                y=36.0,
                theta=0,
                footprint="Resistor_SMD:R_0603_1608Metric",
                value="1k_LED",
                comp_type="res",
            ),
        }
        return _finalize_nets("rp2040_minimal", 50.0, 50.0, components, net_names)

    if preset_name in ("macropad_matrix", "macropad_4x4", "macropad_2x2"):
        from env.footprint_generator import (
            make_keyswitch_mx,
            make_pin_header,
            make_smd_passive,
            make_soic_or_sop,
        )

        net_names = {
            1: "VBUS_5V",
            2: "VCC_3V3",
            3: "GND",
            4: "USB_DP",
            5: "USB_DM",
            6: "COL0",
            7: "COL1",
            8: "ROW0",
            9: "ROW1",
        }
        components = {
            "J1": make_pin_header(
                "J1", "USB_HID_HDR", [1, 5, 4, 3], pitch=2.54, x=6.0, y=25.0, is_locked=True
            ),
            "SW1": make_keyswitch_mx("SW1", "KEY_1", 6, 8, x=32.0, y=14.0, is_locked=True),
            "SW2": make_keyswitch_mx("SW2", "KEY_2", 7, 8, x=42.5, y=14.0, is_locked=True),
            "SW3": make_keyswitch_mx("SW3", "KEY_3", 6, 9, x=32.0, y=36.0, is_locked=True),
            "SW4": make_keyswitch_mx("SW4", "KEY_4", 7, 9, x=42.5, y=36.0, is_locked=True),
            "U1": make_soic_or_sop(
                "U1",
                "CH552G_HID",
                [1, 5, 4, 3, 9, 7, 6, 2],
                x=18.5,
                y=25.0,
            ),
            "D1": make_smd_passive("D1", "ROW0_SCAN", "diode", "0805", 2, 8, x=23.5, y=14.0),
            "R1": make_smd_passive("R1", "10k_ROW0", "res", "0603", 8, 3, x=14.0, y=14.0),
            "C1": make_smd_passive("C1", "10uF_VBUS", "cap", "0805", 1, 3, x=13.5, y=36.0),
            "C2": make_smd_passive("C2", "100nF_VDD", "cap", "0805", 2, 3, x=22.0, y=36.0),
        }
        return _finalize_nets("macropad_matrix", 50.0, 50.0, components, net_names)

    if preset_name == "flight_controller_50":
        from env.footprint_generator import (
            make_pin_header,
            make_qfn,
            make_smd_passive,
            make_soic_or_sop,
            make_sot23,
        )

        net_names = {
            1: "VBAT_16V",
            2: "VCC_5V",
            3: "VCC_3V3",
            4: "GND",
            5: "SW_NODE",
            6: "BOOT_SW",
            7: "FB_5V",
            8: "USB_DP",
            9: "USB_DM",
            10: "USB_DP_MCU",
            11: "USB_DM_MCU",
            12: "HSE_IN",
            13: "HSE_OUT",
            14: "MCU_NRST",
            15: "MCU_BOOT",
            16: "SPI1_SCK",
            17: "SPI1_MOSI",
            18: "SPI1_MISO",
            19: "SPI1_CS_IMU1",
            20: "SPI1_CS_FLASH",
            21: "IMU1_INT",
            22: "SPI2_SCK",
            23: "SPI2_MOSI",
            24: "SPI2_MISO",
            25: "SPI2_CS_IMU2",
            26: "IMU2_INT",
            27: "I2C1_SDA",
            28: "I2C1_SCL",
            29: "BARO_INT",
            30: "M1_PWM",
            31: "M2_PWM",
            32: "M3_PWM",
            33: "M4_PWM",
            34: "SWDIO",
            35: "SWCLK",
            36: "UART1_TX",
            37: "UART1_RX",
            38: "RC_SBUS",
            39: "ARM_LED_SIG",
            40: "GYRO_LED_SIG",
            41: "LED2_K",
            42: "LED3_K",
            43: "LED4_K",
        }
        u1_pins = [
            3, 10, 11, 12, 13, 14, 15, 34, 35, 4,   # Left (1..10)
            36, 37, 38, 39, 40, 3, 4, 22, 23, 24,   # Bottom (11..20)
            25, 26, 27, 28, 29, 30, 31, 32, 33, 4,  # Right (21..30)
            3, 16, 17, 18, 19, 20, 21, 3, 4, 4,     # Top (31..40)
        ]
        components: dict[str, ComponentNode] = {
            # 6 Locked Edge Connectors
            "J1": make_pin_header("J1", "USB_C_PWR", [2, 8, 9, 4], pitch=2.54, x=6.0, y=28.0, is_locked=True),
            "J2": make_pin_header("J2", "VBAT_16V_IN", [1, 4], pitch=2.54, x=20.0, y=74.0, theta=90, is_locked=True),
            "J3": make_pin_header("J3", "ESC_PWM_1_4", [30, 31, 32, 33, 2, 4], pitch=2.54, x=74.0, y=40.0, is_locked=True),
            "J4": make_pin_header("J4", "GPS_UART1", [3, 36, 37, 4], pitch=2.54, x=40.0, y=6.0, theta=90, is_locked=True),
            "J5": make_pin_header("J5", "RC_SBUS_RX", [3, 38, 4], pitch=2.54, x=52.0, y=74.0, theta=90, is_locked=True),
            "J6": make_pin_header("J6", "SWD_DEBUG", [3, 34, 35, 4], pitch=2.54, x=6.0, y=52.0, is_locked=True),
            # 7 Core ICs
            "U1": make_qfn("U1", "STM32F405_FC", u1_pins, pitch=0.8, body_size=7.0, x=40.0, y=38.0),
            "U2": make_qfn("U2", "ICM42688P_IMU1", [3, 16, 17, 18, 4, 19, 21, 3, 4, 3, 4, 3], pitch=0.8, body_size=4.0, x=28.0, y=25.0),
            "U3": make_qfn("U3", "BMI270_IMU2", [3, 22, 23, 24, 4, 25, 26, 3, 4, 3, 4, 3], pitch=0.8, body_size=4.0, x=53.0, y=25.0),
            "U4": make_sot23("U4", "BMP388_BARO", [3, 4, 27, 28, 29, 3], x=55.0, y=44.0, comp_type="ic"),
            "U5": make_soic_or_sop("U5", "TPS54302_5V", [6, 1, 5, 4, 7, 1, 1, 5], x=19.0, y=54.0),
            "U6": make_sot23("U6", "AP2112K_3V3", [2, 4, 2, 3, 3], x=33.0, y=54.0, comp_type="ic"),
            "U7": make_soic_or_sop("U7", "W25Q128_FLASH", [16, 17, 18, 20, 3, 3, 4, 4], x=56.0, y=56.0),
            # 4 Power Magnetics, Schottky, Crystal, Boot Switch
            "L1": ComponentNode(
                comp_id="L1",
                width=7.4,
                height=6.8,
                pads=[Pad("1", -2.2, 0.0, 1.8, 3.0, 5, 1), Pad("2", 2.2, 0.0, 1.8, 3.0, 2, 1)],
                x=31.0,
                y=65.0,
                footprint="Inductor_SMD:L_6.3x6.3_H3",
                value="10uH_BUCK",
                comp_type="inductor",
            ),
            "D1": ComponentNode(
                comp_id="D1",
                width=5.6,
                height=3.6,
                pads=_make_two_pin_smd(4, 5, pitch=2.8),
                x=18.5,
                y=65.0,
                footprint="Diode_SMD:D_SMA",
                value="SS34_SCHOTTKY",
                comp_type="diode",
            ),
            "Y1": ComponentNode(
                comp_id="Y1",
                width=4.6,
                height=3.2,
                pads=[Pad("1", 0.0, -1.0, 1.2, 0.8, 12, 1), Pad("2", 0.0, 1.0, 1.2, 0.8, 13, 1)],
                x=27.0,
                y=38.0,
                footprint="Crystal:Crystal_SMD_3225",
                value="16MHz_HSE",
                comp_type="crystal",
            ),
            "SW1": make_smd_passive("SW1", "BOOT0_BTN", "switch", "0805", 15, 4, x=17.0, y=15.0),
            # 3 Status LEDs
            "D2": make_smd_passive("D2", "PWR_LED_5V", "diode", "0805", 2, 41, x=64.0, y=14.0),
            "D3": make_smd_passive("D3", "ARM_LED", "diode", "0805", 39, 42, x=64.0, y=22.0),
            "D4": make_smd_passive("D4", "GYRO_LED", "diode", "0805", 40, 43, x=64.0, y=30.0),
            # 16 Capacitors (Bulk, Decoupling, Bootstrap, Crystal Load)
            "C1": make_smd_passive("C1", "22uF_VBAT1", "cap", "1206", 1, 4, x=11.0, y=64.0),
            "C2": make_smd_passive("C2", "22uF_VBAT2", "cap", "1206", 1, 4, x=11.0, y=70.0),
            "C3": make_smd_passive("C3", "100nF_BST", "cap", "0603", 6, 5, x=26.5, y=59.5),
            "C4": make_smd_passive("C4", "47uF_5V1", "cap", "1206", 2, 4, x=39.5, y=65.0),
            "C5": make_smd_passive("C5", "47uF_5V2", "cap", "1206", 2, 4, x=46.5, y=65.0),
            "C6": make_smd_passive("C6", "100nF_MCU1", "cap", "0603", 3, 4, x=35.0, y=29.0),
            "C7": make_smd_passive("C7", "100nF_MCU2", "cap", "0603", 3, 4, x=45.0, y=29.0),
            "C8": make_smd_passive("C8", "100nF_MCU3", "cap", "0603", 3, 4, x=35.0, y=47.0),
            "C9": make_smd_passive("C9", "100nF_MCU4", "cap", "0603", 3, 4, x=45.0, y=47.0),
            "C10": make_smd_passive("C10", "100nF_IMU1", "cap", "0603", 3, 4, x=28.0, y=16.5),
            "C11": make_smd_passive("C11", "100nF_IMU2", "cap", "0603", 3, 4, x=53.0, y=16.5),
            "C12": make_smd_passive("C12", "100nF_FLASH", "cap", "0603", 3, 4, x=64.0, y=56.0),
            "C13": make_smd_passive("C13", "18pF_HSE1", "cap", "0603", 12, 4, x=19.5, y=35.0),
            "C14": make_smd_passive("C14", "18pF_HSE2", "cap", "0603", 13, 4, x=19.5, y=41.0),
            "C15": make_smd_passive("C15", "10uF_5V_IN", "cap", "0805", 2, 4, x=13.0, y=25.0),
            "C16": make_smd_passive("C16", "100nF_NRST", "cap", "0603", 14, 4, x=27.0, y=46.0),
            # 14 Resistors (FB Divider, I2C Pull-Ups, LED Limits, USB Series, CS Pull-Ups, Reset/Boot)
            "R1": make_smd_passive("R1", "100k_FB", "res", "0603", 2, 7, x=26.5, y=51.0),
            "R2": make_smd_passive("R2", "18.7k_FB", "res", "0603", 7, 4, x=12.5, y=56.0),
            "R3": make_smd_passive("R3", "4.7k_SDA", "res", "0603", 3, 27, x=64.0, y=41.0),
            "R4": make_smd_passive("R4", "4.7k_SCL", "res", "0603", 3, 28, x=64.0, y=46.5),
            "R5": make_smd_passive("R5", "1k_PWR_LED", "res", "0603", 41, 4, x=71.5, y=14.0),
            "R6": make_smd_passive("R6", "1k_ARM_LED", "res", "0603", 42, 4, x=71.5, y=22.0),
            "R7": make_smd_passive("R7", "1k_GYRO_LED", "res", "0603", 43, 4, x=71.5, y=30.0),
            "R8": make_smd_passive("R8", "22R_USB_DP", "res", "0603", 8, 10, x=14.0, y=30.0),
            "R9": make_smd_passive("R9", "22R_USB_DM", "res", "0603", 9, 11, x=14.0, y=34.0),
            "R10": make_smd_passive("R10", "10k_CS_IMU1", "res", "0603", 3, 19, x=36.0, y=16.5),
            "R11": make_smd_passive("R11", "10k_CS_IMU2", "res", "0603", 3, 25, x=45.0, y=16.5),
            "R12": make_smd_passive("R12", "10k_CS_FLS", "res", "0603", 3, 20, x=64.0, y=62.5),
            "R13": make_smd_passive("R13", "10k_NRST", "res", "0603", 3, 14, x=19.5, y=46.0),
            "R14": make_smd_passive("R14", "10k_BOOT0", "res", "0603", 15, 4, x=23.5, y=12.0),
        }
        return _finalize_nets("flight_controller_50", 80.0, 80.0, components, net_names, layers=4)

    if preset_name == "wearable_oled_keychain_26x26":
        from models.composer.schematic_composer import build_wearable_oled_keychain_netlist

        return build_wearable_oled_keychain_netlist()

    if preset_name == "fpga_custom_protocol_bridge":
        from models.composer.schematic_composer import build_fpga_protocol_bridge_netlist

        return build_fpga_protocol_bridge_netlist()

    if preset_name in ("step_enclosure_keychain", "datasheet_pinout_ingest", "legacy_kicad_reoptimize"):
        from env.importers import build_rev11_preset_board

        return build_rev11_preset_board(preset_name)

    if preset_name == "hdi_fpga_bga_fanout_8l":
        from env.footprint_generator import make_bga, make_smd_passive

        net_names = {
            1: "GND",
            2: "VCC_3V3",
            3: "HS_BUS_0",
            4: "HS_BUS_1",
            5: "HS_BUS_2",
            6: "HS_BUS_3",
            7: "HS_BUS_4",
            8: "HS_BUS_5",
            9: "HS_BUS_6",
            10: "HS_BUS_7",
            11: "QSPI_SCK",
            12: "QSPI_CS",
            13: "QSPI_IO0",
            14: "QSPI_IO1",
            15: "SPI_MOSI",
            16: "SPI_MISO",
            17: "SPI_SCK",
            18: "SPI_CS",
            19: "CLK_25M",
            20: "LED_DRV",
            21: "LED_K",
            22: "RESET_N",
            23: "UART_RX",
        }
        ball_nets: dict[str, int] = {
            # Inner 4x4 (C3..F6) core: GND, VCC_3V3, and 8 high-speed bus pins forced through staggered microvias
            "D4": 1, "D5": 1, "E4": 1, "E5": 1,
            "F4": 1, "F5": 1,
            "C4": 2, "C5": 2,
            "C3": 3, "C6": 4,
            "D3": 5, "D6": 6,
            "E3": 7, "E6": 8,
            "F3": 9, "F6": 10,
            # Second ring (B2..G7): QSPI, SPI, and power/ground anchors
            "B2": 1, "B7": 1, "G2": 1, "G7": 1,
            "D2": 1, "E2": 1, "D7": 1, "E7": 1,
            "C2": 2, "F2": 2, "C7": 2, "F7": 2,
            "B3": 11, "B4": 12, "B5": 13, "B6": 14,
            "G3": 15, "G4": 16, "G5": 17, "G6": 18,
            # Outer ring: corners GND + clock, reset, status LED, UART
            "A1": 1, "A8": 1, "H1": 1, "H8": 1,
            "A4": 19, "D1": 20, "E1": 22, "H5": 23,
        }
        components = {
            "U1": make_bga(
                "U1",
                "ECP5-BGA64",
                ball_nets,
                rows=8,
                cols=8,
                pitch=0.8,
                x=15.0,
                y=23.0,
                is_locked=True,
            ),
            "J1": ComponentNode(
                comp_id="J1",
                width=4.6,
                height=26.0,
                pads=_make_header_pads([3, 4, 5, 6, 7, 8, 9, 10, 2, 1], pitch=2.54),
                is_locked=True,
                x=15.0,
                y=45.0,
                theta=90,
                footprint="Connector_PinHeader_2.54mm:PinHeader_1x10_P2.54mm_Horizontal",
                value="HS_BUS_HDR",
                comp_type="connector",
            ),
            "J2": ComponentNode(
                comp_id="J2",
                width=4.6,
                height=16.0,
                pads=_make_header_pads([2, 1, 15, 16, 17, 18], pitch=2.54),
                is_locked=True,
                x=4.5,
                y=14.0,
                theta=0,
                footprint="Connector_PinHeader_2.54mm:PinHeader_1x06_P2.54mm_Vertical",
                value="SPI_JTAG_HDR",
                comp_type="connector",
            ),
            "U2": ComponentNode(
                comp_id="U2",
                width=8.6,
                height=7.0,
                pads=_make_soic8_pads([12, 14, 2, 1, 13, 11, 22, 2]),
                is_locked=False,
                x=16.5,
                y=8.0,
                theta=0,
                footprint="Package_SO:SOIC-8_3.9x4.9mm_P1.27mm",
                value="W25Q16_QSPI",
                comp_type="ic",
            ),
            "Y1": ComponentNode(
                comp_id="Y1",
                width=4.2,
                height=3.0,
                pads=[
                    Pad("1", -1.1, 0.0, 0.9, 1.2, 19, 1),
                    Pad("2", 1.1, 0.0, 0.9, 1.2, 1, 1),
                ],
                is_locked=False,
                x=15.0,
                y=14.5,
                footprint="Crystal:Crystal_SMD_3225",
                value="25MHz_OSC",
                comp_type="crystal",
            ),
            "C1": make_smd_passive("C1", "100nF", "cap", "0402", 2, 1, x=9.5, y=23.0, compact=True),
            "C2": make_smd_passive("C2", "100nF", "cap", "0402", 2, 1, x=21.5, y=23.0, compact=True),
            "C3": make_smd_passive("C3", "100nF", "cap", "0402", 2, 1, x=15.0, y=31.0, compact=True),
            "C4": make_smd_passive("C4", "100nF", "cap", "0402", 2, 1, x=24.5, y=8.0, compact=True),
            "D1": make_smd_passive("D1", "STAT_LED", "led", "0603", 20, 21, x=24.5, y=29.0, compact=True),
            "R1": make_smd_passive("R1", "10k_CS", "res", "0402", 2, 12, x=24.5, y=14.0, compact=True),
            "R2": make_smd_passive("R2", "330R_LED", "res", "0402", 21, 1, x=24.5, y=34.0, compact=True),
            "R3": make_smd_passive("R3", "10k_RST", "res", "0402", 2, 22, x=24.5, y=19.0, compact=True),
            "R4": make_smd_passive("R4", "10k_RX", "res", "0402", 2, 23, x=9.5, y=31.0, compact=True),
        }
        return _finalize_nets(
            "hdi_fpga_bga_fanout_8l",
            30.0,
            50.0,
            components,
            net_names,
            layers=8,
            design_notes=[
                "8-Layer (2+4+2) HDI stackup: F.Cu / In1.GND / In2.SIG1 / In3.PWR1 / In4.GND2 / In5.SIG2 / In6.GND3 / B.Cu",
                "Sequential lamination budget: max_lamination_cycles = 2 (no blind skip-vias L1<->L3 or L1<->L4)",
                "Staggered-only microvia reliability lockout: delta_xy >= 0.25mm across adjacent buildup layers",
            ],
        )

    if preset_name == "mixed_signal_gan_adc_30x50":
        from env.footprint_generator import make_qfn, make_smd_passive, make_sot23

        net_names = {
            1: "GND",
            2: "VIN_12V",
            3: "VCC_3V3",
            4: "SW_NODE",
            5: "BOOT_SW",
            6: "GAN_PWM",
            7: "SPI_SCK",
            8: "SPI_MOSI",
            9: "SPI_MISO",
            10: "ADC_CS_N",
            11: "ADC_AINP",
            12: "ADC_AINN",
            13: "VREF_2V5",
            14: "SENS_P_IN",
            15: "SENS_N_IN",
        }
        # U1: 5A 2MHz GaN Buck (F.Cu, power_switching)
        u1_gan = make_qfn(
            "U1",
            "LMG5200_GAN_5A_2MHZ",
            [2, 2, 1, 1, 4, 4, 5, 6, 3, 1, 2, 1],
            pitch=0.65,
            body_size=3.5,
            x=11.0,
            y=10.5,
            allow_fine_pitch=True,
            thermal_pad_net=1,
        )
        u1_gan.noise_domain = "power_switching"
        if "high_di_dt_loop" not in u1_gan.special_layout_rules:
            u1_gan.special_layout_rules.append("high_di_dt_loop")
        u1_gan.electrical_meta.update(
            {
                "power_watts": 3.5,
                "switching_freq_hz": 2_000_000,
                "current_a": 5.0,
                "noise_domain": "power_switching",
            }
        )

        # L1: Power inductor for 2MHz GaN buck (F.Cu, power_switching)
        l1_sw = ComponentNode(
            comp_id="L1",
            width=4.4,
            height=4.0,
            pads=[Pad("1", -1.4, 0.0, 1.1, 2.2, 4, 1), Pad("2", 1.4, 0.0, 1.1, 2.2, 3, 1)],
            x=19.0,
            y=10.5,
            footprint="Inductor_SMD:L_3.5x3.5_H1.8",
            value="2.2uH_GAN_SW",
            comp_type="inductor",
            special_layout_rules=["high_di_dt_loop"],
            noise_domain="power_switching",
        )

        c1_vin = make_smd_passive("C1", "10uF_VIN", "cap", "0603", 2, 1, x=11.0, y=15.5, compact=True)
        c1_vin.noise_domain = "power_switching"
        c2_boot = make_smd_passive("C2", "100nF_BOOT", "cap", "0402", 5, 4, x=15.0, y=15.0, compact=True)
        c2_boot.noise_domain = "power_switching"
        c3_vout = make_smd_passive("C3", "22uF_3V3", "cap", "0603", 3, 1, x=23.5, y=10.5, compact=True)
        c3_vout.noise_domain = "power_switching"

        # U4: Digital High-Speed FPGA Controller (mid-board bridge, digital_hs)
        u4_fpga = make_qfn(
            "U4",
            "ECP5_FPGA_CTRL",
            [3, 1, 6, 7, 8, 9, 10, 1, 3, 1, 7, 8],
            pitch=0.65,
            body_size=3.5,
            x=15.0,
            y=24.0,
            allow_fine_pitch=True,
        )
        u4_fpga.noise_domain = "digital_hs"

        c4_dvdd = make_smd_passive("C4", "100nF_DVDD", "cap", "0402", 3, 1, x=8.5, y=24.0, compact=True)
        c4_dvdd.noise_domain = "digital_hs"

        # U2: 24-bit sub-mV Precision ADC (F.Cu, analog_precision)
        u2_adc = make_qfn(
            "U2",
            "ADS1256_24B_ADC",
            [3, 1, 13, 11, 12, 1, 7, 8, 9, 10, 3, 1],
            pitch=0.65,
            body_size=3.5,
            x=14.0,
            y=38.5,
            allow_fine_pitch=True,
        )
        u2_adc.noise_domain = "analog_precision"
        u2_adc.special_layout_rules.append("kelvin_sense_pair")
        u2_adc.electrical_meta.update(
            {
                "noise_floor_mv": 0.05,
                "resolution_bits": 24,
                "noise_domain": "analog_precision",
            }
        )

        # U3: Precision 2.5V VREF (F.Cu, analog_precision)
        u3_vref = make_sot23("U3", "REF5025_VREF", [3, 1, 13, 1, 13], x=22.5, y=38.5, compact=True)
        u3_vref.noise_domain = "analog_precision"

        r1_ainp = make_smd_passive("R1", "100R_AINP", "res", "0402", 14, 11, x=7.5, y=36.5, compact=True)
        r1_ainp.noise_domain = "analog_precision"
        r2_ainn = make_smd_passive("R2", "100R_AINN", "res", "0402", 15, 12, x=7.5, y=40.5, compact=True)
        r2_ainn.noise_domain = "analog_precision"
        c5_vref = make_smd_passive("C5", "100nF_VREF", "cap", "0402", 13, 1, x=22.5, y=42.5, compact=True)
        c5_vref.noise_domain = "analog_precision"

        components = {
            "J1": ComponentNode(
                comp_id="J1",
                width=4.6,
                height=7.0,
                pads=_make_header_pads([2, 1], pitch=2.54),
                is_locked=True,
                x=15.0,
                y=4.5,
                theta=90,
                footprint="Connector_PinHeader_2.54mm:PinHeader_1x02_P2.54mm_Horizontal",
                value="PWR_12V_IN",
                comp_type="connector",
                noise_domain="power_switching",
            ),
            "J2": ComponentNode(
                comp_id="J2",
                width=4.6,
                height=11.0,
                pads=_make_header_pads([14, 15, 13, 1], pitch=2.54),
                is_locked=True,
                x=15.0,
                y=45.5,
                theta=90,
                footprint="Connector_PinHeader_2.54mm:PinHeader_1x04_P2.54mm_Horizontal",
                value="SUBMV_ANALOG_IN",
                comp_type="connector",
                noise_domain="analog_precision",
            ),
            "J3": ComponentNode(
                comp_id="J3",
                width=4.6,
                height=11.0,
                pads=_make_header_pads([3, 7, 8, 9], pitch=2.54),
                is_locked=True,
                x=26.5,
                y=24.0,
                theta=0,
                footprint="Connector_PinHeader_2.54mm:PinHeader_1x04_P2.54mm_Vertical",
                value="HS_SPI_HOST",
                comp_type="connector",
                noise_domain="digital_hs",
            ),
            "U1": u1_gan,
            "L1": l1_sw,
            "C1": c1_vin,
            "C2": c2_boot,
            "C3": c3_vout,
            "U4": u4_fpga,
            "C4": c4_dvdd,
            "U2": u2_adc,
            "U3": u3_vref,
            "R1": r1_ainp,
            "R2": r2_ainn,
            "C5": c5_vref,
        }
        return _finalize_nets(
            "mixed_signal_gan_adc_30x50",
            30.0,
            50.0,
            components,
            net_names,
            layers=8,
            design_notes=[
                "Mixed-signal 30x50mm 8L (2+4+2) HDI stackup with continuous In1.GND reference plane",
                "Both 5A 2MHz GaN converter (U1) and 24-bit sub-mV ADC (U2) constrained to Top F.Cu with spatial domain partitioning",
                "Zero ground splits under high-speed FPGA SPI/PWM traces; J_return switching corridor isolated from ADC >= 3.0mm halo",
            ],
        )

    if preset_name == "ultimate_rigid_flex_8l_hdi_pcie_gan_adc":
        from env.footprint_generator import make_bga, make_qfn, make_smd_passive, make_sot23

        net_names = {
            1: "GND",
            2: "VCC_3V3",
            3: "VIN_12V",
            4: "SW_NODE",
            5: "BOOT_SW",
            6: "PCIE_TX0_P",
            7: "PCIE_TX0_N",
            8: "PCIE_CLK_P",
            9: "PCIE_CLK_N",
            10: "SPI_SCK",
            11: "SPI_MISO",
            12: "ADC_CS_N",
            13: "ADC_AINP",
            14: "ADC_AINN",
            15: "VREF_2V5",
            16: "SENS_P_IN",
            17: "SENS_N_IN",
            18: "QSPI_SCK",
            19: "QSPI_CS",
        }
        zones = [
            SubstrateZone(
                zone_id="Rigid_Main",
                zone_type="rigid",
                x0_mm=0.0,
                y0_mm=0.0,
                x1_mm=30.0,
                y1_mm=34.0,
                er=4.5,
                prepreg_h_mm=0.10,
                layers=8,
                hatch_fill_ratio=1.0,
                allow_components=True,
                allow_vias=True,
                allowed_layers=[0, 1, 2, 3, 4, 5, 6, 7],
            ),
            SubstrateZone(
                zone_id="Dynamic_Flex",
                zone_type="dynamic_flex",
                x0_mm=0.0,
                y0_mm=34.0,
                x1_mm=30.0,
                y1_mm=41.5,
                er=3.4,
                prepreg_h_mm=0.050,
                layers=2,
                hatch_fill_ratio=0.50,
                hatch_angle_deg=45.0,
                allow_components=False,
                allow_vias=False,
                allowed_layers=[0],
            ),
            SubstrateZone(
                zone_id="Rigid_Tail",
                zone_type="rigid",
                x0_mm=0.0,
                y0_mm=41.5,
                x1_mm=30.0,
                y1_mm=50.0,
                er=4.5,
                prepreg_h_mm=0.10,
                layers=8,
                hatch_fill_ratio=1.0,
                allow_components=True,
                allow_vias=True,
                allowed_layers=[0, 1, 2, 3, 4, 5, 6, 7],
            ),
        ]

        # U1: 5A 2MHz GaN Buck (Top F.Cu, power_switching)
        u1_gan = make_qfn(
            "U1",
            "LMG5200_GAN_5A_2MHZ",
            [3, 3, 1, 1, 4, 4, 5, 2, 2, 1, 3, 1],
            pitch=0.65,
            body_size=3.5,
            x=4.5,
            y=4.5,
            allow_fine_pitch=True,
            thermal_pad_net=1,
        )
        u1_gan.noise_domain = "power_switching"
        u1_gan.special_layout_rules.extend(["top_only", "high_di_dt_loop"])
        u1_gan.electrical_meta.update(
            {
                "power_watts": 3.5,
                "switching_freq_hz": 2_000_000,
                "current_a": 5.0,
                "noise_domain": "power_switching",
            }
        )

        l1_sw = ComponentNode(
            comp_id="L1",
            width=4.4,
            height=4.0,
            pads=[Pad("1", -1.4, 0.0, 1.1, 2.2, 4, 1), Pad("2", 1.4, 0.0, 1.1, 2.2, 2, 1)],
            x=4.5,
            y=11.5,
            footprint="Inductor_SMD:L_3.5x3.5_H1.8",
            value="2.2uH_GAN_SW",
            comp_type="inductor",
            special_layout_rules=["top_only", "high_di_dt_loop"],
            noise_domain="power_switching",
        )
        c1_vin = make_smd_passive("C1", "10uF_VIN", "cap", "0603", 3, 1, x=10.5, y=8.0, compact=True)
        c1_vin.noise_domain = "power_switching"
        c1_vin.special_layout_rules.append("top_only")
        c2_boot = make_smd_passive("C2", "100nF_BOOT", "cap", "0402", 5, 4, x=10.5, y=11.0, compact=True)
        c2_boot.noise_domain = "power_switching"
        c2_boot.special_layout_rules.append("top_only")
        c3_vout = make_smd_passive("C3", "22uF_3V3", "cap", "0603", 2, 1, x=10.5, y=14.0, compact=True)
        c3_vout.noise_domain = "power_switching"
        c3_vout.special_layout_rules.append("top_only")

        # U4: 64-ball 0.8mm BGA FPGA (Rigid_Main, staggered HDI fanout + PCIe Gen 4 diff pairs + PDN)
        ball_nets_fpga: dict[str, int] = {
            "D4": 1, "D5": 1, "E4": 1, "E5": 1, "F4": 1, "F5": 1,
            "C4": 2, "C5": 2, "C2": 2, "F2": 2, "C7": 2, "F7": 2,
            "B2": 1, "B7": 1, "G2": 1, "G7": 1,
            "A1": 1, "A8": 1, "H1": 1, "H8": 1,
            "C3": 18, "C6": 19,
            "B4": 10, "B5": 11, "B6": 12,
            "H3": 6, "H4": 7,
            "H6": 8, "H7": 9,
        }
        u4_fpga = make_bga(
            "U4",
            "ECP5_FPGA_BGA64",
            ball_nets_fpga,
            rows=8,
            cols=8,
            pitch=0.8,
            x=11.5,
            y=21.5,
            is_locked=True,
        )
        u4_fpga.noise_domain = "digital_hs"
        u4_fpga.electrical_meta.update(
            {
                "edge_rate_ns": 0.5,
                "delta_i_per_cap_a": 0.025,
                "max_droop_mv": 65.0,
                "pdn_verify": True,
                "noise_domain": "digital_hs",
            }
        )

        # U5: QSPI Flash for inner-ball staggered microvia fanout
        u5_qspi = make_qfn(
            "U5",
            "W25Q16_QSPI",
            [18, 19, 2, 1, 18, 19, 2, 1],
            pitch=0.65,
            body_size=3.0,
            x=19.0,
            y=14.0,
            allow_fine_pitch=True,
        )
        u5_qspi.noise_domain = "digital_hs"

        # Unlocked decoupling caps initially remote on F.Cu (auto-migrated under U4 via-in-pad on B.Cu)
        c4_pdn = make_smd_passive("C4", "100nF_PDN1", "cap", "0402", 2, 1, x=4.5, y=21.5, compact=True)
        c4_pdn.noise_domain = "digital_hs"
        c6_pdn = make_smd_passive("C6", "100nF_PDN2", "cap", "0402", 2, 1, x=17.5, y=19.5, compact=True)
        c6_pdn.noise_domain = "digital_hs"

        # U2: 24-bit sub-mV ADC (Top F.Cu, analog_precision)
        u2_adc = make_qfn(
            "U2",
            "ADS1256_24B_ADC",
            [2, 1, 15, 13, 14, 1, 10, 11, 12, 1, 2, 1],
            pitch=0.65,
            body_size=3.5,
            x=23.5,
            y=26.2,
            allow_fine_pitch=True,
        )
        u2_adc.noise_domain = "analog_precision"
        u2_adc.special_layout_rules.extend(["top_only", "kelvin_sense_pair"])
        u2_adc.electrical_meta.update(
            {
                "noise_floor_mv": 0.05,
                "resolution_bits": 24,
                "noise_domain": "analog_precision",
            }
        )

        u3_vref = make_sot23("U3", "REF5025_VREF", [2, 1, 15, 1, 15], x=25.5, y=20.2, compact=True)
        u3_vref.noise_domain = "analog_precision"
        u3_vref.special_layout_rules.append("top_only")
        c5_vref = make_smd_passive("C5", "100nF_VREF", "cap", "0402", 15, 1, x=26.5, y=16.5, compact=True)
        c5_vref.noise_domain = "analog_precision"
        c5_vref.special_layout_rules.append("top_only")
        r1_ainp = make_smd_passive("R1", "100R_AINP", "res", "0402", 16, 13, x=19.0, y=26.2, compact=True)
        r1_ainp.noise_domain = "analog_precision"
        r1_ainp.special_layout_rules.append("top_only")
        r2_ainn = make_smd_passive("R2", "100R_AINN", "res", "0402", 17, 14, x=19.0, y=28.8, compact=True)
        r2_ainn.noise_domain = "analog_precision"
        r2_ainn.special_layout_rules.append("top_only")

        j1_pwr = ComponentNode(
            comp_id="J1",
            width=6.4,
            height=4.2,
            pads=[
                Pad("1", -1.27, 0.0, 1.3, 1.3, 3, 3),
                Pad("2", 1.27, 0.0, 1.3, 1.3, 1, 3),
            ],
            is_locked=True,
            x=12.5,
            y=3.3,
            theta=0,
            footprint="Connector_PinHeader_2.54mm:PinHeader_1x02_P2.54mm_Horizontal",
            value="PWR_12V_IN",
            comp_type="connector",
            noise_domain="power_switching",
        )
        j2_ain = ComponentNode(
            comp_id="J2",
            width=8.4,
            height=3.6,
            pads=[
                Pad("1", -2.54, 0.0, 1.2, 1.2, 16, 3),
                Pad("2", -0.84, 0.0, 1.2, 1.2, 17, 3),
                Pad("3", 0.84, 0.0, 1.2, 1.2, 15, 3),
                Pad("4", 2.54, 0.0, 1.2, 1.2, 1, 3),
            ],
            is_locked=True,
            x=24.5,
            y=31.8,
            theta=0,
            footprint="Connector_PinHeader_1.70mm:PinHeader_1x04_P1.70mm_Horizontal",
            value="SUBMV_ANALOG_IN",
            comp_type="connector",
            noise_domain="analog_precision",
        )
        j_pcie = ComponentNode(
            comp_id="J_PCIE",
            width=6.2,
            height=3.2,
            pads=[
                Pad("1", -2.0, 0.0, 0.45, 1.2, 1, 1),
                Pad("2", -1.2, 0.0, 0.45, 1.2, 6, 1),
                Pad("3", -0.4, 0.0, 0.45, 1.2, 7, 1),
                Pad("4", 0.4, 0.0, 0.45, 1.2, 1, 1),
                Pad("5", 1.2, 0.0, 0.45, 1.2, 8, 1),
                Pad("6", 2.0, 0.0, 0.45, 1.2, 9, 1),
            ],
            is_locked=True,
            x=11.5,
            y=45.5,
            theta=0,
            footprint="Connector_Hirose:DF40C-6DP-0.8V",
            value="PCIE_GEN4_B2B",
            comp_type="connector",
            noise_domain="digital_hs",
        )

        components = {
            "J1": j1_pwr,
            "J2": j2_ain,
            "J_PCIE": j_pcie,
            "U1": u1_gan,
            "L1": l1_sw,
            "C1": c1_vin,
            "C2": c2_boot,
            "C3": c3_vout,
            "U4": u4_fpga,
            "U5": u5_qspi,
            "C4": c4_pdn,
            "C6": c6_pdn,
            "U2": u2_adc,
            "U3": u3_vref,
            "C5": c5_vref,
            "R1": r1_ainp,
            "R2": r2_ainn,
        }
        return _finalize_nets(
            "ultimate_rigid_flex_8l_hdi_pcie_gan_adc",
            30.0,
            50.0,
            components,
            net_names,
            layers=8,
            substrate_zones=zones,
            design_notes=[
                "Rigid-Flex 8L (2+4+2) HDI FR4 (er=4.5, h=0.10mm) + 2L Polyimide Dynamic Flex (er=3.4, h=0.050mm, 50% cross-hatched In1.GND)",
                "85-Ohm PCIe Gen 4 differential pairs dynamically transition (w_rigid, s_rigid) <-> (w_flex, s_flex) across seam with skew <= 0.125mm",
                "Both 5A 2MHz GaN converter (U1) and 24-bit sub-mV ADC (U2) constrained to Top F.Cu inside Rigid_Main with >=20mm separation & 0% J_return intrusion",
                "Staggered-only microvias (>=0.25mm) + max 2 lamination cycles + under-BGA PDN decoupling (L_loop <= 1.2nH)",
            ],
        )

    raise ValueError(
        f"Unknown preset '{preset_name}'. Available presets: {list(FULL_PRESET_CATALOG)}"
    )


def build_rev13_broadside_trap() -> tuple[NetlistGraph, list[dict[str, Any]]]:
    """Build B14a adversarial thin-core 8L state with pre-seeded parallel L3 analog vs L4 fast clock overlap."""
    from env.footprint_generator import make_smd_passive

    net_names = {
        1: "GND",
        2: "VCC_3V3",
        3: "CLK_100M_HS",
        4: "AIN_SUBMV_P",
    }
    components = {
        "U_CLK_SRC": ComponentNode(
            comp_id="U_CLK_SRC",
            width=4.0,
            height=3.6,
            pads=[Pad("1", 0.0, 0.0, 0.8, 0.8, 3, 1), Pad("2", 1.2, 0.0, 0.8, 0.8, 1, 1)],
            is_locked=True,
            x=6.0,
            y=10.0,
            footprint="Oscillator:Osc_3225",
            value="100MHz_CLK_OSC",
            comp_type="crystal",
            noise_domain="digital_hs",
        ),
        "U_CLK_DST": make_smd_passive("U_CLK_DST", "33R_CLK", "res", "0402", 3, 1, x=24.0, y=10.0, compact=True),
        "J_SENS": ComponentNode(
            comp_id="J_SENS",
            width=4.0,
            height=3.6,
            pads=[Pad("1", 0.0, 0.0, 0.8, 0.8, 4, 1), Pad("2", 0.0, 1.2, 0.8, 0.8, 1, 1)],
            is_locked=True,
            x=6.0,
            y=20.0,
            footprint="Connector_SMD:Pad_SubmV",
            value="SUBMV_SENSOR",
            comp_type="connector",
            noise_domain="analog_precision",
        ),
        "U_ADC_IN": ComponentNode(
            comp_id="U_ADC_IN",
            width=4.0,
            height=3.6,
            pads=[Pad("1", 0.0, 0.0, 0.8, 0.8, 4, 1), Pad("2", 0.0, 1.2, 0.8, 0.8, 1, 1)],
            is_locked=True,
            x=24.0,
            y=20.0,
            footprint="Package_SO:ADC_24B_SUBMV",
            value="ADS1256_24B_ADC",
            comp_type="ic",
            noise_domain="analog_precision",
            electrical_meta={"noise_floor_mv": 0.1, "noise_domain": "analog_precision"},
        ),
    }
    components["U_CLK_DST"].is_locked = True
    components["U_CLK_DST"].noise_domain = "digital_hs"
    graph = _finalize_nets(
        "z_axis_broadside_crosstalk_trap",
        30.0,
        30.0,
        components,
        net_names,
        layers=8,
        design_notes=[
            "Adversarial B14a thin-core 8L state: fast 100MHz clock on L4 (In3.PWR1 idx=3) directly under L3 (In2.SIG1 idx=2) sub-mV analog route."
        ],
    )
    # Pre-seeded adversarial parallel segments (L4 idx=3 clock vs L3 idx=2 analog stacked along y=15.0)
    preseeded_segments = [
        {"net_id": 3, "layer": 3, "x1_mm": 6.0, "y1_mm": 10.0, "x2_mm": 6.0, "y2_mm": 15.0, "width_mm": 0.15},
        {"net_id": 3, "layer": 3, "x1_mm": 6.0, "y1_mm": 15.0, "x2_mm": 24.0, "y2_mm": 15.0, "width_mm": 0.15},
        {"net_id": 3, "layer": 3, "x1_mm": 24.0, "y1_mm": 15.0, "x2_mm": 24.0, "y2_mm": 10.0, "width_mm": 0.15},
        {"net_id": 4, "layer": 2, "x1_mm": 6.0, "y1_mm": 20.0, "x2_mm": 6.0, "y2_mm": 15.0, "width_mm": 0.15},
        {"net_id": 4, "layer": 2, "x1_mm": 6.0, "y1_mm": 15.0, "x2_mm": 24.0, "y2_mm": 15.0, "width_mm": 0.15},
        {"net_id": 4, "layer": 2, "x1_mm": 24.0, "y1_mm": 15.0, "x2_mm": 24.0, "y2_mm": 20.0, "width_mm": 0.15},
    ]
    return graph, preseeded_segments


def build_rev13_star_ground_trap() -> NetlistGraph:
    """Build B14b adversarial mixed-signal board with invalid remote AGND<->DGND star-ground instruction."""
    from env.footprint_generator import make_qfn, make_smd_passive

    net_names = {
        1: "GND",
        2: "VCC_3V3",
        3: "SPI_CLK_HS",
        4: "SPI_MOSI_HS",
        5: "ADC_AINP",
        6: "ADC_AINN",
        7: "VREF_2V5",
    }
    u_mcu = make_qfn(
        "U1",
        "STM32_HS_MCU",
        [2, 1, 3, 4, 1, 2, 3, 4],
        pitch=0.8,
        body_size=4.0,
        x=10.0,
        y=14.0,
        is_locked=True,
    )
    u_mcu.noise_domain = "digital_hs"
    u_adc = make_qfn(
        "U2",
        "ADS1256_24B_ADC",
        [2, 1, 5, 6, 7, 1, 3, 4],
        pitch=0.8,
        body_size=4.0,
        x=30.0,
        y=14.0,
        is_locked=True,
    )
    u_adc.noise_domain = "analog_precision"
    c1 = make_smd_passive("C1", "100nF_MCU", "cap", "0402", 2, 1, x=10.0, y=22.0, compact=True)
    c1.noise_domain = "digital_hs"
    c2 = make_smd_passive("C2", "100nF_ADC", "cap", "0402", 7, 1, x=30.0, y=22.0, compact=True)
    c2.noise_domain = "analog_precision"
    r1 = make_smd_passive("R1", "100R_AIN", "res", "0402", 5, 6, x=30.0, y=28.0, compact=True)
    r1.noise_domain = "analog_precision"

    return _finalize_nets(
        "remote_star_ground_contradiction",
        40.0,
        40.0,
        {"U1": u_mcu, "U2": u_adc, "C1": c1, "C2": c2, "R1": r1},
        net_names,
        layers=4,
        ground_constraints={
            "mode": "remote_star_ground",
            "split_axis": "x",
            "split_pos_mm": 20.0,
            "bridge_xy": [20.0, 36.0],
            "split_nets": ["AGND", "DGND"],
            "reason": "Adversarial remote star-ground split between MCU (x=10) and ADC (x=30) with bridge at y=36mm far below SPI crossing at y=14mm",
        },
        design_notes=[
            "Adversarial B14b state specifies remote AGND<->DGND bridge at (20, 36) while SPI_CLK/MOSI cross x=20 at y=14mm."
        ],
    )


def build_rev13_thermal_inpainting_trap() -> tuple[str, dict[str, Any]]:
    """Build B14c 80%-locked enclosure `.kicad_pcb` + `electrical_meta` JSON with 5W MOSFET pinned near <1mV ADC."""
    from env.board_state import BoardState
    from env.config import BoardConfig
    from env.footprint_generator import make_power_mosfet_or_transistor, make_qfn, make_smd_passive
    from env.kicad_exporter import serialize_kicad_pcb

    net_names = {
        1: "GND",
        2: "VIN_12V",
        3: "SW_NODE",
        4: "VCC_3V3",
        5: "VREF_2V5",
        6: "AIN_SUBMV_P",
        7: "AIN_SUBMV_N",
    }
    q1 = make_power_mosfet_or_transistor(
        "Q1", "GAN_FET_5W_2MHZ", "DFN3x3", [2, 3, 3, 1, 3], x=10.5, y=16.0
    )
    q1.is_locked = True
    u_adc = make_qfn(
        "U_ADC", "ADS1256_24B_ADC", [4, 1, 5, 6, 7, 1, 4, 1], pitch=0.65, body_size=3.5, x=21.5, y=16.0, is_locked=True, allow_fine_pitch=True
    )
    j_pwr = ComponentNode(
        comp_id="J_PWR",
        width=4.2,
        height=7.0,
        pads=_make_header_pads([2, 1], pitch=2.54),
        is_locked=True,
        x=4.0,
        y=16.0,
        footprint="Connector_PinHeader_2.54mm:PinHeader_1x02_P2.54mm_Vertical",
        value="VIN_HDR",
        comp_type="connector",
    )
    j_adc = ComponentNode(
        comp_id="J_ADC",
        width=4.2,
        height=9.5,
        pads=_make_header_pads([6, 7, 1], pitch=2.54),
        is_locked=True,
        x=28.0,
        y=16.0,
        footprint="Connector_PinHeader_2.54mm:PinHeader_1x03_P2.54mm_Vertical",
        value="ADC_IN_HDR",
        comp_type="connector",
    )
    # 20% unlocked components (passives + VREF)
    c_snub = make_smd_passive("C_SNUB", "10uF_VIN", "cap", "0402", 2, 1, x=10.5, y=9.5, compact=True)
    c_vref = make_smd_passive("C_VREF", "100nF_VREF", "cap", "0402", 5, 1, x=21.5, y=24.0, compact=True)
    r_filt = make_smd_passive("R_FILT", "100R_VREF", "res", "0402", 4, 5, x=21.5, y=9.5, compact=True)

    graph = _finalize_nets(
        "thermal_vs_signal_inpainting_80pct",
        32.0,
        32.0,
        {
            "Q1": q1,
            "U_ADC": u_adc,
            "J_PWR": j_pwr,
            "J_ADC": j_adc,
            "C_SNUB": c_snub,
            "C_VREF": c_vref,
            "R_FILT": r_filt,
        },
        net_names,
        layers=4,
    )
    cfg = BoardConfig(board_width_mm=32.0, board_height_mm=32.0, layers=4)
    state = BoardState(netlist=graph, config=cfg)
    pcb_text = serialize_kicad_pcb(state, include_gnd_pour=False)

    electrical_meta = {
        "Q1": {
            "power_watts": 5.0,
            "switching_freq_hz": 2_000_000,
            "noise_domain": "power_switching",
        },
        "U_ADC": {
            "noise_floor_mv": 0.2,
            "resolution_bits": 24,
            "noise_domain": "analog_precision",
        },
        "C_SNUB": {"noise_domain": "power_switching"},
        "C_VREF": {"noise_domain": "analog_precision"},
        "R_FILT": {"noise_domain": "analog_precision"},
    }
    return pcb_text, electrical_meta


def build_rev14_pdn_inductance_trap() -> tuple[NetlistGraph, list[dict[str, Any]]]:
    """Build B15a adversarial PDN mounting inductance trap (remote caps + skinny surface traces)."""
    from env.footprint_generator import make_bga, make_smd_passive

    net_names = {
        1: "GND",
        2: "VCC_3V3",
        3: "SPI_SCK",
        4: "SPI_MOSI",
        5: "SPI_MISO",
        6: "SPI_CS",
    }
    ball_nets: dict[str, int] = {
        "D4": 1, "D5": 1, "E4": 1, "E5": 1,
        "F4": 1, "F5": 1,
        "C4": 2, "C5": 2, "C2": 2, "F2": 2, "C7": 2, "F7": 2,
        "B2": 1, "B7": 1, "G2": 1, "G7": 1,
        "A1": 1, "A8": 1, "H1": 1, "H8": 1,
        "A3": 3, "A4": 4, "A5": 5, "A6": 6,
    }
    u1_fpga = make_bga(
        "U1",
        "ECP5_FPGA_BGA64",
        ball_nets,
        rows=8,
        cols=8,
        pitch=0.8,
        x=15.0,
        y=17.5,
        is_locked=True,
    )
    u1_fpga.noise_domain = "digital_hs"
    u1_fpga.electrical_meta.update(
        {
            "edge_rate_ns": 0.5,
            "delta_i_per_cap_a": 0.025,
            "max_droop_mv": 65.0,
            "pdn_verify": True,
            "noise_domain": "digital_hs",
        }
    )

    j1_io = ComponentNode(
        comp_id="J1",
        width=4.2,
        height=13.0,
        pads=_make_header_pads([3, 4, 5, 6, 1], pitch=2.54),
        is_locked=True,
        x=4.5,
        y=17.5,
        theta=0,
        footprint="Connector_PinHeader_2.54mm:PinHeader_1x05_P2.54mm_Vertical",
        value="FPGA_SPI_HDR",
        comp_type="connector",
        noise_domain="digital_hs",
    )

    # Hostile remote bypass caps connected via long thin 0.15mm F.Cu traces
    c1 = make_smd_passive("C1", "100nF_PDN1", "cap", "0402", 2, 1, x=26.0, y=6.5, compact=True)
    c2 = make_smd_passive("C2", "100nF_PDN2", "cap", "0402", 2, 1, x=26.0, y=28.5, compact=True)
    c1.noise_domain = "digital_hs"
    c2.noise_domain = "digital_hs"

    graph = _finalize_nets(
        "pdn_decoupling_inductance_trap",
        30.0,
        35.0,
        {"U1": u1_fpga, "J1": j1_io, "C1": c1, "C2": c2},
        net_names,
        layers=8,
        design_notes=[
            "Adversarial PDN Trap (B15a): FPGA edge_rate_ns=0.5 with remote decoupling caps & 0.15mm surface traces"
        ],
    )

    # Pre-seed long skinny F.Cu traces from C1/C2 to U1 power & ground balls
    c1_p_wx, c1_p_wy = c1.pad_world_pos(c1.pads[0])
    c1_g_wx, c1_g_wy = c1.pad_world_pos(c1.pads[1])
    c2_p_wx, c2_p_wy = c2.pad_world_pos(c2.pads[0])
    c2_g_wx, c2_g_wy = c2.pad_world_pos(c2.pads[1])
    u1_pwr_top = next(p for p in u1_fpga.pads if p.pad_id == "C7")
    u1_gnd_top = next(p for p in u1_fpga.pads if p.pad_id == "B7")
    u1_pwr_bot = next(p for p in u1_fpga.pads if p.pad_id == "F7")
    u1_gnd_bot = next(p for p in u1_fpga.pads if p.pad_id == "G7")
    u1_pt_wx, u1_pt_wy = u1_fpga.pad_world_pos(u1_pwr_top)
    u1_gt_wx, u1_gt_wy = u1_fpga.pad_world_pos(u1_gnd_top)
    u1_pb_wx, u1_pb_wy = u1_fpga.pad_world_pos(u1_pwr_bot)
    u1_gb_wx, u1_gb_wy = u1_fpga.pad_world_pos(u1_gnd_bot)

    preseeded_segs = [
        {"net_id": 2, "layer": 0, "x1_mm": c1_p_wx, "y1_mm": c1_p_wy, "x2_mm": u1_pt_wx, "y2_mm": u1_pt_wy, "width_mm": 0.15},
        {"net_id": 1, "layer": 0, "x1_mm": c1_g_wx, "y1_mm": c1_g_wy, "x2_mm": u1_gt_wx, "y2_mm": u1_gt_wy, "width_mm": 0.15},
        {"net_id": 2, "layer": 0, "x1_mm": c2_p_wx, "y1_mm": c2_p_wy, "x2_mm": u1_pb_wx, "y2_mm": u1_pb_wy, "width_mm": 0.15},
        {"net_id": 1, "layer": 0, "x1_mm": c2_g_wx, "y1_mm": c2_g_wy, "x2_mm": u1_gb_wx, "y2_mm": u1_gb_wy, "width_mm": 0.15},
    ]
    graph.raw_segments = preseeded_segs
    graph.enclosure_meta["preserve_preseeded_traces"] = True
    return graph, preseeded_segs
