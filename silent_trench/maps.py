"""Bathymetric maps for Silent Trench: 5 Atlantic and 5 Pacific deep sea locations."""
from dataclasses import dataclass, field
from typing import Dict, List, Tuple
from silent_trench.constants import NodeType, OceanRegion


@dataclass
class MapNode:
    id: int
    name: str
    depth_m: int
    node_type: NodeType
    x: float  # Normalized 0.0 - 1.0
    y: float  # Normalized 0.0 - 1.0
    neighbors: List[int] = field(default_factory=list)


@dataclass
class BathymetricMap:
    id: str
    name: str
    region: OceanRegion
    description: str
    nodes: Dict[int, MapNode]
    p0_start: int
    p1_start: int

    def __post_init__(self):
        # Guarantee graph symmetry and uniqueness
        for u, node in self.nodes.items():
            for v in list(node.neighbors):
                if u not in self.nodes[v].neighbors:
                    self.nodes[v].neighbors.append(u)
            node.neighbors = sorted(list(set(node.neighbors)))

    @property
    def num_nodes(self) -> int:
        return len(self.nodes)

    def get_neighbors(self, node_id: int) -> List[int]:
        return self.nodes[node_id].neighbors

    def are_connected(self, u: int, v: int) -> bool:
        return v in self.nodes[u].neighbors


# --- ATLANTIC MAPS ---

def _build_puerto_rico_trench() -> BathymetricMap:
    # 10 nodes
    nodes = {
        0: MapNode(0, "Mona Canyon", 3800, NodeType.CHOKEPOINT, 0.15, 0.65, [1, 2, 7]),
        1: MapNode(1, "San Juan Ridge", 2400, NodeType.LISTENING_POST, 0.35, 0.50, [0, 2, 3]),
        2: MapNode(2, "Brownson Deep", 8100, NodeType.ABYSSAL_FLOOR, 0.35, 0.20, [0, 1, 3, 4]),
        3: MapNode(3, "Milwaukee Deep", 8376, NodeType.ABYSSAL_FLOOR, 0.55, 0.18, [1, 2, 4, 5]),
        4: MapNode(4, "North Escarpment", 4500, NodeType.THERMAL_LAYER, 0.70, 0.25, [2, 3, 5, 9]),
        5: MapNode(5, "Virgin Islands Basin", 4200, NodeType.LISTENING_POST, 0.65, 0.52, [3, 4, 6, 8]),
        6: MapNode(6, "Anegada Passage", 1900, NodeType.CHOKEPOINT, 0.85, 0.60, [5, 9]),
        7: MapNode(7, "Muertos Trough", 5100, NodeType.THERMAL_LAYER, 0.20, 0.85, [0, 8]),
        8: MapNode(8, "Culebra Basin", 3600, NodeType.STANDARD, 0.50, 0.82, [7, 5]),
        9: MapNode(9, "Atlantic Abyssal Plain", 5600, NodeType.STANDARD, 0.90, 0.30, [4, 6]),
    }
    return BathymetricMap(
        id="puerto_rico_trench",
        name="Puerto Rico Trench",
        region=OceanRegion.ATLANTIC,
        description="The deepest trench in the Atlantic with Milwaukee Deep (8,376m) and steep northern escarpments.",
        nodes=nodes,
        p0_start=0,
        p1_start=6,
    )


def _build_romanche_trench() -> BathymetricMap:
    # 10 nodes
    nodes = {
        0: MapNode(0, "Brazil Basin Pass", 4900, NodeType.STANDARD, 0.10, 0.55, [1, 4]),
        1: MapNode(1, "St. Paul Fracture", 4100, NodeType.CHOKEPOINT, 0.25, 0.35, [0, 2, 5]),
        2: MapNode(2, "Mid-Ridge Rift", 4400, NodeType.LISTENING_POST, 0.45, 0.40, [1, 3, 5, 6]),
        3: MapNode(3, "Romanche Abyss", 7760, NodeType.ABYSSAL_FLOOR, 0.55, 0.65, [2, 4, 7, 8]),
        4: MapNode(4, "South Flank Ridge", 3100, NodeType.THERMAL_LAYER, 0.25, 0.75, [0, 3, 7]),
        5: MapNode(5, "North Flank Ridge", 2900, NodeType.STANDARD, 0.40, 0.20, [1, 2, 8]),
        6: MapNode(6, "Equatorial Saddle", 4800, NodeType.THERMAL_LAYER, 0.65, 0.35, [2, 8, 9]),
        7: MapNode(7, "Chain Fracture", 6900, NodeType.ABYSSAL_FLOOR, 0.70, 0.75, [3, 4, 9]),
        8: MapNode(8, "Sierra Leone Rise", 3200, NodeType.LISTENING_POST, 0.80, 0.25, [3, 5, 6, 9]),
        9: MapNode(9, "Guinea Basin Entry", 5100, NodeType.CHOKEPOINT, 0.90, 0.55, [6, 7, 8]),
    }
    return BathymetricMap(
        id="romanche_trench",
        name="Romanche Trench",
        region=OceanRegion.ATLANTIC,
        description="A massive fracture zone slicing through the equatorial Mid-Atlantic Ridge down to 7,760m.",
        nodes=nodes,
        p0_start=0,
        p1_start=9,
    )


def _build_south_sandwich_trench() -> BathymetricMap:
    # 10 nodes
    nodes = {
        0: MapNode(0, "Scotia Sea Basin", 4300, NodeType.STANDARD, 0.15, 0.30, [1, 2]),
        1: MapNode(1, "Zavodovski Seamount", 1400, NodeType.LISTENING_POST, 0.35, 0.25, [0, 3, 4]),
        2: MapNode(2, "Candlemas Ridge", 2500, NodeType.CHOKEPOINT, 0.25, 0.55, [0, 3, 5]),
        3: MapNode(3, "Saunders Chasm", 6200, NodeType.THERMAL_LAYER, 0.45, 0.45, [1, 2, 4, 6]),
        4: MapNode(4, "Meteor Deep", 8264, NodeType.ABYSSAL_FLOOR, 0.58, 0.30, [1, 3, 7]),
        5: MapNode(5, "Montagu Bank", 1800, NodeType.STANDARD, 0.40, 0.75, [2, 6, 8]),
        6: MapNode(6, "Bristol Escarpment", 5800, NodeType.THERMAL_LAYER, 0.60, 0.60, [3, 5, 7, 9]),
        7: MapNode(7, "Trench Floor North", 7900, NodeType.ABYSSAL_FLOOR, 0.75, 0.35, [4, 6, 8]),
        8: MapNode(8, "Southern Thule Rift", 7100, NodeType.LISTENING_POST, 0.70, 0.85, [5, 7, 9]),
        9: MapNode(9, "Weddell Abyss", 5200, NodeType.CHOKEPOINT, 0.90, 0.70, [6, 8]),
    }
    return BathymetricMap(
        id="south_sandwich_trench",
        name="South Sandwich Trench",
        region=OceanRegion.ATLANTIC,
        description="Sub-Antarctic glacial chasm holding Meteor Deep (8,264m) amidst volcanic island arcs.",
        nodes=nodes,
        p0_start=0,
        p1_start=9,
    )


def _build_mid_atlantic_ridge() -> BathymetricMap:
    # 10 nodes
    nodes = {
        0: MapNode(0, "Western Rift Wall", 2200, NodeType.STANDARD, 0.15, 0.45, [1, 2]),
        1: MapNode(1, "Atlantis Massif", 700, NodeType.LISTENING_POST, 0.28, 0.25, [0, 3, 4]),
        2: MapNode(2, "Lost City Vents", 800, NodeType.THERMAL_LAYER, 0.30, 0.70, [0, 3, 5]),
        3: MapNode(3, "Median Valley Rift", 3700, NodeType.CHOKEPOINT, 0.48, 0.50, [1, 2, 6, 7]),
        4: MapNode(4, "Abyssal Hills North", 4100, NodeType.STANDARD, 0.45, 0.15, [1, 6]),
        5: MapNode(5, "Abyssal Hills South", 4200, NodeType.STANDARD, 0.45, 0.85, [2, 7]),
        6: MapNode(6, "Snake Pit Field", 3480, NodeType.THERMAL_LAYER, 0.65, 0.30, [3, 4, 8]),
        7: MapNode(7, "Broken Spur Field", 3080, NodeType.LISTENING_POST, 0.65, 0.70, [3, 5, 8]),
        8: MapNode(8, "Eastern Rift Wall", 2100, NodeType.CHOKEPOINT, 0.80, 0.50, [6, 7, 9]),
        9: MapNode(9, "Kane Fracture Zone", 6000, NodeType.ABYSSAL_FLOOR, 0.92, 0.50, [8]),
    }
    return BathymetricMap(
        id="mid_atlantic_ridge",
        name="Mid-Atlantic Ridge",
        region=OceanRegion.ATLANTIC,
        description="Rugged tectonic mountain range featuring the Atlantis Massif, hydrothermal vents, and deep rifts.",
        nodes=nodes,
        p0_start=0,
        p1_start=9,
    )


def _build_cayman_trough() -> BathymetricMap:
    # 10 nodes
    nodes = {
        0: MapNode(0, "Swan Islands Basin", 5200, NodeType.STANDARD, 0.10, 0.60, [1, 2]),
        1: MapNode(1, "Misteriosa Bank", 1200, NodeType.CHOKEPOINT, 0.25, 0.30, [0, 3, 4]),
        2: MapNode(2, "Swan Ridge", 2200, NodeType.LISTENING_POST, 0.30, 0.75, [0, 3, 5]),
        3: MapNode(3, "Beebe Vent Field", 4960, NodeType.THERMAL_LAYER, 0.45, 0.45, [1, 2, 4, 5, 6]),
        4: MapNode(4, "Grand Cayman Shelf", 1800, NodeType.STANDARD, 0.50, 0.18, [1, 3, 7]),
        5: MapNode(5, "Spreading Center", 5300, NodeType.THERMAL_LAYER, 0.55, 0.82, [2, 3, 8]),
        6: MapNode(6, "Cayman Abyss", 7686, NodeType.ABYSSAL_FLOOR, 0.65, 0.50, [3, 7, 8]),
        7: MapNode(7, "Bartlett Trough", 6800, NodeType.ABYSSAL_FLOOR, 0.78, 0.32, [4, 6, 9]),
        8: MapNode(8, "Oriente Fracture", 6100, NodeType.LISTENING_POST, 0.80, 0.70, [5, 6, 9]),
        9: MapNode(9, "Windward Passage", 3100, NodeType.CHOKEPOINT, 0.92, 0.45, [7, 8]),
    }
    return BathymetricMap(
        id="cayman_trough",
        name="Cayman Trough",
        region=OceanRegion.ATLANTIC,
        description="An ultra-deep pull-apart basin in the Caribbean holding the deepest known hydrothermal vents.",
        nodes=nodes,
        p0_start=0,
        p1_start=9,
    )


# --- PACIFIC MAPS ---

def _build_mariana_trench() -> BathymetricMap:
    # 11 nodes
    nodes = {
        0: MapNode(0, "West Mariana Basin", 4500, NodeType.STANDARD, 0.10, 0.50, [1, 2]),
        1: MapNode(1, "Mariana Arc Seamount", 1600, NodeType.LISTENING_POST, 0.25, 0.25, [0, 3, 4]),
        2: MapNode(2, "Guam Channel", 3400, NodeType.CHOKEPOINT, 0.25, 0.75, [0, 4, 5]),
        3: MapNode(3, "Saipan Ridge", 1900, NodeType.STANDARD, 0.42, 0.15, [1, 6]),
        4: MapNode(4, "Sirena Deep", 10714, NodeType.ABYSSAL_FLOOR, 0.45, 0.50, [1, 2, 6, 7]),
        5: MapNode(5, "Yap Trench Junction", 8527, NodeType.THERMAL_LAYER, 0.40, 0.85, [2, 7]),
        6: MapNode(6, "Nero Deep", 9660, NodeType.ABYSSAL_FLOOR, 0.65, 0.30, [3, 4, 8, 9]),
        7: MapNode(7, "Challenger Deep", 10994, NodeType.ABYSSAL_FLOOR, 0.65, 0.70, [4, 5, 8, 10]),
        8: MapNode(8, "Central Abyssal Sump", 9800, NodeType.THERMAL_LAYER, 0.78, 0.50, [6, 7, 9, 10]),
        9: MapNode(9, "North Marianas Plain", 5900, NodeType.LISTENING_POST, 0.90, 0.25, [6, 8]),
        10: MapNode(10, "Mariana Back-Arc", 4100, NodeType.CHOKEPOINT, 0.92, 0.75, [7, 8]),
    }
    return BathymetricMap(
        id="mariana_trench",
        name="Mariana Trench",
        region=OceanRegion.PACIFIC,
        description="The deepest oceanic chasm on Earth, featuring Challenger Deep (10,994m) and extreme crushing pressures.",
        nodes=nodes,
        p0_start=0,
        p1_start=10,
    )


def _build_tonga_trench() -> BathymetricMap:
    # 10 nodes
    nodes = {
        0: MapNode(0, "Lau Basin Back-Arc", 2800, NodeType.STANDARD, 0.12, 0.50, [1, 2]),
        1: MapNode(1, "Valu Fa Hydrothermal", 1900, NodeType.THERMAL_LAYER, 0.25, 0.25, [0, 3, 4]),
        2: MapNode(2, "Tonga Ridge Shelf", 900, NodeType.LISTENING_POST, 0.28, 0.75, [0, 4, 5]),
        3: MapNode(3, "Ata Fracture", 6200, NodeType.CHOKEPOINT, 0.45, 0.20, [1, 6]),
        4: MapNode(4, "Central Tonga Abyss", 9500, NodeType.ABYSSAL_FLOOR, 0.50, 0.50, [1, 2, 6, 7]),
        5: MapNode(5, "Niue Basin Entry", 4700, NodeType.STANDARD, 0.48, 0.85, [2, 7]),
        6: MapNode(6, "Vityaz Depth", 10200, NodeType.ABYSSAL_FLOOR, 0.70, 0.30, [3, 4, 8]),
        7: MapNode(7, "Horizon Deep", 10800, NodeType.ABYSSAL_FLOOR, 0.72, 0.70, [4, 5, 8, 9]),
        8: MapNode(8, "Louisville Junction", 5600, NodeType.LISTENING_POST, 0.85, 0.35, [6, 7, 9]),
        9: MapNode(9, "Kermadec Gap", 7400, NodeType.CHOKEPOINT, 0.90, 0.75, [7, 8]),
    }
    return BathymetricMap(
        id="tonga_trench",
        name="Tonga Trench",
        region=OceanRegion.PACIFIC,
        description="Fast tectonic subduction zone in the South Pacific harboring Horizon Deep (10,800m).",
        nodes=nodes,
        p0_start=0,
        p1_start=9,
    )


def _build_philippine_trench() -> BathymetricMap:
    # 10 nodes
    nodes = {
        0: MapNode(0, "Samar Basin", 3200, NodeType.STANDARD, 0.12, 0.35, [1, 2]),
        1: MapNode(1, "Surigao Ridge", 1400, NodeType.CHOKEPOINT, 0.25, 0.65, [0, 3, 4]),
        2: MapNode(2, "Benham Rise Crest", 2000, NodeType.LISTENING_POST, 0.30, 0.20, [0, 3, 5]),
        3: MapNode(3, "Leyte Chasm", 7200, NodeType.THERMAL_LAYER, 0.45, 0.45, [1, 2, 4, 6]),
        4: MapNode(4, "Mindanao Escarpment", 5100, NodeType.STANDARD, 0.42, 0.80, [1, 3, 7]),
        5: MapNode(5, "Cape Johnson Deep", 10497, NodeType.ABYSSAL_FLOOR, 0.60, 0.22, [2, 6, 8]),
        6: MapNode(6, "Galathea Deep", 10265, NodeType.ABYSSAL_FLOOR, 0.65, 0.50, [3, 5, 7, 8]),
        7: MapNode(7, "Emden Deep", 10540, NodeType.ABYSSAL_FLOOR, 0.65, 0.80, [4, 6, 9]),
        8: MapNode(8, "Philippine Sea Plain", 5800, NodeType.LISTENING_POST, 0.85, 0.30, [5, 6, 9]),
        9: MapNode(9, "Palau Basin Pass", 4600, NodeType.CHOKEPOINT, 0.90, 0.65, [7, 8]),
    }
    return BathymetricMap(
        id="philippine_trench",
        name="Philippine Trench",
        region=OceanRegion.PACIFIC,
        description="Island arc trench home to Emden Deep (10,540m) and complex tectonic subduction channels.",
        nodes=nodes,
        p0_start=0,
        p1_start=9,
    )


def _build_kuril_kamchatka_trench() -> BathymetricMap:
    # 10 nodes
    nodes = {
        0: MapNode(0, "Okhotsk Deep", 3374, NodeType.STANDARD, 0.12, 0.70, [1, 2]),
        1: MapNode(1, "Bussol Strait", 2300, NodeType.CHOKEPOINT, 0.25, 0.45, [0, 3, 4]),
        2: MapNode(2, "Kuril Basin", 3300, NodeType.STANDARD, 0.28, 0.85, [0, 4, 5]),
        3: MapNode(3, "Vityaz Ridge", 1800, NodeType.LISTENING_POST, 0.40, 0.25, [1, 6]),
        4: MapNode(4, "Simushir Canyon", 6400, NodeType.THERMAL_LAYER, 0.45, 0.60, [1, 2, 6, 7]),
        5: MapNode(5, "Iturup Deep", 7100, NodeType.THERMAL_LAYER, 0.45, 0.90, [2, 7]),
        6: MapNode(6, "Cold Sound Channel", 5100, NodeType.LISTENING_POST, 0.65, 0.35, [3, 4, 8]),
        7: MapNode(7, "Kamchatka Abyss", 10542, NodeType.ABYSSAL_FLOOR, 0.70, 0.75, [4, 5, 8, 9]),
        8: MapNode(8, "Hokkaido Entry", 8200, NodeType.CHOKEPOINT, 0.85, 0.40, [6, 7, 9]),
        9: MapNode(9, "Pacific Far-Plate", 5400, NodeType.STANDARD, 0.92, 0.70, [7, 8]),
    }
    return BathymetricMap(
        id="kuril_kamchatka_trench",
        name="Kuril-Kamchatka Trench",
        region=OceanRegion.PACIFIC,
        description="Sub-arctic trench in the Northwest Pacific with cold acoustic sound channels reaching 10,542m.",
        nodes=nodes,
        p0_start=0,
        p1_start=9,
    )


def _build_kermadec_trench() -> BathymetricMap:
    # 10 nodes
    nodes = {
        0: MapNode(0, "Bay of Plenty Slope", 2100, NodeType.STANDARD, 0.12, 0.75, [1, 2]),
        1: MapNode(1, "Havre Trough", 2600, NodeType.STANDARD, 0.25, 0.45, [0, 3, 4]),
        2: MapNode(2, "Colville Ridge", 1900, NodeType.CHOKEPOINT, 0.30, 0.85, [0, 4, 5]),
        3: MapNode(3, "Monowai Seamount", 1100, NodeType.LISTENING_POST, 0.42, 0.25, [1, 6]),
        4: MapNode(4, "Kermadec Ridge Crest", 800, NodeType.THERMAL_LAYER, 0.45, 0.60, [1, 2, 6, 7]),
        5: MapNode(5, "Hikurangi Plateau", 3500, NodeType.STANDARD, 0.48, 0.90, [2, 7]),
        6: MapNode(6, "Macaulay Island Shelf", 1300, NodeType.LISTENING_POST, 0.65, 0.35, [3, 4, 8]),
        7: MapNode(7, "Scholl Deep", 9994, NodeType.ABYSSAL_FLOOR, 0.68, 0.75, [4, 5, 8, 9]),
        8: MapNode(8, "Kermadec Deep", 10047, NodeType.ABYSSAL_FLOOR, 0.82, 0.45, [6, 7, 9]),
        9: MapNode(9, "Southern Abyssal Trench", 8900, NodeType.CHOKEPOINT, 0.92, 0.70, [7, 8]),
    }
    return BathymetricMap(
        id="kermadec_trench",
        name="Kermadec Trench",
        region=OceanRegion.PACIFIC,
        description="South Pacific volcanic subduction trench harboring Scholl Deep and active underwater volcanoes.",
        nodes=nodes,
        p0_start=0,
        p1_start=9,
    )


# Registry of all 10 maps
ALL_MAPS: Dict[str, BathymetricMap] = {
    # Atlantic (5)
    "puerto_rico_trench": _build_puerto_rico_trench(),
    "romanche_trench": _build_romanche_trench(),
    "south_sandwich_trench": _build_south_sandwich_trench(),
    "mid_atlantic_ridge": _build_mid_atlantic_ridge(),
    "cayman_trough": _build_cayman_trough(),

    # Pacific (5)
    "mariana_trench": _build_mariana_trench(),
    "tonga_trench": _build_tonga_trench(),
    "philippine_trench": _build_philippine_trench(),
    "kuril_kamchatka_trench": _build_kuril_kamchatka_trench(),
    "kermadec_trench": _build_kermadec_trench(),
}


def get_map(map_id: str) -> BathymetricMap:
    if map_id not in ALL_MAPS:
        raise ValueError(f"Unknown map_id: {map_id}. Available: {list(ALL_MAPS.keys())}")
    return ALL_MAPS[map_id]


def list_maps() -> List[Dict[str, str]]:
    return [
        {
            "id": m.id,
            "name": m.name,
            "region": m.region.value,
            "description": m.description,
            "num_nodes": m.num_nodes,
        }
        for m in ALL_MAPS.values()
    ]
