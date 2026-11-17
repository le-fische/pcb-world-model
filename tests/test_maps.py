"""Unit tests for bathymetric maps."""
from collections import deque
import pytest
from silent_trench.constants import NodeType, OceanRegion
from silent_trench.maps import ALL_MAPS, get_map


def test_map_count_and_regions():
    assert len(ALL_MAPS) == 10
    atlantic = [m for m in ALL_MAPS.values() if m.region == OceanRegion.ATLANTIC]
    pacific = [m for m in ALL_MAPS.values() if m.region == OceanRegion.PACIFIC]
    assert len(atlantic) == 5
    assert len(pacific) == 5


@pytest.mark.parametrize("map_id", list(ALL_MAPS.keys()))
def test_map_graph_properties(map_id):
    b_map = get_map(map_id)
    assert b_map.num_nodes >= 10
    assert b_map.p0_start in b_map.nodes
    assert b_map.p1_start in b_map.nodes
    assert b_map.p0_start != b_map.p1_start

    # Check symmetry of edges (undirected graph)
    for u, node in b_map.nodes.items():
        assert len(node.neighbors) > 0, f"Node {u} has no neighbors in {map_id}"
        assert 0.0 <= node.x <= 1.0
        assert 0.0 <= node.y <= 1.0
        assert node.depth_m > 0
        for v in node.neighbors:
            assert v in b_map.nodes, f"Neighbor {v} not in map nodes"
            assert u in b_map.nodes[v].neighbors, f"Edge ({u}, {v}) is not bidirectional in {map_id}"

    # Check connectivity via BFS
    visited = set()
    queue = deque([b_map.p0_start])
    visited.add(b_map.p0_start)
    while queue:
        curr = queue.popleft()
        for nbr in b_map.nodes[curr].neighbors:
            if nbr not in visited:
                visited.add(nbr)
                queue.append(nbr)

    assert len(visited) == b_map.num_nodes, f"Map {map_id} is not fully connected"

    # Check strategic node distribution
    node_types = [n.node_type for n in b_map.nodes.values()]
    assert NodeType.LISTENING_POST in node_types, f"Map {map_id} lacks listening posts"
    assert NodeType.ABYSSAL_FLOOR in node_types, f"Map {map_id} lacks abyssal floors"
