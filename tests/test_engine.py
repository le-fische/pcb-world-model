"""Comprehensive unit tests for the Silent Trench game engine (Two Spies mechanics)."""
import pytest
from silent_trench.constants import (
    ActionType,
    GameStatus,
    NodeType,
    ACTIONS_PER_TURN,
    MAX_POWER,
    STARTING_POWER,
)
from silent_trench.engine import Action, SilentTrenchGame


def test_initial_state():
    game = SilentTrenchGame("mariana_trench")
    assert game.turn == 1
    assert game.active_player == 0
    assert game.actions_remaining_this_turn == ACTIONS_PER_TURN  # 2 actions
    assert game.status == GameStatus.ONGOING
    assert game.players[0].node_id == game.map_data.p0_start
    assert game.players[1].node_id == game.map_data.p1_start
    assert game.players[0].power == STARTING_POWER
    assert game.players[1].power == STARTING_POWER

    legal = game.get_legal_actions(0)
    action_types = {a.action_type for a in legal}
    assert ActionType.WAIT in action_types
    assert ActionType.MOVE in action_types
    assert ActionType.SECURE in action_types
    assert ActionType.TORPEDO in action_types


def test_turn_action_points():
    game = SilentTrenchGame("mariana_trench")
    p0 = game.players[0]

    # Action 1 of 2
    game.step(Action(ActionType.WAIT))
    assert game.active_player == 0
    assert game.actions_remaining_this_turn == 1

    # Action 2 of 2
    game.step(Action(ActionType.WAIT))
    # Turn ends, passes to Player 1
    assert game.active_player == 1
    assert game.actions_remaining_this_turn == 2


def test_locate_does_not_reveal_self():
    """Verify locate/sonar sweep does not reveal the searching submarine."""
    game = SilentTrenchGame("mariana_trench")
    p0 = game.players[0]
    p1 = game.players[1]
    p0.power = 3
    p1.node_id = 4  # Sirena Deep

    # P0 uses LOCATE on node 4
    status, _, _ = game.step(Action(ActionType.ACTIVE_SONAR, target_node=4))

    # P0 stays in Deep Cover!
    assert p0.in_deep_cover is True

    # P0 pinpointed P1 at node 4
    assert game.last_known_contact[0] is not None
    assert game.last_known_contact[0]["node_id"] == 4

    # P1 does NOT learn P0's location!
    assert game.last_known_contact[1] is None


def test_torpedo_direct_hit_wins():
    game = SilentTrenchGame("mariana_trench")
    p0 = game.players[0]
    p1 = game.players[1]

    # Place P1 right next to P0 (node 1)
    p1.node_id = 1
    p0.node_id = 0

    status, winner, reason = game.step(Action(ActionType.TORPEDO, target_node=1))
    assert status == GameStatus.P0_WIN
    assert winner == 0
    assert "destroyed enemy submarine" in reason


def test_torpedo_miss_triggers_strike_reports():
    game = SilentTrenchGame("mariana_trench")
    p0 = game.players[0]
    p1 = game.players[1]

    p0.node_id = 0
    p1.node_id = 10
    p1.has_strike_reports = True

    # P0 fires at node 1 (misses P1)
    status, winner, reason = game.step(Action(ActionType.TORPEDO, target_node=1))
    assert status == GameStatus.ONGOING
    assert not p0.in_deep_cover  # Firing breaks deep cover

    # Check P1 received counter-battery triangulation
    contact = game.last_known_contact[1]
    assert contact is not None
    assert contact["node_id"] == 0  # P0's firing location
    assert contact["type"] == "STRIKE_REPORT_ORIGIN"
    assert not p1.has_strike_reports  # Consumed


def test_overcharge_bonus_move():
    game = SilentTrenchGame("mariana_trench")
    p0 = game.players[0]
    p0.power = 2

    # P0 uses Overcharge (Action 1)
    game.step(Action(ActionType.OVERCHARGE))
    assert p0.extra_action_ready

    # P0 Action 2 (Turn ends)
    game.step(Action(ActionType.WAIT))
    assert game.active_player == 1

    # P1 takes their 2 actions
    game.step(Action(ActionType.WAIT))
    game.step(Action(ActionType.WAIT))

    # P0's turn begins: should have 2 base + 1 overcharge = 3 actions!
    assert game.active_player == 0
    assert game.actions_remaining_this_turn == 3


def test_secure_and_domination_win():
    game = SilentTrenchGame("mariana_trench")
    p0 = game.players[0]
    p0.node_id = 1

    # Secure node 1 (Action 1)
    game.step(Action(ActionType.SECURE, target_node=1))
    assert game.node_control[1] == 0
    assert not p0.in_deep_cover

    # Manually award P0 two more listening posts to test domination trigger
    game.node_control[9] = 0
    p0.node_id = 8
    game.map_data.nodes[8].node_type = NodeType.LISTENING_POST

    # P0 secures node 8 -> 3rd listening post! (Action 2)
    status, winner, reason = game.step(Action(ActionType.SECURE, target_node=8))
    assert status == GameStatus.P0_WIN
    assert winner == 0
    assert "naval domination" in reason


def test_passive_sensor_alert():
    game = SilentTrenchGame("mariana_trench")
    p0 = game.players[0]
    p1 = game.players[1]
    p0.power = 2

    # P0 deploys sensor at current node 0 (Action 1)
    game.step(Action(ActionType.DEPLOY_SENSOR, target_node=0))
    assert 0 in p0.deployed_sensors

    # P0 ends turn (Action 2)
    game.step(Action(ActionType.WAIT))
    assert game.active_player == 1

    # P1 moves from start 10 to neighbor 7 (Sirena Deep)
    # Mariana node 0 neighbors are 1, 2. Let's place P1 at node 1 (neighbor of 0)
    p1.node_id = 1
    # P1 moves from 1 into 0 (where sensor is)
    game.step(Action(ActionType.MOVE, target_node=0))

    # P0 should have received sensor contact alert
    contact = game.last_known_contact[0]
    assert contact is not None
    assert contact["node_id"] == 0
    assert contact["type"] == "SENSOR_CONTACT"


def test_game_cloning():
    game = SilentTrenchGame("mariana_trench")
    game.players[0].power = 5
    game.players[0].deployed_sensors.add(1)
    game.node_control[2] = 0

    clone = game.clone()
    assert clone.turn == game.turn
    assert clone.active_player == game.active_player
    assert clone.actions_remaining_this_turn == game.actions_remaining_this_turn
    assert clone.players[0].power == 5
    assert 1 in clone.players[0].deployed_sensors
    assert clone.node_control[2] == 0

    clone.players[0].power = 1
    clone.players[0].deployed_sensors.add(3)
    assert game.players[0].power == 5
    assert 3 not in game.players[0].deployed_sensors
