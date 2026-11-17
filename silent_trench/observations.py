"""Observation encoder and discrete action space mapping for Silent Trench."""
from typing import Dict, List, Optional, Tuple
import numpy as np

from silent_trench.constants import (
    ActionType,
    NodeType,
    MAX_POWER,
    MAX_TURNS,
)
from silent_trench.engine import Action, SilentTrenchGame

MAX_NODES = 16

# Discrete Action Space Layout:
# 0: WAIT
# 1: DEEP_COVER
# 2: OVERCHARGE
# 3: STRIKE_REPORTS
# 4: ACOUSTIC_MASKING
# 5: SECURE
# 6 .. 6+MAX_NODES-1: MOVE(node)
# 6+MAX_NODES .. 6+2*MAX_NODES-1: TORPEDO(node)
# 6+2*MAX_NODES .. 6+3*MAX_NODES-1: ACTIVE_SONAR(node)
# 6+3*MAX_NODES .. 6+4*MAX_NODES-1: DEPLOY_SENSOR(node)

BASE_ACTIONS_COUNT = 6
ACTION_SPACE_SIZE = BASE_ACTIONS_COUNT + (4 * MAX_NODES)  # 6 + 64 = 70


def action_to_index(action: Action) -> int:
    at = action.action_type
    if at == ActionType.WAIT:
        return 0
    elif at == ActionType.DEEP_COVER:
        return 1
    elif at == ActionType.OVERCHARGE:
        return 2
    elif at == ActionType.STRIKE_REPORTS:
        return 3
    elif at == ActionType.ACOUSTIC_MASKING:
        return 4
    elif at == ActionType.SECURE:
        return 5
    elif at == ActionType.MOVE:
        assert action.target_node is not None and action.target_node < MAX_NODES
        return BASE_ACTIONS_COUNT + action.target_node
    elif at == ActionType.TORPEDO:
        assert action.target_node is not None and action.target_node < MAX_NODES
        return BASE_ACTIONS_COUNT + MAX_NODES + action.target_node
    elif at == ActionType.ACTIVE_SONAR:
        assert action.target_node is not None and action.target_node < MAX_NODES
        return BASE_ACTIONS_COUNT + (2 * MAX_NODES) + action.target_node
    elif at == ActionType.DEPLOY_SENSOR:
        assert action.target_node is not None and action.target_node < MAX_NODES
        return BASE_ACTIONS_COUNT + (3 * MAX_NODES) + action.target_node
    raise ValueError(f"Unknown action: {action}")


def index_to_action(index: int) -> Action:
    if index == 0:
        return Action(ActionType.WAIT)
    elif index == 1:
        return Action(ActionType.DEEP_COVER)
    elif index == 2:
        return Action(ActionType.OVERCHARGE)
    elif index == 3:
        return Action(ActionType.STRIKE_REPORTS)
    elif index == 4:
        return Action(ActionType.ACOUSTIC_MASKING)
    elif index == 5:
        return Action(ActionType.SECURE)
    elif BASE_ACTIONS_COUNT <= index < BASE_ACTIONS_COUNT + MAX_NODES:
        return Action(ActionType.MOVE, target_node=index - BASE_ACTIONS_COUNT)
    elif BASE_ACTIONS_COUNT + MAX_NODES <= index < BASE_ACTIONS_COUNT + (2 * MAX_NODES):
        return Action(ActionType.TORPEDO, target_node=index - (BASE_ACTIONS_COUNT + MAX_NODES))
    elif BASE_ACTIONS_COUNT + (2 * MAX_NODES) <= index < BASE_ACTIONS_COUNT + (3 * MAX_NODES):
        return Action(ActionType.ACTIVE_SONAR, target_node=index - (BASE_ACTIONS_COUNT + (2 * MAX_NODES)))
    elif BASE_ACTIONS_COUNT + (3 * MAX_NODES) <= index < BASE_ACTIONS_COUNT + (4 * MAX_NODES):
        return Action(ActionType.DEPLOY_SENSOR, target_node=index - (BASE_ACTIONS_COUNT + (3 * MAX_NODES)))
    raise ValueError(f"Action index {index} out of bounds")


class ObservationEncoder:
    """Encodes game state into partial observation tensors for neural networks."""

    OBS_DIM = (
        MAX_NODES +          # Own position one-hot (16)
        1 +                  # Normalized power (1)
        1 +                  # In deep cover (1)
        1 +                  # Has strike reports (1)
        1 +                  # Extra action ready (1)
        1 +                  # Acoustic masking active (1)
        MAX_NODES +          # Deployed sensors (16)
        MAX_NODES +          # Node control owned by self (16)
        MAX_NODES +          # Node control owned by opponent (16)
        MAX_NODES +          # Strategic listening posts mask (16)
        MAX_NODES +          # Thermal layers mask (16)
        1 +                  # Normalized turn (1)
        MAX_NODES +          # Last known enemy node (16)
        1 +                  # Turns elapsed since last known contact (1)
        1                    # Any active contact exists (1)
    )  # Total: 16 + 5 + 16*5 + 2 + 1 = 120 floats

    @classmethod
    def get_action_mask(cls, game: SilentTrenchGame, player_id: int) -> np.ndarray:
        mask = np.zeros(ACTION_SPACE_SIZE, dtype=np.bool_)
        legal_actions = game.get_legal_actions(player_id)
        for act in legal_actions:
            if act.target_node is None or act.target_node < MAX_NODES:
                idx = action_to_index(act)
                mask[idx] = True
        return mask

    @classmethod
    def encode_observation(cls, game: SilentTrenchGame, player_id: int) -> np.ndarray:
        obs = np.zeros(cls.OBS_DIM, dtype=np.float32)
        offset = 0

        player = game.players[player_id]
        opp_id = game.get_opponent_id(player_id)

        # 1. Own position one-hot (16)
        if player.node_id < MAX_NODES:
            obs[offset + player.node_id] = 1.0
        offset += MAX_NODES

        # 2. Player scalars (5)
        obs[offset] = player.power / float(MAX_POWER)
        obs[offset + 1] = 1.0 if player.in_deep_cover else 0.0
        obs[offset + 2] = 1.0 if player.has_strike_reports else 0.0
        obs[offset + 3] = 1.0 if player.extra_action_ready else 0.0
        obs[offset + 4] = 1.0 if player.acoustic_masking_turns > 0 else 0.0
        offset += 5

        # 3. Deployed sensors (16)
        for s_node in player.deployed_sensors:
            if s_node < MAX_NODES:
                obs[offset + s_node] = 1.0
        offset += MAX_NODES

        # 4. Node control self (16) and opponent (16)
        for node_id, owner in game.node_control.items():
            if node_id < MAX_NODES:
                if owner == player_id:
                    obs[offset + node_id] = 1.0
                elif owner == opp_id:
                    obs[offset + MAX_NODES + node_id] = 1.0
        offset += (2 * MAX_NODES)

        # 5. Map attributes: Listening Posts (16) and Thermal Layers (16)
        for node_id, node in game.map_data.nodes.items():
            if node_id < MAX_NODES:
                if node.node_type == NodeType.LISTENING_POST:
                    obs[offset + node_id] = 1.0
                elif node.node_type == NodeType.THERMAL_LAYER:
                    obs[offset + MAX_NODES + node_id] = 1.0
        offset += (2 * MAX_NODES)

        # 6. Normalized turn (1)
        obs[offset] = game.turn / float(MAX_TURNS)
        offset += 1

        # 7. Last known enemy contact (16 + 2 = 18)
        contact = game.last_known_contact[player_id]
        if contact is not None:
            c_node = contact["node_id"]
            if c_node < MAX_NODES:
                obs[offset + c_node] = 1.0
            elapsed = (game.turn - contact["turn"]) / float(MAX_TURNS)
            obs[offset + MAX_NODES] = min(1.0, max(0.0, elapsed))
            obs[offset + MAX_NODES + 1] = 1.0  # Contact active flag
        else:
            obs[offset + MAX_NODES] = 1.0
            obs[offset + MAX_NODES + 1] = 0.0
        offset += (MAX_NODES + 2)

        assert offset == cls.OBS_DIM, f"Observation dimension mismatch: {offset} != {cls.OBS_DIM}"
        return obs

    @classmethod
    def get_global_state(cls, game: SilentTrenchGame) -> np.ndarray:
        """Asymmetric full state for training Critic value networks."""
        # Combines p0 obs and p1 obs with exact actual positions
        p0_obs = cls.encode_observation(game, 0)
        p1_obs = cls.encode_observation(game, 1)
        # Append exact enemy positions
        p0_pos = np.zeros(MAX_NODES, dtype=np.float32)
        p1_pos = np.zeros(MAX_NODES, dtype=np.float32)
        p0_pos[game.players[0].node_id] = 1.0
        p1_pos[game.players[1].node_id] = 1.0
        return np.concatenate([p0_obs, p1_obs, p0_pos, p1_pos])
