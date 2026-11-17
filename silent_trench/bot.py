"""Heuristic Baseline AI for Silent Trench."""
from collections import deque
from typing import List, Optional
import random

from silent_trench.constants import ActionType, NodeType
from silent_trench.engine import Action, SilentTrenchGame
from silent_trench.observations import ObservationEncoder, action_to_index


class SonarTacticsBot:
    """A tactical rule-based submarine commander bot."""

    def __init__(self, player_id: int, difficulty: str = "medium"):
        self.player_id = player_id
        self.difficulty = difficulty

    def select_action(self, game: SilentTrenchGame) -> Action:
        legal_actions = game.get_legal_actions(self.player_id)
        if not legal_actions:
            return Action(ActionType.WAIT)

        pid = self.player_id
        opp_id = game.get_opponent_id(pid)
        player = game.players[pid]
        curr_node = player.node_id
        contact = game.last_known_contact[pid]

        # 1. LETHAL STRIKE: If we know the enemy's node and have a torpedo target there -> FIRE!
        if contact is not None:
            c_node = contact["node_id"]
            turns_ago = game.turn - contact["turn"]
            # If recent contact (< 2 turns)
            if turns_ago <= 1:
                torpedo_acts = [
                    a for a in legal_actions
                    if a.action_type == ActionType.TORPEDO and a.target_node == c_node
                ]
                if torpedo_acts:
                    return torpedo_acts[0]

        # 2. SECURE: If standing on an unowned Listening Post -> SECURE IT!
        if game.node_control[curr_node] != pid:
            if game.map_data.nodes[curr_node].node_type == NodeType.LISTENING_POST:
                secure_acts = [a for a in legal_actions if a.action_type == ActionType.SECURE]
                if secure_acts:
                    return secure_acts[0]

        # 3. DEFENSIVE TELEMETRY: If high difficulty and don't have strike reports -> calibrate
        if not player.has_strike_reports and player.power >= 1:
            sr_acts = [a for a in legal_actions if a.action_type == ActionType.STRIKE_REPORTS]
            if sr_acts and random.random() < 0.6:
                return sr_acts[0]

        # 4. RE-ENGAGE DEEP COVER: If exposed and have power -> engage silent running
        if not player.in_deep_cover and player.power >= 1:
            dc_acts = [a for a in legal_actions if a.action_type == ActionType.DEEP_COVER]
            if dc_acts:
                return dc_acts[0]

        # 5. ACTIVE SONAR SWEEP: If power is high (>= 3) and no recent contact -> sweep
        if player.power >= 3 and (contact is None or game.turn - contact["turn"] > 3):
            # Sweep an unowned listening post or enemy-controlled node
            candidates = [
                n for n, node in game.map_data.nodes.items()
                if game.node_control[n] == opp_id or node.node_type == NodeType.LISTENING_POST
            ]
            if candidates:
                sweep_target = random.choice(candidates)
                sonar_acts = [
                    a for a in legal_actions
                    if a.action_type == ActionType.ACTIVE_SONAR and a.target_node == sweep_target
                ]
                if sonar_acts and random.random() < 0.7:
                    return sonar_acts[0]

        # 6. SENSOR DEPLOYMENT: Deploy passive hydrophone at strategic points
        if player.power >= 2 and curr_node not in player.deployed_sensors:
            if game.map_data.nodes[curr_node].node_type in (NodeType.LISTENING_POST, NodeType.CHOKEPOINT):
                sensor_acts = [a for a in legal_actions if a.action_type == ActionType.DEPLOY_SENSOR]
                if sensor_acts and random.random() < 0.5:
                    return sensor_acts[0]

        # 7. TACTICAL MOVEMENT: Move towards uncaptured listening posts or last known contact
        move_acts = [a for a in legal_actions if a.action_type == ActionType.MOVE]
        if move_acts:
            # Target node: uncaptured listening post or contact
            target_goal = None
            if contact is not None and (game.turn - contact["turn"]) <= 2:
                target_goal = contact["node_id"]
            else:
                unowned_posts = [
                    n for n, node in game.map_data.nodes.items()
                    if node.node_type == NodeType.LISTENING_POST and game.node_control[n] != pid
                ]
                if unowned_posts:
                    target_goal = random.choice(unowned_posts)

            if target_goal is not None and target_goal != curr_node:
                # Find shortest path using BFS
                best_next_node = self._bfs_next_step(game, curr_node, target_goal)
                if best_next_node is not None:
                    matching_moves = [a for a in move_acts if a.target_node == best_next_node]
                    if matching_moves:
                        return matching_moves[0]

        # 8. LOW POWER RECOVERY: Wait / Drift
        wait_acts = [a for a in legal_actions if a.action_type == ActionType.WAIT]
        if player.power < 2 and wait_acts:
            return wait_acts[0]

        # 9. Fallback: Prefer move, then secure, then wait
        if move_acts and random.random() < 0.7:
            return random.choice(move_acts)
        if wait_acts:
            return wait_acts[0]

        return random.choice(legal_actions)

    def _bfs_next_step(self, game: SilentTrenchGame, start: int, goal: int) -> Optional[int]:
        if start == goal:
            return None
        queue = deque([[start]])
        visited = {start}
        while queue:
            path = queue.popleft()
            node = path[-1]
            if node == goal:
                return path[1] if len(path) > 1 else None
            for nbr in game.map_data.get_neighbors(node):
                if nbr not in visited:
                    visited.add(nbr)
                    queue.append(path + [nbr])
        return None
