"""Core headless game engine for Silent Trench (Two Spies Submarine Warfare)."""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple
import copy

from silent_trench.constants import (
    ActionType,
    GameStatus,
    NodeType,
    ACTIONS_PER_TURN,
    MAX_POWER,
    STARTING_POWER,
    MAX_TURNS,
    WIN_CONTROL_NODES,
    COST_DEEP_COVER,
    COST_ACTIVE_SONAR,
    COST_OVERCHARGE,
    COST_STRIKE_REPORTS,
    COST_ACOUSTIC_MASKING,
    COST_DEPLOY_SENSOR,
)
from silent_trench.maps import BathymetricMap, get_map


@dataclass
class Action:
    action_type: ActionType
    target_node: Optional[int] = None

    def __repr__(self) -> str:
        if self.target_node is not None:
            return f"{self.action_type.value}(node={self.target_node})"
        return self.action_type.value


@dataclass
class PlayerState:
    player_id: int
    node_id: int
    power: int = STARTING_POWER
    in_deep_cover: bool = True
    has_strike_reports: bool = False
    acoustic_masking_turns: int = 0
    extra_action_ready: bool = False  # From overcharge
    deployed_sensors: Set[int] = field(default_factory=set)


@dataclass
class GameEvent:
    turn: int
    player_id: int
    action_type: ActionType
    target_node: Optional[int]
    public_message: str
    secret_message_p0: Optional[str] = None
    secret_message_p1: Optional[str] = None
    masked: bool = False


class SilentTrenchGame:
    def __init__(self, map_name: str = "mariana_trench"):
        self.map_data: BathymetricMap = get_map(map_name)
        self.reset()

    def reset(self) -> None:
        self.turn: int = 1
        self.active_player: int = 0
        self.actions_remaining_this_turn: int = ACTIONS_PER_TURN  # 2 actions per turn
        self.status: GameStatus = GameStatus.ONGOING
        self.winner: Optional[int] = None
        self.win_reason: str = ""

        # Players
        self.players: Dict[int, PlayerState] = {
            0: PlayerState(player_id=0, node_id=self.map_data.p0_start, power=STARTING_POWER),
            1: PlayerState(player_id=1, node_id=self.map_data.p1_start, power=STARTING_POWER),
        }

        # Node control: node_id -> player_id (or None)
        self.node_control: Dict[int, Optional[int]] = {
            node_id: None for node_id in self.map_data.nodes
        }

        # Two Spies Deduction System: Candidate nodes where each player believes the enemy could be
        # candidate_nodes[pid] = set of nodes where pid thinks opponent could be hiding
        self.candidate_nodes: Dict[int, Set[int]] = {
            0: {self.map_data.p1_start},
            1: {self.map_data.p0_start},
        }

        # Event log
        self.events: List[GameEvent] = []

        # Last known enemy contacts: observer_id -> {"node_id": int, "turn": int, "type": str}
        self.last_known_contact: Dict[int, Optional[Dict]] = {0: None, 1: None}

    def get_opponent_id(self, player_id: int) -> int:
        return 1 - player_id

    def get_controlled_posts_count(self, player_id: int) -> int:
        count = 0
        for node_id, owner in self.node_control.items():
            if owner == player_id:
                if self.map_data.nodes[node_id].node_type == NodeType.LISTENING_POST:
                    count += 1
        return count

    def get_legal_actions(self, player_id: int) -> List[Action]:
        if self.status != GameStatus.ONGOING:
            return []

        player = self.players[player_id]
        curr_node = player.node_id
        neighbors = self.map_data.get_neighbors(curr_node)
        legal: List[Action] = []

        # 1. Free: WAIT (Drift / Pass Action)
        legal.append(Action(ActionType.WAIT))

        # 2. Free: MOVE to adjacent nodes
        for nbr in neighbors:
            legal.append(Action(ActionType.MOVE, target_node=nbr))

        # 3. Free: SECURE current node (if not already controlled by this player)
        if self.node_control[curr_node] != player_id:
            legal.append(Action(ActionType.SECURE, target_node=curr_node))

        # 4. Free: TORPEDO (Strike) current node or adjacent neighbors
        for target in [curr_node] + neighbors:
            legal.append(Action(ActionType.TORPEDO, target_node=target))

        # 5. Tactical: DEEP_COVER (costs 1 power)
        if player.power >= COST_DEEP_COVER and not player.in_deep_cover:
            legal.append(Action(ActionType.DEEP_COVER))

        # 6. Tactical: ACTIVE_SONAR / LOCATE (costs 2 power, can sweep any node on map)
        if player.power >= COST_ACTIVE_SONAR:
            for node_id in self.map_data.nodes:
                legal.append(Action(ActionType.ACTIVE_SONAR, target_node=node_id))

        # 7. Tactical: OVERCHARGE (costs 1 power, prepares extra move next turn)
        if player.power >= COST_OVERCHARGE and not player.extra_action_ready:
            legal.append(Action(ActionType.OVERCHARGE))

        # 8. Bonus: STRIKE_REPORTS (costs 1 power, defensive passive)
        if player.power >= COST_STRIKE_REPORTS and not player.has_strike_reports:
            legal.append(Action(ActionType.STRIKE_REPORTS))

        # 9. Bonus: ACOUSTIC_MASKING (costs 1 power, masks next action)
        if player.power >= COST_ACOUSTIC_MASKING and player.acoustic_masking_turns == 0:
            legal.append(Action(ActionType.ACOUSTIC_MASKING))

        # 10. Bonus: DEPLOY_SENSOR (costs 1 power, drops passive bug at current node)
        if player.power >= COST_DEPLOY_SENSOR and curr_node not in player.deployed_sensors:
            legal.append(Action(ActionType.DEPLOY_SENSOR, target_node=curr_node))

        return legal

    def step(self, action: Action) -> Tuple[GameStatus, Optional[int], str]:
        """Execute action for active player."""
        if self.status != GameStatus.ONGOING:
            return self.status, self.winner, self.win_reason

        pid = self.active_player
        opp_id = self.get_opponent_id(pid)
        player = self.players[pid]
        opponent = self.players[opp_id]
        curr_node = player.node_id

        # Check masking status
        is_masked = player.acoustic_masking_turns > 0
        if player.acoustic_masking_turns > 0:
            player.acoustic_masking_turns -= 1

        action_type = action.action_type
        target = action.target_node

        # --- Resolve Action ---
        if action_type == ActionType.WAIT:
            # Conserves energy / drifts
            player.power = min(MAX_POWER, player.power + 1)
            msg = f"Player {pid} drifted quietly (+1 Power)."
            self._record_event(pid, action_type, target, msg, is_masked)

        elif action_type == ActionType.MOVE:
            assert target is not None and self.map_data.are_connected(curr_node, target), "Invalid move target"
            player.node_id = target
            dest_node = self.map_data.nodes[target]

            # Update opponent's deduction indicators
            if player.in_deep_cover:
                # Expand possible candidate nodes to neighbors of previous candidates
                new_candidates = set()
                for c in self.candidate_nodes[opp_id]:
                    new_candidates.add(c)
                    new_candidates.update(self.map_data.get_neighbors(c))
                self.candidate_nodes[opp_id] = new_candidates
            else:
                # Cover is blown: opponent sees exact node
                self.candidate_nodes[opp_id] = {target}

            # Two Spies Rule: Moving into opponent-controlled territory blows cover!
            if self.node_control[target] == opp_id:
                player.in_deep_cover = False
                self.candidate_nodes[opp_id] = {target}
                self.last_known_contact[opp_id] = {
                    "node_id": target,
                    "turn": self.turn,
                    "type": "TRIPPED_TERRITORY",
                }
                trip_msg = f"SECURITY ALERT: Enemy submarine breached our controlled sector {dest_node.name}!"
                self.events.append(GameEvent(
                    turn=self.turn,
                    player_id=pid,
                    action_type=ActionType.MOVE,
                    target_node=target,
                    public_message=f"Hydrophone tripwire triggered in {dest_node.name}!",
                    secret_message_p0=trip_msg if opp_id == 0 else None,
                    secret_message_p1=trip_msg if opp_id == 1 else None,
                    masked=False,
                ))

            # Thermal layer grants automatic deep cover
            if dest_node.node_type == NodeType.THERMAL_LAYER:
                player.in_deep_cover = True

            # Check if entering opponent's passive sensor
            if target in opponent.deployed_sensors:
                self.last_known_contact[opp_id] = {
                    "node_id": target,
                    "turn": self.turn,
                    "type": "SENSOR_CONTACT",
                }
                self.candidate_nodes[opp_id] = {target}
                sensor_msg = f"Passive sensor alert: contact entering {dest_node.name}!"
                self.events.append(GameEvent(
                    turn=self.turn,
                    player_id=pid,
                    action_type=ActionType.MOVE,
                    target_node=target,
                    public_message="Acoustic anomaly registered on hydrophone array.",
                    secret_message_p0=sensor_msg if opp_id == 0 else None,
                    secret_message_p1=sensor_msg if opp_id == 1 else None,
                    masked=False,
                ))

            if not player.in_deep_cover:
                msg = f"Cavitation detected: Player {pid} moved towards {dest_node.name}."
            else:
                msg = f"Player {pid} executed silent movement maneuvers."

            self._record_event(pid, action_type, target, msg, is_masked)

        elif action_type == ActionType.SECURE:
            target = target if target is not None else curr_node
            self.node_control[target] = pid
            target_name = self.map_data.nodes[target].name

            # Securing a sector is visible and blows cover!
            player.in_deep_cover = False
            self.candidate_nodes[opp_id] = {target}
            self.last_known_contact[opp_id] = {
                "node_id": target,
                "turn": self.turn,
                "type": "SECURED_SECTOR",
            }
            msg = f"Player {pid} secured control of {target_name}!"
            self._record_event(pid, action_type, target, msg, is_masked)

            # Check Win Condition: Control Domination
            posts_held = self.get_controlled_posts_count(pid)
            if posts_held >= WIN_CONTROL_NODES:
                self.status = GameStatus.P0_WIN if pid == 0 else GameStatus.P1_WIN
                self.winner = pid
                self.win_reason = f"Player {pid} achieved naval domination (secured {posts_held} Listening Posts)!"
                return self.status, self.winner, self.win_reason

        elif action_type == ActionType.TORPEDO:
            assert target is not None, "Torpedo requires target node"
            target_name = self.map_data.nodes[target].name

            # Check Hit!
            if opponent.node_id == target:
                # HIT! Immediate Elimination
                self.status = GameStatus.P0_WIN if pid == 0 else GameStatus.P1_WIN
                self.winner = pid
                self.win_reason = f"Player {pid} destroyed enemy submarine with a direct torpedo strike at {target_name}!"
                hit_msg = f"TORPEDO IMPACT: Player {pid} launched a torpedo into {target_name} — DIRECT HIT! Submarine destroyed!"
                self._record_event(pid, action_type, target, hit_msg, is_masked=False)
                return self.status, self.winner, self.win_reason
            else:
                # MISS! Firing blows shooter's cover!
                player.in_deep_cover = False
                self.candidate_nodes[opp_id] = {curr_node}
                miss_msg = f"Player {pid} launched a torpedo targeting {target_name} — MISSED!"

                # Counter-Battery Triangulation (Strike Reports)
                strike_rep_msg = None
                if opponent.has_strike_reports:
                    opponent.has_strike_reports = False  # Consumed
                    self.last_known_contact[opp_id] = {
                        "node_id": curr_node,
                        "turn": self.turn,
                        "type": "STRIKE_REPORT_ORIGIN",
                    }
                    strike_rep_msg = f"STRIKE REPORT: Counter-battery hydrophones triangulated torpedo launch to {self.map_data.nodes[curr_node].name}!"

                self.events.append(GameEvent(
                    turn=self.turn,
                    player_id=pid,
                    action_type=action_type,
                    target_node=target,
                    public_message=miss_msg,
                    secret_message_p0=strike_rep_msg if opp_id == 0 else None,
                    secret_message_p1=strike_rep_msg if opp_id == 1 else None,
                    masked=False,
                ))

        elif action_type == ActionType.DEEP_COVER:
            assert player.power >= COST_DEEP_COVER
            player.power -= COST_DEEP_COVER
            player.in_deep_cover = True
            msg = f"Player {pid} engaged anechoic silent running (Cover restored)."
            self._record_event(pid, action_type, target, msg, is_masked)

        elif action_type == ActionType.ACTIVE_SONAR:
            # Two Spies "LOCATE" mechanic:
            # Costs 2 Power.
            # DOES NOT reveal the searcher! Searcher stays in Deep Cover!
            assert target is not None and player.power >= COST_ACTIVE_SONAR
            player.power -= COST_ACTIVE_SONAR
            target_name = self.map_data.nodes[target].name

            # Check if opponent is at or adjacent to target
            target_neighbors = self.map_data.get_neighbors(target)
            is_direct_hit = (opponent.node_id == target)
            is_echo = (opponent.node_id in target_neighbors)

            p_report = ""
            if is_direct_hit:
                # Pinpointed!
                opponent.in_deep_cover = False
                self.candidate_nodes[pid] = {target}
                self.last_known_contact[pid] = {
                    "node_id": target,
                    "turn": self.turn,
                    "type": "LOCATE_DIRECT",
                }
                p_report = f"SONAR CONTACT CONFIRMED: Enemy submarine pinpointed at {target_name}!"
            elif is_echo:
                # Echo near target
                self.last_known_contact[pid] = {
                    "node_id": opponent.node_id,
                    "turn": self.turn,
                    "type": "LOCATE_ECHO",
                }
                # Target itself is ruled out, but neighbors kept
                self.candidate_nodes[pid].discard(target)
                p_report = f"SONAR ECHO: Acoustic reflection detected near {target_name}!"
            else:
                # Clear of enemy: prune target from candidates!
                self.candidate_nodes[pid].discard(target)
                p_report = f"Sonar sweep of {target_name}: No acoustic returns."

            opp_alert = f"Acoustic sweep swept through {target_name}."

            self.events.append(GameEvent(
                turn=self.turn,
                player_id=pid,
                action_type=action_type,
                target_node=target,
                public_message=f"Sonar pulse analyzed sector {target_name}.",
                secret_message_p0=p_report if pid == 0 else opp_alert,
                secret_message_p1=p_report if pid == 1 else opp_alert,
                masked=False,
            ))

        elif action_type == ActionType.OVERCHARGE:
            assert player.power >= COST_OVERCHARGE
            player.power -= COST_OVERCHARGE
            player.extra_action_ready = True
            msg = f"Player {pid} primed propulsion overcharge (+1 Action next turn)."
            self._record_event(pid, action_type, target, msg, is_masked)

        elif action_type == ActionType.STRIKE_REPORTS:
            assert player.power >= COST_STRIKE_REPORTS
            player.power -= COST_STRIKE_REPORTS
            player.has_strike_reports = True
            msg = f"Player {pid} calibrated counter-battery telemetry array."
            self._record_event(pid, action_type, target, msg, is_masked)

        elif action_type == ActionType.ACOUSTIC_MASKING:
            assert player.power >= COST_ACOUSTIC_MASKING
            player.power -= COST_ACOUSTIC_MASKING
            player.acoustic_masking_turns = 1
            msg = f"Player {pid} deployed acoustic masking decoys."
            self._record_event(pid, action_type, target, msg, is_masked=False)

        elif action_type == ActionType.DEPLOY_SENSOR:
            target = target if target is not None else curr_node
            assert player.power >= COST_DEPLOY_SENSOR
            player.power -= COST_DEPLOY_SENSOR
            player.deployed_sensors.add(target)
            target_name = self.map_data.nodes[target].name
            msg = f"Player {pid} deployed a stationary passive hydrophone sensor."
            self._record_event(pid, action_type, target, msg, is_masked)

        # --- Two Spies Turn Economy (2 Actions per turn) ---
        self.actions_remaining_this_turn -= 1

        if self.actions_remaining_this_turn <= 0:
            # Turn finishes, hand off to other player
            self.active_player = opp_id
            next_player = self.players[self.active_player]

            # Next player gets 2 actions (or 3 if overcharged)
            base_actions = ACTIONS_PER_TURN  # 2
            if next_player.extra_action_ready:
                self.actions_remaining_this_turn = base_actions + 1
                next_player.extra_action_ready = False
            else:
                self.actions_remaining_this_turn = base_actions

            # Turn increment and passive income when cycle returns to Player 0
            if self.active_player == 0:
                self.turn += 1

                # Passive turn recharge: +1 base Power for each player + listening post bonuses
                for p_idx in [0, 1]:
                    posts = self.get_controlled_posts_count(p_idx)
                    gain = 1 + posts
                    self.players[p_idx].power = min(MAX_POWER, self.players[p_idx].power + gain)

                # Check turn limit
                if self.turn > MAX_TURNS:
                    p0_posts = self.get_controlled_posts_count(0)
                    p1_posts = self.get_controlled_posts_count(1)
                    if p0_posts > p1_posts:
                        self.status = GameStatus.P0_WIN
                        self.winner = 0
                        self.win_reason = f"Time expired: Player 0 wins on Listening Posts ({p0_posts} vs {p1_posts})."
                    elif p1_posts > p0_posts:
                        self.status = GameStatus.P1_WIN
                        self.winner = 1
                        self.win_reason = f"Time expired: Player 1 wins on Listening Posts ({p1_posts} vs {p0_posts})."
                    else:
                        self.status = GameStatus.DRAW
                        self.winner = None
                        self.win_reason = "Time expired: Standoff reached (Draw)."

        return self.status, self.winner, self.win_reason

    def _record_event(
        self,
        player_id: int,
        action_type: ActionType,
        target_node: Optional[int],
        msg: str,
        is_masked: bool,
    ) -> None:
        public_msg = "[ACOUSTIC ANOMALY / UNKNOWN SIGNAL]" if is_masked else msg
        self.events.append(GameEvent(
            turn=self.turn,
            player_id=player_id,
            action_type=action_type,
            target_node=target_node,
            public_message=public_msg,
            masked=is_masked,
        ))

    def clone(self) -> "SilentTrenchGame":
        new_game = SilentTrenchGame.__new__(SilentTrenchGame)
        new_game.map_data = self.map_data
        new_game.turn = self.turn
        new_game.active_player = self.active_player
        new_game.actions_remaining_this_turn = self.actions_remaining_this_turn
        new_game.status = self.status
        new_game.winner = self.winner
        new_game.win_reason = self.win_reason

        new_game.players = {
            0: copy.copy(self.players[0]),
            1: copy.copy(self.players[1]),
        }
        new_game.players[0].deployed_sensors = set(self.players[0].deployed_sensors)
        new_game.players[1].deployed_sensors = set(self.players[1].deployed_sensors)

        new_game.node_control = dict(self.node_control)
        new_game.candidate_nodes = {
            0: set(self.candidate_nodes[0]),
            1: set(self.candidate_nodes[1]),
        }
        new_game.events = list(self.events)
        new_game.last_known_contact = {
            0: copy.copy(self.last_known_contact[0]) if self.last_known_contact[0] else None,
            1: copy.copy(self.last_known_contact[1]) if self.last_known_contact[1] else None,
        }
        return new_game
