"""FastAPI server for Silent Trench."""
from typing import Dict, List, Optional
import os
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from silent_trench.constants import (
    ActionType,
    GameStatus,
    NodeType,
    OceanRegion,
    ACTIONS_PER_TURN,
    MAX_POWER,
    STARTING_POWER,
)
from silent_trench.engine import Action, SilentTrenchGame
from silent_trench.maps import ALL_MAPS, list_maps
from silent_trench.bot import SonarTacticsBot
from silent_trench.agent import NeuralSubAgent
from silent_trench.observations import ObservationEncoder

app = FastAPI(title="Silent Trench: Submarine Warfare AI")

# Mount static folder
STATIC_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "static")
os.makedirs(STATIC_DIR, exist_ok=True)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


class ActionRequest(BaseModel):
    action_type: str
    target_node: Optional[int] = None


class NewGameRequest(BaseModel):
    map_name: str = "mariana_trench"
    opponent_type: str = "neural_net"  # "neural_net" or "bot"


class ActiveGameSession:
    def __init__(self, map_name: str = "mariana_trench", opponent_type: str = "neural_net"):
        self.map_name = map_name
        self.opponent_type = opponent_type
        self.game = SilentTrenchGame(map_name)
        self.human_pid = 0
        self.ai_pid = 1

        checkpoint_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)), "checkpoints", "best_model.pt"
        )
        if opponent_type == "neural_net" and os.path.exists(checkpoint_path):
            self.ai_agent = NeuralSubAgent(self.ai_pid, model_path=checkpoint_path, deterministic=True)
        else:
            self.ai_agent = SonarTacticsBot(self.ai_pid, difficulty="medium")

    def execute_ai_turns(self) -> List[str]:
        """Executes AI turns until it's the human's turn or game terminates."""
        max_ai_steps = 6
        steps = 0
        ai_events_before = len(self.game.events)
        while (
            self.game.status == GameStatus.ONGOING
            and self.game.active_player == self.ai_pid
            and steps < max_ai_steps
        ):
            action = self.ai_agent.select_action(self.game)
            self.game.step(action)
            steps += 1

        # Return messages generated during AI's turn
        ai_events = [ev.public_message for ev in self.game.events[ai_events_before:]]
        return ai_events

    def get_human_view(self) -> Dict:
        game = self.game
        pid = self.human_pid
        opp_id = self.ai_pid
        p_state = game.players[pid]
        opp_state = game.players[opp_id]

        # Filter events for human view
        filtered_events = []
        for ev in game.events[-25:]:  # Last 25 events
            secret = ev.secret_message_p0 if pid == 0 else ev.secret_message_p1
            filtered_events.append({
                "turn": ev.turn,
                "player_id": ev.player_id,
                "action_type": ev.action_type.value,
                "public_message": ev.public_message,
                "secret_message": secret,
                "masked": ev.masked,
            })

        # Node view
        nodes_view = []
        for n_id, node in game.map_data.nodes.items():
            control_owner = game.node_control.get(n_id)
            nodes_view.append({
                "id": node.id,
                "name": node.name,
                "depth_m": node.depth_m,
                "node_type": node.node_type.value,
                "x": node.x,
                "y": node.y,
                "neighbors": node.neighbors,
                "controlled_by": control_owner,
                "has_sensor": n_id in p_state.deployed_sensors,
            })

        # Legal actions for human
        legal_actions = []
        if game.status == GameStatus.ONGOING and game.active_player == pid:
            for act in game.get_legal_actions(pid):
                legal_actions.append({
                    "action_type": act.action_type.value,
                    "target_node": act.target_node,
                })

        return {
            "status": game.status.name,
            "winner": game.winner,
            "win_reason": game.win_reason,
            "turn": game.turn,
            "active_player": game.active_player,
            "actions_remaining": game.actions_remaining_this_turn,
            "actions_per_turn": ACTIONS_PER_TURN,
            "is_human_turn": (game.status == GameStatus.ONGOING and game.active_player == pid),
            "map": {
                "id": game.map_data.id,
                "name": game.map_data.name,
                "region": game.map_data.region.value,
                "description": game.map_data.description,
                "nodes": nodes_view,
            },
            "player": {
                "id": pid,
                "node_id": p_state.node_id,
                "power": p_state.power,
                "max_power": MAX_POWER,
                "in_deep_cover": p_state.in_deep_cover,
                "has_strike_reports": p_state.has_strike_reports,
                "extra_action_ready": p_state.extra_action_ready,
                "acoustic_masking_active": p_state.acoustic_masking_turns > 0,
                "controlled_posts": game.get_controlled_posts_count(pid),
            },
            "opponent_intel": {
                "id": opp_id,
                "last_known_contact": game.last_known_contact.get(pid),
                "in_deep_cover": opp_state.in_deep_cover,
                "candidate_nodes": list(game.candidate_nodes[pid]),  # Two Spies "Could be here"
                "controlled_posts": game.get_controlled_posts_count(opp_id),
            },
            "legal_actions": legal_actions,
            "events": filtered_events,
        }


# Global session singleton
CURRENT_SESSION = ActiveGameSession("mariana_trench", "neural_net")


@app.get("/", response_class=HTMLResponse)
def index_page():
    index_file = os.path.join(STATIC_DIR, "index.html")
    if os.path.exists(index_file):
        with open(index_file, "r") as f:
            return HTMLResponse(f.read())
    return HTMLResponse("<h1>Silent Trench server running. Static files missing.</h1>")


@app.get("/api/maps")
def get_maps():
    return list_maps()


@app.post("/api/game/new")
def new_game(req: NewGameRequest):
    global CURRENT_SESSION
    CURRENT_SESSION = ActiveGameSession(req.map_name, req.opponent_type)
    return CURRENT_SESSION.get_human_view()


@app.get("/api/game/state")
def get_game_state():
    return CURRENT_SESSION.get_human_view()


@app.post("/api/game/action")
def submit_action(req: ActionRequest):
    global CURRENT_SESSION
    session = CURRENT_SESSION
    game = session.game

    if game.status != GameStatus.ONGOING:
        return session.get_human_view()

    if game.active_player != session.human_pid:
        raise HTTPException(status_code=400, detail="Not human player's turn")

    try:
        act_enum = ActionType(req.action_type)
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Invalid action type: {req.action_type}")

    action = Action(act_enum, target_node=req.target_node)
    game.step(action)

    return session.get_human_view()


@app.post("/api/game/ai_step")
def ai_step():
    """Trigger AI's turn when human actions are exhausted."""
    global CURRENT_SESSION
    session = CURRENT_SESSION
    game = session.game

    if game.status == GameStatus.ONGOING and game.active_player == session.ai_pid:
        session.execute_ai_turns()

    return session.get_human_view()


@app.post("/api/game/end_turn")
def end_turn():
    """Pass remaining actions and end human's turn."""
    global CURRENT_SESSION
    session = CURRENT_SESSION
    game = session.game

    if game.status == GameStatus.ONGOING and game.active_player == session.human_pid:
        # Spend remaining actions as WAIT (drift)
        while game.actions_remaining_this_turn > 0 and game.active_player == session.human_pid:
            game.step(Action(ActionType.WAIT))

    return session.get_human_view()


@app.post("/api/train/quick")
def quick_train():
    """Runs a quick 5-iteration training burst and reloads agent."""
    from silent_trench.train import PPOTrainer
    trainer = PPOTrainer(checkpoint_dir="checkpoints")
    trainer.train(total_iterations=5, games_per_iter=10)

    if CURRENT_SESSION.opponent_type == "neural_net":
        best_path = os.path.join("checkpoints", "best_model.pt")
        CURRENT_SESSION.ai_agent = NeuralSubAgent(CURRENT_SESSION.ai_pid, model_path=best_path)

    return {"status": "trained", "message": "Completed 5 PPO iterations. Model reloaded."}
