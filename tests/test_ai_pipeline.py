"""Unit tests for observation encoding, action masks, and heuristic bot simulation."""
import pytest
import numpy as np
from silent_trench.constants import GameStatus, ActionType
from silent_trench.engine import Action, SilentTrenchGame
from silent_trench.maps import ALL_MAPS
from silent_trench.observations import (
    ObservationEncoder,
    ACTION_SPACE_SIZE,
    action_to_index,
    index_to_action,
)
from silent_trench.bot import SonarTacticsBot


def test_observation_dimensions_and_mask():
    game = SilentTrenchGame("mariana_trench")
    obs_p0 = ObservationEncoder.encode_observation(game, 0)
    assert obs_p0.shape == (ObservationEncoder.OBS_DIM,)
    assert not np.isnan(obs_p0).any()
    assert (obs_p0 >= 0.0).all() and (obs_p0 <= 1.0).all()

    mask_p0 = ObservationEncoder.get_action_mask(game, 0)
    assert mask_p0.shape == (ACTION_SPACE_SIZE,)
    assert mask_p0.dtype == np.bool_
    assert mask_p0.sum() > 0  # Must have legal moves


def test_action_roundtrip_invariance():
    game = SilentTrenchGame("mariana_trench")
    legal_actions = game.get_legal_actions(0)
    for act in legal_actions:
        idx = action_to_index(act)
        assert 0 <= idx < ACTION_SPACE_SIZE
        reconstructed = index_to_action(idx)
        assert reconstructed.action_type == act.action_type
        if act.action_type in (ActionType.MOVE, ActionType.TORPEDO, ActionType.ACTIVE_SONAR):
            assert reconstructed.target_node == act.target_node


@pytest.mark.parametrize("map_id", list(ALL_MAPS.keys())[:4])
def test_bot_vs_bot_simulation(map_id):
    """Run bot-vs-bot games to ensure full simulation stability."""
    game = SilentTrenchGame(map_id)
    bot0 = SonarTacticsBot(0, difficulty="medium")
    bot1 = SonarTacticsBot(1, difficulty="medium")

    max_steps = 150
    steps = 0
    while game.status == GameStatus.ONGOING and steps < max_steps:
        pid = game.active_player
        active_bot = bot0 if pid == 0 else bot1
        action = active_bot.select_action(game)
        game.step(action)
        steps += 1

    assert steps > 0
    assert game.turn > 1
