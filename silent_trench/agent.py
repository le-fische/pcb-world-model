"""Inference agent wrapper for SubmarineActorCritic."""
from typing import Optional
import os
import torch
import numpy as np

from silent_trench.engine import Action, SilentTrenchGame
from silent_trench.model import SubmarineActorCritic
from silent_trench.observations import ObservationEncoder, index_to_action


class NeuralSubAgent:
    """Neural Network Agent playing Silent Trench using a trained PyTorch model."""

    def __init__(
        self,
        player_id: int,
        model_path: Optional[str] = None,
        device: str = "cpu",
        deterministic: bool = False,
    ):
        self.player_id = player_id
        self.device = torch.device(device)
        self.deterministic = deterministic
        self.model = SubmarineActorCritic().to(self.device)

        if model_path is not None and os.path.exists(model_path):
            state_dict = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(state_dict)
            self.model.eval()

        self.hx: Optional[torch.Tensor] = None
        self.reset()

    def reset(self) -> None:
        self.hx = self.model.initial_state(batch_size=1, device=self.device)

    def select_action(self, game: SilentTrenchGame) -> Action:
        self.model.eval()
        obs = ObservationEncoder.encode_observation(game, self.player_id)
        mask = ObservationEncoder.get_action_mask(game, self.player_id)

        obs_t = torch.from_numpy(obs).unsqueeze(0).to(self.device)
        mask_t = torch.from_numpy(mask).unsqueeze(0).to(self.device)

        with torch.no_grad():
            action_idx, _, _, next_hx = self.model.get_action(
                obs=obs_t,
                hx=self.hx,
                action_mask=mask_t,
                deterministic=self.deterministic,
            )
            self.hx = next_hx

        act_int = int(action_idx.item())
        return index_to_action(act_int)
