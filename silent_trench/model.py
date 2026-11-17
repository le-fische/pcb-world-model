"""Recurrent Actor-Critic Neural Network with Action Masking for Silent Trench."""
from typing import Optional, Tuple
import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical

from silent_trench.observations import ObservationEncoder, ACTION_SPACE_SIZE


class SubmarineActorCritic(nn.Module):
    """Recurrent policy network using a GRU memory layer to maintain belief states."""

    def __init__(
        self,
        obs_dim: int = ObservationEncoder.OBS_DIM,
        action_dim: int = ACTION_SPACE_SIZE,
        hidden_dim: int = 128,
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim

        # Feature extractor
        self.encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # Recurrent memory (maintains hidden state over turns)
        self.gru = nn.GRUCell(hidden_dim, hidden_dim)

        # Actor head (policy logits)
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

        # Critic head (state value baseline)
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=1.0)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
        # Small weights for actor output layer
        nn.init.orthogonal_(self.actor[-1].weight, gain=0.01)

    def initial_state(self, batch_size: int = 1, device: Optional[torch.device] = None) -> torch.Tensor:
        if device is None:
            device = next(self.parameters()).device
        return torch.zeros((batch_size, self.hidden_dim), device=device)

    def forward(
        self,
        obs: torch.Tensor,
        hx: torch.Tensor,
        action_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        obs: [batch_size, obs_dim]
        hx:  [batch_size, hidden_dim]
        action_mask: [batch_size, action_dim] (True for legal, False for illegal)
        Returns: logits, value, next_hx
        """
        feats = self.encoder(obs)
        next_hx = self.gru(feats, hx)
        logits = self.actor(next_hx)
        value = self.critic(next_hx)

        if action_mask is not None:
            # Mask out illegal actions with a large negative number
            huge_neg = -1e9
            logits = torch.where(action_mask, logits, torch.tensor(huge_neg, device=logits.device, dtype=logits.dtype))

        return logits, value, next_hx

    def get_action(
        self,
        obs: torch.Tensor,
        hx: torch.Tensor,
        action_mask: Optional[torch.Tensor] = None,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Selects an action using categorical distribution over masked logits.
        Returns: action, log_prob, value, next_hx
        """
        logits, value, next_hx = self.forward(obs, hx, action_mask)
        dist = Categorical(logits=logits)
        if deterministic:
            action = torch.argmax(logits, dim=-1)
        else:
            action = dist.sample()
        log_prob = dist.log_prob(action)
        return action, log_prob, value.squeeze(-1), next_hx

    def evaluate_actions(
        self,
        obs: torch.Tensor,
        hx: torch.Tensor,
        actions: torch.Tensor,
        action_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Evaluates batch of actions for PPO updates."""
        logits, value, next_hx = self.forward(obs, hx, action_mask)
        dist = Categorical(logits=logits)
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()
        return log_probs, value.squeeze(-1), entropy, next_hx
