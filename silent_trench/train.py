"""Self-play Reinforcement Learning training loop with PPO for Silent Trench."""
from typing import Dict, List, Tuple
import os
import random
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from silent_trench.constants import GameStatus, NodeType
from silent_trench.engine import Action, SilentTrenchGame
from silent_trench.maps import ALL_MAPS
from silent_trench.model import SubmarineActorCritic
from silent_trench.observations import (
    ObservationEncoder,
    action_to_index,
    index_to_action,
)
from silent_trench.bot import SonarTacticsBot
from silent_trench.agent import NeuralSubAgent


class RolloutBuffer:
    def __init__(self):
        self.obs: List[np.ndarray] = []
        self.hx: List[np.ndarray] = []
        self.masks: List[np.ndarray] = []
        self.actions: List[int] = []
        self.log_probs: List[float] = []
        self.rewards: List[float] = []
        self.dones: List[bool] = []
        self.values: List[float] = []

    def clear(self):
        self.obs.clear()
        self.hx.clear()
        self.masks.clear()
        self.actions.clear()
        self.log_probs.clear()
        self.rewards.clear()
        self.dones.clear()
        self.values.clear()


class PPOTrainer:
    def __init__(
        self,
        lr: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_eps: float = 0.2,
        c_val: float = 0.5,
        c_ent: float = 0.01,
        device: str = "cpu",
        checkpoint_dir: str = "checkpoints",
    ):
        self.device = torch.device(device)
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_eps = clip_eps
        self.c_val = c_val
        self.c_ent = c_ent
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)

        self.policy = SubmarineActorCritic().to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)

    def evaluate_against_bot(self, num_games: int = 10) -> float:
        """Evaluates policy win rate against the baseline SonarTacticsBot."""
        wins = 0
        map_keys = list(ALL_MAPS.keys())

        for i in range(num_games):
            map_name = map_keys[i % len(map_keys)]
            game = SilentTrenchGame(map_name)

            # Alternate playing as player 0 or player 1
            agent_pid = i % 2
            bot_pid = 1 - agent_pid

            agent = NeuralSubAgent(agent_pid, device=str(self.device), deterministic=True)
            agent.model = self.policy  # Use current weights
            agent.reset()

            bot = SonarTacticsBot(bot_pid, difficulty="medium")

            steps = 0
            while game.status == GameStatus.ONGOING and steps < 120:
                pid = game.active_player
                if pid == agent_pid:
                    action = agent.select_action(game)
                else:
                    action = bot.select_action(game)
                game.step(action)
                steps += 1

            if (agent_pid == 0 and game.status == GameStatus.P0_WIN) or \
               (agent_pid == 1 and game.status == GameStatus.P1_WIN):
                wins += 1

        win_rate = wins / float(num_games)
        return win_rate

    def collect_self_play_games(self, num_games: int = 20) -> RolloutBuffer:
        """Plays self-play matches between the policy and itself, collecting trajectory data."""
        buffer = RolloutBuffer()
        map_keys = list(ALL_MAPS.keys())
        self.policy.eval()

        for g_idx in range(num_games):
            map_name = random.choice(map_keys)
            game = SilentTrenchGame(map_name)

            hx_p0 = self.policy.initial_state(batch_size=1, device=self.device)
            hx_p1 = self.policy.initial_state(batch_size=1, device=self.device)

            p0_trajectory = []
            p1_trajectory = []

            steps = 0
            while game.status == GameStatus.ONGOING and steps < 100:
                pid = game.active_player
                curr_hx = hx_p0 if pid == 0 else hx_p1

                obs = ObservationEncoder.encode_observation(game, pid)
                mask = ObservationEncoder.get_action_mask(game, pid)

                obs_t = torch.from_numpy(obs).unsqueeze(0).to(self.device)
                mask_t = torch.from_numpy(mask).unsqueeze(0).to(self.device)

                with torch.no_grad():
                    action_t, log_prob_t, val_t, next_hx = self.policy.get_action(
                        obs_t, curr_hx, mask_t, deterministic=False
                    )

                if pid == 0:
                    hx_p0 = next_hx
                else:
                    hx_p1 = next_hx

                act_idx = int(action_t.item())
                action = index_to_action(act_idx)

                # Prior state for reward calculation
                prev_posts = game.get_controlled_posts_count(pid)
                prev_contact = game.last_known_contact[pid] is not None

                # Execute
                status, winner, _ = game.step(action)

                # Intermediate reward shaping
                reward = 0.0
                curr_posts = game.get_controlled_posts_count(pid)
                if curr_posts > prev_posts:
                    reward += 0.2  # Reward for capturing Listening Posts
                if not prev_contact and game.last_known_contact[pid] is not None:
                    reward += 0.1  # Reward for detecting enemy

                done = (status != GameStatus.ONGOING)

                step_record = {
                    "obs": obs,
                    "hx": curr_hx.squeeze(0).cpu().numpy(),
                    "mask": mask,
                    "action": act_idx,
                    "log_prob": float(log_prob_t.item()),
                    "value": float(val_t.item()),
                    "reward": reward,
                    "done": done,
                }

                if pid == 0:
                    p0_trajectory.append(step_record)
                else:
                    p1_trajectory.append(step_record)

                steps += 1

            # Assign terminal rewards based on game outcome
            p0_terminal = 0.0
            p1_terminal = 0.0
            if game.status == GameStatus.P0_WIN:
                p0_terminal = 1.0
                p1_terminal = -1.0
            elif game.status == GameStatus.P1_WIN:
                p0_terminal = -1.0
                p1_terminal = 1.0

            if p0_trajectory:
                p0_trajectory[-1]["reward"] += p0_terminal
                p0_trajectory[-1]["done"] = True
            if p1_trajectory:
                p1_trajectory[-1]["reward"] += p1_terminal
                p1_trajectory[-1]["done"] = True

            # Append to buffer
            for rec in p0_trajectory + p1_trajectory:
                buffer.obs.append(rec["obs"])
                buffer.hx.append(rec["hx"])
                buffer.masks.append(rec["mask"])
                buffer.actions.append(rec["action"])
                buffer.log_probs.append(rec["log_prob"])
                buffer.rewards.append(rec["reward"])
                buffer.dones.append(rec["done"])
                buffer.values.append(rec["value"])

        return buffer

    def train_epoch(self, buffer: RolloutBuffer, ppo_epochs: int = 4, batch_size: int = 64) -> Dict[str, float]:
        """Runs PPO gradient updates over collected rollouts."""
        self.policy.train()

        num_samples = len(buffer.obs)
        if num_samples == 0:
            return {}

        obs_t = torch.tensor(np.array(buffer.obs), dtype=torch.float32, device=self.device)
        hx_t = torch.tensor(np.array(buffer.hx), dtype=torch.float32, device=self.device)
        masks_t = torch.tensor(np.array(buffer.masks), dtype=torch.bool, device=self.device)
        actions_t = torch.tensor(np.array(buffer.actions), dtype=torch.long, device=self.device)
        old_log_probs_t = torch.tensor(np.array(buffer.log_probs), dtype=torch.float32, device=self.device)
        rewards = np.array(buffer.rewards, dtype=np.float32)
        dones = np.array(buffer.dones, dtype=np.bool_)
        values = np.array(buffer.values, dtype=np.float32)

        # Compute GAE advantages and returns
        advantages = np.zeros_like(rewards)
        last_gae = 0.0
        for t in reversed(range(num_samples)):
            next_val = 0.0 if dones[t] or (t == num_samples - 1) else values[t + 1]
            next_non_terminal = 0.0 if dones[t] else 1.0
            delta = rewards[t] + self.gamma * next_val * next_non_terminal - values[t]
            advantages[t] = last_gae = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae

        returns = advantages + values
        adv_t = torch.tensor(advantages, dtype=torch.float32, device=self.device)
        returns_t = torch.tensor(returns, dtype=torch.float32, device=self.device)

        # Normalize advantages
        if adv_t.std() > 1e-6:
            adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)

        total_loss = 0.0
        indices = np.arange(num_samples)

        for _ in range(ppo_epochs):
            np.random.shuffle(indices)
            for start in range(0, num_samples, batch_size):
                end = min(start + batch_size, num_samples)
                idx = indices[start:end]

                b_obs = obs_t[idx]
                b_hx = hx_t[idx]
                b_masks = masks_t[idx]
                b_acts = actions_t[idx]
                b_old_lp = old_log_probs_t[idx]
                b_adv = adv_t[idx]
                b_ret = returns_t[idx]

                new_lp, new_val, entropy, _ = self.policy.evaluate_actions(
                    b_obs, b_hx, b_acts, b_masks
                )

                # Ratio
                ratio = torch.exp(new_lp - b_old_lp)

                # Clipped surrogate objective
                surr1 = ratio * b_adv
                surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * b_adv
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value loss
                val_loss = nn.MSELoss()(new_val, b_ret)

                # Total loss
                loss = policy_loss + (self.c_val * val_loss) - (self.c_ent * entropy.mean())

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=0.5)
                self.optimizer.step()

                total_loss += loss.item()

        return {
            "loss": total_loss,
            "samples": num_samples,
        }

    def train(self, total_iterations: int = 10, games_per_iter: int = 20) -> None:
        """Full training loop."""
        print(f"=== Starting RL Self-Play Training on {self.device} ===")
        best_winrate = -1.0

        for it in range(1, total_iterations + 1):
            t0 = time.time()
            buffer = self.collect_self_play_games(num_games=games_per_iter)
            train_metrics = self.train_epoch(buffer)
            elapsed = time.time() - t0

            # Evaluate against bot
            eval_winrate = self.evaluate_against_bot(num_games=10)
            print(
                f"[Iter {it:03d}/{total_iterations}] "
                f"Samples: {train_metrics.get('samples', 0):04d} | "
                f"Eval Win Rate vs Bot: {eval_winrate * 100:.1f}% | "
                f"Time: {elapsed:.2f}s"
            )

            # Save checkpoint
            latest_path = os.path.join(self.checkpoint_dir, "latest_model.pt")
            torch.save(self.policy.state_dict(), latest_path)

            if eval_winrate > best_winrate:
                best_winrate = eval_winrate
                best_path = os.path.join(self.checkpoint_dir, "best_model.pt")
                torch.save(self.policy.state_dict(), best_path)
                print(f"  -> New best model saved to {best_path} (Win rate: {best_winrate * 100:.1f}%)")


if __name__ == "__main__":
    trainer = PPOTrainer()
    trainer.train(total_iterations=5, games_per_iter=15)
