"""E1.5: DQN with the adapter's action mask (SB3 DQN has no masking).

Invalid actions are excluded in both places where DQN takes a maximum or chooses:
- acting: epsilon-greedy picks a random VALID action, or the valid action with the highest Q-value;
- learning: the target max over next actions is taken over the actions valid in the next marking.
The next marking's mask is stored in the replay buffer with each transition (info["action_mask"] of the
step, which DummyVecEnv keeps for the terminal step too). If no action is valid in the next marking (a
dead end, which ends the episode under D1), the bootstrap term is 0. One environment only.
Also logs train/q_max_mean (mean over the batch of the largest valid next Q-value) to watch for divergence.
"""
import numpy as np
import torch as th
from torch.nn import functional as F
from sb3_contrib.common.maskable.utils import get_action_masks
from stable_baselines3 import DQN
from stable_baselines3.common.buffers import ReplayBuffer


class MaskedReplayBuffer(ReplayBuffer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.n_envs != 1:
            raise ValueError("MaskedReplayBuffer supports one environment")
        self.next_masks = np.ones((self.buffer_size, self.n_envs, self.action_space.n), dtype=bool)

    def add(self, obs, next_obs, action, reward, done, infos):
        mask = infos[0].get("action_mask")
        if mask is None:
            raise RuntimeError("step info has no action_mask")
        self.next_masks[self.pos, 0] = np.asarray(mask, dtype=bool)
        super().add(obs, next_obs, action, reward, done, infos)

    def sample_with_masks(self, batch_size):
        upper = self.buffer_size if self.full else self.pos
        inds = np.random.randint(0, upper, size=batch_size)
        return self._get_samples(inds), th.as_tensor(self.next_masks[inds, 0], device=self.device)


def masked_max(q: th.Tensor, masks: th.Tensor) -> th.Tensor:
    """Largest Q-value over valid actions per row; 0 for a row with no valid action."""
    best = q.masked_fill(~masks, float("-inf")).max(dim=1).values
    return th.where(masks.any(dim=1), best, th.zeros_like(best))


class MaskedDQN(DQN):
    def __init__(self, *args, **kwargs):
        kwargs["replay_buffer_class"] = MaskedReplayBuffer
        super().__init__(*args, **kwargs)

    def _sample_action(self, learning_starts, action_noise=None, n_envs=1):
        mask = np.asarray(get_action_masks(self.env), dtype=bool)[0]
        valid = np.flatnonzero(mask)
        if len(valid) == 0:
            raise RuntimeError("no valid action in a non-terminal marking")
        if self.num_timesteps < learning_starts or np.random.rand() < self.exploration_rate:
            action = int(np.random.choice(valid))
        else:
            obs = self.policy.obs_to_tensor(self._last_obs)[0]
            with th.no_grad():
                q = self.q_net(obs)[0].cpu().numpy()
            q[~mask] = -np.inf
            action = int(np.argmax(q))
        a = np.array([action])
        return a, a

    def train(self, gradient_steps: int, batch_size: int = 100) -> None:
        self.policy.set_training_mode(True)
        self._update_learning_rate(self.policy.optimizer)
        losses, q_max = [], []
        for _ in range(gradient_steps):
            replay_data, next_masks = self.replay_buffer.sample_with_masks(batch_size)
            with th.no_grad():
                next_q = masked_max(self.q_net_target(replay_data.next_observations), next_masks).reshape(-1, 1)
                target_q = replay_data.rewards + (1 - replay_data.dones) * self.gamma * next_q
                q_max.append(next_q.mean().item())
            current_q = th.gather(self.q_net(replay_data.observations), dim=1, index=replay_data.actions.long())
            loss = F.smooth_l1_loss(current_q, target_q)
            losses.append(loss.item())
            self.policy.optimizer.zero_grad()
            loss.backward()
            th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.policy.optimizer.step()
        self._n_updates += gradient_steps
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/loss", np.mean(losses))
        self.logger.record("train/q_max_mean", np.mean(q_max))
