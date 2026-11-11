"""E1.5: tabular Q-learning with the adapter's action mask, for graphs with few reachable markings.

State = the observation (included, executed, pending bits per event; plus the budget when it is part of
the observation). Q(s, a) starts at 0. Epsilon-greedy over VALID actions only (ties broken at random),
epsilon falls linearly from 1 to 0.05 over the first 20% of training (as for the masked DQN).
Update: Q(s, a) += alpha x (r + gamma x max over valid a' of Q(s', a') - Q(s, a)); no bootstrap after a
terminated step (acceptance or a dead end), bootstrap after a step-limit truncation. alpha = 0.1, gamma = 1.

The training loop feeds the same StepDebugCallback as the SB3 runs, so the trace CSV has the same columns.
"""
import json
import numpy as np


class _Stub:
    """The parts of an SB3 model that BaseCallback uses."""

    def __init__(self, env, logger):
        self.num_timesteps = 0
        self.logger = logger
        self._env = env

    def get_env(self):
        return self._env


class TabularQ:
    def __init__(self, env, seed: int, alpha: float = 0.1, gamma: float = 1.0, eps_start: float = 1.0,
                 eps_end: float = 0.05, eps_fraction: float = 0.2):
        self.env = env
        self.rng = np.random.default_rng(seed)
        self.alpha, self.gamma = alpha, gamma
        self.eps_start, self.eps_end, self.eps_fraction = eps_start, eps_end, eps_fraction
        self.n_actions = env.action_space.n
        self.q = {}

    def _row(self, obs):
        key = np.asarray(obs).tobytes()
        if key not in self.q:
            self.q[key] = np.zeros(self.n_actions)
        return self.q[key]

    def _epsilon(self, t, total):
        frac = min(1.0, t / max(1, self.eps_fraction * total))
        return self.eps_start + frac * (self.eps_end - self.eps_start)

    def _choose(self, obs, mask, eps):
        valid = np.flatnonzero(mask)
        if len(valid) == 0:
            raise RuntimeError("no valid action in a non-terminal marking")
        if self.rng.random() < eps:
            return int(self.rng.choice(valid))
        q = self._row(obs)[valid]
        return int(self.rng.choice(valid[q == q.max()]))

    def learn(self, total_timesteps: int, callback, logger, rollout_len: int = 2048):
        model = _Stub(self.env, logger)
        callback.init_callback(model)
        callback.on_training_start({}, {})
        obs, _ = self.env.reset()
        for t in range(total_timesteps):
            mask = np.asarray(self.env.action_masks(), dtype=bool)
            action = self._choose(obs, mask, self._epsilon(t, total_timesteps))
            obs2, reward, terminated, truncated, info = self.env.step(action)
            row = self._row(obs)
            if terminated:
                target = reward
            else:
                next_mask = np.asarray(info.get("action_mask"), dtype=bool)
                next_row = self._row(obs2)
                target = reward + self.gamma * (next_row[next_mask].max() if next_mask.any() else 0.0)
            row[action] += self.alpha * (target - row[action])
            model.num_timesteps = t + 1
            done = terminated or truncated
            callback.update_locals({"infos": [info], "actions": np.array([action]),
                                    "rewards": np.array([reward]), "dones": np.array([done])})
            callback.on_step()
            if (t + 1) % rollout_len == 0:
                callback.on_rollout_end()
            obs = self.env.reset()[0] if done else obs2
        callback.on_training_end()

    def save(self, path):
        with open(path, "w") as fh:
            json.dump({"alpha": self.alpha, "gamma": self.gamma, "states": len(self.q),
                       "q": {k.hex(): v.tolist() for k, v in self.q.items()}}, fh)
