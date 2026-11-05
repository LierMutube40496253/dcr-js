import gymnasium as gym
from gymnasium import spaces
import numpy as np
import os
import sys

# Make imports work whether package is installed or running from src/ directly
if __package__ is None:
    # add project src/ to path
    this_dir = os.path.dirname(__file__)
    src_root = os.path.abspath(os.path.join(this_dir, ".."))
    if src_root not in sys.path:
        sys.path.insert(0, src_root)

from utils.node_connector import connect_to_dcr_environment
from envs.partway import TraceFrontier

def state_to_obs(state, event_list):
    obs = []
    for ev in event_list:
        obs.append(1 if ev in state.get("included", []) else 0)
        obs.append(1 if ev in state.get("executed", []) else 0)
        obs.append(1 if ev in state.get("pending", []) else 0)
    return np.array(obs, dtype=np.int8)

def state_to_mask(state, event_list):
    # Mask 1 if event is valind (pending/allowed). We prefer server-provided mask when available.
    # Fallback: mark events with pending==1 as valid.
    mask = []
    for ev in event_list:
        mask.append(1 if ev in state.get("pending", []) else 0)
    return mask

class DCRGymEnv(gym.Env):
    metadata = {"render.modes": []}

    def __init__(self, node_url="http://localhost:5001", truncation_fix=False, dead_end_truncate=False,
                 fix_dead_end=False, partway_starts="none", partway_initial_share=0.2, partway_prob=1.0,
                 partway_max=20, partway_traces=None, partway_timelimit_factor=None, partway_seed=0):
        super().__init__()
        # Special-course switch (issue A2), off by default: when on, step() returns
        # terminated = real acceptance and truncated = step limit reached without acceptance.
        self.truncation_fix = truncation_fix
        # Special-course switch for masked runs, off by default: a non-accepting marking with no
        # allowed action (all-zero adapter mask) ends the episode as truncated, not accepting.
        self.dead_end_truncate = dead_end_truncate
        # Special-course switch (issue D1), off by default: masked or not, a non-accepting marking
        # with no enabled event ends the episode as terminated (no future, value 0), not accepting.
        # Takes precedence over dead_end_truncate.
        self.fix_dead_end = fix_dead_end
        # Special-course E1.4b partway starts, "none" by default (see reset() and envs/partway.py).
        self.partway_starts = partway_starts
        self.partway_initial_share = partway_initial_share
        self.partway_prob = partway_prob
        self.partway_max = partway_max
        self.partway_traces = partway_traces or []
        self.partway_timelimit_factor = partway_timelimit_factor
        self.partway_frontiers = [TraceFrontier(len(t)) for t in self.partway_traces]
        self._partway_rng = np.random.default_rng(partway_seed)  # separate from the agent's RNG
        self.start_type, self.prefix_len, self.partway_trace = "initial", 0, None
        self._episode_limit, self._episode_steps = None, 0
        self.node = connect_to_dcr_environment(node_url)
        init = self.node["reset"]()
        events = init["events"]
        self.event_list = events
        self.label_map = init.get("labelMap", {}) if isinstance(init, dict) else {}

        # Role-extended action space: server returns eventRolePairs when roles are defined.
        # Each pair is {"event": str, "role": str}. Action index maps to a pair.
        self.event_role_pairs = init.get("eventRolePairs") or [{"event": ev, "role": ""} for ev in events]
        self.roles = init.get("roles", [])
        self.n_roles = init.get("nRoles", 1)

        n_events = len(self.event_list)
        n_actions = len(self.event_role_pairs)
        # Budget dim appended only when k is finite; k=None → 36 dims, byte-identical to today.
        self.expert_budget_k = init.get("expertBudgetK")         # None when EXPERT_BUDGET_K unset
        self.expert_budget_remaining = init.get("expertBudgetRemaining")
        obs_dim = n_events * 3 + (1 if self.expert_budget_k is not None else 0)
        obs_dtype = np.float32 if self.expert_budget_k is not None else np.int8
        self.observation_space = spaces.Box(low=0, high=1, shape=(obs_dim,), dtype=obs_dtype)
        self.action_space = spaces.Discrete(n_actions)

        self.action_mask = init.get("actionMask") if isinstance(init, dict) else None
        self._adapter_mask = self.action_mask  # raw adapter mask for action_masks(), no fallback
        if self.action_mask is None:
            self.action_mask = state_to_mask(init.get("state", {}), self.event_list)
        if partway_starts == "trace":
            if not self.partway_traces:
                raise ValueError("partway_starts='trace' needs partway_traces")
            for trace in self.partway_traces:  # a demonstration must be legal and end accepting
                self.node["reset"]()
                if not self.node["replay"](trace)["accepting"]:
                    raise ValueError(f"demonstration does not reach an accepting marking: {trace}")

    def _state_to_obs(self, state, expert_budget_remaining=None):
        arr = []
        for ev in self.event_list:
            included = 1 if ev in state.get("included", []) else 0
            executed = 1 if ev in state.get("executed", []) else 0
            pending = 1 if ev in state.get("pending", []) else 0
            arr.extend([included, executed, pending])
        if self.expert_budget_k is not None:
            frac = float(expert_budget_remaining or 0) / self.expert_budget_k
            arr.append(min(1.0, max(0.0, frac)))
        return np.array(arr, dtype=np.float32 if self.expert_budget_k is not None else np.int8)

    def _partway_start(self, r):
        """Replay a prefix before the episode (no reward, no cost, no step, nothing for training).
        Takes the /reset answer, returns the answer describing the marking the episode starts from."""
        rng = self._partway_rng
        self.start_type, self.prefix_len, self.partway_trace = "initial", 0, None
        self._episode_limit, self._episode_steps = None, 0
        if rng.random() < self.partway_initial_share:
            return r
        if self.partway_starts == "random":
            if rng.random() >= self.partway_prob:
                return r
            r = self._random_prefix(r, int(rng.integers(1, self.partway_max + 1)))
        else:  # trace
            self.partway_trace = int(rng.integers(len(self.partway_traces)))
            trace = self.partway_traces[self.partway_trace]
            self.prefix_len = self.partway_frontiers[self.partway_trace].sample_prefix_len(rng)
            if self.prefix_len > 0:
                r = self.node["replay"](trace[:self.prefix_len])
            if self.partway_timelimit_factor is not None:
                self._episode_limit = self.partway_timelimit_factor * (len(trace) - self.prefix_len) + 10
        self.start_type = "partway" if self.prefix_len > 0 else "initial"
        return r

    def _random_prefix(self, r, length):
        """A random legal prefix, one event at a time. A prefix must leave work for the agent, so
        an event that reaches an accepting marking or a dead end is undone and the prefix ends."""
        chosen = []
        for _ in range(length):
            enabled = self._enabled_events(r)
            if not enabled:
                break
            event = enabled[int(self._partway_rng.integers(len(enabled)))]
            after = self.node["replay"]([event])
            if after["accepting"] or not after["validActions"]:
                r = self.node["reset"]()
                if chosen:
                    r = self.node["replay"](chosen)
                break
            chosen.append(event)
            r = after
        self.prefix_len = len(chosen)
        return r

    def _enabled_events(self, r):
        if "validActions" in r:
            return r["validActions"]
        mask = r.get("actionMask") or []  # the /reset answer: first role of each enabled event
        return list(dict.fromkeys(p["event"] for p, m in zip(self.event_role_pairs, mask) if m))

    def _partway_step(self, info, result, terminated, truncated):
        """Per-episode log fields, optional shorter limit, frontier update at the end of an episode."""
        self._episode_steps += 1
        if (self._episode_limit is not None and self._episode_steps >= self._episode_limit
                and not terminated and not truncated):
            truncated = True
            info["partway_timelimit"] = True
        info["start_type"] = self.start_type
        info["prefix_len"] = self.prefix_len
        info["partway_trace"] = self.partway_trace
        if self.partway_trace is not None:
            frontier = self.partway_frontiers[self.partway_trace]
            info["partway_j"] = frontier.j
            if terminated or truncated:
                success = bool(result.get("accepting", False)) and result.get("pendingAfter", 0) == 0
                move = frontier.record(self.prefix_len, success)
                if move:
                    info["partway_frontier_move"] = (self.partway_trace, *move)
        return truncated

    def reset(self, seed=None, options=None):
        r = self.node["reset"]()
        if self.partway_starts != "none":
            r = self._partway_start(r)
        self.expert_budget_remaining = r.get("expertBudgetRemaining")
        obs = self._state_to_obs(r["state"], self.expert_budget_remaining)
        self.label_map = r.get("labelMap", self.label_map)
        self.action_mask = r.get("actionMask", state_to_mask(r.get("state", {}), self.event_list))
        self._adapter_mask = r.get("actionMask")
        self.episode_cost = 0.0
        self.episode_duration = 0.0
        return obs, {}

    def step(self, action):
        # Send numeric action index as plain int (numpy int64 is not JSON serializable)
        r = self.node["send_action"](int(action))
        result = r.get("result", {})
        self.expert_budget_remaining = result.get("expertBudgetRemaining", self.expert_budget_remaining)
        obs = self._state_to_obs(result.get("state", {}), self.expert_budget_remaining)
        reward = result.get("stepReward", result.get("reward", 0))
        done = bool(result.get("done", False))
        self.action_mask = r.get("actionMask") or state_to_mask(result.get("state", {}), self.event_list)
        self._adapter_mask = r.get("actionMask")

        self.episode_cost     = result.get("episodeCost", self.episode_cost)
        self.episode_duration = result.get("episodeDuration", self.episode_duration)

        pair = self.event_role_pairs[action] if action < len(self.event_role_pairs) else {}
        info = {
            "engine_result":    result,
            "action_mask":      self.action_mask,
            "event_cost":       result.get("eventCost"),
            "event_duration":   result.get("eventDuration"),
            "episode_cost":     self.episode_cost,
            "episode_duration": self.episode_duration,
            "accepting":               result.get("accepting", False),
            "action_event":            pair.get("event", ""),
            "action_role":             pair.get("role", ""),
            "intercepted_reason":      result.get("interceptedReason"),
            "expert_budget_remaining": result.get("expertBudgetRemaining"),
            "expert_budget_initial":   result.get("expertBudgetInitial"),
            "num_budget_blocks":       result.get("numBudgetBlocks"),
        }
        terminated, truncated = done, False
        dead_end = False
        if self.dead_end_truncate or self.fix_dead_end:
            mask = self._adapter_mask
            dead_end = (not done and not result.get("accepting", False)
                        and mask is not None and not any(mask))
            info["dead_end"] = dead_end
        if dead_end:
            terminated, truncated = (True, False) if self.fix_dead_end else (False, True)
        elif self.truncation_fix:
            terminated = bool(result.get("accepting", False))
            truncated = done and not terminated  # adapter: done = accepting or maxStepReached
        if self.partway_starts != "none":
            truncated = self._partway_step(info, result, terminated, truncated)
        return obs, reward, terminated, truncated, info

    def action_masks(self):
        """Boolean mask over (event, role) actions for sb3-contrib MaskablePPO, taken from the
        adapter's actionMask. Never falls back to state_to_mask (issue B2)."""
        mask = getattr(self, "_adapter_mask", None)
        if mask is None:
            raise RuntimeError("adapter sent no actionMask; cannot mask actions")
        if len(mask) != self.action_space.n:
            raise RuntimeError(f"actionMask has {len(mask)} entries, action space has {self.action_space.n}")
        mask = np.asarray(mask, dtype=bool)
        if not mask.any():
            # With dead_end_truncate, step() already ended such episodes; this is only reached at
            # reset (a graph whose initial marking is a dead end) or with the switch off.
            raise RuntimeError("actionMask is all zero: no legal action in a non-terminal marking")
        return mask

    def close(self):
        pass