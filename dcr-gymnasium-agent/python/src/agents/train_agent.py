import os
import sys
import time
import argparse
import csv
from pathlib import Path

# Use stable-baselines3 PPO directly (no action masking).
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.utils import set_random_seed

if __package__ is None:
    this_dir = os.path.dirname(__file__)
    src_root = os.path.abspath(os.path.join(this_dir, ".."))
    if src_root not in sys.path:
        sys.path.insert(0, src_root)

from envs.dcr_env import DCRGymEnv
from envs.partway import load_traces
from agents.lagrange import lagrange_update

MASKED_ALGOS = ("maskable_ppo", "maskable_a2c", "maskable_dqn", "tabular_q")
# E1.5 masked DQN: SB3 defaults (lr 1e-4, batch 32, train every 4 steps, 1 gradient step, Huber loss,
# max grad norm 10, final epsilon 0.05, net 64x64) except where the default is tied to a much longer
# run: replay buffer 100,000 (= the whole run), learning starts after 1,000 steps, target network
# copied every 1,000 steps (default 10,000 would give 10 copies in the run), epsilon decays over 20%.
DQN_SETTINGS = dict(learning_rate=1e-4, buffer_size=100_000, learning_starts=1_000, batch_size=32,
                    train_freq=4, gradient_steps=1, target_update_interval=1_000,
                    exploration_fraction=0.2, exploration_initial_eps=1.0, exploration_final_eps=0.05,
                    max_grad_norm=10)


class StepDebugCallback(BaseCallback):
    """Per-step debug trace: actions, rewards, and all reward components."""

    def __init__(self, env_ref, csv_path: Path, log_every: int = 500, verbose: int = 0,
                 violation_fix: bool = False, node_url: str = "", lagrange: tuple | None = None):
        super().__init__(verbose)
        # E1.8: violation columns only with --violation-fix; lagrange = (lambda_0, lr) or None
        self.violation_fix = violation_fix
        self.node_url = node_url
        self.lagrange = lagrange
        self.lam = lagrange[0] if lagrange else None
        self._rollout_steps = 0
        self._rollout_violations = 0
        self._rollouts = 0
        self.lagrange_csv_path = csv_path.with_name(csv_path.name.replace("train_trace_", "lagrange_"))
        self._lagrange_file = None
        self._lagrange_writer = None
        self.env_ref = env_ref
        self.csv_path = csv_path
        self.log_every = log_every
        self.episode_idx = 1
        self.step_in_episode = 0
        self._episode_reward_sum = 0.0
        self._ep_illegal_count = 0
        self._ep_budget_block_count = 0
        self._csv_file = None
        self._writer = None
        # Dead-end logging only for masked runs, so her CSV format is unchanged otherwise
        self.log_dead_ends = getattr(env_ref, "dead_end_truncate", False) or getattr(env_ref, "fix_dead_end", False)
        self.dead_end_episodes = 0
        # E1.4b: start columns and a frontier log only for partway runs, her CSV format otherwise
        self.log_partway = getattr(env_ref, "partway_starts", "none") != "none"
        self.frontier_csv_path = csv_path.with_name(csv_path.name.replace("train_trace_", "partway_frontier_"))
        self._frontier_file = None
        self._frontier_writer = None

    def _on_training_start(self) -> None:
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        self._csv_file = open(self.csv_path, "w", newline="")
        self._writer = csv.DictWriter(
            self._csv_file,
            fieldnames=[
                "global_timestep",
                "episode",
                "step_in_episode",
                "action_idx",
                "action_event",
                "action_role",
                "action_label",
                "reward",
                "ep_rew_sum",
                "done",
                # Reward components
                "base_mapped",
                "novelty_delta",
                "progress_delta",
                # DCR state info
                "pending_before",
                "pending_after",
                "accepting",
                "goal_reached",
                "max_step_reached",
                # Convergence analysis
                "illegal_traces_count",
                "episode_steps",
                "illegal_traces_ratio",
                # Multi-objective metrics
                "event_cost",
                "event_duration",
                "episode_cost",
                "episode_duration",
                # Ablation: DCR compliance regardless of shield state
                "action_compliant",
                "message",
                "intercepted_reason",
                "expert_budget_remaining",
                "expert_budget_initial",
            ] + (["dead_end"] if self.log_dead_ends else [])
              + (["start_type", "prefix_len", "partway_trace", "partway_j"] if self.log_partway else [])
              + (["violation", "violation_price"] if self.violation_fix else []),
        )
        self._writer.writeheader()
        if self.lagrange:
            self._set_price(self.lam)
            self._lagrange_file = open(self.lagrange_csv_path, "w", newline="")
            self._lagrange_writer = csv.writer(self._lagrange_file)
            self._lagrange_writer.writerow(["global_timestep", "rollout", "steps", "violations",
                                            "violations_per_step", "lambda"])
            self._lagrange_writer.writerow([0, 0, 0, 0, "", self.lam])
            self._lagrange_file.flush()
        if self.log_partway and self.env_ref.partway_frontiers:
            self._frontier_file = open(self.frontier_csv_path, "w", newline="")
            self._frontier_writer = csv.writer(self._frontier_file)
            self._frontier_writer.writerow(["global_timestep", "episode", "trace", "j_old", "j_new", "success_rate"])
            for i, frontier in enumerate(self.env_ref.partway_frontiers):
                self._frontier_writer.writerow([0, 0, i, frontier.j, frontier.j, ""])
            self._frontier_file.flush()

    def _on_step(self) -> bool:
        infos = self.locals.get("infos") or []
        actions = self.locals.get("actions")
        rewards = self.locals.get("rewards")
        dones = self.locals.get("dones")

        if not infos:
            return True

        idx = 0
        info = infos[idx] if len(infos) > idx else {}
        action_idx = int(actions[idx]) if actions is not None else -1
        reward = float(rewards[idx]) if rewards is not None else 0.0
        done = bool(dones[idx]) if dones is not None else False

        engine_result = info.get("engine_result", {})
        pairs = getattr(self.env_ref, "event_role_pairs", None)
        if pairs and 0 <= action_idx < len(pairs):
            action_event = pairs[action_idx].get("event", "")
            action_role  = pairs[action_idx].get("role", "")
        elif 0 <= action_idx < len(self.env_ref.event_list):
            action_event = self.env_ref.event_list[action_idx]
            action_role  = info.get("action_role", "")
        else:
            action_event = ""
            action_role  = ""
        action_label = self.env_ref.label_map.get(action_event, action_event)

        self.step_in_episode += 1
        self._episode_reward_sum += reward

        base_mapped = engine_result.get("baseMapped", None)
        intercepted_reason = engine_result.get("interceptedReason", "")
        # dcr_illegal and budget_block both have baseMapped==-10; keep the two counts separate.
        if (base_mapped == -10 and intercepted_reason != "budget_block") or intercepted_reason == "violation":
            self._ep_illegal_count += 1
        self._rollout_steps += 1
        self._rollout_violations += intercepted_reason == "violation"
        if intercepted_reason == "budget_block":
            self._ep_budget_block_count += 1
        
        # Get illegal traces count and episode steps from server
        illegal_traces_count = engine_result.get("illegalTracesCount", 0)
        episode_steps = engine_result.get("episodeSteps", self.step_in_episode)
        
        # Calculate illegal traces ratio (percentage of steps that were illegal)
        illegal_traces_ratio = (
            (illegal_traces_count / episode_steps * 100) if episode_steps > 0 else 0
        )
        
        row = {
            "global_timestep": int(self.num_timesteps),
            "episode": self.episode_idx,
            "step_in_episode": self.step_in_episode,
            "action_idx": action_idx,
            "action_event": action_event,
            "action_role": action_role,
            "action_label": action_label,
            "reward": reward,
            "ep_rew_sum": round(self._episode_reward_sum, 4),
            "done": done,
            "base_mapped": engine_result.get("baseMapped", ""),
            "novelty_delta": engine_result.get("noveltyDelta", ""),
            "progress_delta": engine_result.get("progressDelta", ""),
            "pending_before": engine_result.get("pendingBefore", ""),
            "pending_after": engine_result.get("pendingAfter", ""),
            "accepting": engine_result.get("accepting", ""),
            "goal_reached": engine_result.get("goalReached", ""),
            "max_step_reached": engine_result.get("maxStepReached", ""),
            "illegal_traces_count": illegal_traces_count,
            "episode_steps": episode_steps,
            "illegal_traces_ratio": round(illegal_traces_ratio, 2),
            "event_cost":       info.get("event_cost", ""),
            "event_duration":   info.get("event_duration", ""),
            "episode_cost":     info.get("episode_cost", ""),
            "episode_duration": info.get("episode_duration", ""),
            "action_compliant":        engine_result.get("actionCompliant", True),
            "message":                 engine_result.get("msg", ""),
            "intercepted_reason":      intercepted_reason,
            "expert_budget_remaining": engine_result.get("expertBudgetRemaining", ""),
            "expert_budget_initial":   engine_result.get("expertBudgetInitial", ""),
        }
        if self.log_dead_ends:
            row["dead_end"] = bool(info.get("dead_end", False))
            if row["dead_end"]:
                self.dead_end_episodes += 1
        if self.log_partway:
            row["start_type"] = info.get("start_type", "")
            row["prefix_len"] = info.get("prefix_len", "")
            row["partway_trace"] = info.get("partway_trace", "")
            row["partway_j"] = info.get("partway_j", "")
            move = info.get("partway_frontier_move")
            if move:
                trace, j_old, j_new, rate = move
                self._frontier_writer.writerow([int(self.num_timesteps), self.episode_idx, trace, j_old, j_new, rate])
                self._frontier_file.flush()
                self.logger.record(f"partway/j_trace{trace}", j_new)
        if self.violation_fix:
            row["violation"] = bool(engine_result.get("violation", False))
            row["violation_price"] = engine_result.get("violationPrice", "")
        self._writer.writerow(row)

        if (self.num_timesteps % self.log_every == 0) or done:
            print(
                f"[TRAIN] t={row['global_timestep']} ep={row['episode']} "
                f"step={row['step_in_episode']} action={action_label} "
                f"reward={reward:.2f} progress={row['progress_delta']} "
                f"pending={row['pending_before']}→{row['pending_after']} "
                f"accepting={row['accepting']} "
                f"illegal_ratio={row['illegal_traces_ratio']}%"
            )

        if done:
            illegal_traces_ratio = (
                self._ep_illegal_count / self.step_in_episode * 100
                if self.step_in_episode > 0
                else 0
            )
            self.logger.record("train/ep_rew_sum", self._episode_reward_sum)
            self.logger.record("train/illegal_traces_count", self._ep_illegal_count)
            self.logger.record("train/budget_block_count", self._ep_budget_block_count)
            self.logger.record("train/illegal_traces_ratio", illegal_traces_ratio)
            self.logger.record("train/episode_steps", episode_steps)
            if self.log_dead_ends:
                self.logger.record("train/dead_end_episodes_total", self.dead_end_episodes)

            # Multi-objective: log cost/duration on accepting episodes
            if info.get("accepting"):
                ep_cost = info.get("episode_cost")
                ep_dur  = info.get("episode_duration")
                if ep_cost is not None:
                    self.logger.record("pareto/episode_cost", ep_cost)
                if ep_dur is not None:
                    self.logger.record("pareto/episode_duration", ep_dur)

            self.logger.dump(self.num_timesteps)
            self.episode_idx += 1
            self.step_in_episode = 0
            self._episode_reward_sum = 0.0
            self._ep_illegal_count = 0
            self._ep_budget_block_count = 0

        return True

    def _set_price(self, price: float) -> None:
        import requests
        r = requests.post(f"{self.node_url}/violation_price", json={"price": price}, timeout=10)
        if r.status_code != 200:
            raise RuntimeError(f"/violation_price answered {r.status_code}: {r.text}")

    def _on_rollout_end(self) -> None:
        # E1.8 Lagrangian: one multiplier update per rollout, used as the price from the next rollout on
        if self.lagrange:
            self._rollouts += 1
            rate = self._rollout_violations / self._rollout_steps if self._rollout_steps else 0.0
            self.lam = lagrange_update(self.lam, self.lagrange[1], self._rollout_violations, self._rollout_steps)
            self._set_price(self.lam)
            self._lagrange_writer.writerow([int(self.num_timesteps), self._rollouts, self._rollout_steps,
                                            self._rollout_violations, round(rate, 6), round(self.lam, 6)])
            self._lagrange_file.flush()
            self.logger.record("lagrange/lambda", self.lam)
            self.logger.record("lagrange/violations_per_step", rate)
        self._rollout_steps = 0
        self._rollout_violations = 0

    def _on_training_end(self) -> None:
        if self._lagrange_file is not None:
            self._lagrange_file.close()
        if self._csv_file is not None:
            self._csv_file.close()
        if self._frontier_file is not None:
            self._frontier_file.close()
        if self.log_dead_ends:
            how = "terminated" if getattr(self.env_ref, "fix_dead_end", False) else "truncated"
            print(f"Dead-end episodes ({how}, no allowed action): {self.dead_end_episodes} "
                  f"of {self.episode_idx - 1}")


def train(
    node_url: str,
    exp_id: str,
    total_timesteps: int,
    out_dir: Path,
    ent_coef: float,
    tb_log_dir: Path,
    seed: int,
    algo: str = "ppo",
    truncation_fix: bool = False,
    gamma: float = 0.99,
    fix_dead_end: bool = False,
    partway_starts: str = "none",
    partway_initial_share: float = 0.2,
    partway_prob: float = 1.0,
    partway_max: int = 20,
    partway_traces: list | None = None,
    partway_timelimit_factor: float | None = None,
    partway_step: int = 1,
    violation_fix: bool = False,
    lagrange: tuple | None = None,
):
    ts = time.strftime("%Y%m%dT%H%M%S")
    run_name = f"exp_{exp_id}_{ts}"

    out_dir.mkdir(parents=True, exist_ok=True)
    tb_log_dir.mkdir(parents=True, exist_ok=True)

    print(f"--- Starting Experiment ---")
    print(f"Experiment ID:  {exp_id}")
    print(f"Entropy Coef:   {ent_coef}")
    print(f"Seed:           {seed}")
    print(f"Total Steps:    {total_timesteps}")
    print(f"TB Log Dir:     {tb_log_dir}")
    if algo == "maskable_ppo":
        print(f"Policy:         MaskablePPO (adapter action mask)")
    elif algo == "maskable_a2c":
        print(f"Policy:         Masked A2C (MaskablePPO with the A2C settings of Huang et al. 2022)")
        print(f"A2C settings:   n_steps 5, 1 epoch, full batch, gae_lambda 1, no advantage normalisation, "
              f"RMSprop(alpha 0.99, eps 1e-5), lr 7e-4, vf_coef 0.5, max_grad_norm 0.5")
    elif algo == "maskable_dqn":
        print(f"Policy:         Masked DQN (invalid actions excluded in action choice and target max)")
        print(f"DQN settings:   {DQN_SETTINGS}")
    elif algo == "tabular_q":
        print(f"Policy:         Tabular Q-learning with the action mask")
        print(f"Q settings:     alpha 0.1, epsilon 1 -> 0.05 over the first 20% of steps, Q starts at 0")
    else:
        print(f"Policy:         PPO (no action masking)")
    if truncation_fix:
        print(f"Truncation fix: on (step limit -> truncated)")
    if gamma != 0.99:
        print(f"Gamma:          {gamma}")
    if fix_dead_end:
        print(f"Dead ends:      terminated, not accepting (D1)")
    if partway_starts != "none":
        print(f"Partway starts: {partway_starts}")
        print(f"Partway initial share: {partway_initial_share}")
        if partway_starts == "random":
            print(f"Partway prob, max: {partway_prob}, {partway_max}")
        else:
            print(f"Partway traces: {partway_traces}")
            print(f"Partway time limit factor: {partway_timelimit_factor}")
            print(f"Partway step: {partway_step}")
    if violation_fix:
        print(f"Violations:     priced as legal steps plus a violation price (VIOLATION_FIX)")
    if lagrange:
        print(f"Lagrangian:     lambda_0 {lagrange[0]}, lr {lagrange[1]}, target 0 violations per step")
    print(f"---------------------------")

    logs_dir = out_dir.parent / "scripts" / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    train_trace_csv = logs_dir / f"train_trace_{run_name}.csv"

    # --- Environment factory (no ActionMasker here) ---
    # The environment communicates valid actions via actionMask,
    # but the agent explores all actions and learns illegal ones by negative reward.
    def make_env(seed_offset: int):
        def _init():
            e = DCRGymEnv(node_url=node_url, truncation_fix=truncation_fix,
                          dead_end_truncate=(algo in MASKED_ALGOS), fix_dead_end=fix_dead_end,
                          partway_starts=partway_starts, partway_initial_share=partway_initial_share,
                          partway_prob=partway_prob, partway_max=partway_max, partway_traces=partway_traces,
                          partway_timelimit_factor=partway_timelimit_factor, partway_seed=seed + seed_offset,
                          partway_step=partway_step)
            # Ensure each env instance gets a deterministic seed.
            e.reset(seed=seed + seed_offset)
            return Monitor(e)
        return _init

    set_random_seed(seed)
    vec_env = DummyVecEnv([make_env(0)])

    # env_ref points to the inner DCRGymEnv (unwrap Monitor → DCRGymEnv)
    env_ref = vec_env.envs[0].env

    callback = StepDebugCallback(
        env_ref=env_ref, csv_path=train_trace_csv, log_every=500,
        violation_fix=violation_fix, node_url=node_url, lagrange=lagrange,
    )
    tb = str(tb_log_dir) if tb_log_dir and str(tb_log_dir) else None

    # E1.5 tabular Q-learning: own loop, same callback and trace CSV, TensorBoard log via SB3's logger
    if algo == "tabular_q":
        from stable_baselines3.common.logger import configure
        from agents.tabular_q import TabularQ
        logger = configure(str(Path(tb) / f"{run_name}_1") if tb else None, ["tensorboard"] if tb else [])
        agent = TabularQ(env_ref, seed=seed, gamma=gamma)
        agent.learn(total_timesteps, callback, logger)
        model_path = out_dir / f"{algo}_{run_name}.json"
        agent.save(model_path)
        print(f"Tabular states visited: {len(agent.q)}")
        print(f"Model saved to: {model_path}")
        return model_path

    # --- PPO (standard, no action masking), MaskablePPO with the same hyperparameters,
    # masked A2C (MaskablePPO with A2C settings) or masked DQN ---
    common = dict(verbose=1, tensorboard_log=tb, seed=seed, gamma=gamma)  # gamma 0.99 = SB3 default (her runs)
    learn_kwargs = {}
    if algo == "maskable_ppo":
        from sb3_contrib import MaskablePPO  # masks come from DCRGymEnv.action_masks()
        model = MaskablePPO("MlpPolicy", vec_env, ent_coef=ent_coef, **common)
    elif algo == "maskable_a2c":
        # A2C as a special case of PPO (Huang et al. 2022): one epoch on one full batch of a 5-step
        # rollout (the ratio is 1, so clipping never acts), GAE off, no advantage normalisation, A2C's
        # RMSprop; SB3 A2C defaults for lr, vf_coef, max_grad_norm and the log interval
        import torch as th
        from sb3_contrib import MaskablePPO
        model = MaskablePPO("MlpPolicy", vec_env, ent_coef=ent_coef, n_steps=5, batch_size=5, n_epochs=1,
                            learning_rate=7e-4, gae_lambda=1.0, normalize_advantage=False, vf_coef=0.5,
                            max_grad_norm=0.5,
                            policy_kwargs=dict(optimizer_class=th.optim.RMSprop,
                                               optimizer_kwargs=dict(alpha=0.99, eps=1e-5, weight_decay=0)),
                            **common)
        learn_kwargs["log_interval"] = 100
    elif algo == "maskable_dqn":
        from agents.masked_dqn import MaskedDQN
        model = MaskedDQN("MlpPolicy", vec_env, **DQN_SETTINGS, **common)
        learn_kwargs["log_interval"] = 100
    else:
        model = PPO("MlpPolicy", vec_env, ent_coef=ent_coef, **common)

    model.learn(
        total_timesteps=total_timesteps,
        tb_log_name=run_name,
        callback=callback,
        **learn_kwargs,
    )

    model_path = out_dir / f"{algo}_{run_name}.zip"
    model.save(str(model_path))
    print(f"Model saved to: {model_path}")
    return model_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-id", type=str, default="B")
    parser.add_argument("--steps", type=int, default=30000)
    parser.add_argument(
        "--node-url",
        type=str,
        default=os.environ.get("NODE_URL", "http://localhost:5001"),
    )
    parser.add_argument("--out-dir", type=Path, default=Path("./models"))
    parser.add_argument(
        "--tb-log-dir",
        type=Path,
        default=Path(os.environ.get("TB_LOG_DIR", "./dcr_tensorboard")),
    )
    parser.add_argument("--ent-coef", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--algo", choices=["ppo", "maskable_ppo", "maskable_a2c", "maskable_dqn", "tabular_q"],
                        default="ppo",
                        help="ppo = Sofía's unmasked PPO (default); maskable_ppo = sb3-contrib MaskablePPO; "
                             "E1.5: maskable_a2c, maskable_dqn, tabular_q (all masked)")
    parser.add_argument("--truncation-fix", action="store_true",
                        help="issue A2: report the step limit as truncated, not terminated")
    parser.add_argument("--gamma", type=float, default=0.99,
                        help="discount factor (0.99 = SB3 default, her runs)")
    parser.add_argument("--fix-dead-end", action="store_true",
                        help="issue D1: a dead end ends the episode as terminated, not accepting")
    parser.add_argument("--partway-starts", choices=["none", "random", "trace"], default="none",
                        help="E1.4b: replay a prefix before the episode (none = her behaviour)")
    parser.add_argument("--partway-initial-share", type=float, default=0.2,
                        help="share of episodes that always start at the initial marking")
    parser.add_argument("--partway-prob", type=float, default=1.0,
                        help="random mode: probability of a prefix for the other episodes")
    parser.add_argument("--partway-max", type=int, default=20,
                        help="random mode: prefix length is uniform on 1..max")
    parser.add_argument("--partway-trace-file", type=Path, default=None,
                        help="trace mode: JSON list of event-label lists (default: l_min_trace from "
                             "analysis/E1_3_state_counts.csv for --partway-graph)")
    parser.add_argument("--partway-graph", type=str, default=None,
                        help="trace mode: graph name in the E1.3 CSV, e.g. Synthetic_chain_40")
    parser.add_argument("--partway-timelimit-factor", type=float, default=None,
                        help="trace mode: partway episode limit = F x (events left in trace) + 10 (default off)")
    parser.add_argument("--partway-step", type=int, default=1,
                        help="trace mode: events the frontier moves per stage")
    parser.add_argument("--violation-fix", action="store_true",
                        help="E1.8: the adapter runs with VIOLATION_FIX=1; log violation columns")
    parser.add_argument("--lagrange", action="store_true",
                        help="E1.8: violation price = Lagrange multiplier, updated after each rollout")
    parser.add_argument("--lagrange-init", type=float, default=1.0, help="lambda_0 (Ray et al. 2019)")
    parser.add_argument("--lagrange-lr", type=float, default=3.0,
                        help="multiplier learning rate on violations per step (see agents/lagrange.py)")

    args = parser.parse_args()
    if args.lagrange and not args.violation_fix:
        parser.error("--lagrange needs --violation-fix")
    traces = None
    if args.partway_starts == "trace":
        traces = load_traces(args.partway_trace_file, args.partway_graph)
    train(
        node_url=args.node_url,
        exp_id=args.exp_id,
        total_timesteps=args.steps,
        out_dir=args.out_dir,
        ent_coef=args.ent_coef,
        tb_log_dir=args.tb_log_dir,
        seed=args.seed,
        algo=args.algo,
        truncation_fix=args.truncation_fix,
        gamma=args.gamma,
        fix_dead_end=args.fix_dead_end,
        partway_starts=args.partway_starts,
        partway_initial_share=args.partway_initial_share,
        partway_prob=args.partway_prob,
        partway_max=args.partway_max,
        partway_traces=traces,
        partway_timelimit_factor=args.partway_timelimit_factor,
        partway_step=args.partway_step,
        violation_fix=args.violation_fix,
        lagrange=(args.lagrange_init, args.lagrange_lr) if args.lagrange else None,
    )