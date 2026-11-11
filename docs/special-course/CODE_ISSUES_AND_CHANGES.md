# Sofía's code: issues found and changes planned for the special course

Commit dbd7c49, branch feature/cost-roles. Line numbers as of 21 Oct.

## A. Issues found

### Major: change results
| # | Issue | Location | Effect | Status |
| --- | --- | --- | --- | --- |
| A1 | Final step's cost and duration never enter the reward (flat +100) | dcr-engine/src/reward.ts, computeStepReward, lines 54-58 | Expense Report: ...Approve>PayOut (cost 50) and ...Approve>Withdraw (cost 735) get identical reward under every weight pair. Explains Table 6.3 (median cost 735 for the cost-focused agent) | Confirmed by replay, 21 Oct. **Fixed behind a switch** (`FIX_FINAL_STEP=1`, `run_local.py --fix-final-step`, run-id tag `fixA1`), 26 Oct; test f. With the switch the accepting step gets its normal step reward (base +1, step penalty, progress, repetition, cost and duration with role multipliers) plus 100; `baseMapped` stays 100. Also covered by `SHAPING_FIX=1` (3 Nov, test g/h), which prices every step by the objective only |
| A2 | Hitting the step limit is reported as a normal end (terminated), never as truncated | dcr-gymnasium-agent/python/src/envs/dcr_env.py, step(), line 116 | Value estimates learn "nothing more to earn" at the cut-off. Small for her (about 100% acceptance); large on big graphs where most episodes hit the limit (graph 09: all of them) | Read in code; thesis 9.5 admits it. **Fixed behind a switch** (`--truncation-fix`), 24 Oct; test d |
| A3 | After 100 legal steps the episode is marked accepting and gets +100, even though the process is not finished | dcr-engine/src/generation.ts, RLDCREnvironment.step (done = accepting OR currentStep >= maxSteps, maxSteps defaults to 100); node-adapter/src/server.ts line 30 (created without maxSteps) and the /action route (structuralAccepting = result.done) | Replay 21 Oct: Expense Report, FillOut 100 times: step 100 gives +100, accepting=True, with 2 events still pending. Inflates acceptance rates and rewards looping early in training. How often it happened in her runs is unknown: check logs for accepting episodes with pendingAfter > 0 | Confirmed by replay, 21 Oct. **Fixed behind a switch** (`FIX_FALSE_ACCEPT=1`), 24 Oct; test b. Note: the fix also requires that the action executed, since a blocked action leaves an initially accepting marking (e.g. LoanApp_junior_senior) unchanged |

### Medium: traps for your experiments
| # | Issue | Location | Effect |
| --- | --- | --- | --- |
| B1 | Expert budget not in the action mask (deliberate) | node-adapter/src/server.ts, BUDGET GATE lines 185-230; buildActionMask line 87 | A masked agent can still pick Expert with no budget left and be blocked at -10. **Fixed behind a switch** (`BUDGET_IN_MASK=1`), 24 Oct; test c (without it, a masked agent on LoanApp k=1 was blocked in 31% of training steps) |
| B2 | Python fallback mask marks pending events, not enabled ones, and has one entry per event instead of per (event, role) | dcr_env.py, state_to_mask line 25; used at lines 62, 82, 95 | Never runs with her adapter; wrong if it ever does. Left as is; the new `action_masks()` never uses it and raises an error if the adapter mask is missing or has the wrong length |
| B3 | Defaults differ from the thesis settings | server.ts lines 33-34 (100 steps, -0.1); train_agent.py lines 284, 296 (30,000 steps, entropy 0.01) | Running by hand without her runner gives different results. Thesis: 300 steps, -1.5, 100,000 steps, entropy 0.1 |
| B4 | Experiment runner is Linux/HPC only | scripts/run_experiments.py, lines 169-171 (lsof), 265 (bash + nvm) | Fails on Windows; fine on DTU HPC |
| B5 | One adapter = one global process state, one environment | server.ts line 30 (single env); train_agent.py line 252 (DummyVecEnv with one env) | No parallel environments; each step is an HTTP round trip (about 4.9 ms). Parallel training needs one adapter per environment on its own port |
| B6 | Lockfile out of date; adapter relies on a package it does not list | yarn.lock; node-adapter/package.json (fast-xml-parser missing) | Install changes yarn.lock; the root yarn install must run before the adapter works |
| B7 | Windows port exhaustion: node_connector.py opened a new TCP connection per step; fixed with a requests.Session on special-course (no behaviour change) | dcr-gymnasium-agent/python/src/utils/node_connector.py | Before the fix, 100,000-step runs on Windows crashed after a few thousand steps with WinError 10048 (16,384 ephemeral ports, closed sockets held in TIME_WAIT) |

### Understanding: thesis vs code, and thesis inconsistencies
| # | Point | Where |
| --- | --- | --- |
| C1 | Thesis says the budget is "enforced by the same shield"; in code it is a separate gate in the adapter (same effect) | Thesis 4.6 vs server.ts lines 185-230 |
| C2 | Two step counters: the adapter counts blocked attempts toward the limit, the engine counts only legal steps | server.ts line 183 vs generation.ts, RLDCREnvironment.step. Matters when comparing masked and unmasked runs at the same step limit |
| C3 | Table 4.2: rows say cost and duration penalties apply to "each legal action"; caption says non-terminal only (the code does non-terminal) | Thesis 4.3 |
| C4 | Expense Report text says the duration agent finds both Pareto points and the cost agent finds none; Table 6.3 says the opposite | Thesis 6.1.4, 9.1.3 vs Table 6.3 |
| C5 | Seed-study numbers disagree (1.13 +/- 0.34 vs 1.67 +/- 0.89; "below 8%" vs 9.15% and 11.58%) | Thesis 7.3, Table 7.7, Fig. 7.6 |
| C6 | "Collapse beyond ~42 event-role pairs": 42 is the largest size that worked; the break is between 42 and 60 | Abstract, 9.5, 10.4 vs 8.3.1-8.3.2 |
| C7 | AGENTS.md (written by Sofía, commits of 22-23 Jun) describes an older setup (Synthetic Review, 3 roles, 42 actions); the final runner uses Loan Application with 2 roles | AGENTS.md vs run_experiments.py line 75 |
| C8 | Loan Application log is simulated (AgentSimulator), not a real company log | Thesis 5.2.2 |

### Model issues and open decisions
| # | Issue | Location | Effect / decision |
| --- | --- | --- | --- |
| M1 | A Loan Application case is accepted right after Assess loan risk, without a decision: the decision events (Approve, Reject, Cancel, Design offer) only have conditions from Assess loan risk, no response | app/public/examples/diagrams/LoanApp_junior_senior.xml | Learned traces stop after the risk assessment (99.9% of accepting episodes, 22 Oct runs). Question 5 for Hugo. Possible fix: response Assess loan risk -> Approve application, with Reject and Cancel excluding Approve application |
| D1 | Dead ends in masked runs (non-accepting marking, no allowed action) end as truncated, so SB3 bootstraps a value for a state that has no future; in unmasked runs the agent is blocked until the step limit | dcr_env.py (dead-end handling, commit b37e17f) | **Fixed behind a switch** (`--fix-dead-end`, tag `D1`), 5 Nov; test e. Masked or not: terminated, not accepting, no +100. Off keeps both old behaviours. No dead ends seen so far (smoke tests, E1.3 random walks) |

## Switches added (24 Oct, all off by default; tests in TESTS.md)

Baseline for all later Study 1 experiments (decision 5 Nov): `--shaping-fix --gamma 1.0 --fix-false-accept --truncation-fix --fix-dead-end`, weights (w, 1 - w). Sofía's behaviour stays the default with no flags.

| Switch | Fixes | Files |
| --- | --- | --- |
| `FIX_FALSE_ACCEPT=1` (adapter env; `run_local.py --fix-false-accept`) | A3 | node-adapter/src/server.ts (/action), dist/server.mjs rebuilt |
| `--truncation-fix` (`train_agent.py`, `run_local.py`) | A2 | dcr_env.py step(), train_agent.py |
| `BUDGET_IN_MASK=1` (adapter env; `run_local.py --budget-in-mask`) | B1 | node-adapter/src/server.ts buildActionMask |
| `--algo maskable_ppo` (`train_agent.py`, `run_local.py`) | masking (replaces B2 for masked runs) | dcr_env.py action_masks(), train_agent.py; sb3-contrib 2.7.1 in requirements.txt |
| `FIX_FINAL_STEP=1` (adapter env; `run_local.py --fix-final-step`), 26 Oct | A1 | dcr-engine/src/reward.ts, node-adapter/src/server.ts |
| `SHAPING_FIX=1` (adapter env; `run_local.py --shaping-fix`), 3 Nov; with `SHAPING_GAMMA` (`--gamma`) and `SHAPING_NO_PROGRESS=1` (`--no-progress`) | A1 and the Expense Report reward-scale problem (cheap path loses to the short one, `analysis/A1_A3_switch_results.md`) | reward.ts computeStepReward (shaping branch, shared objectivePenalties), server.ts /action (dead-end flag, no step penalty) |
| `--fix-dead-end` (`train_agent.py`, `run_local.py`), 5 Nov | D1 | dcr_env.py step() |
| `--partway-starts` and its options (`train_agent.py`, `run_local.py`), 5 Nov | E1.4b | generation.ts replayPrefix; server.ts new `/replay` route (`/reset` and `/action` unchanged); node_connector.py; dcr_env.py reset(), step(); new src/envs/partway.py; train_agent.py callback |
| `--gamma` (`train_agent.py`, `run_local.py`), 3 Nov | (discount factor, default 0.99 = hers) | train_agent.py train(); run_local.py tag `g<gamma>` |
| `run_local.py --shield-off` (adapter SHIELD_DISABLED=1, her RL-only mode, existed in generation.ts), 9 Nov | E1.8 | run_local.py only (sets the existing variable; tag `shieldOff`) |
| `VIOLATION_FIX=1` (adapter env; `run_local.py --violation-fix`, `--violation-price X`), 9 Nov; needs SHIELD_DISABLED and SHAPING_FIX | Her RL-only path gives an executed forbidden event a flat -10: no cost, duration or progress term, no +100 even when it ends accepting; it does not use the Expert budget; it is logged as `dcr_illegal` like a blocked move. With the switch: priced like a legal step (cost, duration, progress, +100 if accepting) minus the violation price (default 10), uses the Expert budget, logged as `violation`; new route POST /violation_price | server.ts /action, /violation_price; train_agent.py columns `violation`, `violation_price`; tags `violFix`, `vp<X>`; test l |
| `--lagrange` (`run_local.py`, `train_agent.py`; `--lagrange-init`, `--lagrange-lr`), 9 Nov; needs VIOLATION_FIX | E1.8 Lagrangian variant: the violation price is a multiplier, lambda <- max(0, lambda + lr x violations per step) after each rollout (defaults 1 and 3, see src/agents/lagrange.py); log `lagrange_exp_<run>.csv` | new src/agents/lagrange.py; train_agent.py callback (_on_rollout_end); tag `lagI<init>L<lr>`; test m |
| `--algo maskable_a2c`, `maskable_dqn`, `tabular_q` (`train_agent.py`, `run_local.py`), 11 Nov | E1.5 other algorithms, all masked. A2C = MaskablePPO with the A2C settings of Huang et al. 2022; DQN = SB3 DQN with the mask in action choice and in the target max (src/agents/masked_dqn.py); tabular Q-learning (src/agents/tabular_q.py), same trace CSV through the same callback | train_agent.py train() (her PPO path unchanged, test a); tags `maskedA2C`, `maskedDQN`, `tabQ`; tests n, o |
| `node-adapter/scripts/exact_optimum.ts`, 11 Nov | E1.5 exact optimum per w (Dijkstra over reachable markings, the adapter's normalised step cost, roles, optional Expert budget and default duration); no learning | test o (agrees with the MiniZinc results in python/minizinc_ground_truth and with Diaz et al. for the Expense Report) |
| `DEFAULT_DURATION=X` (adapter env; `run_local.py --default-duration X`), 6 Nov; not part of the baseline | Events with no duration (the synthetic chains; `analysis/Graph_costs_durations.md`) get duration X; a stated duration, including 0, stays | server.ts applyDefaultDuration (start-up and /load); run_local.py tag `defDur<X>`; test k |

With all switches off, a 2,048-step run is identical row for row to commit fe01d1c (test a).
A run with any switch on gets a tag in its id (`masked`, `fixA2A3B1`, `shapeFix`, `g1p0`, ...).

`SHAPING_FIX` reward of a legal step: -alpha x cost/maxCost - eps x duration/maxDuration (role
multipliers as before) + gamma x Phi(after) - Phi(before), Phi = -2 x included pending events, +100
on the accepting step; no +1 base, step penalty or repetition penalty; blocked actions -10. Phi is 0
for the marking an episode ends in (accepting or dead end), so the progress terms are potential-based
and, with gamma = 1, a finished case's total is 100 - its weighted normalised cost and duration
(Expense Report: P1 100 - 50/700, P2 100 - 710/700, D 100 - 735/700; test g). The cheap path then
wins at any alpha > 0 with gamma = 1. Remaining gamma < 1 effect: the +100 is still discounted by
path length (about 1 unit per extra step at 0.99), which is why `--gamma` exists. Without A3 (her
engine's 100-legal-step accept) a fake accept still gets +100 here; combine with `--fix-false-accept`.

## B. Functions to change during the special course

| Deliverable / experiment | File | Function | Change |
| --- | --- | --- | --- |
| SC-D2, E1.1 reproduce | (none) | | Run unchanged with her settings (B3) to get her baseline first |
| SC-D2, all runs | generation.ts RLDCREnvironment.step; server.ts /action | | Use the real acceptance check (isAcceptingS) for 'accepting', not the engine's done flag (fixes A3) |
| SC-D2, all runs | dcr_env.py | step() | Return terminated = accepting, truncated = maxStepReached (fixes A2). Keep a switch to reproduce her version |
| SC-D2, E1.4 masking | dcr_env.py | new action_masks() | Return the adapter's mask as booleans; raise an error if it is missing (replaces B2) |
| SC-D2, E1.4 masking with k | server.ts | buildActionMask / BUDGET GATE | Set Expert pairs to 0 when the budget is used up (fixes B1) |
| SC-D2, E1.4-E1.5 algorithms | train_agent.py | train(), argument parser | Add --algo (PPO, MaskablePPO, DQN, A2C), thesis defaults (100,000 steps, entropy 0.1); add sb3-contrib to requirements.txt |
| SC-D2, reward fix (A1, done 26 Oct) | reward.ts; server.ts | computeStepReward (fixFinalStep); /action step penalty | `FIX_FINAL_STEP=1`: the accepting step is priced like any other step, plus 100; her flat +100 stays the default |
| SC-D2, reward that follows the objective (done 3 Nov, no runs yet) | reward.ts; server.ts; train_agent.py; run_local.py | computeStepReward (shapingFix); /action; --gamma; --weights-sweep | `SHAPING_FIX=1`: objective penalties + potential-based progress, +100 on acceptance; `--no-progress` baseline; `--weights-sweep` w = 0..1 by 0.1 |
| SC-D2, E1.2 scaling | minizinc_ground_truth/random_dcr_generator.py; external_validation/run_scaling_rl_local.py | generate_graph | Reuse to make the 20-100 event graphs |
| SC-D2, E1.3 state counts | minizinc_ground_truth/compute_lmin.py (BFS) | | Extend the BFS to count reachable markings and average enabled events per step |
| SC-D2, E1.6 state abstraction | dcr_env.py | _state_to_obs | Drop bits that never change; optionally merge markings with the same enabled and pending sets |
| SC-D2, E1.7 factored actions | server.ts getEventRolePairs; dcr_env.py action_space | | Event first, then role (MultiDiscrete) |
| SC-D2, runs on Windows/HPC | run_experiments.py | free_adapter_port, adapter launch | Cross-platform launcher (no lsof, no bash/nvm), one adapter per run/port (B4, B5) |
| E1.4b partway starts (done 5 Nov; also for the thesis) | generation.ts; server.ts; dcr_env.py; partway.py | RLDCREnvironment.replayPrefix; new route /replay; reset(), step() | Replays a prefix (ids or labels), reports the first illegal event; a separate route, so /reset stays as it was. Decisions not in the design notes: --partway-prob default 1.0, --partway-max default 20; k = 0, 1, 2, ...; the window is emptied when j moves and only moves when full; a prefix of length 0 logs as `initial`; a random prefix stops before an accepting marking or dead end. With a prefix, the first step's progress term includes -Phi(start marking), so a case's terms add up to -Phi(start) instead of 0; a constant per start, the policy is unaffected |
| Thesis (was SC-D3), choose the trade-off | server.ts | COST_WEIGHT / DURATION_WEIGHT (lines 37-38) | Today fixed at start-up, so one policy per weight pair. Either pick the matching trained policy, or send weights per reset and add them to the observation |
| Thesis (was SC-D3), recommender | new file, e.g. src/recommend.py | | Given a prefix and weights: replay, list enabled next activities, rank them with the policy and rollouts, return expected remaining cost and duration |
| Thesis (was SC-D3), exact check | minizinc_ground_truth/dcrjs_converter.py | | Write the marking after the prefix as the initial marking, then run the solver |
| SC-D4 figures | scripts/pareto_analysis.py, internal_validation/, external_validation/ | | Reuse her figure scripts for comparison plots |
