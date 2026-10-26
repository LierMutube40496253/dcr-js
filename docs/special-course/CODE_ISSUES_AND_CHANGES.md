# Sofía's code: issues found and changes planned for the special course

Commit dbd7c49, branch feature/cost-roles. Line numbers as of 21 Oct.

## A. Issues found

### Major: change results
| # | Issue | Location | Effect | Status |
| --- | --- | --- | --- | --- |
| A1 | Final step's cost and duration never enter the reward (flat +100) | dcr-engine/src/reward.ts, computeStepReward, lines 54-58 | Expense Report: ...Approve>PayOut (cost 50) and ...Approve>Withdraw (cost 735) get identical reward under every weight pair. Explains Table 6.3 (median cost 735 for the cost-focused agent) | Confirmed by replay, 21 Oct |
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
| D1 | Dead ends in masked runs (non-accepting marking, no allowed action) currently end as truncated, so SB3 bootstraps a value for a state that has no future | dcr_env.py (dead-end handling, commit b37e17f) | OPEN, decide before the next runs. Recommendation: terminated and not accepting (value 0). No dead ends seen so far (smoke tests, E1.3 random walks) |

## Switches added (24 Oct, all off by default; tests in TESTS.md)

| Switch | Fixes | Files |
| --- | --- | --- |
| `FIX_FALSE_ACCEPT=1` (adapter env; `run_local.py --fix-false-accept`) | A3 | node-adapter/src/server.ts (/action), dist/server.mjs rebuilt |
| `--truncation-fix` (`train_agent.py`, `run_local.py`) | A2 | dcr_env.py step(), train_agent.py |
| `BUDGET_IN_MASK=1` (adapter env; `run_local.py --budget-in-mask`) | B1 | node-adapter/src/server.ts buildActionMask |
| `--algo maskable_ppo` (`train_agent.py`, `run_local.py`) | masking (replaces B2 for masked runs) | dcr_env.py action_masks(), train_agent.py; sb3-contrib 2.7.1 in requirements.txt |

With all switches off, a 2,048-step run is identical row for row to commit fe01d1c (test a).
A run with any switch on gets a tag in its id (`masked`, `fixA2A3B1`, ...).

## B. Functions to change during the special course

| Deliverable / experiment | File | Function | Change |
| --- | --- | --- | --- |
| SC-D2, E1.1 reproduce | (none) | | Run unchanged with her settings (B3) to get her baseline first |
| SC-D2, all runs | generation.ts RLDCREnvironment.step; server.ts /action | | Use the real acceptance check (isAcceptingS) for 'accepting', not the engine's done flag (fixes A3) |
| SC-D2, all runs | dcr_env.py | step() | Return terminated = accepting, truncated = maxStepReached (fixes A2). Keep a switch to reproduce her version |
| SC-D2, E1.4 masking | dcr_env.py | new action_masks() | Return the adapter's mask as booleans; raise an error if it is missing (replaces B2) |
| SC-D2, E1.4 masking with k | server.ts | buildActionMask / BUDGET GATE | Set Expert pairs to 0 when the budget is used up (fixes B1) |
| SC-D2, E1.4-E1.5 algorithms | train_agent.py | train(), argument parser | Add --algo (PPO, MaskablePPO, DQN, A2C), thesis defaults (100,000 steps, entropy 0.1); add sb3-contrib to requirements.txt |
| SC-D2, reward fix (if Hugo agrees) | reward.ts | computeStepReward | Optional switch: also subtract the final step's cost and duration (fixes A1); her version stays the default baseline |
| SC-D2, E1.2 scaling | minizinc_ground_truth/random_dcr_generator.py; external_validation/run_scaling_rl_local.py | generate_graph | Reuse to make the 20-100 event graphs |
| SC-D2, E1.3 state counts | minizinc_ground_truth/compute_lmin.py (BFS) | | Extend the BFS to count reachable markings and average enabled events per step |
| SC-D2, E1.6 state abstraction | dcr_env.py | _state_to_obs | Drop bits that never change; optionally merge markings with the same enabled and pending sets |
| SC-D2, E1.7 factored actions | server.ts getEventRolePairs; dcr_env.py action_space | | Event first, then role (MultiDiscrete) |
| SC-D2, runs on Windows/HPC | run_experiments.py | free_adapter_port, adapter launch | Cross-platform launcher (no lsof, no bash/nvm), one adapter per run/port (B4, B5) |
| E1.4b partway starts (and thesis) | generation.ts | RLDCREnvironment.reset | Accept a prefix of events and replay it; report the first illegal event if any (6.4% of her log) |
| E1.4b partway starts (and thesis) | server.ts | /reset route | Read an optional prefix (and weights) from the request body and pass it on |
| E1.4b partway starts (and thesis) | dcr_env.py | reset(options) | Pass options={"prefix": [...]} to the adapter; sample prefixes from the log during training |
| Thesis (was SC-D3), choose the trade-off | server.ts | COST_WEIGHT / DURATION_WEIGHT (lines 37-38) | Today fixed at start-up, so one policy per weight pair. Either pick the matching trained policy, or send weights per reset and add them to the observation |
| Thesis (was SC-D3), recommender | new file, e.g. src/recommend.py | | Given a prefix and weights: replay, list enabled next activities, rank them with the policy and rollouts, return expected remaining cost and duration |
| Thesis (was SC-D3), exact check | minizinc_ground_truth/dcrjs_converter.py | | Write the marking after the prefix as the initial marking, then run the solver |
| SC-D4 figures | scripts/pareto_analysis.py, internal_validation/, external_validation/ | | Reuse her figure scripts for comparison plots |
