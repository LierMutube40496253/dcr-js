# Tests for the special-course switches

The switches (all OFF by default, so off = Sofía's behaviour):

| Switch | Where | Issue | Effect when on |
| --- | --- | --- | --- |
| `FIX_FALSE_ACCEPT=1` | adapter env var; `run_local.py --fix-false-accept` | A3 | Accepting only if the executed step reaches a real DCR accepting marking (`isAcceptingS`); the engine's 100-legal-step limit never gives accepting or +100. Episodes end on real acceptance or `MAX_EPISODE_STEPS`. |
| `--truncation-fix` | `train_agent.py`, `run_local.py` | A2 | `dcr_env.step` returns terminated = acceptance, truncated = step limit without acceptance, so SB3 bootstraps the cut-off state (`TimeLimit.truncated`). |
| `FIX_FINAL_STEP=1` | adapter env var; `run_local.py --fix-final-step` | A1 | The accepting step gets its normal step reward (+1, step penalty, progress, repetition, cost and duration with role multipliers) plus 100, instead of a flat +100. `baseMapped` stays 100. |
| `SHAPING_FIX=1` | adapter env var; `run_local.py --shaping-fix` | A1 + reward scale | A legal step's reward follows the objective: minus cost and duration penalties (role multipliers as before) plus a progress term gamma * Phi(after) - Phi(before), Phi(marking) = -2 x included pending events; no +1 base, no step penalty, no repetition penalty. The accepting step gets the same plus 100 (covers A1; takes precedence over `FIX_FINAL_STEP`). Phi = 0 for the marking an episode ends in (accepting, or a dead end: no enabled event), so with gamma = 1 the progress terms of a finished case add up to -Phi(initial), 0 when no event starts pending. Blocked actions (DCR-illegal, budget-blocked) stay a flat -10. Episodes cut by the step limit keep Phi of their last marking (a truncation). |
| `SHAPING_GAMMA=g` | adapter env var; set by `run_local.py --gamma` (always, default 0.99) | (with `SHAPING_FIX`) | gamma of the progress term; should equal the agent's gamma. Read only with `SHAPING_FIX=1`. |
| `SHAPING_NO_PROGRESS=1` | adapter env var; `run_local.py --no-progress` (needs `--shaping-fix`) | baseline | `SHAPING_FIX` reward without the progress term. |
| `--gamma g` | `train_agent.py`, `run_local.py` | | PPO discount factor; default 0.99 = SB3 default = her runs. `run_local.py` passes it to the trainer only when not 0.99. |
| `BUDGET_IN_MASK=1` | adapter env var; `run_local.py --budget-in-mask` | B1 | Expert pairs are 0 in the action mask once the Expert budget is used up. The budget gate itself is unchanged. |
| `--fix-dead-end` | `train_agent.py`, `run_local.py` (tag: `D1` in the fix codes, e.g. `fixA2A3D1`) | D1 | Masked or not, a non-accepting marking with no enabled event ends the episode as terminated (no bootstrapping), not accepting, no +100. Takes precedence over the masked `dead_end_truncate` below. Logged in the `dead_end` trace column. |
| `--partway-starts {none,random,trace}` | `train_agent.py`, `run_local.py` (tags `psRandom`, `psTrace`) | E1.4b | Replays a prefix before the episode through the adapter's new `/replay` route: no reward, no cost, no episode step, not stored for training. `--partway-initial-share` (0.2) of episodes always start at the initial marking; results use `start_type = initial` only. Random: with `--partway-prob` (1.0), a random legal prefix of length 1..`--partway-max` (20), cut short before an accepting marking or a dead end. Trace: one frontier j per trace from `--partway-trace-file` (default `l_min_trace` in `analysis/E1_3_state_counts.csv`), see `src/envs/partway.py`; optional `--partway-timelimit-factor`; `--partway-step N` (default 1, tag `psTraceStep<N>`) moves j by N events per stage. The trace CSV gets `start_type`, `prefix_len`, `partway_trace`, `partway_j`; j over time is in `scripts/logs/partway_frontier_exp_<run>.csv`. With `none`, `/replay` is never called and the CSV format is unchanged. |
| `--algo maskable_ppo` | `train_agent.py`, `run_local.py` | (masking) | sb3-contrib MaskablePPO with the same hyperparameters as PPO; masks from `DCRGymEnv.action_masks()` (the adapter's `actionMask`, never the `state_to_mask` fallback, issue B2). Also turns on dead-end handling (below). |
| dead-end handling (`DCRGymEnv(dead_end_truncate=True)`, on with `--algo maskable_ppo` only) | `dcr_env.py`, `train_agent.py` | (masking) | A non-accepting marking with no allowed action (all-zero mask) ends the episode as truncated, not accepting, no +100. Logged in a `dead_end` trace column (masked runs only) and counted per run ("Dead-end episodes: N of M" at the end of the run log). |

Any switch on adds a tag to the run id: `masked` for MaskablePPO, `fix` + the issue codes for the
fixes, e.g. `E1_4_Synth60_masked_fixA2A3B1_s1_a0p0_b0p0_kunlim` (A1 comes first: `fixA1A3`). No tag = her baseline.
Then `shapeFix` (or `shapeFixNoProg` with `--no-progress`) and `g<gamma>` when gamma is not 0.99,
e.g. `ExpenseReport_Diaz_fixA2_shapeFix_g1p0_s1_a1p0_b0p0_kunlim`.

`run_local.py --weights-sweep` replaces `--pairs` with cost weight w = 0, 0.1, ..., 1 and duration
weight 1 - w (11 pairs, run ids `a0p0_b1p0` ... `a0p3_b0p7` ... `a1p0_b0p0`). Not a switch: no tag.

## Running the tests

From `dcr-gymnasium-agent/python`, with either venv (`.venv\Scripts\python` on Windows,
`.venv-linux/bin/python` in WSL; in WSL load nvm first so `node` is on PATH). Each test starts its
own adapter on its own port (5011-5014), prints `PASS` or `FAIL` and exits 0 or 1, and deletes its
outputs.

```bash
# a. Switches off = her code: 2,048 steps, Expense Report, pair (1,0), seed 1, run with the
#    reference commit (git archive of fe01d1c = her code + transport fix) and with the working
#    tree; the train_trace CSVs must be identical row for row. About 1 minute.
python scripts/special_course_tests/test_a_baseline_identical.py --ref fe01d1c

# b. A3: FillOut 105 times on the Expense Report. Off: step 100 accepting with +100 (2 events
#    still pending). FIX_FALSE_ACCEPT=1: no accepting, no +100, no done.
python scripts/special_course_tests/test_b_false_accept.py

# c. Masking: MaskablePPO trained 10,000 steps, then 1,000 sampled episodes under the mask:
#    zero dcr_illegal and zero budget_block steps (training and evaluation). A few minutes each.
python scripts/special_course_tests/test_c_masking.py --graph Expense_Report_Diaz.xml
python scripts/special_course_tests/test_c_masking.py --graph LoanApp_junior_senior.xml --k 1 --budget-in-mask
#    Contrast (optional): without BUDGET_IN_MASK the masked agent IS budget-blocked (B1)
python scripts/special_course_tests/test_c_masking.py --graph LoanApp_junior_senior.xml --k 1 --expect-blocks

# d. A2: an episode that hits MAX_EPISODE_STEPS=300 (FillOut loop, FIX_FALSE_ACCEPT=1) is
#    truncated=True, terminated=False with --truncation-fix (terminated=True without), and
#    DummyVecEnv sets TimeLimit.truncated, which PPO and MaskablePPO use to bootstrap.
python scripts/special_course_tests/test_d_truncation.py

# e. Dead ends (masked runs): a 3-event graph that dead-ends after one step (A makes B pending;
#    B and C are each other's condition). With dead_end_truncate: truncated=True, not accepting,
#    dead_end logged; a short MaskablePPO run does not crash and counts every episode as a dead
#    end. Without (her env): the episode goes on and action_masks() raises.
python scripts/special_course_tests/test_e_dead_end.py

# f. A1: LoanApp_junior_senior, COST_WEIGHT=1, DURATION_WEIGHT=0.5. A seeded masked random walk
#    to real acceptance, replayed with the final event under each role. Off: flat +100 for every
#    role. FIX_FINAL_STEP=1: 1 + progress + repetition - cost - duration + STEP_PENALTY + 100, cost
#    and duration with the role multipliers; earlier steps unchanged. Port 5016, a few seconds.
python scripts/special_course_tests/test_f_final_step.py

# g. SHAPING_FIX: Expense Report, COST_WEIGHT=1, DURATION_WEIGHT=0, SHAPING_GAMMA=1. Replays the
#    three paths; totals must be 100 - 50/700 (Approve > PayOut), 100 - 710/700 (Withdraw) and
#    100 - 735/700 (Approve > Withdraw), progress terms adding up to 0; with SHAPING_NO_PROGRESS=1
#    the same totals and every progress term 0. Dead end (test e graph, gamma 1 and 0.99): the step
#    into it has progress 0 (Phi = 0) and reward -1. Ports 5017-5018, a few seconds.
python scripts/special_course_tests/test_g_shaping_paths.py

# h. SHAPING_FIX: LoanApp_junior_senior, COST_WEIGHT=1, DURATION_WEIGHT=0, gamma 0.99. A seeded
#    masked random walk to real acceptance ending in Assess loan risk, replayed with each role:
#    the Junior final step must earn more than the Expert one, each equal to
#    -Phi(before) - cost x multiplier / maxCost + 100. Port 5019, a few seconds.
python scripts/special_course_tests/test_h_shaping_roles.py

# i. E1.4b: for every prefix of an accepting trace (Expense Report, Synthetic_chain_40), /replay
#    gives the same marking and mask as playing the events, and no reward; the next step counts as
#    step 1 with only its own cost. Illegal prefix: HTTP 400 naming the event; after the first
#    action: HTTP 409. Environment: initial share, start_type, prefix_len, random prefixes, time
#    limit. Port 5020, under a minute.
python scripts/special_course_tests/test_i_prefix_replay.py

# j. E1.4b: frontier with a fake success stream (back at >= 0.8 over 20 episodes at j, forward at
#    <= 0.15 when below the start value, limits, other prefix lengths ignored, geometric offset,
#    --partway-step), then through the environment on Synthetic_chain_40. Port 5020, a few seconds.
python scripts/special_course_tests/test_j_frontier.py

# k. --default-duration (adapter DEFAULT_DURATION): off, Synthetic_chain_20 has no durations. X = 1 and
#    X = 7: every event gets X, each step's reward is 0.5 lower at weights (0.5, 0.5), whatever X is, and
#    the case duration is 5 x X. One duration removed from the Expense Report: only that event gets X.
#    Loan_Application_Diaz: a stated 0 stays. /load applies it too; X = 0 stops the adapter. Port 5021,
#    under a minute. "Nothing changes when off" is test a.
python scripts/special_course_tests/test_k_default_duration.py

# l. E1.8 VIOLATION_FIX (shield off, new reward, Expense, weights (0.5, 0.5)): off, a violation that ends
#    accepting gets a flat -10 (her RL-only path); on, it gets the legal-step reward (cost, duration,
#    progress, +100) minus 10 and is labelled "violation"; legal steps unchanged; /violation_price changes
#    the price (400 for a negative one, 409 without the switch); an Expert violation uses the Expert budget
#    only with the switch on (LoanApp, k = 2); VIOLATION_FIX without SHIELD_DISABLED stops the adapter.
#    Port 5023, under a minute.
python scripts/special_course_tests/test_l_violations.py

# m. E1.8 Lagrangian: the update rule max(0, lambda + lr x violations per step), then a 4,096-step PPO run
#    on the Expense Report with --shield-off --violation-fix --lagrange: the multiplier log follows the
#    rule and each rollout's steps used the multiplier set after the previous rollout. Port 5024, about a
#    minute; its files are deleted afterwards.
python scripts/special_course_tests/test_m_lagrange.py
```

## Results, 24 Oct 2026 (Windows .venv; c also run in WSL .venv-linux with identical numbers)

| Test | Result |
| --- | --- |
| a | PASS: 2,048 rows each, identical row for row |
| b | PASS: off: accepting and +100 at steps 100-105, pendingAfter = 2; on: none |
| c Expense Report | PASS: training 10,240 legal steps; evaluation 1,000 episodes, 2,192 steps, all legal, 1,000 accepting |
| c LoanApp, k=1, BUDGET_IN_MASK=1 | PASS: training 10,240 legal; evaluation 1,000 episodes, 8,228 steps, all legal, 1,000 accepting |
| c contrast, k=1, BUDGET_IN_MASK off | budget_block in 3,145 of 10,240 training steps and 1,323 of 11,294 evaluation steps |
| d | PASS: on: truncated at step 300, TimeLimit.truncated=True; off: terminated at step 300; both algorithms bootstrap on TimeLimit.truncated |

Rerun on 24 Oct after adding dead-end handling: a, b, c (both graphs), d and e all PASS with the
same numbers as above. e: on: truncated at step 1, dead_end=True, reward -0.5 (not +100); off:
episode continues, action_masks() raises; MaskablePPO: 2,048 steps, 2,048 episodes, all dead
ends, 0 accepting.

Also checked: rebuilding `node-adapter/dist/server.mjs` from the unchanged source gave a
byte-identical bundle, so the committed bundle differs from hers only by the switch code.

## Results, 26 Oct 2026 (FIX_FINAL_STEP added; Windows .venv)

| Test | Result |
| --- | --- |
| a | PASS: 2,048 rows each, identical row for row to fe01d1c (switch off = her code) |
| b | PASS: same numbers as 24 Oct |
| f | PASS: walk of 21 steps ending in Assess loan risk (base cost 318, duration 175). Off: +100 for both roles. On: Expert 97.5793, Junior 97.8437 (= 1 - 1 repetition - cost - duration - 1.5 + 100; Expert cost -0.8292, duration -0.0915; Junior cost -0.2902, duration -0.3661) |

## Results, 3 Nov 2026 (SHAPING_FIX, --gamma, --no-progress added; Windows .venv)

| Test | Result |
| --- | --- |
| a | PASS: 2,048 rows each, identical row for row to fe01d1c (all switches off) |
| b, c (3 variants), d, e, f | PASS, same numbers as 24 and 26 Oct (c and e need `PYTHONIOENCODING=utf-8` when output is redirected to a file on Windows: the trainer prints a "->" arrow character) |
| g | PASS: with progress: P1 99.928571, P2 98.985714, D 98.950000 (= 100 - 50/700, 100 - 710/700, 100 - 735/700); progress terms [-4, 2, 2], [-4, 4], [-4, 2, 2]. Without progress: same totals, all progress terms 0. Dead end, gamma 1 and 0.99: progress 0, reward -1 |
| h | PASS: walk seed 1, 21 steps, final Assess loan risk (base cost 318, maxCost 767), 1 pending before: Junior 101.709778 (cost -0.290222) > Expert 101.170795 (cost -0.829205), each = 0.99 x 0 + 2 - cost + 100 |

## Results, 5 Nov 2026 (--fix-dead-end, --partway-starts added; Windows .venv, PYTHONIOENCODING=utf-8)

| Test | Result |
| --- | --- |
| a | PASS: 2,048 rows each, identical row for row to fe01d1c (all switches off) |
| b, c (3 variants), d, f, g, h | PASS, same numbers as 3 Nov |
| e | PASS, as before; new part D1, with dead_end_truncate on and off: terminated=True, truncated=False, not accepting, reward -0.5 (no shaping in this test) |
| i | PASS: same marking and mask for every prefix on both graphs, no reward in the answer; next step episodeSteps=1, only its own cost; illegal prefix 400, after first action 409; trace mode 55 of 300 resets initial; random prefix lengths 1..6 with max 6; time limit 2 x 1 + 10 = 12 steps, truncated |
| j | PASS: all frontier rules with the fake stream; P(k = 0) = 0.502; environment: 20 finished partway episodes move j from 4 to 3 |

