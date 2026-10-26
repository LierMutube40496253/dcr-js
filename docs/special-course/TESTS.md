# Tests for the special-course switches

The switches (all OFF by default, so off = Sofía's behaviour):

| Switch | Where | Issue | Effect when on |
| --- | --- | --- | --- |
| `FIX_FALSE_ACCEPT=1` | adapter env var; `run_local.py --fix-false-accept` | A3 | Accepting only if the executed step reaches a real DCR accepting marking (`isAcceptingS`); the engine's 100-legal-step limit never gives accepting or +100. Episodes end on real acceptance or `MAX_EPISODE_STEPS`. |
| `--truncation-fix` | `train_agent.py`, `run_local.py` | A2 | `dcr_env.step` returns terminated = acceptance, truncated = step limit without acceptance, so SB3 bootstraps the cut-off state (`TimeLimit.truncated`). |
| `FIX_FINAL_STEP=1` | adapter env var; `run_local.py --fix-final-step` | A1 | The accepting step gets its normal step reward (+1, step penalty, progress, repetition, cost and duration with role multipliers) plus 100, instead of a flat +100. `baseMapped` stays 100. |
| `BUDGET_IN_MASK=1` | adapter env var; `run_local.py --budget-in-mask` | B1 | Expert pairs are 0 in the action mask once the Expert budget is used up. The budget gate itself is unchanged. |
| `--algo maskable_ppo` | `train_agent.py`, `run_local.py` | (masking) | sb3-contrib MaskablePPO with the same hyperparameters as PPO; masks from `DCRGymEnv.action_masks()` (the adapter's `actionMask`, never the `state_to_mask` fallback, issue B2). Also turns on dead-end handling (below). |
| dead-end handling (`DCRGymEnv(dead_end_truncate=True)`, on with `--algo maskable_ppo` only) | `dcr_env.py`, `train_agent.py` | (masking) | A non-accepting marking with no allowed action (all-zero mask) ends the episode as truncated, not accepting, no +100. Logged in a `dead_end` trace column (masked runs only) and counted per run ("Dead-end episodes: N of M" at the end of the run log). |

Any switch on adds a tag to the run id: `masked` for MaskablePPO, `fix` + the issue codes for the
fixes, e.g. `E1_4_Synth60_masked_fixA2A3B1_s1_a0p0_b0p0_kunlim` (A1 comes first: `fixA1A3`). No tag = her baseline.

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
