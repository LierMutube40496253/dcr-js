# Session log (special course, branch special-course)

Read this at the start of every session and append to it at the end. Newest entries last.
Result files live in `C:\Users\lierm\Desktop\SpecialCourse\analysis\` (outside the repo).

## 21 Oct 2026: setup on Windows, E1.1 Expense Report
- Windows: Python 3.12 venv `dcr-gymnasium-agent/python/.venv`, Node 24, Yarn 1.
- Wrote CLAUDE.md, docs/special-course/CODE_MAP.md and CODE_ISSUES_AND_CHANGES.md (issues A1-A3,
  B1-B7, C1-C8).
- Added `scripts/run_local.py` (cross-platform runner, her run-id format) and fixed Windows port
  exhaustion with one HTTP session (issue B7, no behaviour change).
- E1.1 Expense Report, seed 1, 6 pairs: matches thesis Table 6.3 / 6.2 closely; 0 false
  acceptances; cost-focused agent prefers the dominated Withdraw ending (issue A1).
  Result: `analysis/E1_1_expense_report_results.md`.

## 22 Oct 2026: WSL, E1.1 Loan Application, E1.3
- WSL Ubuntu 24.04: nvm + Node 22, python3.12-venv, Linux venv `.venv-linux` (torch 2.8.0 CPU
  build). Reuses the Windows node_modules. Git only from Windows.
- `run_local.py --fast-io DIR`: logs and TensorBoard on the Linux disk, copied back after each
  run (writing to /mnt/c from WSL was 10x slower: 52 vs 527 steps/s).
- E1.1 Loan Application, 24 runs (4 seeds x 6 pairs), 2 h 48 min: 26.0 +/- 0.8 Pareto points
  (thesis 26.25 +/- 0.5), identical min-cost, min-duration and balanced points, 100% acceptance.
  New: Assess loan risk ends 99.9% of accepting episodes, so its role is never priced (A1).
  Result: `analysis/E1_1_loanapp_results.md`.
- E1.3 state-space statistics (no training), 9 repo graphs + 5 synthetic chains: L_min matches
  the thesis (graph 05 = 6, graph 09 = 51). Synthetic chain graphs rebuilt with her generator
  (seed 1, noise 2.0) and our own XML exporter (our reconstruction, not her files).
  Results: `analysis/E1_3_state_counts.csv`, `.md`, `E1_3_expense_markings.txt`.

## 22-23 Oct 2026: E1.2 scaling baseline (her unmasked PPO), overnight in WSL
- 28 runs (Synth20-100, Mined11, Mined09; pair (0,0); seeds 1-4), all OK.
- Real acceptance only at 20 events; from 40 events every "accepting" episode is fake (A3: the
  agent learned to make exactly 100 legal moves). Mined11 works but is trivial (L_min = 1);
  Mined09 0%. The scaling limit is between 20 and 40 events, not 42-60.
  Result: `analysis/E1_2_scaling_results.md` (script `analysis/e1_2_real_vs_fake.py`).

## 24 Oct 2026: switches and tests (E1.4 preparation)
- Fixes behind off-by-default switches: `FIX_FALSE_ACCEPT=1` (A3), `--truncation-fix` (A2),
  `BUDGET_IN_MASK=1` (B1), `--algo maskable_ppo` (sb3-contrib 2.7.1, masks from
  `DCRGymEnv.action_masks()`). run_local.py passes them through and tags the run id
  (`masked`, `fixA2A3B1`, ...). Bundle rebuilt (esbuild 0.18.20 devDependency).
- Dead ends in masked runs end the episode as truncated, logged in a `dead_end` trace column
  and counted per run.
- Tests a-e (docs/special-course/TESTS.md) all PASS; test a: with switches off, a 2,048-step
  run is identical row for row to fe01d1c (her code + transport fix).
- E1.4 smoke test (Mined09, 2,048 steps, both fixes): unmasked PPO 13 legal moves, 0 accepting;
  MaskablePPO 25/25 episodes really accepting, about 82 steps each (E1.3 random-walk prediction
  82.1). No dead ends.
- Repo checked for files from other projects (costream, co-stream, Discord, esports): none.
- METHOD_NOTES.md copied into docs/special-course/.
- E1.4 overnight sweeps prepared (not started): A = PPO + fixes A2/A3, B = MaskablePPO + fixes,
  each over Synth20-100, Mined11, Mined09, pair (0,0), seeds 1-4, 28 runs per sweep, about
  4-4.5 h. Outputs `scripts/logs/sweep_E1_4_ppo.out` and `sweep_E1_4_masked.out`.

## Commits on special-course (after her final commit dbd7c49)
| Commit | Date | What |
| --- | --- | --- |
| d6f3e9b | 21 Oct | Update yarn.lock to match package.json |
| ddc3ff3 | 21 Oct | Declare fast-xml-parser dependency in node-adapter |
| 543c68b | 21 Oct | Special-course context: CLAUDE.md, code map, issues notes |
| 6f92239 | 21 Oct | Cross-platform experiment runner run_local.py |
| fe01d1c | 21 Oct | One HTTP connection in node_connector.py (Windows port exhaustion); reference for test a |
| ecde07b | 21 Oct | Ignore run_local.py sweep console output |
| 99edb74 | 21 Oct | .gitignore: sweep_*.out on its own line |
| a4e9243 | 22 Oct | run_local.py --fast-io for WSL; ignore .venv-linux |
| b08ab58 | 22 Oct | E1.3 state-space statistics script and synthetic chain-graph exporter |
| 2e4fcbb | 22 Oct | E1.3 synthetic chain-plus-noise graphs (20-100 events) |
| 83bf48f | 24 Oct | A2, A3, B1 fixes and MaskablePPO behind off-by-default switches |
| e1d79b7 | 24 Oct | Switch tests a-d, TESTS.md; A2, A3, B1 marked fixed behind a switch |
| b37e17f | 24 Oct | Dead ends in masked runs end as truncated; test e |
| 8b45edf | 24 Oct | This log, METHOD_NOTES.md copy, CLAUDE.md session-log rule |
| 1e4631e | 24 Oct | Issues doc: M1 (LoanApp accepts without a decision), D1 (dead-end handling, open) |
| e4d09ef | 26 Oct | Docs: 26 Oct meeting outcome (SC-D3 analysis only, E1.2 scaling limit) |
| 22dbcaf | 26 Oct | A1 fix behind a switch (`FIX_FINAL_STEP`, `--fix-final-step`); test f |
| 749b64d | 3 Nov  | Session log: A1/A3 sweeps of 26 Oct-3 Nov |
| 1e61d86 | 3 Nov  | Analysis entry; `SHAPING_FIX`, `--gamma`, `--no-progress`, `--weights-sweep`; tests g, h |

## Open issues
- A1 (flat +100 on the accepting step, so the final event's cost and role are never priced):
  fixed behind `FIX_FINAL_STEP=1` (26 Oct); runs done, analysis still to do.
- Dead ends are ended as truncated (as specified in METHOD_NOTES). A dead end has no future, so
  terminated would be the more standard choice for value learning; worth deciding before final runs.
- Masked and unmasked runs use the same 300-step limit, but blocked attempts count toward it
  (C2), so masked runs get more legal steps per episode.
- LoanApp: the graph lets a case finish without a decision event (ask Hugo whether intended).
- Thesis claims on the scaling ceiling (about 42 pairs) need revising after E1.2 (for Hugo).
- B2 fallback mask left in her code (unused by the new masking).

## 26 Oct: meeting with Hugo
- SC-D3 changed: no recommender prototype in the special course. New SC-D3 = comparative
  analysis and benchmark design for next-activity recommendation (inputs, outputs, datasets,
  evaluation of existing approaches; specification of ours). Building it is thesis work.
- Study 1 / SC-D2 unchanged. Next: code D1 (dead end = terminated, not accepting), then E1.4b
  (partway starts or reward shaping) for the synthetic chains.
- Official period 1-30 Nov (2.5 ECTS). Plan: SPECIAL_COURSE_MAP.md in the course folder.
- Updated: CLAUDE.md (Study 1 limit, Study 2 scope), CODE_ISSUES_AND_CHANGES.md (SC-D3 rows now
  E1.4b / thesis), METHOD_NOTES.md.

## 26 Oct - 3 Nov 2026: A1 switch, A1 and A3 sweeps (WSL, --fast-io, 100,000 steps)
- A1 switch: `FIX_FINAL_STEP=1` / `run_local.py --fix-final-step`, run-id tag `fixA1`. The accepting
  step gets its normal step reward (+1, step penalty, progress, repetition, cost and duration with
  role multipliers) plus 100; `baseMapped` stays 100. Test f PASS (LoanApp, final Assess loan risk:
  off +100 for both roles; on Expert 97.58, Junior 97.84); tests a and b still PASS.
- Sweeps (her settings otherwise; 74 runs, all OK, each with one complete trace of 100,352 steps):
  (a) `ExpenseReport_Diaz_s{2,3,4}_<pair>_kunlim`, no switches, 18 runs (seed 1 from 21 Oct);
  (b) `ExpenseReport_Diaz_fixA1_s{1-4}_<pair>_kunlim`, 24 runs;
  (c) `LoanApp_roles_fixA1_s{1-4}_<pair>_kunlim`, 24 runs;
  (d) `E1_2_Synth{20,40}_fixA3_s{1-4}_a0p0_b0p0_kunlim`, `--fix-false-accept` only, 8 runs.
- Wall time: first attempt 26 Oct 16:46-18:56, four sweeps in parallel, 11-23 min per run; (d)
  finished (Synth20 57 min, Synth40 1 h 11 min). At 18:56 C: was full (30 MB free; not our
  files: logs 3.4 GB, WSL disk 5.3 GB) and writes to /mnt/c failed with EIO: 29 runs had finished,
  3 broke (partial files kept in `scripts/logs/failed_26oct_diskfull/`, not used), 42 not started.
  Rerun of the 45 (`~/sweeps_26oct_rerun.sh`, three sweeps in parallel) 3 Nov 02:02-04:22,
  2 h 20 min, 5.5-9.8 min per run, all OK.
- WSL failed to start a few times on 26 Oct (Wsl/Service/CreateInstance E_UNEXPECTED / E_FAIL);
  it worked again without a reset. Long sweeps are run by hand in the Ubuntu app, not from Claude.
- Next: analyse (b)/(c) against (a)/her LoanApp runs (does the cost-focused agent still prefer the
  Withdraw ending? are final-event roles now priced?) and (d) against E1.2 (real acceptance at 20
  and 40 events without false finishes).

## 3 Nov 2026: analysis of the A1 and A3 runs
Result: `analysis/A1_A3_switch_results.md` (scripts `a1_a3_analysis.py`, `expense_path_returns.py`).
- LoanApp, fixA1: works as intended. Assess loan risk (ends 99.9% of cases) now follows the
  objective: Expert 2-8% at (1,0), 91-99% at (0,1) (her version about 50% everywhere). Mean cost
  at (1,0) 1437 -> 1273, mean duration at (0,1) 645 -> 503; front 26.75 vs 25.75 points; same
  min-cost and min-duration points.
- Expense Report, fixA1: the dominated D path disappears, but every weight pair converges to P2
  (710, 2,700), including cost-focused ones (her version: P1 and D tied at about 45% each).
  Cause: with +100 for completion and gamma 0.99, the shorter path's earlier bonus is worth about
  1 unit, more than the normalised cost saving of P1 (0.94 at alpha 1). Reward scale needs a decision.
- Synth20/40, A3 only: no fake accepts; Synth40 0 real (1 completion in 4 runs). Synth20 fails in
  3 of 4 seeds (1.5-10% real) while A3 + A2 (E1.4) succeeds in 4 of 4; first real completion
  identical per seed, so the A2 fix decides it. Use A3 only together with A2.
- For Hugo: A1 result, the reward-scale issue on the Expense Report, A3 needs A2.

## 3 Nov 2026: SHAPING_FIX, --gamma, --no-progress, --weights-sweep (no runs)
Follows from the Expense Report result above (cheap path loses to the short one).
- `SHAPING_FIX=1` / `run_local.py --shaping-fix` (tag `shapeFix`): a legal step's reward is
  -alpha x cost/maxCost - eps x duration/maxDuration (role multipliers) + gamma x Phi(after) -
  Phi(before), Phi = -2 x included pending events, +100 on the accepting step (covers A1); no +1
  base, no step penalty, no repetition penalty; blocked actions stay -10. Phi = 0 for the marking
  an episode ends in (accepting, or a dead end with no enabled event; user's addition), so the
  progress terms add up to 0 over a finished case when gamma = 1 and no event starts pending.
  Step-limit ends keep Phi (truncation; the last step there may be a blocked one).
- `--no-progress` (needs `--shaping-fix`, tag `shapeFixNoProg`): adapter `SHAPING_NO_PROGRESS=1`,
  the same reward without the progress term, as a baseline.
- `--gamma` in train_agent.py and run_local.py (default 0.99 = SB3 default = her runs; tag
  `g<gamma>` otherwise); run_local.py always passes it to the adapter as `SHAPING_GAMMA`.
- `run_local.py --weights-sweep`: w = 0, 0.1, ..., 1 for cost, 1 - w for duration (11 pairs).
- Tests: g (Expense Report, alpha 1, gamma 1: P1 99.928571 = 100 - 50/700, P2 98.985714, D 98.95,
  progress terms [-4, 2, 2] / [-4, 4] summing to 0; same totals with --no-progress; dead-end step
  progress 0, reward -1) PASS; h (LoanApp, alpha 1: final Assess loan risk Junior 101.7098 >
  Expert 101.1708) PASS; a (switches off, identical to fe01d1c) PASS; b, c (3 variants), d, e, f PASS
  with the same numbers as before (c, e need PYTHONIOENCODING=utf-8 when redirected on Windows).
- dist/server.mjs rebuilt. No training runs started.
- Open: which runs to do with it (Expense Report and LoanApp, gamma 0.99 vs 1, with and without
  progress, the 11-pair weights sweep); with gamma 0.99 the +100 is still discounted by path
  length (about 1 unit per extra step), so the cheap Expense path needs alpha above about 1.06.

## 3 Nov 2026: Expense Report weights sweeps with A2 + A3 (132 runs, not analysed yet)
Commit 1e61d86, WSL, `~/sweeps_3nov.sh`, `--weights-sweep` (11 pairs, w = 0, 0.1, ..., 1 for cost,
1 - w for duration), seeds 1-4, 100,000 steps, `--fix-false-accept --truncation-fix --fast-io`:
  (a) `ExpenseReport_Diaz_fixA2A3_shapeFix_g1p0_s{1-4}_<pair>_kunlim`, `--shaping-fix --gamma 1.0`, 44 runs;
  (b) `ExpenseReport_Diaz_fixA2A3_shapeFixNoProg_g1p0_s{1-4}_<pair>_kunlim`, also `--no-progress`, 44 runs;
  (c) `ExpenseReport_Diaz_fixA1A2A3_s{1-4}_<pair>_kunlim`, `--fix-final-step`, gamma 0.99, 44 runs.
- Wall time: three sweeps in parallel, 15:58-22:27, (a) 6 h 24 min, (b) 6 h 28 min, (c) 6 h 28 min;
  7.6-10.7 min per run. No failures: 132 runs OK, each with a 100,353-line trace, "Model saved" in
  the run log, one TensorBoard dir and one model.
- Disk: free space on C: went from 82 to 69 GB (about 90 MB per run counting the `--fast-io` copies
  in `~/dcr_io/sh_{a,b,c}`, 5 GB, still there), more than the 3-4 GB estimated beforehand.
- Next: analysis; LoanApp sweeps (`~/sweeps_4nov_loanapp.sh`) to be run next.

## 4 Nov 2026: LoanApp weights sweeps with A2 + A3 (88 runs)
Commit 17f6f6c (code as in 1e61d86), WSL, `~/sweeps_4nov_loanapp.sh`, graph LoanApp_junior_senior,
run-id base `LoanApp_roles`, `--weights-sweep` (11 pairs), seeds 1-4, 100,000 steps,
`--fix-false-accept --truncation-fix --fast-io`:
  (a) `LoanApp_roles_fixA2A3_shapeFix_g1p0_s{1-4}_<pair>_kunlim`, `--shaping-fix --gamma 1.0`, 44 runs;
  (b) `LoanApp_roles_fixA1A2A3_s{1-4}_<pair>_kunlim`, `--fix-final-step`, gamma 0.99, 44 runs.
- Wall time: two sweeps in parallel, 02:29-07:12, 4 h 43 min each; 6.0-7.2 min per run. No failures:
  88 runs OK, each with a 100,353-line trace, "Model saved" in the run log, one TensorBoard dir and
  one model.
- Disk: free space on C: 69 -> 64 GB; `--fast-io` copies in `~/dcr_io/loan_{a,b}` (1.2 GB) still there.
- Both reward checks are already analysed: `analysis/Reward_check_expense_results.md` (3 Nov sweeps,
  script `reward_check_expense.py`) and `analysis/Reward_check_loan_results.md` (these sweeps, script
  `reward_check_loan.py`).

## 5 Nov 2026: larger-graph reward check (144 runs); E1.4b partway starts implemented
RW_ sweeps (`~/sweeps_4nov_rw.sh`, commit 4ef2be2 code as in 1e61d86), 5 Nov 01:29-08:18, three at a
time, 5.2-11.6 min per run: 144 of 144 OK, and `~/check_4nov_rw.py` confirmed the switches, graph,
seed, weights, gamma, algorithm and outputs of every run. Results: `analysis/Reward_check_larger_graphs_results.md`.

Quick check, no training (`node-adapter/scripts/chain_trace_check.ts`, E1.3 files not changed):
| Graph | Act_Event_1 > ... > 5 | Greedy best-first (fewest included pending first) |
| --- | --- | --- |
| Synthetic_chain_60 | legal, does not accept (2 pending left) | found, 213 steps, verified by replay, 205 s |
| Synthetic_chain_80 | legal, does not accept (3 pending left) | found, 154 steps, verified, 14 s |
| Synthetic_chain_100 | legal, does not accept (7 pending left) | none in the 10-minute cap (4.7 million markings, never below 3 pending) |
Traces, not shortest: `analysis/E1_4b_trace_Synthetic_chain_{60,80}.json`, usable with
`--partway-trace-file`. So 60 and 80 are not unsolvable for trace mode; E1.3 only lacked a trace.
The design notes are in `SpecialCourse/METHOD_NOTES.md`; the repo copy of that file stops at Oct 26.

Implemented (details in TESTS.md and CODE_ISSUES_AND_CHANGES.md): `--partway-starts {none,random,trace}`
with `--partway-initial-share`, `--partway-prob`, `--partway-max`, `--partway-trace-file`,
`--partway-timelimit-factor`; `--fix-dead-end` (D1). Reward not changed. Tests a-j PASS.
Committed as 428a890 (not pushed).

Smoke test, WSL, seed 1, 2,000 steps (2,048 taken), all 8 versions at w = 0.5, including chain 40
trace mode: 8 of 8 OK with `~/check_5nov_ps.py`; masked runs 0 illegal steps; 0 dead ends; j stays 4
(only 4-6 frontier episodes fit in one rollout, a move needs 20). Smoke files deleted.
Weights on the chains: w = 0 and w = 1, same seed, trace mode, masked: the two trace CSVs are byte
identical. The chains have no costs or durations, so w changes nothing: of the 72 chain 40 runs
asked for, 48 would be exact copies, and cost and duration of accepted cases are 0 there.
Sweep script `~/sweeps_5nov_ps.sh` (80 runs; `--w05-only`: 32) ready, not started.

Decision (user, 5 Nov): chain 40 at w = 0.5 only, `~/sweeps_5nov_ps.sh --w05-only`, 32 runs (24 on chain 40, 8 on chain 20).

While the PS_ sweep runs (started 15:16; code at 428a890, not touched):
- Short traces (`node-adapter/scripts/shorten_trace.ts`: remove one event at a time while the replay stays legal and accepting): chain 60 213 -> 55 events, chain 80 154 -> 77; `analysis/E1_4b_trace_Synthetic_chain_{60,80}_short.json`. No single event can be removed; not proved shortest.
- `--partway-step N` written and tested in a temporary worktree, `../dcr-js-wt-step` (tests a, i, j PASS; 4 smoke runs OK), NOT yet in this tree: running sweeps start Python from these files. To apply when the sweep ends.
- Prepared, not started: `~/sweeps_ps6080.sh`, chain 60 and 80, none vs trace, masked, w = 0.5, seeds 1-4, 16 runs; step 3 (18 stages) and step 4 (19 stages); `~/check_ps6080.py`. It refuses to start while training runs or before the step option is applied.
- Budget note: 20 episodes at each j mean about 50 episodes per stage, of which 10 start at the initial marking (up to 300 steps each). Even with perfect play that is about 74,000 steps on chain 60 and 87,000 on chain 80, so 100,000 steps may not reach j = 0. `STEPS` is at the top of the script.

PS_ sweep finished: 15:16-17:37 (2 h 21 min), 32 of 32 OK with `~/check_5nov_ps.py --w05-only`, no failures. Run ids `PS_Synth40_[masked_]fixA2A3D1_shapeFix_g1p0[_psRandom|_psTrace]_s{1-4}_a0p5_b0p5_kunlim` and `PS_Synth20_masked_fixA2A3D1_shapeFix_g1p0[_psTrace]_s{1-4}_a0p5_b0p5_kunlim`. Results: `analysis/E1_4b_partway_results.md`; scripts `analysis/e1_4b_partway.py`, `analysis/e1_4b_decay_check.py`.
- Chain 40, accepted from the initial marking, final 20%: masked trace 92-98%, masked none 0-29%, PPO trace 58-80%, PPO none 0-1.5%, random 0% (1 accepted case in 8 runs). j = 0 at 36,338-85,341 masked; 84,041-95,754 PPO, one seed not reached. The demonstration is only 5 events (4 stages).
- Cost and duration of accepted cases are 0 (chains have none). False finishes 0, dead ends 0 (D1 did not fire).
- Other traces: chain 40 no, all 8,518 accepted trace-mode cases contain `1 2 3 4 5` in order (44% exactly), and so do all 56 accepted cases without partway starts. Chain 20 yes, all four length-5 orders in all 8 runs.
- Problem found: chain 20, masked, new reward: acceptance reaches 99.9% at 50-60% of training, then falls to 60% (none) and 34% (trace) in the last bin, while legal steps per case rise from 12-15 to 94, entropy from 1.50 to 2.33, explained variance 0.99. The E1.4 runs (old reward, same entropy) stay at 100%. Likely cause, not tested: no cost, no step penalty and gamma 1 make every finish worth the same, so the entropy bonus takes over. This qualifies the 4-5 Nov reward check, whose Synth20 runs were unmasked and still learning at 100,000 steps. Chain 40's 97% may be before the same fall.
- Open decision before chain 60/80: test `--ent-coef 0.01` and `--gamma 0.99` on chain 20 first. `~/sweeps_ps6080.sh` not started. Chain 60/80 probably need several hundred thousand steps (extrapolation).

`--partway-step` applied to this tree after the sweep (identical to the worktree files, not committed): tests a, i, j PASS. Temporary worktree and its junction removed. `partway_frontier_exp_*.csv` added to .gitignore, like the other training logs. Committed as ea5cffb (not pushed).

Decision (user, 5 Nov): test the late fall on chain 20 before chain 60/80. Prepared, not started: `~/sweeps_fall20.sh`, `~/check_fall20.py`; report with `analysis/e1_4b_decay_check.py <logs> "fall test"`.
- Synthetic_chain_20, masked, no partway starts, w = 0.5, seeds 1-4, 100,000 steps, baseline flags. Entropy 0.01 with gamma 1: `FT_Synth20_ent0p01_masked_fixA2A3D1_shapeFix_g1p0_s*`. Entropy 0.1 with gamma 0.99: `FT_Synth20_gam0p99_masked_fixA2A3D1_shapeFix_s*` (run_local gives no gamma tag at 0.99, so the setting is in the base name). 8 runs. Control, not rerun: `PS_Synth20_masked_fixA2A3D1_shapeFix_g1p0_s*`.
- Entropy default: `train_agent.py` --ent-coef 0.01 (her code, unchanged since dbd7c49); her runner `run_experiments.py` passes 0.1, recorded in our notes as the thesis value (CODE_ISSUES B3, CODE_MAP 4.5; thesis itself not reopened); our `run_local.py` defaults to 0.1 to match it. All our runs so far use 0.1. SB3's own default is 0.0.
- Smoke test at ea5cffb (seed 1, 6,000 steps, 3 runs OK, files deleted): control settings repeat the PS_ run in 6,144 of 6,144 rows, so the PS_ runs are a valid control. Both versions choose the same actions as the control in the 2,048 rows before the first update and differ after. Saved models hold the intended gamma and ent_coef. With gamma 0.99 the potential term pays 0.02 x pending on a step that resolves nothing (0.08 with 4 pending), as the formula gives.

Chain 20 test finished: 21:28-22:01 on 5 Nov (32 min), 8 of 8 OK with `~/check_fall20.py`, at ea5cffb. Results: `analysis/E1_4b_fall_test_results.md`.
- Accepted from the initial marking, final 20%, seeds 1-4: control 81.5, 88.7, 34.1, 85.6; entropy 0.01: 99.2, 35.2, 99.8, 98.0; gamma 0.99: 100, 100, 99.9, 100. Last-window steps per case 94 / 10 / 5.8, entropy 2.33 / 1.55 / 0.60.
- Gamma 0.99 removes the fall; entropy 0.01 only slows it (seed 2 falls to 10.5%, entropy rising in all seeds). Limits: 4 seeds, 100,000 steps.
- Gamma 0.99 is not adopted. By hand calculation (no run) it would stop P1 being preferred on the Expense Report at any w: one extra step loses 0.99 of the +100, against an objective gap of at most 0.94. At gamma 0.999 the extra step loses 0.10 and the switch moves from w = 0.535 to 0.58.
- Open decision: test gamma 0.999 on chain 20, or run chain 60/80 with gamma 0.99 as a chain-only setting. Chain 60/80 not started.

## 6 Nov 2026: costs and durations per graph; --default-duration
Decision (user, 6 Nov): no gamma 0.999 test, keep gamma 1. Test instead whether giving events a duration removes the late fall.
- Table: `analysis/Graph_costs_durations.md` (script `node-adapter/scripts/graph_costs_durations.ts`, through the engine's parser). Every graph is all or nothing: all events have both cost and duration, or none has either. Stated zeros: Loan_Application_Diaz "request loan" (cost and duration), Synthetic_review_roles (cost, 5 events). The graphs we train on are in `app/public/examples/diagrams/`; `test_models/Mined DCR Models/` holds 9 originals in another format that the engine reads as 0 events, with no values.
- Mined logs: none in the working tree, git history or SpecialCourse folder; nothing downloaded. Referenced in thesis Ch. 8 (footnote), `github.com/s205358/dcr-js/tree/s205358/bench`, with a download script; link not opened. Timestamps cannot be confirmed from the repo. Mined costs and durations are seeded random (seed 1), not data from the logs. Mined 02, 06, 07, 08, 10 are not in the repo.
- `--default-duration X` (adapter `DEFAULT_DURATION`, off by default, tag `defDur<X>`): events with no duration get X; a stated value, including 0, stays. Tests: a PASS (off: 2,048 rows identical to fe01d1c), k PASS (new), g, h, i PASS.
- On a chain X does not change the reward, because durations are divided by the largest: every event scores 1. So it acts as a step penalty of (1 - w): 0.5 per step at w = 0.5, none at w = 1.
- Prepared, not started: `~/sweeps_defdur20.sh`, `~/check_defdur20.py`; chain 20, masked, no partway starts, w = 0.5, seeds 1-4, 100,000 steps, gamma 1, entropy 0.1, 4 runs, `FD_Synth20_masked_fixA2A3D1_shapeFix_g1p0_defDur1_s*`. Control, not rerun: `PS_Synth20_masked_fixA2A3D1_shapeFix_g1p0_s*`. Report: `analysis/e1_4b_decay_check.py <logs> "duration test"`.
- Smoke test (seed 1, 6,000 steps, 2 runs OK, files deleted): control settings repeat the PS_ run in 6,144 of 6,144 rows; the default-duration run chooses the same actions in the 2,048 rows before the first update, with each reward 0.5 lower, and differs after. Chain 60/80 not started.

Chain 20 default-duration test finished: 15:20-15:43 on 6 Nov (23 min), 4 of 4 OK with `~/check_defdur20.py`, at 0bb8e71. Results: `analysis/E1_4b_default_duration_results.md`.
- Accepted from the initial marking, final 20%, seeds 1-4: 100, 100, 99.9, 100 (control 81.5, 88.7, 34.1, 85.6). Last-window steps per case 6.9 (control 94.1), entropy 0.86 (control 2.33). No dead ends.
- The fall disappears at gamma 1 with a price of 0.5 per step. It matches gamma 0.99 on acceptance, with slightly longer cases (6.9 against 5.8). Entropy rose a little in the last window in seeds 2 and 3. Limits: 4 seeds, 100,000 steps.
- Every non-chain graph has a price on every step at every w (except Loan_Application_Diaz "request loan"), so the fall is a chain problem among our graphs. Whether a much smaller price (0.0013-0.014 per step at w = 1 on LoanApp and Expense) is enough is untested.
- Open decision: run chain 60/80 with `--default-duration 1` at w = 0.5. `~/sweeps_ps6080.sh` does not have the flag yet. Chain 60/80 not started.

## 9 Nov 2026: chain 60/80 sweep updated; plan for E1.8 and E1.5
- Decision (user): synthetic chains always run with `--default-duration 1` and at w = 0.5 only. `~/sweeps_ps6080.sh` now passes the flag to all four jobs (run ids gain `_defDur1`), and `~/check_ps6080.py` checks it on every row. It also has a `--smoke` mode. Smoke test (seed 1, 2,000 steps): 4 of 4 OK, files deleted. Still not started.
- Script fix: a variable named PREFIX set before loading nvm stops nvm from loading (node missing, all runs fail at once). Renamed to RUN_PREFIX.
- Privacy rule added to CLAUDE.md: no user messages or prompts stored verbatim; the session log keeps short summaries only.
- Next: rerun chain 40 with `--default-duration 1` (PD_), then E1.8 (safe RL variants) and E1.5 (other algorithms). E1.8 checks of the shield-off path come before any code change.
- Prepared, not started: `~/sweeps_pd40.sh` + `~/check_pd40.py`. Chain 40, w = 0.5, seeds 1-4, 100,000 steps, `--default-duration 1`: masked none, masked trace, PPO trace (12 runs, `PD_Synth40_...defDur1...`), all three jobs at once, about 1.5 h. Smoke 3 of 3 OK after a fix (RUN_PREFIX was not exported, so run ids had no prefix; those stray files were deleted).
- `~/sweeps_ps6080.sh` updated (not started): 300,000 steps; trace seeds 1-4, none seeds 1-2 (12 runs); trace jobs first. Smoke 4 of 4 OK; traces 55 and 77 events, frontier starts at 54 and 76, steps 3 and 4 (18 and 19 stages).
- Late-fall check of the 3-4 Nov new-reward runs (existing logs; `analysis/Late_fall_check_results.md`): no fall in Expense, LoanApp, Mined 05 or Mined 09 at w = 0, 0.5, 1. 100% acceptance in every window; steps flat or falling. Expense w = 1 rises from 2.1 to 3.4 steps because it moves to the cheap 3-step path.
- E1.8 check of the shield-off path (`SHIELD_DISABLED=1` with `--shaping-fix`, no code change, probed on the adapter). A forbidden event is executed and its cost and duration go into the episode totals, but its reward is a flat -10: no cost, duration or progress term. If the new marking accepts, the episode ends as accepted with -10 and no +100. Expense: 6 of 7 violations tried end accepted, e.g. PayOut at the start. LoanApp: 22 of 346. Also: Phi is not charged on a violation step, so pendings created by a violation are not charged but are paid when resolved later; violations do not use up the Expert budget; the log calls them `dcr_illegal` even though they executed. Waiting for the user's decision before any fix.
- Decisions (user): price a violation like a legal step plus the violation price, +100 if it ends accepted, counted as non-compliant; violations use the Expert budget and are logged as "violation"; behind a switch. A2C for E1.5: masked, MaskablePPO with A2C settings. The masked E1.8 runs are the MaskablePPO reference for E1.5.
- PD_ chain 40 rerun: started 16:01 (in a terminal tab), ALL_DONE 16:56, 12 of 12 OK with `~/check_pd40.py`. About 8 min per run alone, 27 min while the E1.8 smoke test ran beside it. Not analysed yet.
- E1.8 implemented (in a temporary worktree while PD_ ran, then applied): `--shield-off` (her SHIELD_DISABLED), `--violation-fix` (adapter VIOLATION_FIX, price 10, `--violation-price`), `--lagrange` (multiplier after each rollout, `src/agents/lagrange.py`, log `lagrange_exp_<run>.csv`). Lambda_0 = 1 (Ray et al. 2019 code default; Stooke et al. 2020). lr = 3: Stooke et al.'s K_I = 1e-2 on episode cost, times our 300-step limit. Our variant is a reward penalty, not their separate cost critic with 1/(1 + lambda) scaling; with target 0, lambda only rises. New tests l and m; tests a-m PASS (c on Expense and LoanApp k = 1).
- A2C settings checked (Huang et al. 2022, A2C as a special case of PPO): RMSprop alpha 0.99, eps 1e-5; lr 7e-4, no annealing; n_steps 5; gae_lambda 1; 1 epoch, full batch; no advantage normalisation, no value clipping. SB3 A2C defaults agree (vf 0.5, max grad norm 0.5). Not built yet.
- E1.8 sweep `~/sweeps_e18.sh` + `~/check_e18.py` (160 runs, SV_, 16 jobs, 3 at a time; estimate about 7 h). Smoke from the worktree: 16 of 16 OK.
- PD_ analysis (`analysis/E1_4b_PD_rerun_results.md`). With the step price, masked PPO without partway starts learns chain 40: 77-98% in the final 20% (PS_: 0-29%). Trace starts reach 90% sooner (42-65k vs 63-96k steps), final level similar (90-99%). Two masked trace seeds dip at the end with entropy rising. PPO trace is worse with the price: 39-70%, j = 0 in 2 of 4 seeds.
- Correction to the test-m description given in chat: violations per step rose from 57% to 73% over the two rollouts, not fell.

## 11 Nov 2026: chain 60/80 sweep interrupted
- The laptop lost power during `~/sweeps_ps6080.sh` (started 14:51); no run finished. Leftovers were 3 sweep .out files and 3 run logs in `~/dcr_io/ps6080` holding only the adapter start-up lines: no trainer output, no traces, no models. All deleted. No processes left. Git intact (fsck clean); only the main checkout exists (no E1.5 worktree: E1.5 not started; the E1.8 worktree was removed on 9 Nov).
- Smoke test of the sweep on the current code (43512bb): 4 of 4 OK, files deleted. Ready to restart (300,000 steps, 12 runs). Free space on C: 23 GB (the scripts stop below 20 GB).

## 11 Nov 2026: E1.8 results; E1.5 built; chain 60/80 finished
- Chain 60/80 restarted 16:46. 10 runs OK by 18:26 (18-39 min each at 300,000 steps). The chain 80 control was skipped by the disk guard (free space fell below 20 GB through Windows files, not our runs; the user emptied the Recycle Bin), then run separately with `~/sweeps_ps6080_s80none.sh` (both seeds in parallel, 18:29-18:48). `~/check_ps6080.py`: 12 of 12 OK. Not analysed yet.
- E1.8 (160 SV_ runs, 160 of 160 OK with `~/check_e18.py`): `analysis/E1_8_safe_rl_results.md` (script `analysis/e1_8_safe_rl.py`). The user's quickcheck matches the logs except small differences on Mined 09 (window rule). Shield and mask: 0 violations; mask also solves Mined 09, where the shield accepts no case. Penalty and Lagrangian learn a one-step loophole on LoanApp ("Cancel application") and Mined 05 ("get review X") and keep it (100% violating episodes, 0% compliant). Lambda passes 100 at about 72k steps and ends at 134-144; on Expense it settles at 24-33 with 68-100% compliant acceptance. Our Lagrangian is the reward-penalty form.
- E1.5 built (in a temporary worktree while chain 60/80 ran, then applied): `--algo maskable_a2c` (MaskablePPO with the A2C settings of Huang et al. 2022), `maskable_dqn` (`src/agents/masked_dqn.py`: mask in action choice and target max; SB3 defaults except buffer 100,000, learning starts 1,000, target every 1,000 steps, epsilon over 20%; gamma 1), `tabular_q` (`src/agents/tabular_q.py`: alpha 0.1, gamma 1, same epsilon schedule). Exact optimum: `node-adapter/scripts/exact_optimum.ts` (Dijkstra over markings, roles, optional Expert budget); results in `analysis/E1_5_exact/`. MiniZinc is not installed; the optima agree with the stored MiniZinc results (LoanApp without roles, Mined 05) and with Diaz et al. (Expense). New tests n, o; tests a-o PASS. The MaskablePPO path gives the same 4,096 trace rows before and after the change.
- AL_ sweep `~/sweeps_al.sh` + `~/check_al.py`: 116 runs (A2C and DQN on Expense, LoanApp, Mined 05 at w = 0, 0.5, 1 and chain 20 at w = 0.5 with default duration 1; tabular Q on the first three), 11 jobs, 3 at a time; each job deletes its WSL working folder when done; disk guard 10 GB. Smoke from the worktree: 11 of 11 OK. Reference: SV_*_masked and FD_Synth20 (not rerun). Analysis script ready: `analysis/e1_5_algorithms.py`.

## 11 Nov 2026 (evening, AL_ sweep running): export, chain 60/80, overview
- Episode export for sharing: `analysis/short_path_bias_episodes.zip` (7.9 MB; 1,041,421 episodes ending in the final 20%: Expense and LoanApp with both rewards at all w, Mined 09 both rewards at 3 w, chain 20 FD_ and PS_ control; README with both reward definitions in words, no flag names or paths). Script `analysis/export_short_path_bias.py`. A zip of the full per-step CSVs of the same runs (208 files, 3.59 GB) would be about 0.25 GB.
- Expense, new reward, w >= 0.8 (`analysis/Expense_off_path_traces.md`): 24-31% of accepted cases are the cheap path with repeated activities (extra FillOut, Approve or Reject; cost 60-100), 0.1-0.6% are P2 or D. With no step or repetition penalty, a detour costs about 0.014-0.025 at these weights.
- Chain 60/80 (`analysis/E1_4b_chain6080_results.md`): 0% acceptance from the initial marking in all 12 runs. The trace frontier never passed the first stage (4-5 events left: 0-31% success). Few episodes (failures run 300 steps). Options: frontier step 1, shorter time limit for partway episodes.
- Disk (read-only survey): repo run files 33.7 GB (traces 13.7, run logs 11.1, TensorBoard 8.7); WSL working folders 15.6 GB, duplicates of repo files. AL_ projected at 7-10 GB. Nothing deleted.
- Study 1 overview started: `analysis/Study1_results_overview.md`; E1.5 and E1.6/E1.7 rows pending.
