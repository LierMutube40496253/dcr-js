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
| (next)  | 24 Oct | This log, METHOD_NOTES.md copy, CLAUDE.md session-log rule |

## Open issues
- A1 (flat +100 on the accepting step, so the final event's cost and role are never priced):
  not fixed; optional switch only if Hugo agrees.
- Dead ends are ended as truncated (as specified in METHOD_NOTES). A dead end has no future, so
  terminated would be the more standard choice for value learning; worth deciding before final runs.
- Masked and unmasked runs use the same 300-step limit, but blocked attempts count toward it
  (C2), so masked runs get more legal steps per episode.
- LoanApp: the graph lets a case finish without a decision event (ask Hugo whether intended).
- Thesis claims on the scaling ceiling (about 42 pairs) need revising after E1.2 (for Hugo).
- B2 fallback mask left in her code (unused by the new masking).
