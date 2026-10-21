# Code map: Sofía's thesis to her code (branch feature/cost-roles, commit dbd7c49)

Checked on 21 Oct against thesis chapter 4 (4.1-4.6), chapter 5.2 and chapter 8, and
against the code. Paths are relative to `dcr-js/`. "OK" = code and thesis agree.

## The three parts and how one step flows

Python agent (PPO)  --HTTP-->  Node adapter (port 5001)  -->  DCR engine (TypeScript)

Example, Expense Report at the start (only FillOut is enabled):
1. PPO picks action index 0 (Approve; events are sorted by id: approve, fillout, payout,
   reject, withdraw). `dcr_env.step()` sends it to the adapter (`POST /action`).
2. The adapter turns the index into an (event, role) pair, here (Approve, no role).
3. The engine checks whether Approve is enabled. It is not (condition FillOut -> Approve),
   so the shield blocks it: marking unchanged, reward -10.
4. The adapter returns reward, new state and the current action mask. Python turns the
   state into the 15-number observation.

## Concept to file and function

| Thesis concept (section) | File | Function / place | Check |
| --- | --- | --- | --- |
| Architecture: agent, adapter, engine (4.1) | the three folders above | | OK |
| DCR semantics: enabled, execute, accepting (4.2.3) | dcr-engine/src/executionEngine.ts | isEnabledS, executeS, isAcceptingS | OK. Accepting = no event both included and pending |
| RL environment, engine side | dcr-engine/src/generation.ts | class RLDCREnvironment: reset, getValidActions, step | OK |
| Interception shield (4.2.4) | dcr-engine/src/generation.ts | RLDCREnvironment.step | OK. Not enabled: -10, marking unchanged, done=false. Thesis gives the reason for interception instead of masking: illegal attempts stay visible as a learning signal and a metric |
| RL-only ablation (7.1) | same | env variable SHIELD_DISABLED=1 | OK. Illegal event executed anyway, still -10 |
| Observation (4.2.1) | dcr-gymnasium-agent/python/src/envs/dcr_env.py | _state_to_obs | OK. 3 bits per event (included, executed, pending); +1 budget fraction when k is set. Expense 15, Loan 36 |
| Action space (4.2.2, 4.6) | node-adapter/src/server.ts | getEventRolePairs | OK. Index -> (event, role); events and roles sorted alphabetically |
| Action mask, logged only (4.1) | node-adapter/src/server.ts; dcr_env.py | buildActionMask; self.action_mask | OK. Thesis: "logs the enabled set but never uses it to restrict the agent" |
| Reward (4.3, Eq. 4.3, Table 4.2) | dcr-engine/src/reward.ts | computeStepReward | OK. +1 base, +2 per pending event resolved for the first time, -1 repeat (per event, any role), -alpha*c/cmax, -eps*d/dmax. Accepting step: flat +100 that replaces everything (see finding 1) |
| Step penalty -1.5 (4.3) | node-adapter/src/server.ts | /action, effectiveReward | OK with her runner. Adapter default is -0.1 |
| Normalisation cmax, dmax (4.3) | node-adapter/src/server.ts | maxGraphCost, maxGraphDuration | OK. Max base value in the graph, before role multipliers |
| Role multipliers (4.6, Table 4.4) | node-adapter/src/server.ts | /action, eventCost / eventDuration | OK. Base x multiplier (Junior 0.7/2.0, Expert 2.0/0.5) from the XML roleOptions |
| Expert budget k (4.6, 6.2.4) | node-adapter/src/server.ts | BUDGET GATE in /action | CORRECTION: thesis says "enforced by the same shield"; in code it is a separate gate in the adapter, not the engine shield. Same effect: -10, state unchanged. Not in the mask (finding 3) |
| Episode end and step limit (4.5.3) | node-adapter/src/server.ts; dcr_env.py | MAX_EPISODE_STEPS; step() | OK. Default 100, runner sets 300; blocked attempts count as steps. Truncation reported as termination (finding 2) |
| PPO and hyperparameters (4.5.1-4.5.2) | dcr-gymnasium-agent/python/src/agents/train_agent.py | train(): PPO("MlpPolicy", ...) | OK. Only ent_coef and seed set; the rest are SB3 defaults = Table 4.3. WARNING: the script's own default is --ent-coef 0.01; the thesis value 0.1 comes from the runner |
| Experiment runner (5.3) | dcr-gymnasium-agent/python/scripts/run_experiments.py | experiment list + env variables | OK. DCR_XML, MAX_EPISODE_STEPS=300, STEP_PENALTY=-1.5, weights, ent_coef 0.1, 100k steps. Linux only |
| Benchmarks (5.2) | app/public/examples/diagrams/ | Expense_Report_Diaz.xml, Loan_Application_Diaz.xml, LoanApp_junior_senior.xml | OK. Note 5.2.2: the Loan log is simulated with AgentSimulator, not a real log |
| Expense Report figures (6.1) | scripts/internal_validation/ | figures_expense_report.py | Present |
| Ablation (7.1-7.2) | scripts/run_ablation.py | | Present |
| Seed study (7.3) | scripts/run_seed_robustness*.py | | Present |
| Mined graphs (8.1) | app/public/examples/diagrams/Mined_*.xml | 01, 03, 04, 05, 09, 11 | OK, all six present |
| Cost/duration policy variants (8.4) | same folder | Mined_05_*_linear / _inverse / _policyrandom | OK |
| Mined-graph ablation (8.2-8.3) | scripts/run_ablation_mined.py; external_validation/analyze_mined_ablation.py | | Present |
| Front quality vs exact solver (8.3.4) | external_validation/analyze_front_quality_metrics.py | | Present |
| Scaling diagnostic 20-100 events (8.3.2) | minizinc_ground_truth/random_dcr_generator.py; external_validation/run_scaling_rl_local.py | generate_graph(num_events, seed, density=6.0) | OK. Seeded generator, density scales with size, all events included at start. Reuse it on Wed 28 Oct |
| Exact solver (6.1, 8) | minizinc_ground_truth/ | | Needs github.com/JuanK120/dcrGraph (public) |
| Heuristics (6.2.4) | scripts/run_heuristics.py | | Present |

## Findings

### 1. The final step's cost and duration never enter the reward (CONFIRMED by a run)

Thesis Table 4.2 says it in its caption ("the terminal bonus replaces the whole step
reward"), but the consequence is not discussed. Replaying the three accepting traces of
the Expense Report through her adapter with her settings (step penalty -1.5, 300 steps):

| Weights | P1 FillOut>Approve>PayOut (cost 50, dur 10,500) | FillOut>Approve>Withdraw (cost 735, dur 9,900) | P2 FillOut>Withdraw (cost 710, dur 2,700) |
| --- | --- | --- | --- |
| alpha=1, eps=0 | 98.945 | 98.945 | 98.486 |
| alpha=0, eps=1 | 97.797 | 97.797 | 98.292 |
| alpha=0.5, eps=0.5 | 98.371 | 98.371 | 98.389 |
| alpha=0, eps=0 | 98.995 | 98.995 | 98.500 |

(discounted return, gamma 0.99; step rewards match thesis Table 6.4: FillOut -0.71,
Approve +0.50, final +100)

- PayOut (cost 15) and Withdraw (cost 700) are both final steps, so both get a flat +100.
  The two traces after Approve score identically under every weight pair, including
  the cost-focused one.
- This explains Table 6.3: the cost-focused (1,0) and unweighted (0,0) settings have a
  median cost of exactly 735 = the Withdraw ending, and reach P1 only 44% / 41% of the time.
- The thesis (9.1.2) explains it as "if duration is not part of the reward function, the
  agent has little reason to prefer one over the other". The run shows that cost is in
  the reward and still cannot separate them.
- Duration-focused settings prefer P2 because it is one step shorter (discount and step
  penalty), not because of the durations of PayOut or Withdraw.
- Consequence: any process whose most expensive or slowest activity can be the last
  step is optimised blind on that activity. A one-line fix (apply the cost and duration
  penalty on the accepting step too) is a candidate for Study 1, but changes her baseline.

### 2. Truncation is reported as termination
dcr_env.step returns truncated=False always. Statistics are fine (truncated episodes
count as non-accepting and are excluded from the front, 4.5.3); the problem is learning.
Fix for Study 1: terminated = accepting, truncated = maxStepReached (both already sent
by the adapter).

### 3. The Expert budget is not in the mask
Deliberate (comment in server.ts: "the policy must learn not to attempt Expert when
budget=0"). No effect today because nothing uses the mask. With MaskablePPO it must be
added, or masked Expert actions will still be blocked with -10 when k runs out.

### 4. The Python fallback mask uses pending events
state_to_mask in dcr_env.py. Only runs if the adapter sends no mask, which never happens
with her adapter. In the code since the first RL commit (30 Apr), so it looks like a
placeholder. It is wrong in three ways (see notes of 21 Oct) and should not be used.
