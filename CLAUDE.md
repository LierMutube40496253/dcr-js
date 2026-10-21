# Special Course context (branch special-course)

This is a fork of Sofía Ortiz-Arce Prada's master thesis code ("Multi-objective Optimisation
of Business Processes via Reinforcement Learning: A Shielded Reinforcement Learning Approach
over Declarative DCR Models", DTU 2026). Her final code is branch `feature/cost-roles`,
commit dbd7c49. This branch, `special-course`, is where the DTU special course work happens.
Supervisor: Hugo A. López (DTU Compute).

## Goal of the special course
- Study 1: measure where her PPO agent stops scaling (works at 42 actions, collapses between
  42 and 60), add action masking, compare other algorithms (MaskablePPO, DQN, A2C, tabular
  where feasible), state abstraction and factored actions, always keeping 100% valid traces.
- Study 2: a next-activity recommender that starts from a running case (a prefix of events)
  instead of the initial marking, under the same DCR guarantees.
- First step: reproduce her results unchanged (Expense Report, Loan Application).

## How the code is organised
- `dcr-engine/src/` (TypeScript): DCR semantics (`executionEngine.ts`), RL environment and the
  interception shield (`generation.ts`, class `RLDCREnvironment`), reward (`reward.ts`).
- `node-adapter/src/server.ts`: HTTP server on port 5001 used by the Python agent. Decodes
  action index -> (event, role), applies role multipliers, step penalty, step limit, Expert
  budget gate, calls the reward. Run with `node dist/server.mjs` (prebuilt bundle).
- `dcr-gymnasium-agent/python/src/envs/dcr_env.py`: Gymnasium environment (observation = 3
  bits per event: included, executed, pending).
- `dcr-gymnasium-agent/python/src/agents/train_agent.py`: PPO training (SB3 defaults).
- `dcr-gymnasium-agent/python/scripts/run_experiments.py`: her batch runner. Linux/HPC only
  (bash, nvm, lsof). On Windows use `scripts/run_local.py` (ours).
- Graphs: `app/public/examples/diagrams/` (Expense_Report_Diaz.xml, LoanApp_junior_senior.xml,
  Mined_*.xml).

## Settings that must match the thesis (otherwise results differ)
Engine environment variables: MAX_EPISODE_STEPS=300, STEP_PENALTY=-1.5, COST_WEIGHT=alpha,
DURATION_WEIGHT=epsilon, EXPERT_BUDGET_K=k (unset = unlimited), DCR_XML=absolute path.
Training: --steps 100000 --ent-coef 0.1 --seed N. Weight pairs (alpha, epsilon): (0,0), (1,0),
(0,1), (0.5,0.5), (2,0.5), (0.5,2). Run ids: `<Base>_s<seed>_a<alpha>_b<eps>_k<k|unlim>` with
dots as `p` (e.g. ExpenseReport_Diaz_s1_a1p0_b0p0_kunlim) so her analysis scripts parse them.
Statistics use the accepting episodes of the final 20% of training.

## Known issues in her code (details: docs/special-course/CODE_ISSUES_AND_CHANGES.md)
- A1: the step that reaches acceptance gets a flat +100; its cost and duration are ignored.
- A2: hitting the step limit is returned as terminated, never truncated.
- A3: after 100 legal steps the engine's internal limit marks the episode accepting (+100)
  even with events pending.
- B1: the Expert budget is not reflected in the action mask.
- The mask is computed every step but never used by the agent (her design: interception).
Do not fix these silently: every fix goes behind a switch so her behaviour stays the default
baseline, and is noted in docs/special-course/.

## Working rules
- Do not change her code's behaviour without a switch that keeps her version as default.
- Do not commit or push without asking.
- Windows machine: Python 3.12 venv in `dcr-gymnasium-agent/python/.venv`, Node 24, Yarn 1.
- AGENTS.md is a symlink to this file.
