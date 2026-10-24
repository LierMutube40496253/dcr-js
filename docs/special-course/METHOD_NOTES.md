# Method notes for implementation (special course)

Short, implementation-level notes from the reading list, added when an experiment starts.
Full paper notes: project doc "Special Course Reading Guide and Paper Notes" (Part D).

## Action masking (E1.4) - sources: Huang & Ontanon [2], valid-action suppression paper [14], CACM shields [4]

What to implement
- Mask = the engine's enabled set, per (event, role) action. With an Expert budget, Expert
  actions are also masked when the budget is used up (issue B1).
- Masking must be applied BOTH when sampling an action AND when computing the policy gradient:
  invalid logits are replaced by a very large negative number, so their probability is ~0 and
  their gradient is 0. This is a valid policy gradient (Huang & Ontanon, Proposition 1).
- Do NOT do "naive masking" (mask only when sampling, update with the unmasked distribution):
  in Huang & Ontanon the KL divergence explodes and learning is much slower.
- sb3-contrib MaskablePPO does it correctly: it stores the masks in the rollout buffer and
  uses them in the update. The environment must expose action_masks().
- Evaluate with the mask as well (model.predict(obs, action_masks=...)). Removing the mask
  after training degrades performance (Huang & Ontanon).

Pitfalls specific to DCR
- Dead ends: a marking can be non-accepting with no enabled event (E1.3 CSV has a dead-end
  count per graph). An all-false mask crashes MaskablePPO (NaN probabilities). Detect it and end
  the episode (truncated, not accepting), and count how often it happens.
- Fake acceptance (A3) must be fixed first: a masked agent makes every step legal and reaches
  the engine's 100-legal-step limit quickly.
- The -10 interception and the illegal-attempt rate disappear with masking: report "steps to
  first real acceptance" and real vs fake acceptance instead.

What to measure (besides the E0 protocol)
- Real acceptance rate (no included event pending), final 20% of training, per graph size.
- Steps until the first real acceptance and until 95% real acceptance.
- Episode length in legal steps vs the shortest finish L_min (E1.3).
- Optional diagnostic for the unmasked runs ([14]): probability mass the policy puts on enabled
  actions at markings it has rarely visited. If it stays low, valid-action suppression explains
  part of the failure on large graphs.

Predictions from E1.3 to check
- Graph 09: unmasked random pick legal 2.6%; masked random walk finishes 100%. Masking should
  turn 0% into high acceptance.
- Synthetic chains: masked random walk finishes 8.8% (20 events), 0.17% (40), 0% (60+).
  Masking alone may not be enough from 40 events; if so, the next step is starting episodes
  partway through (shared with Study 2, E2.3).

## Later sections (add when the experiment starts)
- E1.5 other algorithms: DQN/Double DQN with masked argmax and masked targets; tabular
  Q-learning only on Expense (17 states), Loan (358), graph 05 (324).
- E1.6 state abstraction: Ahmetoglu et al. [19] (group by enabled set, refine by transition
  error); Process Pilot [7] action-set and structural abstraction.
- E1.7 factored actions: Huang & Ontanon factored design; Tang et al. [20] linear Q decomposition
  (biased because role multipliers make cost a product).
