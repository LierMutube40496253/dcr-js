"""
Test c (masking): train MaskablePPO briefly through train_agent.train(algo="maskable_ppo"),
then run 1,000 episodes with the trained policy, sampling (not argmax) under the adapter's
mask. Passes when no step during training or evaluation is dcr_illegal or budget_block.

    cd dcr-gymnasium-agent/python
    python scripts/special_course_tests/test_c_masking.py --graph Expense_Report_Diaz.xml
    python scripts/special_course_tests/test_c_masking.py --graph LoanApp_junior_senior.xml --k 1 --budget-in-mask
Without --budget-in-mask and with --k 1, budget_block steps are expected (issue B1); run it
with --expect-blocks to check that contrast.
"""
import argparse
import csv
import shutil
import sys
import tempfile
from collections import Counter
from pathlib import Path

from sb3_contrib import MaskablePPO

from _adapter import adapter, DIAGRAMS
from agents.train_agent import train
from envs.dcr_env import DCRGymEnv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--graph", required=True, help="file name in app/public/examples/diagrams")
    ap.add_argument("--k", type=int, default=None, help="EXPERT_BUDGET_K")
    ap.add_argument("--budget-in-mask", action="store_true")
    ap.add_argument("--steps", type=int, default=10_000)
    ap.add_argument("--episodes", type=int, default=1_000)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--port", type=int, default=5013)
    ap.add_argument("--expect-blocks", action="store_true",
                    help="invert the check: expect budget_block > 0 (BUDGET_IN_MASK off, k set)")
    args = ap.parse_args()

    extra = {}
    if args.k is not None:
        extra["EXPERT_BUDGET_K"] = str(args.k)
    if args.budget_in_mask:
        extra["BUDGET_IN_MASK"] = "1"
    tmp = Path(tempfile.mkdtemp(prefix="testc_"))
    with adapter(DIAGRAMS / args.graph, args.port, tmp / "adapter.log", **extra) as url:
        model_path = train(node_url=url, exp_id="TESTC", total_timesteps=args.steps,
                           out_dir=tmp / "models", ent_coef=0.1, tb_log_dir=tmp / "tb",
                           seed=args.seed, algo="maskable_ppo")
        trace = next((tmp / "scripts" / "logs").glob("train_trace_exp_TESTC_*.csv"))
        rows = list(csv.DictReader(open(trace, newline="")))
        train_counts = Counter(r["intercepted_reason"] or "legal" for r in rows)

        env = DCRGymEnv(node_url=url)
        model = MaskablePPO.load(str(model_path))
        eval_counts, accepted, steps = Counter(), 0, 0
        for _ in range(args.episodes):
            obs, _ = env.reset()
            while True:
                action, _ = model.predict(obs, action_masks=env.action_masks(), deterministic=False)
                obs, _, terminated, truncated, info = env.step(action)
                steps += 1
                eval_counts[info["intercepted_reason"] or "legal"] += 1
                if terminated or truncated:
                    accepted += bool(info["accepting"])
                    break

    print(f"{args.graph} k={args.k} BUDGET_IN_MASK={int(args.budget_in_mask)}")
    print(f"  training {len(rows)} steps: {dict(train_counts)}")
    print(f"  evaluation {args.episodes} episodes, {steps} steps: {dict(eval_counts)}, "
          f"accepting {accepted}/{args.episodes}")
    bad = sum(c.get("dcr_illegal", 0) for c in (train_counts, eval_counts))
    blocks = sum(c.get("budget_block", 0) for c in (train_counts, eval_counts))
    ok = bad == 0 and (blocks > 0 if args.expect_blocks else blocks == 0)
    shutil.rmtree(tmp, ignore_errors=True)
    print("PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
