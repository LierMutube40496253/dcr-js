"""
Test h (SHAPING_FIX): on LoanApp_junior_senior with COST_WEIGHT=1, DURATION_WEIGHT=0 (and
SHAPING_GAMMA 0.99, the default), the final Assess loan risk step must give a higher reward with
Junior (cost x 0.7) than with Expert (cost x 2.0). A seeded random walk under the action mask
is played until it really accepts with Assess loan risk; the prefix is replayed and the final
event executed once with each role. Each final reward must equal
    gamma * 0 - Phi(before) - cost x multiplier / maxCost + 100,  Phi = -2 x included pending.

    cd dcr-gymnasium-agent/python
    python scripts/special_course_tests/test_h_shaping_roles.py
"""
import random
import shutil
import sys
import tempfile
from pathlib import Path

import requests

from _adapter import adapter, DIAGRAMS

FINAL_LABEL = "assess loan risk"


def find_walk(url, pairs, labels, max_steps=200):
    """Action indices of a masked random walk that really accepts with FINAL_LABEL."""
    s = requests.Session()
    for seed in range(1, 500):
        rng = random.Random(seed)
        mask = s.post(f"{url}/reset").json()["actionMask"]
        actions = []
        for _ in range(max_steps):
            a = rng.choice([i for i, m in enumerate(mask) if m])
            body = s.post(f"{url}/action", json={"action": a}).json()
            actions.append(a)
            if body["result"]["accepting"] or body["result"]["done"]:
                ev = pairs[a]["event"]
                if body["result"]["pendingAfter"] == 0 and labels.get(ev, ev).strip().lower() == FINAL_LABEL:
                    return seed, actions
                break
            mask = body["actionMask"]
    raise RuntimeError("no walk ending with Assess loan risk found")


def main():
    tmp = Path(tempfile.mkdtemp(prefix="testh_"))
    ok = True
    with adapter(DIAGRAMS / "LoanApp_junior_senior.xml", 5019, tmp / "adapter.log",
                 COST_WEIGHT=1, DURATION_WEIGHT=0, SHAPING_FIX="1") as url:
        init = requests.post(f"{url}/reset").json()
        pairs, labels = init["eventRolePairs"], init["labelMap"]
        seed, walk = find_walk(url, pairs, labels)
        final_event = pairs[walk[-1]]["event"]
        max_cost = max(v for v in init["costMap"].values() if v > 0)
        rewards = {}
        for i, p in enumerate(pairs):
            if p["event"] != final_event:
                continue
            s = requests.Session()
            s.post(f"{url}/reset")
            rows = [s.post(f"{url}/action", json={"action": a}).json()["result"] for a in walk[:-1] + [i]]
            f = rows[-1]
            cost_pen = init["costMap"][final_event] * init["roleMultipliers"][p["role"]]["costMultiplier"] / max_cost
            expected = 0.99 * 0 + 2 * f["pendingBefore"] - cost_pen + 100
            ok &= not any(r["accepting"] for r in rows[:-1])
            ok &= bool(f["accepting"] and f["pendingAfter"] == 0 and f["noveltyDelta"] == 0)
            ok &= abs(f["costPenalty"] - cost_pen) < 1e-9 and abs(f["stepReward"] - expected) < 1e-9
            rewards[p["role"]] = f["stepReward"]
            print(f"role {p['role']:7s}: final reward {f['stepReward']:.6f} (expected {expected:.6f}; "
                  f"pending before {f['pendingBefore']}, progress {f['progressDelta']}, cost -{cost_pen:.6f})")
    print(f"walk seed {seed}, {len(walk)} steps, final event {labels.get(final_event, final_event)}, "
          f"base cost {init['costMap'][final_event]}, maxCost {max_cost}")
    ok &= rewards["Junior"] > rewards["Expert"]
    shutil.rmtree(tmp, ignore_errors=True)
    print("PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
