"""
Test f (issue A1): the accepting step's reward on LoanApp_junior_senior, COST_WEIGHT=1,
DURATION_WEIGHT=0.5. A seeded random walk under the action mask is played until it really
accepts (no pending included events left); then the same prefix is replayed and the final
event is executed once with each role.
  without FIX_FINAL_STEP: the accepting step is a flat +100 for every role, no cost or
                          duration penalty (her behaviour)
  with FIX_FINAL_STEP=1:  the accepting step is 1 + progress + repetition - cost - duration
                          + STEP_PENALTY + 100, cost and duration use the role multipliers
                          (so the roles get different rewards); every earlier step is
                          unchanged

    cd dcr-gymnasium-agent/python
    python scripts/special_course_tests/test_f_final_step.py
"""
import random
import shutil
import sys
import tempfile
from pathlib import Path

import requests

from _adapter import adapter, DIAGRAMS, THESIS_ENV

ALPHA, EPS = 1.0, 0.5
STEP_PENALTY = float(THESIS_ENV["STEP_PENALTY"])


def find_accepting_walk(url, seed=1, max_steps=200):
    """Action indices of a masked random walk that ends in a real accepting marking."""
    rng = random.Random(seed)
    s = requests.Session()
    for _ in range(100):
        mask = s.post(f"{url}/reset").json()["actionMask"]
        actions = []
        for _ in range(max_steps):
            a = rng.choice([i for i, m in enumerate(mask) if m])
            body = s.post(f"{url}/action", json={"action": a}).json()
            actions.append(a)
            if body["result"]["accepting"]:
                if body["result"]["pendingAfter"] == 0:
                    return actions
                break
            mask = body["actionMask"]
    raise RuntimeError("no accepting walk found")


def play(url, prefix, final):
    s = requests.Session()
    init = s.post(f"{url}/reset").json()
    rows = [s.post(f"{url}/action", json={"action": a}).json()["result"] for a in prefix + [final]]
    return init, rows


def main():
    xml = DIAGRAMS / "LoanApp_junior_senior.xml"
    log_dir = Path(tempfile.mkdtemp(prefix="testf_"))
    weights = {"COST_WEIGHT": ALPHA, "DURATION_WEIGHT": EPS}
    results = {}
    for fix in (False, True):
        extra = {"FIX_FINAL_STEP": "1"} if fix else {}
        with adapter(xml, 5016, log_dir / f"adapter_fix{int(fix)}.log", **weights, **extra) as url:
            if not fix:
                walk = find_accepting_walk(url)
                init = requests.post(f"{url}/reset").json()
                pairs = init["eventRolePairs"]
                final_event = pairs[walk[-1]]["event"]
                finals = [i for i, p in enumerate(pairs) if p["event"] == final_event]
            results[fix] = {i: play(url, walk[:-1], i) for i in finals}

    max_cost = max(v for v in init["costMap"].values() if v > 0)
    max_dur = max(v for v in init["durationMap"].values() if v > 0)
    print(f"walk of {len(walk)} steps; final event {init['labelMap'].get(final_event, final_event)}, "
          f"base cost {init['costMap'].get(final_event)}, duration {init['durationMap'].get(final_event)}")
    ok = True
    for i in finals:
        role = pairs[i]["role"]
        mult = init["roleMultipliers"][role]
        _, off = results[False][i]
        _, on = results[True][i]
        f_off, f_on = off[-1], on[-1]
        ok &= all(r["accepting"] is False for r in off[:-1] + on[:-1])
        ok &= [r["stepReward"] for r in off[:-1]] == [r["stepReward"] for r in on[:-1]]
        ok &= f_off["accepting"] and f_off["stepReward"] == 100 and f_off["costPenalty"] == 0
        cost = init["costMap"][final_event] * mult["costMultiplier"]
        dur = init["durationMap"][final_event] * mult["durationMultiplier"]
        cost_pen, dur_pen = ALPHA * cost / max_cost, EPS * dur / max_dur
        expected = 1 + f_on["progressDelta"] + f_on["noveltyDelta"] - cost_pen - dur_pen + STEP_PENALTY + 100
        ok &= bool(f_on["accepting"] and f_on["done"] and f_on["baseMapped"] == 100)
        ok &= abs(f_on["costPenalty"] - cost_pen) < 1e-9 and abs(f_on["durationPenalty"] - dur_pen) < 1e-9
        ok &= abs(f_on["stepReward"] - expected) < 1e-9
        print(f"role {role:8s}: off {f_off['stepReward']:.4f} | on {f_on['stepReward']:.4f} "
              f"(expected {expected:.4f}; progress {f_on['progressDelta']}, repetition "
              f"{f_on['noveltyDelta']}, cost -{cost_pen:.4f}, duration -{dur_pen:.4f}, step {STEP_PENALTY})")
    on_finals = {round(results[True][i][1][-1]["stepReward"], 9) for i in finals}
    ok &= len(on_finals) == len(finals)  # roles priced differently
    shutil.rmtree(log_dir, ignore_errors=True)
    print("PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
