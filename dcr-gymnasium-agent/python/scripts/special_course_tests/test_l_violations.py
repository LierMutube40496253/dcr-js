"""
Test l (VIOLATION_FIX, E1.8): with the shield off (SHIELD_DISABLED=1) and the new reward, weights (0.5, 0.5),
gamma 1, on the Expense Report:
1. Switch off (her RL-only path): paying out at the start is executed, ends accepting, reward a flat -10.
2. Switch on: the same violation is priced like a legal step: 100 + progress - cost/maxCost x 0.5
   - duration/maxDuration x 0.5 - 10, labelled "violation"; one that creates a pending event pays the
   progress term (-2). A legal step's reward is the same with the switch on and off.
3. /violation_price 2.5 makes the same violation 7.5 better; a negative price is refused (400); without the
   switch the route answers 409.
4. LoanApp, Expert budget 2: an Expert violation uses one unit with the switch on, none with it off.
5. VIOLATION_FIX without SHIELD_DISABLED stops the adapter at start-up.

    cd dcr-gymnasium-agent/python
    python scripts/special_course_tests/test_l_violations.py
"""
import shutil
import sys
import tempfile
from pathlib import Path

import requests

from _adapter import adapter, DIAGRAMS

BASE = dict(SHIELD_DISABLED="1", SHAPING_FIX="1", SHAPING_GAMMA="1", FIX_FALSE_ACCEPT="1",
            COST_WEIGHT=0.5, DURATION_WEIGHT=0.5)
PORT = 5023


def first(url, label, role=""):
    """Reset, then play one action by label (and role); return the /reset answer and the step result."""
    s = requests.Session()
    init = s.post(f"{url}/reset").json()
    index = next(i for i, p in enumerate(init["eventRolePairs"])
                 if init["labelMap"][p["event"]] == label and p["role"] == role)
    return init, s.post(f"{url}/action", json={"action": index}).json()["result"]


def expected(init, label, progress, accepting, price):
    ev = next(e for e, l in init["labelMap"].items() if l == label)
    max_cost = max(v for v in init["costMap"].values() if v > 0)
    max_dur = max(v for v in init["durationMap"].values() if v > 0)
    return progress - 0.5 * init["costMap"][ev] / max_cost - 0.5 * init["durationMap"][ev] / max_dur \
        + (100 if accepting else 0) - price


def main():
    tmp = Path(tempfile.mkdtemp(prefix="testl_"))
    ok = True
    expense = DIAGRAMS / "Expense_Report_Diaz.xml"

    with adapter(expense, PORT, tmp / "off.log", **BASE) as url:
        _, r = first(url, "PayOut")
        ok &= r["stepReward"] == -10 and r["accepting"] and r["interceptedReason"] == "dcr_illegal"
        print(f"off: PayOut at the start: reward {r['stepReward']}, accepting {r['accepting']}, reason {r['interceptedReason']}")
        _, legal_off = first(url, "FillOutExpenseReport")
        ok &= requests.post(f"{url}/violation_price", json={"price": 1}).status_code == 409

    with adapter(expense, PORT, tmp / "on.log", VIOLATION_FIX="1", **BASE) as url:
        init, r = first(url, "PayOut")
        want = expected(init, "PayOut", 0, True, 10)
        ok &= abs(r["stepReward"] - want) < 1e-9 and r["accepting"] and r["done"]
        ok &= r["violation"] is True and r["interceptedReason"] == "violation" and r["actionCompliant"] is False
        print(f"on: PayOut at the start: reward {r['stepReward']:.6f} (expected {want:.6f}), accepting {r['accepting']}, "
              f"reason {r['interceptedReason']}, episode cost {r['episodeCost']}")
        init, r = first(url, "Reject")
        want = expected(init, "Reject", -2, False, 10)
        ok &= abs(r["stepReward"] - want) < 1e-9 and r["progressDelta"] == -2 and not r["accepting"]
        print(f"on: Reject at the start: reward {r['stepReward']:.6f} (expected {want:.6f}), progress {r['progressDelta']}, "
              f"pending {r['pendingBefore']}->{r['pendingAfter']}")
        _, legal_on = first(url, "FillOutExpenseReport")
        ok &= legal_on["stepReward"] == legal_off["stepReward"] and legal_on["violation"] is False
        print(f"legal FillOut: reward {legal_on['stepReward']} on, {legal_off['stepReward']} off")
        ok &= requests.post(f"{url}/violation_price", json={"price": 2.5}).json()["violationPrice"] == 2.5
        _, r2 = first(url, "PayOut")
        ok &= abs(r2["stepReward"] - (want_pay := expected(init, "PayOut", 0, True, 2.5))) < 1e-9 and r2["violationPrice"] == 2.5
        print(f"price 2.5: PayOut reward {r2['stepReward']:.6f} (expected {want_pay:.6f})")
        ok &= requests.post(f"{url}/violation_price", json={"price": -1}).status_code == 400

    loan = DIAGRAMS / "LoanApp_junior_senior.xml"
    for fix in (False, True):
        extra = {"VIOLATION_FIX": "1"} if fix else {}
        with adapter(loan, PORT, tmp / f"loan_{fix}.log", EXPERT_BUDGET_K=2, **BASE, **extra) as url:
            _, r = first(url, "Approve application", "Expert")
            want = 1 if fix else 2
            ok &= r["expertBudgetRemaining"] == want and r["actionCompliant"] is False
            print(f"LoanApp, budget 2, Expert violation, switch {'on' if fix else 'off'}: budget left {r['expertBudgetRemaining']}")

    try:
        with adapter(expense, PORT, tmp / "bad.log", VIOLATION_FIX="1", SHAPING_FIX="1"):
            ok = False
            print("VIOLATION_FIX without SHIELD_DISABLED: adapter started, it should not")
    except RuntimeError:
        print("VIOLATION_FIX without SHIELD_DISABLED: adapter stops at start-up")

    shutil.rmtree(tmp, ignore_errors=True)
    print("PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
