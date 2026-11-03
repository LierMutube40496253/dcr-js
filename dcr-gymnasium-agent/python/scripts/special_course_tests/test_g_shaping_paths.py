"""
Test g (SHAPING_FIX): on the Expense Report with COST_WEIGHT=1, DURATION_WEIGHT=0 and
SHAPING_GAMMA=1, the three accepting paths are replayed and their total rewards checked:
  FillOut > Approve > PayOut     = 100 - 50/700
  FillOut > Withdraw             = 100 - 710/700
  FillOut > Approve > Withdraw   = 100 - 735/700
and the progress terms of each path must add up to 0. With SHAPING_NO_PROGRESS=1 every
progress term is 0 and the totals are the same.
Dead end (graph of test e: A makes B pending, B can never fire): the step into the dead-end
marking takes Phi = 0, so its progress term is 0 instead of -2 * gamma, also with gamma 0.99.

    cd dcr-gymnasium-agent/python
    python scripts/special_course_tests/test_g_shaping_paths.py
"""
import shutil
import sys
import tempfile
from pathlib import Path

import requests

from _adapter import adapter, DIAGRAMS
from test_e_dead_end import DEAD_END_XML

PATHS = {
    "FillOut > Approve > PayOut": (["FillOutExpenseReport", "Approve", "PayOut"], 50),
    "FillOut > Withdraw": (["FillOutExpenseReport", "WithdrawExpenseReport"], 710),
    "FillOut > Approve > Withdraw": (["FillOutExpenseReport", "Approve", "WithdrawExpenseReport"], 735),
}


def play(url, labels):
    s = requests.Session()
    init = s.post(f"{url}/reset").json()
    by_label = {init["labelMap"].get(p["event"], p["event"]): i for i, p in enumerate(init["eventRolePairs"])}
    return [s.post(f"{url}/action", json={"action": by_label[lab]}).json()["result"] for lab in labels]


def main():
    tmp = Path(tempfile.mkdtemp(prefix="testg_"))
    xml = DIAGRAMS / "Expense_Report_Diaz.xml"
    ok = True
    for no_progress in (False, True):
        extra = {"SHAPING_NO_PROGRESS": "1"} if no_progress else {}
        with adapter(xml, 5017, tmp / f"adapter_np{int(no_progress)}.log", COST_WEIGHT=1, DURATION_WEIGHT=0,
                     SHAPING_FIX="1", SHAPING_GAMMA="1", **extra) as url:
            print("without progress term" if no_progress else "with progress term")
            for name, (labels, cost) in PATHS.items():
                rows = play(url, labels)
                total = sum(r["stepReward"] for r in rows)
                progress = [r["progressDelta"] for r in rows]
                expected = 100 - cost / 700
                ok &= all(r["actionCompliant"] and not r["accepting"] for r in rows[:-1])
                ok &= bool(rows[-1]["accepting"] and rows[-1]["done"] and rows[-1]["pendingAfter"] == 0)
                ok &= abs(total - expected) < 1e-9 and abs(sum(progress)) < 1e-9
                ok &= rows[-1]["episodeCost"] == cost
                if no_progress:
                    ok &= all(p == 0 for p in progress)
                print(f"  {name:30s} total {total:.6f} (expected {expected:.6f}), "
                      f"progress terms {progress}, steps {[round(r['stepReward'], 6) for r in rows]}")

    dead_xml = tmp / "dead_end.xml"
    dead_xml.write_text(DEAD_END_XML, encoding="utf-8")
    for gamma in ("1", "0.99"):
        with adapter(dead_xml, 5018, tmp / f"adapter_dead_g{gamma}.log", COST_WEIGHT=1, DURATION_WEIGHT=0,
                     SHAPING_FIX="1", SHAPING_GAMMA=gamma) as url:
            r = play(url, ["A"])[0]
            ok &= r["actionCompliant"] and not r["accepting"] and r["pendingAfter"] == 1
            ok &= r["progressDelta"] == 0 and abs(r["stepReward"] - (-1)) < 1e-9
            print(f"dead end, gamma {gamma}: pending after A {r['pendingAfter']}, progress {r['progressDelta']}, "
                  f"reward {r['stepReward']} (expected -1: cost 1/1, Phi(dead end) = 0)")
    shutil.rmtree(tmp, ignore_errors=True)
    print("PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
