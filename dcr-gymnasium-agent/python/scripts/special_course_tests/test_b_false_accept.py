"""
Test b (issue A3): execute FillOutExpenseReport 105 times on the Expense Report.
  without FIX_FALSE_ACCEPT: step 100 is reported accepting with +100 although events are
                            still pending (her behaviour, the engine's 100-legal-step limit)
  with FIX_FALSE_ACCEPT=1:  no step is accepting, none gets +100, the episode does not end

    cd dcr-gymnasium-agent/python
    python scripts/special_course_tests/test_b_false_accept.py
"""
import shutil
import sys
import tempfile
from pathlib import Path

import requests

from _adapter import adapter, DIAGRAMS

N_STEPS = 105


def fill_out_105(url):
    s = requests.Session()
    init = s.post(f"{url}/reset").json()
    idx = next(i for i, p in enumerate(init["eventRolePairs"])
               if init["labelMap"].get(p["event"]) == "FillOutExpenseReport")
    steps = []
    for t in range(1, N_STEPS + 1):
        r = s.post(f"{url}/action", json={"action": idx}).json()["result"]
        steps.append((t, r["accepting"], r["stepReward"], r["done"], r["pendingAfter"], r["interceptedReason"]))
    return steps


def main():
    xml = DIAGRAMS / "Expense_Report_Diaz.xml"
    log_dir = Path(tempfile.mkdtemp(prefix="testb_"))
    ok = True
    for fix in (False, True):
        extra = {"FIX_FALSE_ACCEPT": "1"} if fix else {}
        with adapter(xml, 5012, log_dir / f"adapter_fix{int(fix)}.log", **extra) as url:
            steps = fill_out_105(url)
        assert all(s[5] is None for s in steps), "FillOut was blocked; the test needs 105 legal steps"
        acc = [s for s in steps if s[1]]
        plus100 = [s for s in steps if s[2] == 100]
        done = [s for s in steps if s[3]]
        print(f"FIX_FALSE_ACCEPT={int(fix)}: accepting at steps {[s[0] for s in acc][:6]}"
              f"{' ...' if len(acc) > 6 else ''}, +100 at {[s[0] for s in plus100][:6]}, "
              f"done at {[s[0] for s in done][:6]}; step 100 pendingAfter={steps[99][4]}")
        if fix:
            ok &= not acc and not plus100 and not done
        else:
            ok &= steps[99][1] is True and steps[99][2] == 100 and not any(s[1] for s in steps[:99])
    shutil.rmtree(log_dir, ignore_errors=True)
    print("PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
