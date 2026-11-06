"""
Test k (DEFAULT_DURATION): events with no duration get duration X; nothing else changes.
1. Synthetic_chain_20 (no durations), off: no durations, a demonstration's reward has no duration term.
2. Same, X = 1 and X = 7, weights (0.5, 0.5), SHAPING_FIX, gamma 1: every step pays 0.5 more than with
   the switch off, for any X (durations are divided by the largest in the graph), and the case's duration
   is 5 x X.
3. Graph with stated durations, one removed (Expense Report): only that event gets X, the others and
   the largest duration stay.
4. Loan_Application_Diaz (a stated duration of 0): nothing changes, the 0 stays 0.
5. /load applies it too. X = 0 stops the adapter at start-up.

    cd dcr-gymnasium-agent/python
    python scripts/special_course_tests/test_k_default_duration.py
"""
import re
import shutil
import sys
import tempfile
from pathlib import Path

import requests

from _adapter import adapter, DIAGRAMS

CHAIN = DIAGRAMS / "synthetic" / "Synthetic_chain_20.xml"
TRACE = ["Act_Event_1", "Act_Event_17", "Act_Event_2", "Act_Event_11", "Act_Event_12"]
WEIGHTS = dict(COST_WEIGHT=0.5, DURATION_WEIGHT=0.5, SHAPING_FIX="1", SHAPING_GAMMA="1", FIX_FALSE_ACCEPT="1")


def play(url):
    """Rewards and last answer of the demonstration on chain 20."""
    s = requests.Session()
    init = s.post(f"{url}/reset").json()
    index = {init["labelMap"][p["event"]]: i for i, p in enumerate(init["eventRolePairs"])}
    rewards, body = [], None
    for label in TRACE:
        body = s.post(f"{url}/action", json={"action": index[label]}).json()["result"]
        rewards.append(body["stepReward"])
    return init, rewards, body


def main():
    tmp = Path(tempfile.mkdtemp(prefix="testk_"))
    ok = True

    with adapter(CHAIN, 5021, tmp / "off.log", **WEIGHTS) as url:
        init, off, last = play(url)
        ok &= init["durationMap"] == {} and last["accepting"] and last["episodeDuration"] == 0
        print(f"off: durations {init['durationMap']}, rewards {off}, case duration {last['episodeDuration']}")

    for x in (1, 7):
        with adapter(CHAIN, 5021, tmp / f"x{x}.log", DEFAULT_DURATION=x, **WEIGHTS) as url:
            init, on, last = play(url)
            diffs = [round(a - b, 6) for a, b in zip(off, on)]
            ok &= set(init["durationMap"].values()) == {x} and len(init["durationMap"]) == 20
            ok &= diffs == [0.5] * 5 and last["accepting"] and last["episodeDuration"] == 5 * x
            print(f"X = {x}: 20 events get {x}, rewards {on}, each 0.5 lower: {diffs == [0.5] * 5}, "
                  f"case duration {last['episodeDuration']}")

    xml = (DIAGRAMS / "Expense_Report_Diaz.xml").read_text(encoding="utf-8")
    stripped, n = re.subn(r'(<event id="E_approve">.*?)<duration>[^<]*</duration>', r"\1", xml, count=1, flags=re.S)
    partial = tmp / "expense_one_missing.xml"
    partial.write_text(stripped, encoding="utf-8")
    with adapter(DIAGRAMS / "Expense_Report_Diaz.xml", 5021, tmp / "exp.log") as url:
        stated = requests.post(f"{url}/reset").json()["durationMap"]
    with adapter(partial, 5021, tmp / "exp_x.log", DEFAULT_DURATION=11) as url:
        got = requests.post(f"{url}/reset").json()["durationMap"]
        loaded = requests.post(f"{url}/load", json={"xml": CHAIN.read_text(encoding="utf-8")}).json()["durationMap"]
    want = dict(stated, E_approve=11)
    ok &= n == 1 and got == want and max(got.values()) == max(stated.values())
    ok &= set(loaded.values()) == {11} and len(loaded) == 20
    print(f"Expense, E_approve's duration removed, X = 11: E_approve {got['E_approve']} (stated {stated['E_approve']}), "
          f"others unchanged: {got == want}; /load of chain 20: {len(loaded)} events get {set(loaded.values())}")

    loan = DIAGRAMS / "Loan_Application_Diaz.xml"
    with adapter(loan, 5021, tmp / "loan.log") as url:
        stated = requests.post(f"{url}/reset").json()["durationMap"]
    with adapter(loan, 5021, tmp / "loan_x.log", DEFAULT_DURATION=11) as url:
        got = requests.post(f"{url}/reset").json()["durationMap"]
    ok &= got == stated and 0 in got.values()
    print(f"Loan_Application_Diaz, X = 11: unchanged: {got == stated}; stated zeros stay: {0 in got.values()}")

    try:
        with adapter(CHAIN, 5021, tmp / "zero.log", DEFAULT_DURATION=0):
            ok = False
            print("X = 0: adapter started, it should not")
    except RuntimeError:
        print("X = 0: adapter stops at start-up")

    shutil.rmtree(tmp, ignore_errors=True)
    print("PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
