"""
Test o (E1.5 exact optimum, node-adapter/scripts/exact_optimum.ts) against known results:
1. Expense Report: w = 0 gives the fast path P2 (cost 710, duration 2,700), w = 1 the cheap path P1 (50, 10,500)
   (Diaz et al.; the switch lies between w = 0.5 and 0.6, METHOD_NOTES 3 Nov).
2. LoanApp without roles: the single Pareto point of the MiniZinc result in the repo (1,560, 816).
3. Mined 05: the two MiniZinc Pareto points (335, 410) and (339, 387) (tobias_sweep_results.json).
4. LoanApp with roles: cheapest (1,092, 1,632) and fastest (3,120, 408), as found by every RL seed (4 Nov).
5. Synthetic_chain_20 with --default-duration 1, w = 0.5: a 5-step trace (E1.3 L_min), objective 2.5.

    cd dcr-gymnasium-agent/python
    python scripts/special_course_tests/test_o_exact_optimum.py
"""
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

PY_DIR = Path(__file__).resolve().parents[2]
ROOT = PY_DIR.parents[1]
DIAGRAMS = ROOT / "app" / "public" / "examples" / "diagrams"
MZ = PY_DIR / "minizinc_ground_truth"


def exact(graph, *extra):
    out = Path(tempfile.mkdtemp(prefix="testo_")) / "out.json"
    npx = shutil.which("npx") or "npx"
    subprocess.run([npx, "tsx", "scripts/exact_optimum.ts", str(graph), str(out), *extra],
                   cwd=ROOT / "node-adapter", capture_output=True, text=True, check=True)
    result = json.loads(out.read_text())
    shutil.rmtree(out.parent, ignore_errors=True)
    return result


def points(result):
    return sorted((round(p["cost"], 6), round(p["duration"], 6)) for p in result["supportedFront"])


def main():
    ok = True
    r = exact(DIAGRAMS / "Expense_Report_Diaz.xml")
    o = {x["w"]: (x["cost"], x["duration"]) for x in r["optima"]}
    good = o[0] == (710, 2700) and o[1] == (50, 10500)
    print(f"Expense: w=0 {o[0]}, w=0.5 {o[0.5]}, w=1 {o[1]}: {'ok' if good else 'WRONG'}")
    ok &= good

    mz = json.loads((MZ / "loanapp_ground_truth_result.json").read_text())["pareto_front"]
    r = exact(DIAGRAMS / "LoanApp_junior_senior.xml", "--no-roles")
    good = points(r) == sorted((p["cost"], p["duration"]) for p in mz)
    print(f"LoanApp, no roles: {points(r)} vs MiniZinc {[(p['cost'], p['duration']) for p in mz]}: {'ok' if good else 'WRONG'}")
    ok &= good

    tobias = json.loads((MZ / "tobias_sweep_results.json").read_text())
    mz05 = next(x for x in tobias if x["graph"].startswith("05"))["pareto_front"]
    r = exact(DIAGRAMS / "Mined_05_SyntheticReviewLarge.xml")
    good = points(r) == sorted((p["cost"], p["duration"]) for p in mz05)
    print(f"Mined 05: {points(r)} vs MiniZinc {sorted((p['cost'], p['duration']) for p in mz05)}: {'ok' if good else 'WRONG'}")
    ok &= good

    r = exact(DIAGRAMS / "LoanApp_junior_senior.xml")
    o = {x["w"]: (round(x["cost"], 6), x["duration"]) for x in r["optima"]}
    good = o[0] == (3120, 408) and o[1] == (1092, 1632)
    print(f"LoanApp with roles: fastest {o[0]}, cheapest {o[1]}: {'ok' if good else 'WRONG'}")
    ok &= good

    r = exact(DIAGRAMS / "synthetic" / "Synthetic_chain_20.xml", "--w", "0.5", "--default-duration", "1")
    x = r["optima"][0]
    good = x["steps"] == 5 and abs(x["objective"] - 2.5) < 1e-9
    print(f"chain 20, default duration 1, w = 0.5: {x['steps']} steps, objective {x['objective']}: {'ok' if good else 'WRONG'}")
    ok &= good
    print("PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
