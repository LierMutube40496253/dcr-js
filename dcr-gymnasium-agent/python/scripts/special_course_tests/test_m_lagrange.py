"""
Test m (E1.8 Lagrangian): the multiplier update and its use as the violation price.
1. lagrange_update: lambda <- max(0, lambda + lr x violations per step); no violations leaves it; no steps
   leaves it; it never goes below 0.
2. A 4,096-step PPO run on the Expense Report with --shield-off --violation-fix --lagrange: the lagrange log
   has the start value and one row per rollout, each lambda follows the rule from the logged counts, and every
   step of a rollout used the lambda set at the end of the previous one (trace column violation_price).

    cd dcr-gymnasium-agent/python
    python scripts/special_course_tests/test_m_lagrange.py
"""
import csv
import glob
import subprocess
import sys
from pathlib import Path

PY_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PY_DIR / "src"))
from agents.lagrange import lagrange_update  # noqa: E402

DIAGRAMS = PY_DIR.parents[1] / "app" / "public" / "examples" / "diagrams"
LOGS, TB, MODELS = PY_DIR / "scripts" / "logs", PY_DIR / "dcr_tensorboard", PY_DIR / "models"
BASE = "TESTM"


def main():
    ok = True
    ok &= lagrange_update(1.0, 3.0, 10, 100) == 1.3
    ok &= lagrange_update(1.0, 3.0, 0, 100) == 1.0 and lagrange_update(1.0, 3.0, 5, 0) == 1.0
    ok &= lagrange_update(0.0, 3.0, 0, 100, target=0.5) == 0.0
    print(f"update rule: {'ok' if ok else 'wrong'}")

    for f in glob.glob(str(LOGS / f"*{BASE}_*")) + glob.glob(str(TB / f"*{BASE}_*")) + glob.glob(str(MODELS / f"*{BASE}_*")):
        Path(f).unlink() if Path(f).is_file() else None
    cmd = [sys.executable, "scripts/run_local.py", "--graph", str(DIAGRAMS / "Expense_Report_Diaz.xml"),
           "--base", BASE, "--pairs", "0.5,0.5", "--seeds", "1", "--steps", "4096", "--port", "5024",
           "--shaping-fix", "--gamma", "1.0", "--fix-false-accept", "--truncation-fix", "--fix-dead-end",
           "--shield-off", "--violation-fix", "--lagrange"]
    run = subprocess.run(cmd, cwd=PY_DIR, capture_output=True, text=True)
    ok &= run.returncode == 0 and "[OK]" in run.stdout
    run_id = f"{BASE}_fixA2A3D1_shapeFix_g1p0_shieldOff_violFix_lagI1L3_s1_a0p5_b0p5_kunlim"
    lag = glob.glob(str(LOGS / f"lagrange_exp_{run_id}_*.csv"))
    trace = glob.glob(str(LOGS / f"train_trace_exp_{run_id}_*.csv"))
    if len(lag) != 1 or len(trace) != 1:
        print(f"run failed or logs missing ({len(lag)} lagrange, {len(trace)} trace); see the run log")
        print(run.stdout[-1500:])
        sys.exit(1)
    rows = list(csv.DictReader(open(lag[0], newline="")))
    lam, follows = float(rows[0]["lambda"]), True
    for r in rows[1:]:
        lam = lagrange_update(lam, 3.0, int(r["violations"]), int(r["steps"]))
        follows &= abs(lam - float(r["lambda"])) < 1e-5
    ok &= follows and len(rows) == 3 and float(rows[0]["lambda"]) == 1.0
    print(f"lagrange log: {[(r['global_timestep'], r['violations'], r['steps'], r['lambda']) for r in rows]}, "
          f"follows the rule: {follows}")
    steps = list(csv.DictReader(open(trace[0], newline="")))
    prices = [round(float(s["violation_price"]), 5) for s in steps]
    want = [round(float(rows[0]["lambda"]), 5)] * 2048 + [round(float(rows[1]["lambda"]), 5)] * 2048
    ok &= prices == want
    violations = sum(s["violation"] == "True" for s in steps)
    ok &= violations == sum(int(r["violations"]) for r in rows[1:]) and violations > 0
    print(f"trace: price per step matches the multiplier of the rollout: {prices == want}; {violations} violations")

    for f in glob.glob(str(LOGS / f"*{BASE}_*")) + glob.glob(str(MODELS / f"*{BASE}_*")):
        Path(f).unlink()
    for d in glob.glob(str(TB / f"*{BASE}_*")):
        import shutil
        shutil.rmtree(d, ignore_errors=True)
    print("PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
