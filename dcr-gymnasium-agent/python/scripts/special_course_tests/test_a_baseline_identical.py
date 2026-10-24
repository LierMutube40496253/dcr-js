"""
Test a: with every switch off, the new code reproduces the reference commit exactly.

Runs 2,048 steps on the Expense Report, pair (1,0), seed 1, twice through run_local.py:
once with the reference commit (default fe01d1c = her code + the HTTP transport fix),
extracted with `git archive` into a temporary folder, and once with the working tree.
The two train_trace CSVs must be identical row for row. All outputs are deleted afterwards.

    cd dcr-gymnasium-agent/python
    python scripts/special_course_tests/test_a_baseline_identical.py [--ref fe01d1c]
"""
import argparse
import csv
import io
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

from _adapter import ROOT, PYTHON_PROJECT_DIR, DIAGRAMS

PARTS = ["node-adapter/dist", "dcr-gymnasium-agent/python/src", "dcr-gymnasium-agent/python/scripts/run_local.py"]


def link_dir(link: Path, target: Path):
    if os.name == "nt":  # directory junction: no admin rights needed
        subprocess.run(["cmd", "/c", "mklink", "/J", str(link), str(target)], check=True, capture_output=True)
    else:
        link.symlink_to(target, target_is_directory=True)


def run(tree: Path, base: str, port: int):
    py_dir = tree / "dcr-gymnasium-agent" / "python"
    cmd = [sys.executable, str(py_dir / "scripts" / "run_local.py"),
           "--graph", str(DIAGRAMS / "Expense_Report_Diaz.xml"), "--base", base,
           "--pairs", "1,0", "--seeds", "1", "--steps", "2048", "--port", str(port)]
    subprocess.run(cmd, cwd=str(py_dir), check=True)
    traces = sorted((py_dir / "scripts" / "logs").glob(f"train_trace_exp_{base}_s1_*.csv"))
    assert len(traces) == 1, traces
    return traces[0]


def cleanup_new(base: str):
    for pattern in (f"scripts/logs/*{base}_*", f"models/*{base}_*", f"dcr_tensorboard/*{base}_*"):
        for p in PYTHON_PROJECT_DIR.glob(pattern):
            shutil.rmtree(p) if p.is_dir() else p.unlink()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", default="fe01d1c")
    ap.add_argument("--port", type=int, default=5011)
    args = ap.parse_args()

    tmp = Path(tempfile.mkdtemp(prefix="testa_ref_"))
    links = []
    try:
        tar = subprocess.run(["git", "archive", "--format=tar", args.ref, *PARTS],
                             cwd=str(ROOT), check=True, capture_output=True).stdout
        tarfile.open(fileobj=io.BytesIO(tar)).extractall(tmp, filter="data")
        for sub in ("node-adapter/node_modules", "node_modules"):
            if (ROOT / sub).exists():
                link = tmp / sub
                link_dir(link, ROOT / sub)
                links.append(link)

        ref_csv = run(tmp, "TESTA_ref", args.port)
        new_csv = run(ROOT, "TESTA_new", args.port)
        ref_rows = list(csv.reader(open(ref_csv, newline="")))
        new_rows = list(csv.reader(open(new_csv, newline="")))
        print(f"ref ({args.ref}): {len(ref_rows) - 1} rows   new: {len(new_rows) - 1} rows")
        diffs = [i for i, (a, b) in enumerate(zip(ref_rows, new_rows)) if a != b]
        if len(ref_rows) != len(new_rows) or diffs:
            print(f"FAIL: {len(diffs)} differing rows, first at row {diffs[0] if diffs else 'n/a'}")
            if diffs:
                print("  ref:", ref_rows[diffs[0]]); print("  new:", new_rows[diffs[0]])
            sys.exit(1)
        print("PASS: train_trace CSVs identical row for row")
    finally:
        for link in links:  # remove the junctions/symlinks first, never their targets
            os.rmdir(link) if os.name == "nt" else link.unlink()
        shutil.rmtree(tmp, ignore_errors=True)
        cleanup_new("TESTA_new")


if __name__ == "__main__":
    main()
