"""
Test n (E1.5 algorithms, all masked):
1. Units: masked_max takes the max over valid actions only and gives 0 when none is valid; the masked
   replay buffer returns, with each sampled transition, the mask stored for that transition.
2. A 2,000-step run of each algorithm on the Expense Report (w = 0.5, baseline flags) through run_local.py:
   no forbidden action is ever chosen; masked A2C saved with the A2C settings (n_steps 5, 1 epoch, batch 5,
   gae_lambda 1, no advantage normalisation, lr 7e-4, RMSprop); masked DQN saved with gamma 1 and its
   settings; tabular Q visited at most the 17 reachable markings of the Expense Report (E1.3).
   The run files are deleted afterwards.

    cd dcr-gymnasium-agent/python
    python scripts/special_course_tests/test_n_algorithms.py
"""
import csv
import glob
import json
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import numpy as np
import torch as th
from gymnasium import spaces

PY_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PY_DIR / "src"))
from agents.masked_dqn import MaskedReplayBuffer, masked_max  # noqa: E402

DIAGRAMS = PY_DIR.parents[1] / "app" / "public" / "examples" / "diagrams"
LOGS, TB, MODELS = PY_DIR / "scripts" / "logs", PY_DIR / "dcr_tensorboard", PY_DIR / "models"
BASE = "TESTN"


def units():
    ok = True
    q = th.tensor([[1.0, 5.0, 3.0], [2.0, 9.0, 4.0], [7.0, 8.0, 9.0]])
    masks = th.tensor([[True, False, True], [False, False, False], [False, True, False]])
    ok &= masked_max(q, masks).tolist() == [3.0, 0.0, 8.0]
    buf = MaskedReplayBuffer(16, spaces.Box(0, 1, (3,), np.int8), spaces.Discrete(4), device="cpu")
    for i in range(10):
        mask = [j <= i % 4 for j in range(4)]
        buf.add(np.full((1, 3), i % 2, np.int8), np.zeros((1, 3), np.int8), np.array([[i % 4]]),
                np.array([float(i)]), np.array([False]), [{"action_mask": mask}])
    data, m = buf.sample_with_masks(64)
    rewards = data.rewards.flatten().int().tolist()
    ok &= all(m[k].tolist() == [j <= r % 4 for j in range(4)] for k, r in enumerate(rewards))
    print(f"units: masked max and stored masks {'ok' if ok else 'WRONG'}")
    return ok


def run(algo):
    cmd = [sys.executable, "scripts/run_local.py", "--graph", str(DIAGRAMS / "Expense_Report_Diaz.xml"),
           "--base", BASE, "--pairs", "0.5,0.5", "--seeds", "1", "--steps", "2000", "--port", "5025",
           "--algo", algo, "--shaping-fix", "--gamma", "1.0", "--fix-false-accept", "--truncation-fix", "--fix-dead-end"]
    r = subprocess.run(cmd, cwd=PY_DIR, capture_output=True, text=True)
    if r.returncode != 0 or "[OK]" not in r.stdout:
        print(r.stdout[-1500:], r.stderr[-1500:])
        return None, None
    tag = {"maskable_a2c": "maskedA2C", "maskable_dqn": "maskedDQN", "tabular_q": "tabQ"}[algo]
    run_id = f"{BASE}_{tag}_fixA2A3D1_shapeFix_g1p0_s1_a0p5_b0p5_kunlim"
    trace = glob.glob(str(LOGS / f"train_trace_exp_{run_id}_*.csv"))
    model = glob.glob(str(MODELS / f"{algo}_exp_{run_id}_*"))
    return (list(csv.DictReader(open(trace[0], newline=""))) if len(trace) == 1 else None,
            model[0] if len(model) == 1 else None)


def cleanup():
    for f in glob.glob(str(LOGS / f"*{BASE}_*")) + glob.glob(str(MODELS / f"*{BASE}_*")):
        Path(f).unlink()
    for d in glob.glob(str(TB / f"*{BASE}_*")):
        shutil.rmtree(d, ignore_errors=True)


def main():
    ok = units()
    cleanup()
    for algo in ("maskable_a2c", "maskable_dqn", "tabular_q"):
        rows, model = run(algo)
        if rows is None or model is None:
            print(f"{algo}: run failed or files missing")
            ok = False
            continue
        illegal = sum(r["action_compliant"] != "True" for r in rows)
        good = len(rows) == 2000 and illegal == 0
        if algo == "tabular_q":
            states = json.load(open(model))["states"]
            good &= 0 < states <= 17
            detail = f"{states} states"
        else:
            data = json.loads(zipfile.ZipFile(model).read("data"))
            if algo == "maskable_a2c":
                lr = data["learning_rate"]
                good &= (data["n_steps"], data["n_epochs"], data["batch_size"], data["gae_lambda"],
                         data["normalize_advantage"], data["gamma"], lr) == (5, 1, 5, 1.0, False, 1.0, 0.0007)
                good &= "RMSprop" in json.dumps(data["policy_kwargs"])
                detail = f"n_steps {data['n_steps']}, epochs {data['n_epochs']}, batch {data['batch_size']}, lr {lr}"
            else:
                good &= (data["gamma"], data["target_update_interval"], data["buffer_size"],
                         data["learning_starts"]) == (1.0, 1000, 100000, 1000)
                detail = f"gamma {data['gamma']}, target every {data['target_update_interval']}"
        print(f"{algo}: {len(rows)} steps, {illegal} forbidden actions, {detail}: {'ok' if good else 'WRONG'}")
        ok &= good
    cleanup()
    print("PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
