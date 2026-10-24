"""Cross-platform (Windows/Linux/macOS) replacement for run_experiments.py.

Runs one adapter + PPO training per (seed, alpha, epsilon) with the thesis
settings, without bash/nvm/lsof. Experiment ids match run_experiments.py so
the analysis scripts parse them:

    <base>_s<seed>_a<alpha>_b<eps>_<kunlim|k<k>>   e.g. ExpenseReport_Diaz_s1_a1p0_b0p0_kunlim

Example (from anywhere):
    .venv\\Scripts\\python scripts\\run_local.py \\
        --graph ..\\..\\app\\public\\examples\\diagrams\\Expense_Report_Diaz.xml \\
        --base ExpenseReport_Diaz
"""
import argparse
import os
import shutil
import subprocess
import sys
import time
import urllib.request
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]  # repo root
NODE_ADAPTER_DIR = ROOT / "node-adapter"
PYTHON_PROJECT_DIR = Path(__file__).resolve().parents[1]
LOGS_DIR = PYTHON_PROJECT_DIR / "scripts" / "logs"
TB_DIR = PYTHON_PROJECT_DIR / "dcr_tensorboard"  # train_agent.py default


def parse_pairs(raw: str) -> list[tuple[float, float]]:
    pairs = []
    for chunk in raw.split(";"):
        chunk = chunk.strip()
        if not chunk:
            continue
        alpha, eps = chunk.split(",")
        pairs.append((float(alpha), float(eps)))
    return pairs


def parse_seeds(raw: str) -> list[int]:
    return [int(s.strip()) for s in raw.split(",") if s.strip()]


def switch_tag(args) -> str:
    """Short tag for the special-course switches, '' when all are off (her baseline).
    e.g. masked_fixA2A3B1 = MaskablePPO + truncation fix + false-accept fix + budget in mask."""
    parts = ["masked"] if args.algo == "maskable_ppo" else []
    fixes = "".join(code for code, on in (("A2", args.truncation_fix), ("A3", args.fix_false_accept),
                                          ("B1", args.budget_in_mask)) if on)
    if fixes:
        parts.append(f"fix{fixes}")
    return "_".join(parts)


def make_exp_id(base: str, seed: int, alpha: float, eps: float, k, tag: str = "") -> str:
    # Same format as run_experiments.py; the switch tag goes into the base name
    weight_tag = f"a{float(alpha)}_b{float(eps)}".replace(".", "p")
    k_tag = "kunlim" if k is None else f"k{k}"
    base = f"{base}_{tag}" if tag else base
    return f"{base}_s{seed}_{weight_tag}_{k_tag}"


def adapter_answers(url: str) -> bool:
    try:
        with urllib.request.urlopen(f"{url}/state", timeout=2) as r:
            return r.status == 200
    except Exception:
        return False


def wait_for_adapter(url: str, proc: subprocess.Popen, timeout: float = 30) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if proc.poll() is not None:
            return False  # adapter exited early
        if adapter_answers(url):
            return True
        time.sleep(0.5)
    return False


def stop_process(proc: subprocess.Popen):
    if proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()


def run_one(args, xml: Path, seed: int, alpha: float, eps: float) -> bool:
    exp_id = make_exp_id(args.base, seed, alpha, eps, args.k, switch_tag(args))
    node_url = f"http://localhost:{args.port}"
    # With --fast-io the chatty run log and TensorBoard files are written on a fast local
    # disk and copied into the repo afterwards (WSL: many small writes to /mnt/c are ~10x slower)
    work_dir = args.fast_io if args.fast_io else LOGS_DIR
    run_log = work_dir / f"run_{exp_id}.log"
    tb_dir = args.fast_io / "dcr_tensorboard" if args.fast_io else None

    start = datetime.now()
    print(f"[RUN] {exp_id} | start {start:%Y-%m-%d %H:%M:%S} | alpha={alpha} eps={eps} "
          f"seed={seed} steps={args.steps}", flush=True)

    if adapter_answers(node_url):
        print(f"[ERROR] Port {args.port} is already in use by another adapter; skipping {exp_id}",
              flush=True)
        return False

    env = os.environ.copy()
    env["DCR_XML"] = str(xml)
    env["PORT"] = str(args.port)
    env["MAX_EPISODE_STEPS"] = str(args.max_steps)
    env["STEP_PENALTY"] = str(args.step_penalty)
    env["COST_WEIGHT"] = str(alpha)
    env["DURATION_WEIGHT"] = str(eps)
    env.pop("EXPERT_BUDGET_K", None)
    if args.k is not None:
        env["EXPERT_BUDGET_K"] = str(args.k)
    # Special-course adapter switches: set only when on, so off = her environment exactly
    for name, on in (("FIX_FALSE_ACCEPT", args.fix_false_accept), ("BUDGET_IN_MASK", args.budget_in_mask)):
        env.pop(name, None)
        if on:
            env[name] = "1"
    # Unbuffered UTF-8 output so the log is live and Windows code pages cannot crash prints
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"

    ok = False
    run_log.write_bytes(b"")
    with open(run_log, "ab") as lf:
        lf.write(f"=== {exp_id} started {start.isoformat()} ===\n".encode())
        lf.flush()
        proc = subprocess.Popen(
            ["node", "dist/server.mjs"],
            cwd=str(NODE_ADAPTER_DIR),
            env=env,
            stdout=lf,
            stderr=subprocess.STDOUT,
        )
        try:
            if not wait_for_adapter(node_url, proc, timeout=30):
                print(f"[ERROR] Node adapter did not answer on {node_url}/state within 30 s. "
                      f"See {run_log}", flush=True)
            else:
                train_cmd = [
                    sys.executable,
                    "src/agents/train_agent.py",
                    "--exp-id", exp_id,
                    "--steps", str(args.steps),
                    "--ent-coef", str(args.ent_coef),
                    "--seed", str(seed),
                    "--node-url", node_url,
                ]
                if args.fast_io:
                    train_cmd += ["--tb-log-dir", str(tb_dir)]
                if args.algo != "ppo":
                    train_cmd += ["--algo", args.algo]
                if args.truncation_fix:
                    train_cmd += ["--truncation-fix"]
                print("Launching trainer:", " ".join(train_cmd), flush=True)
                ret = subprocess.run(
                    train_cmd,
                    cwd=str(PYTHON_PROJECT_DIR),
                    env=env,
                    stdout=lf,
                    stderr=subprocess.STDOUT,
                )
                if ret.returncode != 0:
                    print(f"[ERROR] Trainer failed (code {ret.returncode}). See {run_log}", flush=True)
                else:
                    ok = True
        except Exception as e:
            print(f"[ERROR] {exp_id}: {e}", flush=True)
        finally:
            stop_process(proc)

    if args.fast_io:
        shutil.copy2(run_log, LOGS_DIR / run_log.name)
        for d in tb_dir.glob(f"exp_{exp_id}_*"):
            shutil.copytree(d, TB_DIR / d.name, dirs_exist_ok=True)
        run_log = LOGS_DIR / run_log.name

    end = datetime.now()
    status = "OK" if ok else "FAILED"
    print(f"[{status}] {exp_id} | start {start:%H:%M:%S} | end {end:%H:%M:%S} | "
          f"duration {end - start} | log {run_log}", flush=True)
    return ok


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--graph", type=Path, required=True, help="path to the DCR XML")
    parser.add_argument("--base", required=True, help="experiment id prefix, e.g. ExpenseReport_Diaz")
    parser.add_argument("--pairs", default="0,0;1,0;0,1;0.5,0.5;2,0.5;0.5,2",
                        help='weight pairs "alpha,eps;alpha,eps;..."')
    parser.add_argument("--seeds", default="1", help='comma-separated seeds, e.g. "1,2,3"')
    parser.add_argument("--steps", type=int, default=100000)
    parser.add_argument("--ent-coef", type=float, default=0.1)
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--step-penalty", type=float, default=-1.5)
    parser.add_argument("--port", type=int, default=5001)
    parser.add_argument("--k", type=int, default=None, help="Expert budget (omit = unlimited)")
    parser.add_argument("--fast-io", type=Path, default=None,
                        help="write run logs and TensorBoard here first, copy into the repo "
                             "after each run (use a Linux path such as ~/dcr_io under WSL)")
    # Special-course switches, all off by default (= Sofía's baseline); see
    # docs/special-course/CODE_ISSUES_AND_CHANGES.md. Any switch on adds a tag to the run id.
    parser.add_argument("--algo", choices=["ppo", "maskable_ppo"], default="ppo")
    parser.add_argument("--fix-false-accept", action="store_true",
                        help="issue A3: adapter FIX_FALSE_ACCEPT=1")
    parser.add_argument("--truncation-fix", action="store_true",
                        help="issue A2: step limit reported as truncated")
    parser.add_argument("--budget-in-mask", action="store_true",
                        help="issue B1: adapter BUDGET_IN_MASK=1")
    args = parser.parse_args()

    if args.fast_io:
        args.fast_io = args.fast_io.expanduser().resolve()
        args.fast_io.mkdir(parents=True, exist_ok=True)
    xml = args.graph.expanduser().resolve()
    if not xml.exists():
        sys.exit(f"XML not found: {xml}")
    pairs = parse_pairs(args.pairs)
    seeds = parse_seeds(args.seeds)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Repository root: {ROOT}")
    print(f"Graph: {xml}")
    print(f"Runs: {len(seeds) * len(pairs)} (seeds {seeds} x pairs {pairs})", flush=True)

    results = []
    sweep_start = datetime.now()
    for seed in seeds:
        for alpha, eps in pairs:
            ok = run_one(args, xml, seed, alpha, eps)
            results.append((make_exp_id(args.base, seed, alpha, eps, args.k, switch_tag(args)), ok))
            time.sleep(2)

    print(f"\n=== Summary ({datetime.now() - sweep_start} total) ===")
    for exp_id, ok in results:
        print(f"{'OK    ' if ok else 'FAILED'} {exp_id}")
    sys.exit(0 if all(ok for _, ok in results) else 1)


if __name__ == "__main__":
    main()
