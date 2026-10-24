"""Shared helper for the special-course switch tests: start/stop one node adapter."""
import os
import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parents[1]
PYTHON_PROJECT_DIR = SCRIPTS_DIR.parent
ROOT = PYTHON_PROJECT_DIR.parents[1]
DIAGRAMS = ROOT / "app" / "public" / "examples" / "diagrams"

sys.path.insert(0, str(SCRIPTS_DIR))
sys.path.insert(0, str(PYTHON_PROJECT_DIR / "src"))
from run_local import wait_for_adapter, stop_process, adapter_answers  # noqa: E402

THESIS_ENV = {"MAX_EPISODE_STEPS": "300", "STEP_PENALTY": "-1.5"}
SWITCH_VARS = ("FIX_FALSE_ACCEPT", "BUDGET_IN_MASK", "EXPERT_BUDGET_K")


@contextmanager
def adapter(xml: Path, port: int, log_path: Path, adapter_dir: Path = ROOT / "node-adapter", **env_vars):
    """Run `node dist/server.mjs` from adapter_dir with the thesis settings plus env_vars."""
    url = f"http://localhost:{port}"
    if adapter_answers(url):
        raise RuntimeError(f"port {port} is already in use")
    env = os.environ.copy()
    for name in SWITCH_VARS:
        env.pop(name, None)
    env.update(THESIS_ENV)
    env.update({"DCR_XML": str(xml), "PORT": str(port), "COST_WEIGHT": "0", "DURATION_WEIGHT": "0"})
    env.update({k: str(v) for k, v in env_vars.items()})
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "wb") as lf:
        proc = subprocess.Popen(["node", "dist/server.mjs"], cwd=str(adapter_dir), env=env,
                                stdout=lf, stderr=subprocess.STDOUT)
        try:
            if not wait_for_adapter(url, proc, timeout=60):
                raise RuntimeError(f"adapter did not start, see {log_path}")
            yield url
        finally:
            stop_process(proc)
