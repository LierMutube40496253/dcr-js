"""
Test d (issue A2): an episode that hits MAX_EPISODE_STEPS (300) without acceptance.
  with --truncation-fix:    truncated=True, terminated=False, and SB3's DummyVecEnv sets
                            info["TimeLimit.truncated"]=True, which PPO and MaskablePPO use
                            to bootstrap the value of the cut-off state
  without (her behaviour):  terminated=True, truncated=False

The episode loops FillOutExpenseReport on the Expense Report with FIX_FALSE_ACCEPT=1, so
the engine's 100-step limit (A3) does not end it early.

    cd dcr-gymnasium-agent/python
    python scripts/special_course_tests/test_d_truncation.py
"""
import inspect
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.on_policy_algorithm import OnPolicyAlgorithm
from stable_baselines3.common.vec_env import DummyVecEnv
from sb3_contrib.ppo_mask import MaskablePPO

from _adapter import adapter, DIAGRAMS
from envs.dcr_env import DCRGymEnv


def run_episode(url, truncation_fix):
    venv = DummyVecEnv([lambda: Monitor(DCRGymEnv(node_url=url, truncation_fix=truncation_fix))])
    raw = venv.envs[0].env
    idx = next(i for i, p in enumerate(raw.event_role_pairs)
               if raw.label_map.get(p["event"]) == "FillOutExpenseReport")
    venv.reset()
    raw.reset()
    # 1) the env itself
    for t in range(1, 301):
        _, _, terminated, truncated, info = raw.step(idx)
        if terminated or truncated:
            break
    env_result = (t, terminated, truncated, info["accepting"])
    # 2) what SB3 sees through DummyVecEnv
    venv.reset()
    for t in range(1, 301):
        _, _, dones, infos = venv.step(np.array([idx]))
        if dones[0]:
            break
    vec_result = (t, bool(infos[0].get("TimeLimit.truncated", False)), "terminal_observation" in infos[0])
    return env_result, vec_result


def main():
    xml = DIAGRAMS / "Expense_Report_Diaz.xml"
    log_dir = Path(tempfile.mkdtemp(prefix="testd_"))
    ok = True
    with adapter(xml, 5014, log_dir / "adapter.log", FIX_FALSE_ACCEPT="1") as url:
        for fix in (True, False):
            (t, term, trunc, acc), (vt, tl_trunc, has_term_obs) = run_episode(url, fix)
            print(f"--truncation-fix={fix}: env ends at step {t}: terminated={term} truncated={trunc} "
                  f"accepting={acc} | DummyVecEnv done at step {vt}: TimeLimit.truncated={tl_trunc} "
                  f"terminal_observation={has_term_obs}")
            if fix:
                ok &= t == 300 and trunc and not term and not acc and vt == 300 and tl_trunc and has_term_obs
            else:
                ok &= t == 300 and term and not trunc and vt == 300 and not tl_trunc
    for cls in (OnPolicyAlgorithm, MaskablePPO):
        src = inspect.getsource(cls.collect_rollouts)
        uses = 'infos[idx].get("TimeLimit.truncated", False)' in src and "terminal_value" in src
        print(f"{cls.__name__}.collect_rollouts bootstraps on TimeLimit.truncated: {uses}")
        ok &= uses
    shutil.rmtree(log_dir, ignore_errors=True)
    print("PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
