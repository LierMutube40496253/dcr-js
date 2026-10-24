"""
Test e (dead ends, masked runs only): a non-accepting marking with no allowed action ends the
episode as truncated (not accepting, no +100), is logged in the trace CSV and counted per run.

Forced with a 3-event graph: A makes B pending and excludes itself; B and C are each a
condition for the other, so neither can ever fire. After A: nothing enabled, B pending and
included -> dead end at step 1.
  dead_end_truncate on (what --algo maskable_ppo uses): truncated=True, terminated=False,
      accepting=False, info["dead_end"]=True; a short MaskablePPO run does not crash, every
      episode is a dead end, and the trace has a dead_end column with the count printed.
  off (her env): the episode does not end, and action_masks() raises on the all-zero mask.

    cd dcr-gymnasium-agent/python
    python scripts/special_course_tests/test_e_dead_end.py
"""
import csv
import shutil
import sys
import tempfile
from pathlib import Path

from _adapter import adapter
from agents.train_agent import train
from envs.dcr_env import DCRGymEnv

DEAD_END_XML = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<dcrgraph title="Dead-end test graph">
  <specification>
    <resources>
      <events>
        <event id="A"><cost>1</cost><duration>1</duration></event>
        <event id="B"><cost>1</cost><duration>1</duration></event>
        <event id="C"><cost>1</cost><duration>1</duration></event>
      </events>
      <labels><label id="A"/><label id="B"/><label id="C"/></labels>
      <labelMappings>
        <labelMapping eventId="A" labelId="A"/>
        <labelMapping eventId="B" labelId="B"/>
        <labelMapping eventId="C" labelId="C"/>
      </labelMappings>
    </resources>
    <constraints>
      <conditions>
        <condition sourceId="C" targetId="B"/>
        <condition sourceId="B" targetId="C"/>
      </conditions>
      <responses><response sourceId="A" targetId="B"/></responses>
      <excludes><exclude sourceId="A" targetId="A"/></excludes>
      <includes></includes>
    </constraints>
  </specification>
</dcrgraph>
"""


def main():
    tmp = Path(tempfile.mkdtemp(prefix="teste_"))
    xml = tmp / "dead_end.xml"
    xml.write_text(DEAD_END_XML, encoding="utf-8")
    ok = True
    with adapter(xml, 5015, tmp / "adapter.log") as url:
        # 1) env with dead-end handling
        env = DCRGymEnv(node_url=url, dead_end_truncate=True)
        env.reset()
        mask0 = env.action_masks().tolist()
        idx_a = next(i for i, p in enumerate(env.event_role_pairs) if p["event"] == "A")
        _, reward, terminated, truncated, info = env.step(idx_a)
        print(f"on:  initial mask {mask0}; after A: terminated={terminated} truncated={truncated} "
              f"accepting={info['accepting']} dead_end={info['dead_end']} reward={reward} "
              f"mask={env._adapter_mask}")
        ok &= mask0 == [True, False, False] and truncated and not terminated
        ok &= not info["accepting"] and info["dead_end"] and reward != 100

        # 2) her env: the episode just goes on, and masking cannot continue
        env = DCRGymEnv(node_url=url)
        env.reset()
        _, _, terminated, truncated, info = env.step(idx_a)
        try:
            env.action_masks()
            raised = False
        except RuntimeError:
            raised = True
        print(f"off: after A: terminated={terminated} truncated={truncated} "
              f"dead_end in info={'dead_end' in info}; action_masks() raises={raised}")
        ok &= not terminated and not truncated and "dead_end" not in info and raised

        # 3) a short MaskablePPO run on the graph: no crash, every episode a dead end, logged
        train(node_url=url, exp_id="TESTE", total_timesteps=256, out_dir=tmp / "models",
              ent_coef=0.1, tb_log_dir=tmp / "tb", seed=1, algo="maskable_ppo")
        trace = next((tmp / "scripts" / "logs").glob("train_trace_exp_TESTE_*.csv"))
        rows = list(csv.DictReader(open(trace, newline="")))
        done_rows = [r for r in rows if r["done"] == "True"]
        dead = [r for r in done_rows if r["dead_end"] == "True"]
        print(f"MaskablePPO: {len(rows)} steps, {len(done_rows)} episodes, {len(dead)} dead ends, "
              f"accepting episodes {sum(r['accepting'] == 'True' for r in done_rows)}, "
              f"'dead_end' column present={'dead_end' in rows[0]}")
        ok &= len(rows) > 0 and len(dead) == len(done_rows) == len(rows)
        ok &= not any(r["accepting"] == "True" for r in rows)
    shutil.rmtree(tmp, ignore_errors=True)
    print("PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
