"""
Test i (E1.4b partway starts): replaying a prefix gives the same marking as playing its events,
and the prefix earns nothing. On the Expense Report (has costs) and Synthetic_chain_40, for every
prefix of a known accepting trace, compared with the same events played through /action:
  same included, executed and pending sets and the same action mask; the /replay answer carries no
  reward; the next /action reports episodeSteps = 1 and only that event's cost and duration.
Also: an illegal prefix event is refused and named, /replay after the first action is refused, and
ids and labels both work. Then the environment: --partway-initial-share is respected, start_type
and prefix_len are correct in info, random prefixes are legal and within --partway-max, and the
optional time limit ends a partway episode as truncated.

    cd dcr-gymnasium-agent/python
    python scripts/special_course_tests/test_i_prefix_replay.py
"""
import shutil
import sys
import tempfile
from pathlib import Path

import requests

from _adapter import adapter, DIAGRAMS
from envs.dcr_env import DCRGymEnv
from envs.partway import load_traces

BASELINE = {"SHAPING_FIX": "1", "SHAPING_GAMMA": "1", "FIX_FALSE_ACCEPT": "1", "COST_WEIGHT": 1, "DURATION_WEIGHT": 0}
CASES = [
    ("Expense_Report_Diaz.xml", ["FillOutExpenseReport", "Approve", "PayOut"]),
    ("synthetic/Synthetic_chain_40.xml", ["Act_Event_1", "Act_Event_2", "Act_Event_3", "Act_Event_4", "Act_Event_5"]),
]


def same_marking(a, b):
    return all(set(a[k]) == set(b[k]) for k in ("included", "executed", "pending"))


def check_graph(url, trace):
    s = requests.Session()
    init = s.post(f"{url}/reset").json()
    index = {init["labelMap"].get(p["event"], p["event"]): i for i, p in enumerate(init["eventRolePairs"])}
    event_of = {init["labelMap"].get(ev, ev): ev for ev in init["events"]}
    ok = True
    for n in range(1, len(trace)):
        s.post(f"{url}/reset")
        played = {}
        for label in trace[:n]:
            played = s.post(f"{url}/action", json={"action": index[label]}).json()
        s.post(f"{url}/reset")
        replayed = s.post(f"{url}/replay", json={"events": trace[:n]}).json()
        ok &= replayed["ok"] and replayed["replayed"] == n
        ok &= same_marking(replayed["state"], played["result"]["state"])
        ok &= replayed["actionMask"] == played["actionMask"]
        ok &= "stepReward" not in replayed and "reward" not in replayed
        # ids work as well as labels
        s.post(f"{url}/reset")
        by_id = s.post(f"{url}/replay", json={"events": [event_of[e] for e in trace[:n]]}).json()
        ok &= same_marking(by_id["state"], replayed["state"])
        # the agent's first step after the prefix: nothing of the prefix is counted
        s.post(f"{url}/reset")
        s.post(f"{url}/replay", json={"events": trace[:n]})
        first = s.post(f"{url}/action", json={"action": index[trace[n]]}).json()["result"]
        ev = event_of[trace[n]]
        ok &= first["episodeSteps"] == 1
        ok &= first["episodeCost"] == init["costMap"].get(ev, first["episodeCost"])
        ok &= first["episodeDuration"] == init["durationMap"].get(ev, first["episodeDuration"])
        ok &= first["pendingBefore"] == replayed["pendingAfter"]
        ok &= first["illegalTracesCount"] == 0
        print(f"  prefix {n}: same marking and mask, no reward; next step episodeSteps={first['episodeSteps']}, "
              f"episodeCost={first['episodeCost']}, pendingBefore={first['pendingBefore']}, "
              f"reward={first['stepReward']:.4f}")
    # an illegal event is refused and named; the last trace event alone is not enabled at the start
    s.post(f"{url}/reset")
    bad = s.post(f"{url}/replay", json={"events": [trace[-1]]})
    ok &= bad.status_code == 400 and bad.json()["illegalEvent"] == trace[-1]
    # replay is only allowed before the episode's first action
    s.post(f"{url}/reset")
    s.post(f"{url}/action", json={"action": index[trace[0]]})
    late = s.post(f"{url}/replay", json={"events": [trace[1]]})
    ok &= late.status_code == 409
    print(f"  illegal prefix -> HTTP {bad.status_code}, after first action -> HTTP {late.status_code}")
    return ok


def check_env(url):
    ok = True
    trace = load_traces(graph_name="Synthetic_chain_40")[0]
    ok &= trace == CASES[1][1]
    env = DCRGymEnv(node_url=url, truncation_fix=True, fix_dead_end=True, partway_starts="trace",
                    partway_traces=[trace], partway_initial_share=0.2, partway_seed=1)
    index = {env.label_map.get(p["event"], p["event"]): i for i, p in enumerate(env.event_role_pairs)}
    n_initial = 0
    for _ in range(300):
        env.reset()
        n_initial += env.start_type == "initial"
        ok &= (env.start_type == "initial") == (env.prefix_len == 0)
        if env.start_type == "partway":
            ok &= env.prefix_len == 4  # frontier at its start value
    print(f"trace mode: {n_initial} of 300 resets at the initial marking (share 0.2)")
    ok &= 40 <= n_initial <= 80
    while True:  # finish one partway episode: info fields, success counted for the frontier
        env.reset()
        if env.start_type == "partway":
            break
    _, reward, terminated, truncated, info = env.step(index[trace[4]])
    print(f"partway episode: start_type={info['start_type']} prefix_len={info['prefix_len']} j={info['partway_j']} "
          f"reward={reward:.2f} terminated={terminated} accepting={info['accepting']} cost={info['episode_cost']}")
    ok &= info["start_type"] == "partway" and info["prefix_len"] == 4 and info["partway_j"] == 4
    ok &= terminated and not truncated and info["accepting"] and reward >= 100
    ok &= len(env.partway_frontiers[0].results) == 1 and env.partway_frontiers[0].results[0]

    env = DCRGymEnv(node_url=url, truncation_fix=True, partway_starts="random", partway_prob=1.0,
                    partway_max=6, partway_initial_share=0.0, partway_seed=2)
    lengths = []
    for _ in range(200):
        obs, _ = env.reset()
        lengths.append(env.prefix_len)
        ok &= 0 <= env.prefix_len <= 6 and any(env.action_masks())
    print(f"random mode: prefix lengths {min(lengths)}..{max(lengths)} over 200 resets")
    ok &= min(lengths) >= 1 and max(lengths) == 6

    env = DCRGymEnv(node_url=url, truncation_fix=True, partway_starts="trace", partway_traces=[trace],
                    partway_initial_share=0.0, partway_timelimit_factor=2, partway_seed=3)
    env.reset()  # one event left: limit 2 x 1 + 10 = 12 steps
    enabled = [i for i, m in enumerate(env.action_masks()) if m and i != index[trace[4]]]
    steps = 0
    while True:
        steps += 1
        _, _, terminated, truncated, info = env.step(enabled[0])
        if terminated or truncated:
            break
    print(f"time limit: ended after {steps} steps, truncated={truncated}, flagged={info.get('partway_timelimit')}")
    ok &= steps == 12 and truncated and not terminated and info.get("partway_timelimit") is True
    return ok


def main():
    tmp = Path(tempfile.mkdtemp(prefix="testi_"))
    ok = True
    for graph, trace in CASES:
        print(graph)
        with adapter(DIAGRAMS / graph, 5020, tmp / "adapter.log", **BASELINE) as url:
            ok &= check_graph(url, trace)
    with adapter(DIAGRAMS / CASES[1][0], 5020, tmp / "adapter2.log", **BASELINE) as url:
        ok &= check_env(url)
    shutil.rmtree(tmp, ignore_errors=True)
    print("PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
