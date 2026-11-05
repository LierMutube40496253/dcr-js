"""
Test j (E1.4b trace mode): the frontier moves back on success and forward on failure, with a fake
success stream (no adapter, no training), on a trace of 5 events (j starts at 4):
  20 successes at j -> j - 1, and the window is emptied; 19 are not enough; a rate of exactly 0.8
  moves back, 0.75 does not; 20 failures -> j + 1 only if j is below its start value; a rate of
  exactly 0.15 moves forward, 0.2 does not; episodes started at another prefix length are ignored;
  j never goes below 0 or above its start value. The start is j + k, capped at len(trace) - 1, with
  P(k = 0) = 0.5. --partway-step N: j moves N events per stage in both directions, stopping at 0
  and at its start value. Then the same through the environment, on Synthetic_chain_40: 20 partway
  episodes finished by the demonstration move j from 4 to 3 (step 1) or from 4 to 2 (step 2).

    cd dcr-gymnasium-agent/python
    python scripts/special_course_tests/test_j_frontier.py
"""
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

from _adapter import adapter, DIAGRAMS
from envs.dcr_env import DCRGymEnv
from envs.partway import TraceFrontier, load_traces


def feed(frontier, stream, prefix_len=None):
    """Record a stream of outcomes at the current frontier; returns the moves."""
    moves = []
    for success in stream:
        move = frontier.record(frontier.j if prefix_len is None else prefix_len, success)
        if move:
            moves.append(move)
    return moves


def check_logic():
    ok = True
    f = TraceFrontier(5)
    ok &= f.j == 4 and f.start == 4

    ok &= feed(f, [True] * 19) == [] and f.j == 4              # 19 successes: window not full
    ok &= feed(f, [True]) == [(4, 3, 1.0)] and f.j == 3        # the 20th moves j back
    ok &= len(f.results) == 0                                  # window emptied after a move
    ok &= feed(f, [True] * 20, prefix_len=4) == [] and f.j == 3  # episodes not exactly at j are ignored
    print(f"20 successes: j 4 -> 3; other prefix lengths ignored; j = {f.j}")

    ok &= feed(f, [True] * 16 + [False] * 4) == [(3, 2, 0.8)] and f.j == 2  # exactly 0.8 moves back
    ok &= feed(f, [True] * 15 + [False] * 5) == [] and f.j == 2             # 0.75 does not
    print(f"rate 0.8 moves back, 0.75 does not; j = {f.j}")

    f = TraceFrontier(5)
    f.j = 2
    ok &= feed(f, [False] * 20) == [(2, 3, 0.0)] and f.j == 3  # failures move j forward
    ok &= feed(f, [True] * 3 + [False] * 17) == [(3, 4, 0.15)] and f.j == 4  # exactly 0.15 moves forward
    ok &= feed(f, [False] * 40) == [] and f.j == 4             # never above the start value
    print(f"20 failures: j 2 -> 3; rate 0.15: j 3 -> 4; at the start value failures do not move it; j = {f.j}")

    f = TraceFrontier(5)
    f.j = 2
    ok &= feed(f, [True] * 4 + [False] * 16) == [] and f.j == 2  # 0.2 does not move forward
    f.j = 0
    f.results.clear()
    ok &= feed(f, [True] * 60) == [] and f.j == 0              # never below 0
    print(f"rate 0.2 does not move forward; at j = 0 successes do not move it; j = {f.j}")

    # A full walk: successes down to 0, a run of failures back up by one, successes again
    f = TraceFrontier(5)
    moves = feed(f, [True] * 80)
    ok &= [m[1] for m in moves] == [3, 2, 1, 0]
    moves = feed(f, [False] * 20 + [True] * 20)
    ok &= [m[1] for m in moves] == [1, 0]
    print(f"walk: {[m[1] for m in moves]}, ends at j = {f.j}")

    # Start sampling: geometric offset toward the goal, capped at len(trace) - 1
    rng = np.random.default_rng(1)
    f = TraceFrontier(9)
    f.j = 2
    starts = [f.sample_prefix_len(rng) for _ in range(20000)]
    p0 = np.mean(np.array(starts) == 2)
    print(f"j = 2 of 9 events: starts {min(starts)}..{max(starts)}, P(k = 0) = {p0:.3f}")
    ok &= min(starts) == 2 and max(starts) <= 8 and abs(p0 - 0.5) < 0.02
    f.j = 8
    ok &= all(f.sample_prefix_len(rng) == 8 for _ in range(200))

    # --partway-step: 8 events, start value 7, step 3
    f = TraceFrontier(8, step=3)
    moves = feed(f, [True] * 80)
    ok &= [m[1] for m in moves] == [4, 1, 0] and f.j == 0      # stops at 0, not -2
    ok &= feed(f, [True] * 20) == [] and f.j == 0
    print(f"step 3, successes: j 7 -> {[m[1] for m in moves]}")
    moves = feed(f, [False] * 80)
    ok &= [m[1] for m in moves] == [3, 6, 7] and f.j == 7      # stops at the start value, not 9
    print(f"step 3, failures: j 0 -> {[m[1] for m in moves]}")
    f = TraceFrontier(8, step=3)
    ok &= feed(f, [True] * 19) == [] and f.j == 7              # same window rule as step 1
    ok &= feed(f, [True] * 20, prefix_len=6) == [] and f.j == 7  # an episode inside the stage is not at j
    try:
        TraceFrontier(8, step=0)
        ok = False
    except ValueError:
        pass
    return ok


def check_env():
    tmp = Path(tempfile.mkdtemp(prefix="testj_"))
    trace = load_traces(graph_name="Synthetic_chain_40")[0]
    with adapter(DIAGRAMS / "synthetic" / "Synthetic_chain_40.xml", 5020, tmp / "adapter.log", SHAPING_FIX="1",
                 SHAPING_GAMMA="1", FIX_FALSE_ACCEPT="1", COST_WEIGHT=0.5, DURATION_WEIGHT=0.5) as url:
        env = DCRGymEnv(node_url=url, truncation_fix=True, partway_starts="trace", partway_traces=[trace],
                        partway_initial_share=0.0, partway_seed=1)
        index = {env.label_map.get(p["event"], p["event"]): i for i, p in enumerate(env.event_role_pairs)}
        moves = []
        for episode in range(20):
            env.reset()
            ok_start = env.prefix_len == 4
            _, _, terminated, _, info = env.step(index[trace[env.prefix_len]])
            if info.get("partway_frontier_move"):
                moves.append((episode + 1, info["partway_frontier_move"]))
        print(f"environment: {moves}, j = {env.partway_frontiers[0].j}")
        ok = ok_start and terminated and moves == [(20, (0, 4, 3, 1.0))] and env.partway_frontiers[0].j == 3

        env = DCRGymEnv(node_url=url, truncation_fix=True, partway_starts="trace", partway_traces=[trace],
                        partway_initial_share=0.0, partway_seed=1, partway_step=2)
        moves = []
        for episode in range(20):
            env.reset()
            _, _, _, _, info = env.step(index[trace[env.prefix_len]])
            if info.get("partway_frontier_move"):
                moves.append((episode + 1, info["partway_frontier_move"]))
        print(f"environment, step 2: {moves}, j = {env.partway_frontiers[0].j}")
        ok &= moves == [(20, (0, 4, 2, 1.0))] and env.partway_frontiers[0].j == 2
    shutil.rmtree(tmp, ignore_errors=True)
    return ok


def main():
    ok = check_logic()
    ok &= check_env()
    print("PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
