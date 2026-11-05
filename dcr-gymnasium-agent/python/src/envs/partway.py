"""E1.4b partway starts: demonstration traces and the per-trace frontier (special course).

Design and sources: METHOD_NOTES.md, "E1.4b partway starts" (Salimans & Chen 2018; Tao et al. 2024,
per-demonstration frontier and geometric offset; Aphale & Singh 2026, advance at >= 0.8, move back
at <= 0.15).
"""
import csv
import json
from collections import deque
from pathlib import Path

# Next to the repository: SpecialCourse/analysis/E1_3_state_counts.csv
STATE_COUNTS_CSV = Path(__file__).resolve().parents[5] / "analysis" / "E1_3_state_counts.csv"


def load_traces(trace_file=None, graph_name=None, csv_path=STATE_COUNTS_CSV):
    """Traces as lists of event labels: from a JSON file (list of event-label lists), or else the
    l_min_trace of the graph in the E1.3 CSV."""
    if trace_file:
        traces = json.loads(Path(trace_file).read_text(encoding="utf-8"))
        if not traces or not all(isinstance(t, list) and t for t in traces):
            raise ValueError(f"{trace_file}: expected a non-empty list of non-empty event-label lists")
        return traces
    with open(csv_path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            if row["graph"] == graph_name:
                trace = [ev.strip() for ev in row["l_min_trace"].split(">") if ev.strip()]
                if not trace:
                    raise ValueError(f"{graph_name}: no l_min_trace in {csv_path}; use --partway-trace-file")
                return [trace]
    raise ValueError(f"{graph_name}: not in {csv_path}; use --partway-trace-file")


class TraceFrontier:
    """Frontier j of one trace: episodes start after a prefix of length j + k.

    Starts at len(trace) - 1 (one event left). Over the last `window` episodes started exactly at j:
    success rate >= up moves j back by one (longer task), <= down moves it forward by one if j is
    below its start value. The window is emptied after a move: it only holds episodes at the
    current j.
    """

    def __init__(self, trace_len, window=20, up=0.8, down=0.15, offset_p=0.5):
        if trace_len < 1:
            raise ValueError("empty trace")
        self.trace_len = trace_len
        self.start = trace_len - 1
        self.j = self.start
        self.up = up
        self.down = down
        self.offset_p = offset_p
        self.window = window
        self.results = deque(maxlen=window)

    def sample_prefix_len(self, rng):
        """j + k, k = 0, 1, 2, ... geometric (P(k = 0) = offset_p), capped at len(trace) - 1."""
        k = int(rng.geometric(self.offset_p)) - 1
        return min(self.j + k, self.start)

    def record(self, prefix_len, success):
        """Record a finished episode. Returns (old j, new j, rate) if the frontier moved, else None."""
        if prefix_len != self.j:
            return None
        self.results.append(bool(success))
        if len(self.results) < self.window:
            return None
        rate = sum(self.results) / len(self.results)
        old = self.j
        if rate >= self.up and self.j > 0:
            self.j -= 1
        elif rate <= self.down and self.j < self.start:
            self.j += 1
        else:
            return None
        self.results.clear()
        return old, self.j, rate
