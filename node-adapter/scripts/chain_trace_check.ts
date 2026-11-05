// E1.4b quick check (special course): look for an accepting trace on graphs where the E1.3 BFS
// found none, no training. Two tries, using the same engine functions as state_space_stats.ts:
//   1. the pattern of Synthetic_chain_40, Act_Event_1 > ... > Act_Event_5, replayed as is;
//   2. greedy best-first search: always expand the marking with the fewest included pending events
//      (ties: the most recently found), skipping markings already seen, under a time cap.
//
// Usage (from node-adapter/):
//   npx tsx scripts/chain_trace_check.ts --out <results.json> [--minutes 10] [--max-markings N] <graph.xml> ...
import fs from "fs";
import path from "path";
import { xmlToDCR } from "../../dcr-engine/src/graphConversion.js";
import { isEnabledS, executeS, isAcceptingS } from "../../dcr-engine/src/executionEngine.js";
import { copyMarking } from "../../dcr-engine/src/utility.js";
import type { DCRGraphS, Marking } from "../../dcr-engine/src/types.js";

const log = console.error;
console.log = () => {};  // engine logging, as in state_space_stats.ts

const argv = process.argv.slice(2);
let outPath = "";
let minutes = 10;
let maxMarkings = 3_000_000;
const xmlPaths: string[] = [];
for (let i = 0; i < argv.length; i++) {
  if (argv[i] === "--out") outPath = argv[++i];
  else if (argv[i] === "--minutes") minutes = Number(argv[++i]);
  else if (argv[i] === "--max-markings") maxMarkings = Number(argv[++i]);
  else xmlPaths.push(argv[i]);
}
if (!outPath || xmlPaths.length === 0) {
  log("usage: chain_trace_check.ts --out <results.json> [--minutes 10] [--max-markings N] <graph.xml> ...");
  process.exit(1);
}

// A node keeps only the encoded marking (full Markings for millions of nodes do not fit in memory)
type Node = { key: string; parent: Node | null; event: string; depth: number };

const analyse = (xmlPath: string) => {
  const graph: DCRGraphS = xmlToDCR(fs.readFileSync(xmlPath, "utf-8"));
  const initial = copyMarking(graph.marking);
  const events = Array.from(graph.events).sort();
  const nameOf = (ev: string) => graph.labelMap[ev] || ev;
  const enabledAt = (m: Marking) => {
    graph.marking = copyMarking(m);
    return events.filter(ev => {
      const group = graph.subProcessMap[ev] ? graph.subProcessMap[ev] : graph;
      return isEnabledS(ev, graph, group).enabled;
    });
  };
  const pendingIncluded = (m: Marking) => {
    let n = 0;
    for (const ev of m.pending) if (m.included.has(ev)) n++;
    return n;
  };
  const encode = (m: Marking) => {
    let k = "";
    for (const ev of events) {
      k += String.fromCharCode(48 + (m.included.has(ev) ? 1 : 0) + (m.executed.has(ev) ? 2 : 0) + (m.pending.has(ev) ? 4 : 0));
    }
    return k;
  };
  const decode = (k: string): Marking => {
    const m: Marking = { included: new Set(), executed: new Set(), pending: new Set() };
    events.forEach((ev, i) => {
      const c = k.charCodeAt(i) - 48;
      if (c & 1) m.included.add(ev);
      if (c & 2) m.executed.add(ev);
      if (c & 4) m.pending.add(ev);
    });
    return m;
  };

  // --- 1. the pattern of the 40 graph ---
  graph.marking = copyMarking(initial);
  const pattern = [1, 2, 3, 4, 5].map(i => `Act_Event_${i}`);
  const byLabel = new Map(events.map(ev => [nameOf(ev), ev]));
  const patternSteps: { label: string; enabled: boolean; pendingAfter: number; accepting: boolean }[] = [];
  let patternLegal = true;
  for (const label of pattern) {
    const ev = byLabel.get(label);
    const enabled = ev !== undefined && enabledAt(graph.marking).includes(ev);
    if (!enabled) { patternLegal = false; patternSteps.push({ label, enabled, pendingAfter: -1, accepting: false }); break; }
    executeS(ev!, graph);
    patternSteps.push({ label, enabled, pendingAfter: pendingIncluded(graph.marking), accepting: isAcceptingS(graph, graph) });
  }
  const patternAccepts = patternLegal && patternSteps[patternSteps.length - 1].accepting;

  // --- 2. greedy best-first search; buckets by included pending count, LIFO inside a bucket ---
  const t0 = Date.now();
  const seen = new Set<string>([encode(initial)]);
  const buckets: Node[][] = [];
  const push = (n: Node, p: number) => {
    (buckets[p] = buckets[p] || []).push(n);
  };
  push({ key: encode(initial), parent: null, event: "", depth: 0 }, pendingIncluded(initial));
  let found: Node | null = null;
  let expanded = 0;
  let minPending = Infinity;
  let capped = "";
  search: while (true) {
    const bucket = buckets.find(b => b && b.length > 0);
    if (!bucket) break;
    const node = bucket.pop()!;
    expanded++;
    if (expanded % 1000 === 0 && Date.now() - t0 > minutes * 60 * 1000) { capped = "time"; break; }
    const marking = decode(node.key);
    for (const ev of enabledAt(marking)) {
      graph.marking = copyMarking(marking);
      executeS(ev, graph);
      const accepting = isAcceptingS(graph, graph);
      const key = encode(graph.marking);
      const pending = pendingIncluded(graph.marking);
      const child: Node = { key, parent: node, event: ev, depth: node.depth + 1 };
      minPending = Math.min(minPending, pending);
      if (accepting) { found = child; break search; }
      if (!seen.has(key)) { seen.add(key); push(child, pending); }
    }
    if (seen.size >= maxMarkings) { capped = "markings"; break; }
  }
  const greedyEvents: string[] = [];
  for (let n = found; n && n.parent; n = n.parent) greedyEvents.unshift(n.event);

  // Replay the found trace from the initial marking as a check
  let greedyVerified = false;
  if (found) {
    graph.marking = copyMarking(initial);
    greedyVerified = greedyEvents.every(ev => {
      const ok = enabledAt(graph.marking).includes(ev);
      if (ok) executeS(ev, graph);
      return ok;
    }) && isAcceptingS(graph, graph);
  }
  graph.marking = copyMarking(initial);

  return {
    graph: path.basename(xmlPath, ".xml"),
    pattern,
    pattern_legal: patternLegal,
    pattern_accepts: patternAccepts,
    pattern_steps: patternSteps,
    greedy_found: Boolean(found),
    greedy_verified: greedyVerified,
    greedy_trace: greedyEvents.map(nameOf),
    greedy_len: greedyEvents.length,
    greedy_capped: capped,
    greedy_expanded: expanded,
    greedy_markings_seen: seen.size,
    greedy_min_pending_seen: minPending,
    greedy_seconds: (Date.now() - t0) / 1000,
  };
};

const results = xmlPaths.map(p => {
  log(`[check] ${p}`);
  const r = analyse(p);
  log(`[check] ${r.graph}: pattern legal=${r.pattern_legal} accepts=${r.pattern_accepts}; greedy found=${r.greedy_found} len=${r.greedy_len} verified=${r.greedy_verified} capped=${r.greedy_capped || "no"} expanded=${r.greedy_expanded} seen=${r.greedy_markings_seen} minPending=${r.greedy_min_pending_seen} ${r.greedy_seconds.toFixed(1)} s`);
  return r;
});
fs.writeFileSync(outPath, JSON.stringify(results, null, 2));
