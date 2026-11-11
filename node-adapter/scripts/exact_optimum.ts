// E1.5 (special course): exact optimum per weight w, no learning. Dijkstra over reachable markings
// from the initial marking to the first accepting marking reached after at least one step (an episode
// ends there), with step cost  w x cost / maxCost + (1 - w) x duration / maxDuration  as in the adapter's
// SHAPING_FIX reward: role multipliers on cost and duration, normalisation by the largest base value
// above 0, an event without a value costs 0 on that objective. The reward of an accepted case is
// 100 - (this objective) - Phi(initial marking), so the cheapest path here is the best accepted case
// for the agent. Optional: --k N (Expert budget: Expert actions allowed N times), --default-duration X
// (as DEFAULT_DURATION), --no-roles (base values, for comparison with the MiniZinc results).
// Also returns the points (cost, duration) found over a fine grid of w (the supported Pareto points).
//
// Usage (from node-adapter/):
//   npx tsx scripts/exact_optimum.ts <graph.xml> <out.json> [--w 0,0.5,1] [--k N] [--default-duration X] [--no-roles]
import fs from "fs";
import { xmlToDCR } from "../../dcr-engine/src/graphConversion.js";
import { isEnabledS, executeS, isAcceptingS } from "../../dcr-engine/src/executionEngine.js";
import { copyMarking } from "../../dcr-engine/src/utility.js";
import type { DCRGraphS, Marking } from "../../dcr-engine/src/types.js";

const log = console.error;
console.log = () => {};

const argv = process.argv.slice(2);
const [xmlPath, outPath] = argv;
let ws = [0, 0.5, 1];
let budget: number | null = null;
let defaultDuration: number | null = null;
let noRoles = false;
for (let i = 2; i < argv.length; i++) {
  if (argv[i] === "--w") ws = argv[++i].split(",").map(Number);
  else if (argv[i] === "--k") budget = Number(argv[++i]);
  else if (argv[i] === "--default-duration") defaultDuration = Number(argv[++i]);
  else if (argv[i] === "--no-roles") noRoles = true;
}
if (!outPath) {
  log("usage: exact_optimum.ts <graph.xml> <out.json> [--w 0,0.5,1] [--k N] [--default-duration X] [--no-roles]");
  process.exit(1);
}

const graph: DCRGraphS = xmlToDCR(fs.readFileSync(xmlPath, "utf-8"));
const events = Array.from(graph.events).sort();
if (defaultDuration !== null) for (const e of events) if (graph.durationMap[e] === undefined) graph.durationMap[e] = defaultDuration;
const positive = (vals: number[]) => vals.filter(v => v > 0);
const maxCost = Math.max(1, ...positive(Object.values(graph.costMap))) === 1 && positive(Object.values(graph.costMap)).length === 0
  ? 1 : Math.max(...positive(Object.values(graph.costMap)));
const maxDur = positive(Object.values(graph.durationMap)).length === 0 ? 1 : Math.max(...positive(Object.values(graph.durationMap)));
const roleNames = noRoles ? [] : Object.keys(graph.roleMultipliers || {}).sort();
const roles = roleNames.length ? roleNames : [""];
const initial = copyMarking(graph.marking);

const encode = (m: Marking) => {
  let k = "";
  for (const e of events) k += String.fromCharCode(48 + (m.included.has(e) ? 1 : 0) + (m.executed.has(e) ? 2 : 0) + (m.pending.has(e) ? 4 : 0));
  return k;
};
const enabled = (m: Marking) => {
  graph.marking = m;
  return events.filter(e => isEnabledS(e, graph, graph.subProcessMap[e] ? graph.subProcessMap[e] : graph).enabled);
};
const accepting = (m: Marking) => { graph.marking = m; return isAcceptingS(graph, graph); };
const fire = (m: Marking, e: string) => { const n = copyMarking(m); graph.marking = n; executeS(e, graph); return n; };
const stepValues = (e: string, role: string) => {
  const mult = role ? graph.roleMultipliers[role] : undefined;
  const c = graph.costMap[e] ?? 0, d = graph.durationMap[e] ?? 0;
  return { cost: mult ? c * mult.costMultiplier : c, duration: mult ? d * mult.durationMultiplier : d };
};

// Minimal binary heap on [priority, id]
class Heap {
  a: [number, number][] = [];
  push(x: [number, number]) {
    const a = this.a; a.push(x); let i = a.length - 1;
    while (i > 0) { const p = (i - 1) >> 1; if (a[p][0] <= a[i][0]) break; [a[p], a[i]] = [a[i], a[p]]; i = p; }
  }
  pop(): [number, number] | undefined {
    const a = this.a; if (!a.length) return undefined;
    const top = a[0], last = a.pop()!;
    if (a.length) {
      a[0] = last; let i = 0;
      for (;;) {
        const l = 2 * i + 1, r = l + 1; let s = i;
        if (l < a.length && a[l][0] < a[s][0]) s = l;
        if (r < a.length && a[r][0] < a[s][0]) s = r;
        if (s === i) break; [a[s], a[i]] = [a[i], a[s]]; i = s;
      }
    }
    return top;
  }
}

const solve = (w: number) => {
  // node: marking, budget left, at least one step taken
  type Node = { m: Marking; b: number | null; started: boolean; dist: number; prev: number; step: string; cost: number; duration: number };
  const nodes: Node[] = [];
  const index = new Map<string, number>();
  const keyOf = (m: Marking, b: number | null, started: boolean) => `${encode(m)}|${b ?? "-"}|${started ? 1 : 0}`;
  nodes.push({ m: initial, b: budget, started: false, dist: 0, prev: -1, step: "", cost: 0, duration: 0 });
  index.set(keyOf(initial, budget, false), 0);
  const heap = new Heap();
  heap.push([0, 0]);
  const done = new Set<number>();
  let popped = 0;
  for (let top = heap.pop(); top; top = heap.pop()) {
    const [d, id] = top;
    if (done.has(id) || d > nodes[id].dist) continue;
    done.add(id);
    popped++;
    const node = nodes[id];
    if (node.started && accepting(node.m)) {
      const trace: string[] = [];
      for (let i = id; nodes[i].prev >= 0; i = nodes[i].prev) trace.unshift(nodes[i].step);
      return { w, objective: node.dist, cost: node.cost, duration: node.duration, steps: trace.length, trace, expanded: popped };
    }
    for (const e of enabled(node.m)) {
      for (const role of roles) {
        if (role === "Expert" && node.b !== null && node.b <= 0) continue;
        const v = stepValues(e, role);
        const nd = node.dist + w * v.cost / maxCost + (1 - w) * v.duration / maxDur;
        const m2 = fire(node.m, e);
        const b2 = role === "Expert" && node.b !== null ? node.b - 1 : node.b;
        const key = keyOf(m2, b2, true);
        const label = (graph.labelMap[e] || e) + (role ? `/${role}` : "");
        const existing = index.get(key);
        if (existing === undefined || nd < nodes[existing].dist - 1e-12) {
          const rec = { m: m2, b: b2, started: true, dist: nd, prev: id, step: label, cost: node.cost + v.cost, duration: node.duration + v.duration };
          if (existing === undefined) { index.set(key, nodes.length); nodes.push(rec); heap.push([nd, nodes.length - 1]); }
          else { nodes[existing] = rec; heap.push([nd, existing]); }
        }
      }
    }
  }
  return { w, objective: null, cost: null, duration: null, steps: null, trace: [], expanded: popped };
};

const t0 = Date.now();
const optima = ws.map(solve);
const front = new Map<string, any>();
for (let i = 0; i <= 100; i++) {
  const r = solve(i / 100);
  if (r.cost !== null) front.set(`${r.cost}|${r.duration}`, { cost: r.cost, duration: r.duration, w_first: i / 100 });
}
const result = { graph: xmlPath.split(/[\\/]/).pop(), maxCost, maxDuration: maxDur, roles: roleNames, budget, defaultDuration,
                 noRoles, optima, supportedFront: Array.from(front.values()), seconds: (Date.now() - t0) / 1000 };
fs.writeFileSync(outPath, JSON.stringify(result, null, 1));
for (const o of optima) log(`w=${o.w}: objective ${o.objective?.toFixed(6)}, cost ${o.cost}, duration ${o.duration}, ${o.steps} steps, expanded ${o.expanded}: ${o.trace.join(" > ")}`);
log(`supported front: ${result.supportedFront.map(p => `(${p.cost}, ${p.duration})`).join(" ")}; ${result.seconds} s`);
