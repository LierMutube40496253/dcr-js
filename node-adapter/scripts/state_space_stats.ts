// E1.3 (special course): state-space statistics for DCR graphs, no training.
// Uses Sofía's engine functions exactly as conformance_replay.ts does, and the same
// enabled-event rule as RLDCREnvironment.getValidActions (generation.ts).
//
// Usage (from node-adapter/):
//   npx tsx scripts/state_space_stats.ts --out <results.json> [--list-markings <file.txt>] <graph.xml> ...
// --list-markings writes every reachable marking of the FIRST graph given.
import fs from "fs";
import path from "path";
import { xmlToDCR } from "../../dcr-engine/src/graphConversion.js";
import { isEnabledS, executeS, isAcceptingS } from "../../dcr-engine/src/executionEngine.js";
import { copyMarking } from "../../dcr-engine/src/utility.js";
import type { DCRGraphS, Marking } from "../../dcr-engine/src/types.js";

// Suppress verbose engine logging, as in conformance_replay.ts
const log = console.error;
console.log = () => {};

const MAX_MARKINGS = 2_000_000;
const MAX_BFS_MS = 10 * 60 * 1000;
const N_WALKS = 10_000;
const MAX_WALK_STEPS = 300; // = MAX_EPISODE_STEPS in the thesis runs
const WALK_SEED = 1;

// --- arguments ---
const argv = process.argv.slice(2);
let outPath = "";
let listPath = "";
const xmlPaths: string[] = [];
for (let i = 0; i < argv.length; i++) {
  if (argv[i] === "--out") outPath = argv[++i];
  else if (argv[i] === "--list-markings") listPath = argv[++i];
  else xmlPaths.push(argv[i]);
}
if (!outPath || xmlPaths.length === 0) {
  log("usage: state_space_stats.ts --out <results.json> [--list-markings <file>] <graph.xml> ...");
  process.exit(1);
}

// Seeded PRNG (mulberry32) so the random walks are reproducible
const mulberry32 = (seed: number) => () => {
  seed |= 0; seed = (seed + 0x6d2b79f5) | 0;
  let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
  t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
  return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
};

const analyse = (xmlPath: string, listMarkings: boolean) => {
  const graph: DCRGraphS = xmlToDCR(fs.readFileSync(xmlPath, "utf-8"));
  const initialMarking = copyMarking(graph.marking);
  const events = Array.from(graph.events).sort();
  const nEvents = events.length;
  const roles = Object.keys(graph.roleMultipliers || {}).sort();
  const nRoles = roles.length || 1;
  const nRelations = ["conditionsFor", "responseTo", "includesTo", "excludesTo", "milestonesFor"]
    .map(k => Object.values((graph as any)[k] as Record<string, Set<string>>).reduce((s, v) => s + v.size, 0));

  // Marking key: one char per event (bit0 included, bit1 executed, bit2 pending)
  const encode = (m: Marking) => {
    let k = "";
    for (const e of events) {
      k += String.fromCharCode(48 + (m.included.has(e) ? 1 : 0) + (m.executed.has(e) ? 2 : 0) + (m.pending.has(e) ? 4 : 0));
    }
    return k;
  };
  const decode = (k: string): Marking => {
    const m: Marking = { included: new Set(), executed: new Set(), pending: new Set() };
    for (let i = 0; i < nEvents; i++) {
      const c = k.charCodeAt(i) - 48;
      if (c & 1) m.included.add(events[i]);
      if (c & 2) m.executed.add(events[i]);
      if (c & 4) m.pending.add(events[i]);
    }
    return m;
  };
  // Same rule as RLDCREnvironment.getValidActions
  const enabledEvents = (m: Marking) => {
    graph.marking = m;
    return events.filter(e => {
      const group = graph.subProcessMap[e] ? graph.subProcessMap[e] : graph;
      return isEnabledS(e, graph, group).enabled;
    });
  };
  const accepting = (m: Marking) => {
    graph.marking = m;
    return isAcceptingS(graph, graph);
  };
  const fire = (m: Marking, e: string) => {
    const next = copyMarking(m);
    graph.marking = next;
    executeS(e, graph);
    return next;
  };

  // --- BFS over reachable markings ---
  const t0 = Date.now();
  const startKey = encode(initialMarking);
  const visited = new Set<string>([startKey]);
  let frontier = [startKey];
  let depth = 0;
  let capped = "";
  let lMin: number | null = null;
  let lMinTrace: string[] = [];
  const parent = listMarkings || nEvents <= 200 ? new Map<string, [string, string]>() : null;
  const enabledHist = new Array(nEvents + 1).fill(0);
  let expanded = 0, sumEnabled = 0, sumShare = 0, maxEnabled = 0, nAccepting = 0, nDeadEnds = 0;
  const listing: string[] = [];

  bfs: while (frontier.length > 0) {
    const next: string[] = [];
    for (const key of frontier) {
      if (Date.now() - t0 > MAX_BFS_MS) { capped = "time"; break bfs; }
      const m = decode(key);
      const en = enabledEvents(m);
      const acc = accepting(m);
      expanded++;
      enabledHist[en.length]++;
      sumEnabled += en.length;
      sumShare += en.length / nEvents;
      if (en.length > maxEnabled) maxEnabled = en.length;
      if (acc) nAccepting++;
      if (en.length === 0) nDeadEnds++;
      if (listMarkings) {
        const lab = (s: Set<string>) => "{" + events.filter(e => s.has(e)).map(e => graph.labelMap[e] || e).join(", ") + "}";
        listing.push(`#${expanded} depth=${depth} accepting=${acc} enabled=${en.length}\n` +
          `  included=${lab(m.included)}\n  executed=${lab(m.executed)}\n  pending=${lab(m.pending)}\n` +
          `  enabled=[${en.map(e => graph.labelMap[e] || e).join(", ")}]`);
      }
      for (const e of en) {
        const nm = fire(m, e);
        const nk = encode(nm);
        // L_min: shortest accepting trace with at least one step (BFS order => shortest)
        if (lMin === null && accepting(nm)) {
          lMin = depth + 1;
          lMinTrace = [e];
          if (parent) {
            let k = key;
            while (parent.has(k)) { const [pk, pe] = parent.get(k)!; lMinTrace.unshift(pe); k = pk; }
          }
        }
        if (!visited.has(nk)) {
          if (visited.size >= MAX_MARKINGS) { capped = "markings"; continue; }
          visited.add(nk);
          parent?.set(nk, [key, e]);
          next.push(nk);
        }
      }
    }
    frontier = next;
    depth++;
  }
  const bfsSeconds = (Date.now() - t0) / 1000;
  let median = 0;
  for (let i = 0, c = 0; i <= nEvents; i++) { c += enabledHist[i]; if (c >= expanded / 2) { median = i; break; } }

  // --- random valid walks (uniform over ENABLED events) ---
  const rand = mulberry32(WALK_SEED);
  let walkSteps = 0, walkShareSum = 0, walkLegalSum = 0, walksAccepted = 0, acceptedLenSum = 0, walksStuck = 0;
  const t1 = Date.now();
  for (let w = 0; w < N_WALKS; w++) {
    let m = copyMarking(initialMarking);
    for (let s = 0; s < MAX_WALK_STEPS; s++) {
      const en = enabledEvents(m);
      walkSteps++;
      walkShareSum += en.length / nEvents;
      // P(uniform pick over all events x roles is legal); role does not affect legality (no budget)
      walkLegalSum += (en.length * nRoles) / (nEvents * nRoles);
      if (en.length === 0) { walksStuck++; break; }
      m = fire(m, en[Math.floor(rand() * en.length)]);
      if (accepting(m)) { walksAccepted++; acceptedLenSum += s + 1; break; }
    }
  }
  const walkSeconds = (Date.now() - t1) / 1000;

  const nameOf = (e: string) => graph.labelMap[e] || e;
  return {
    result: {
      graph: path.basename(xmlPath, ".xml"),
      events: nEvents,
      roles: roles.length,
      actions: nEvents * nRoles,
      relations: nRelations.reduce((a, b) => a + b, 0),
      relations_by_type: { condition: nRelations[0], response: nRelations[1], include: nRelations[2], exclude: nRelations[3], milestone: nRelations[4] },
      initial_accepting: accepting(copyMarking(initialMarking)),
      reachable_markings: visited.size,
      reachable_capped: capped,           // "" | "markings" | "time"
      markings_expanded: expanded,        // statistics below are over these
      bfs_depth_reached: depth,
      accepting_markings: nAccepting,
      dead_end_markings: nDeadEnds,
      enabled_mean: sumEnabled / expanded,
      enabled_median: median,
      enabled_max: maxEnabled,
      enabled_share_mean: sumShare / expanded,
      l_min: lMin,
      l_min_trace: lMinTrace.map(nameOf),
      walks: N_WALKS,
      walk_enabled_share_mean: walkShareSum / walkSteps,
      walk_p_random_action_legal: walkLegalSum / walkSteps,
      walk_accept_share: walksAccepted / N_WALKS,
      walk_stuck_share: walksStuck / N_WALKS,
      walk_accepted_len_mean: walksAccepted ? acceptedLenSum / walksAccepted : null,
      bfs_seconds: bfsSeconds,
      walk_seconds: walkSeconds,
    },
    listing,
  };
};

const results = [];
for (let i = 0; i < xmlPaths.length; i++) {
  const { result, listing } = analyse(xmlPaths[i], i === 0 && listPath !== "");
  if (listing.length) fs.writeFileSync(listPath, listing.join("\n") + "\n");
  log(JSON.stringify(result));
  results.push(result);
  fs.writeFileSync(outPath, JSON.stringify(results, null, 2) + "\n");
}
