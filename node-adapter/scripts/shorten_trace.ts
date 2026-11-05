// E1.4b (special course): shorten an accepting trace, no training. Repeatedly try removing one
// event; keep the removal if the replay is still legal and reaches an accepting marking. Stops when
// no single event can be removed, so the result is not guaranteed to be the shortest trace.
// Uses the same engine functions and enabled-event rule as state_space_stats.ts.
//
// Usage (from node-adapter/):
//   npx tsx scripts/shorten_trace.ts <graph.xml> <trace.json> <out.json>
// trace.json and out.json: a JSON list of event-label lists, as for --partway-trace-file.
import fs from "fs";
import { xmlToDCR } from "../../dcr-engine/src/graphConversion.js";
import { isEnabledS, executeS, isAcceptingS } from "../../dcr-engine/src/executionEngine.js";
import { copyMarking } from "../../dcr-engine/src/utility.js";
import type { DCRGraphS } from "../../dcr-engine/src/types.js";

const log = console.error;
console.log = () => {};  // engine logging, as in state_space_stats.ts

const [xmlPath, tracePath, outPath] = process.argv.slice(2);
if (!outPath) {
  log("usage: shorten_trace.ts <graph.xml> <trace.json> <out.json>");
  process.exit(1);
}

const graph: DCRGraphS = xmlToDCR(fs.readFileSync(xmlPath, "utf-8"));
const initial = copyMarking(graph.marking);
const eventOf = new Map(Array.from(graph.events).map(ev => [graph.labelMap[ev] || ev, ev]));

// Index of the first step whose marking is accepting, -1 if the replay is illegal or never accepts
const firstAccepting = (trace: string[]): number => {
  graph.marking = copyMarking(initial);
  let first = -1;
  for (let i = 0; i < trace.length; i++) {
    const ev = eventOf.get(trace[i]);
    if (ev === undefined) return -1;
    const group = graph.subProcessMap[ev] ? graph.subProcessMap[ev] : graph;
    if (!isEnabledS(ev, graph, group).enabled) return -1;
    executeS(ev, graph);
    if (first < 0 && isAcceptingS(graph, graph)) first = i;
  }
  return isAcceptingS(graph, graph) ? first : -1;
};

const shorten = (start: string[]) => {
  let trace = start.slice();
  let passes = 0;
  let removed = true;
  while (removed) {
    removed = false;
    passes++;
    // From the end backwards, so a removal does not move the events still to be tried
    for (let i = trace.length - 1; i >= 0; i--) {
      if (trace.length < 2) break;
      const shorter = trace.filter((_, k) => k !== i);
      if (firstAccepting(shorter) >= 0) {
        trace = shorter;
        removed = true;
      }
    }
  }
  return { trace, passes };
};

const traces: string[][] = JSON.parse(fs.readFileSync(tracePath, "utf-8"));
const results = traces.map(t => {
  if (firstAccepting(t) < 0) throw new Error("input trace is illegal or does not accept");
  const { trace, passes } = shorten(t);
  const first = firstAccepting(trace);
  log(`${t.length} -> ${trace.length} events in ${passes} passes; first accepting marking at step ${first + 1}; distinct events ${new Set(trace).size}`);
  return trace;
});
fs.writeFileSync(outPath, JSON.stringify(results));
