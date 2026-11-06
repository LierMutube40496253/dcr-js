// Special course: which events of a graph have a cost, a duration, both or neither, as the engine
// reads them (the same xmlToDCR the adapter uses). Also lists events whose stated value is 0.
//
// Usage (from node-adapter/):
//   npx tsx scripts/graph_costs_durations.ts <out.json> <graph.xml> [<graph.xml> ...]
import fs from "fs";
import path from "path";
import { xmlToDCR } from "../../dcr-engine/src/graphConversion.js";

const log = console.error;
console.log = () => {};  // engine logging

const [outPath, ...files] = process.argv.slice(2);
if (!outPath || files.length === 0) {
  log("usage: graph_costs_durations.ts <out.json> <graph.xml> [<graph.xml> ...]");
  process.exit(1);
}

const range = (values: number[]) => (values.length ? [Math.min(...values), Math.max(...values)] : null);

const rows = files.map(file => {
  const graph = xmlToDCR(fs.readFileSync(file, "utf-8"));
  const events = Array.from(graph.events) as string[];
  const hasCost = (e: string) => graph.costMap[e] !== undefined;
  const hasDuration = (e: string) => graph.durationMap[e] !== undefined;
  const label = (e: string) => graph.labelMap[e] || e;
  const pick = (test: (e: string) => boolean) => events.filter(test).map(label);
  return {
    file: path.basename(file),
    events: events.length,
    both: pick(e => hasCost(e) && hasDuration(e)).length,
    costOnly: pick(e => hasCost(e) && !hasDuration(e)),
    durationOnly: pick(e => !hasCost(e) && hasDuration(e)),
    neither: pick(e => !hasCost(e) && !hasDuration(e)),
    costZero: pick(e => graph.costMap[e] === 0),
    durationZero: pick(e => graph.durationMap[e] === 0),
    costRange: range(events.filter(hasCost).map(e => graph.costMap[e])),
    durationRange: range(events.filter(hasDuration).map(e => graph.durationMap[e])),
    roles: graph.roleMultipliers || {},
  };
});
fs.writeFileSync(outPath, JSON.stringify(rows, null, 1));
log(`${rows.length} graphs written to ${outPath}`);
