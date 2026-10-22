"""
E1.3 (special course): write the chain-plus-noise graphs of thesis Section 8.3.2
as DCR-js XML, in the same format as app/public/examples/diagrams/Mined_*.xml.

The graphs come from Sofía's generator (minizinc_ground_truth/random_dcr_generator.py,
generate_chain_graph) with the parameters of scripts/external_validation/
run_scaling_rl_local.py (seed=1, noise_density=2.0, default noise_mix). Her XML
exporter (build_engine_xml in tobias_converter.py) is not in the repo, so the XML
writer below is OUR reconstruction, not her exact files:
  - no <cost>/<duration> (the generator produces none; her scaling runs used
    alpha=beta=0, so costs did not enter the reward)
  - no <runtime> marking: xmlToDCR then starts fully included, not executed, not
    pending, which is the generator's initial marking
  - duplicate relations from the generator are written once (the engine stores
    relations as sets, so duplicates would collapse anyway)

Also writes synthetic_manifest.json with the generator's raw and distinct
relation counts per type, to check against what the engine loads.

Usage:
    cd dcr-gymnasium-agent/python
    python scripts/make_synthetic_graphs.py            # 20 40 60 80 100
    python scripts/make_synthetic_graphs.py 20 40 --seed 1 --noise-density 2.0
"""
import argparse
import json
import sys
from pathlib import Path
from xml.sax.saxutils import quoteattr

PYTHON_PROJECT_DIR = Path(__file__).resolve().parents[1]
ROOT = PYTHON_PROJECT_DIR.parents[1]
OUT_DIR = ROOT / "app" / "public" / "examples" / "diagrams" / "synthetic"

sys.path.insert(0, str(PYTHON_PROJECT_DIR / "minizinc_ground_truth"))
from random_dcr_generator import generate_chain_graph  # noqa: E402

# relation type -> (XML section, XML element), in the Mined_*.xml section order
SECTIONS = [
    ("condition", "conditions", "condition"),
    ("response", "responses", "response"),
    ("exclude", "excludes", "exclude"),
    ("include", "includes", "include"),
]


def distinct_relations(parsed):
    seen, out = set(), []
    for rel in parsed["relations"]:
        if rel not in seen:
            seen.add(rel)
            out.append(rel)
    return out


def to_dcrjs_xml(parsed, title):
    rels = distinct_relations(parsed)
    lines = ['<?xml version="1.0" encoding="UTF-8" standalone="yes"?>',
             f"<dcrgraph title={quoteattr(title)}>",
             "  <specification>",
             "    <resources>",
             "      <events>"]
    lines += [f"        <event id={quoteattr(e)}></event>" for e in parsed["event_ids"]]
    lines += ["      </events>", "      <labels>"]
    lines += [f"        <label id={quoteattr(parsed['descriptions'][e])}/>" for e in parsed["event_ids"]]
    lines += ["      </labels>", "      <labelMappings>"]
    lines += [f"        <labelMapping eventId={quoteattr(e)} labelId={quoteattr(parsed['descriptions'][e])}/>"
              for e in parsed["event_ids"]]
    lines += ["      </labelMappings>", "    </resources>", "    <constraints>"]
    for rtype, section, element in SECTIONS:
        lines.append(f"      <{section}>")
        lines += [f"        <{element} sourceId={quoteattr(s)} targetId={quoteattr(t)}/>"
                  for r, s, t in rels if r == rtype]
        lines.append(f"      </{section}>")
    lines += ["    </constraints>", "  </specification>", "</dcrgraph>", ""]
    return "\n".join(lines)


def count_by_type(rels):
    return {rtype: sum(1 for r, _, _ in rels if r == rtype) for rtype, _, _ in SECTIONS}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("event_counts", type=int, nargs="*", default=[20, 40, 60, 80, 100])
    ap.add_argument("--seed", type=int, default=1)               # run_scaling_rl_local.py default
    ap.add_argument("--noise-density", type=float, default=2.0)  # run_scaling_rl_local.py default
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest = []
    for n in args.event_counts:
        parsed = generate_chain_graph(n, seed=args.seed, noise_density=args.noise_density)
        name = f"Synthetic_chain_{n}"
        path = OUT_DIR / f"{name}.xml"
        path.write_text(to_dcrjs_xml(parsed, f"Synthetic chain+noise {n} events (seed {args.seed}, "
                                             f"noise {args.noise_density}), E1.3 reconstruction"),
                        encoding="utf-8")
        distinct = distinct_relations(parsed)
        manifest.append({
            "graph": name,
            "events": parsed["num_events"],
            "num_relations_generator": parsed["num_relations"],
            "num_relations_distinct": len(distinct),
            "by_type_generator": count_by_type(parsed["relations"]),
            "by_type_distinct": count_by_type(distinct),
            "seed": args.seed,
            "noise_density": args.noise_density,
        })
        print(f"{path.name}: {n} events, {parsed['num_relations']} relations "
              f"({len(distinct)} distinct)")
    (OUT_DIR / "synthetic_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
