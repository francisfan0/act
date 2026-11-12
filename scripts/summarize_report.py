"""Print a validation report written by the functional bridge, and exit nonzero on failure.

Usage: summarize_report.py <run_dir> <actsim.log>
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path


def main(run_dir: str, log_path: str) -> int:
    log = Path(log_path).read_text()
    report_path = Path(run_dir) / "report.json"
    problems = []
    if "ASSERTION failed" in log:
        problems.append("actsim reported a failed assertion")
    if not report_path.exists():
        problems.append(f"no report at {report_path}; the sink never ran")
        print(log, file=sys.stderr)
        print("\n".join(problems), file=sys.stderr)
        return 1

    r = json.loads(report_path.read_text())
    rows = r["rows"]
    by_stage: dict[str, dict] = {}
    for row in rows:
        phase = "prefill" if row["invocation_id"] == 0 else "decode"
        stage = f"{phase:7} " + re.sub(r"^layer\.\d+\.", "", row["producer"]) + "." + row["tensor"]
        cur = by_stage.get(stage)
        if cur is None or row.get("tolerance_ratio", 1e9) > cur.get("tolerance_ratio", -1):
            by_stage[stage] = row

    print(f"run {r['run_id']} ({r['mode']}, case {r['case']}): {len(rows)} tensor comparisons, "
          f"{r['artifacts']} artifacts, {r['retained_tensor_bytes'] / 2**20:.1f} MiB retained")
    if rows:
        print("pass when every element has |act - ref| <= atol + rtol*|ref|; "
              "ratio is the worst element's share of that bound (worst over layers and invocations)")
        print(f"{'phase   stage.tensor':48} {'dtype':>7} {'max_abs':>10} {'max_rel':>10} {'atol=rtol':>9} "
              f"{'ratio':>9}  ok")
    for stage, row in by_stage.items():
        if "max_abs" in row:
            print(f"{stage:48} {row['dtype']:>7} {row['max_abs']:10.2e} {row['max_rel']:10.2e} "
                  f"{row['atol']:9.0e} {row['tolerance_ratio']:9.2e}  {'yes' if row['ok'] else 'NO'}")
        else:
            print(f"{stage:48} {row['error']}")
    g = r.get("generation")
    if g:
        print(f"prompt: {g['prompt']!r} ({len(g['prompt_ids'])} tokens)")
        for inv in g["invocations"]:
            print(f"  invocation {inv['invocation_id']} {inv['phase']:7} positions "
                  f"{inv['position_start']}..{inv['position_start'] + inv['sequence_length'] - 1} "
                  f"-> selected {inv['selected_id']}")
        print(f"ACT output:       {g['text']!r}")
        if g["agree"] is None:
            print(f"reference output: none ({g['note']})")
        else:
            print(f"reference output: {g['reference_text']!r} ({'agree' if g['agree'] else 'DISAGREE'})")
        c = g["cache_expected"]
        print(f"KV cache: {c['layers']} layers, each expected length {c['valid_length']} version {c['version']}, "
              f"{len(c['mismatched'])} mismatched")
    else:
        print("cache:", ", ".join(f"{k} len {v['valid_length']} v{v['version']}"
                                   for k, v in list(r["cache"].items())[:3]),
              "..." if len(r["cache"]) > 3 else "")
    for f in r["failures"]:
        problems.append(f"bridge failure: {f}")
    if r["unconsumed_artifacts"]:
        problems.append(f"unconsumed artifacts: {r['unconsumed_artifacts'][:5]}")
    if r["uncommitted_handles"]:
        problems.append(f"uncommitted handles: {r['uncommitted_handles'][:5]}")
    if not r["ok"]:
        problems.append("report marked not ok")
    if problems:
        print("\n".join(problems), file=sys.stderr)
        return 1
    print(f"PASS -> {run_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:]))
