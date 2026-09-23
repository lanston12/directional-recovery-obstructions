"""Read-only consistency checks for the isolated 24-run OVM/FVD extension."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "results" / "ovm_prefix_extension"

with (DATA / "prefix_classification.csv").open(newline="", encoding="utf-8") as stream:
    rows = list(csv.DictReader(stream))
with (DATA / "witness_validation.csv").open(newline="", encoding="utf-8") as stream:
    witnesses = list(csv.DictReader(stream))
with (DATA / "raw_added_runs.csv").open(newline="", encoding="utf-8") as stream:
    raw = list(csv.DictReader(stream))
with (ROOT / "results" / "nonlinear_revision" / "raw_runs.csv").open(newline="", encoding="utf-8") as stream:
    original_count = sum(1 for _ in csv.DictReader(stream))
assert original_count == 1848
assert len(raw) == len(rows) == 24
assert len({r["run_id"] for r in rows}) == 24
assert len(witnesses) == 3

events: dict[str, list[dict]] = {}
with (DATA / "failure_log.jsonl").open(encoding="utf-8") as stream:
    for line in stream:
        event = json.loads(line)
        events.setdefault(event["run_id"], []).append(event)

traces = np.load(DATA / "prefix_gap_traces.npz")
for row in rows:
    rid = row["run_id"]
    assert f"gap_{rid}" in traces
    assert row["hdv_model"] == "ovm" and row["controller"] == "margin"
    if row["category"] != "B":
        continue
    prefix = json.loads(row["prefix_indices"])
    g = traces[f"gap_{rid}"][:, np.array(prefix) - 1]
    mask = (g > 0) & (g < 4)
    time_index, vehicle_index = np.where(mask)
    assert len(time_index) > 0
    k, j = int(time_index[0]), int(vehicle_index[0])
    assert np.isclose(traces["time_s"][k], float(row["witness_time_s"]))
    assert prefix[j] == int(row["witness_vehicle"])
    assert np.isclose(g[k, j], float(row["witness_gap_m"]))
    collisions = [e["time_s"] for e in events.get(rid, []) if e["kind"] == "collision"]
    assert not collisions or float(row["witness_time_s"]) < min(collisions)

assert all(w["validated"] == "1" for w in witnesses)
print(json.dumps({"original_runs_unchanged_count": original_count,
                  "extension_runs": len(rows), "witnesses_trace_validated": len(witnesses)}, indent=2))
