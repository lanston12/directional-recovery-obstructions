"""Read-only validation of archived prefix witnesses and pre-collision loss."""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "results" / "nonlinear_revision"
DERIVED = ROOT / "derived_results"
grid = pd.read_csv(DERIVED / "nonlinear_directional_mechanism_runs.csv")
events = defaultdict(list)
with (RAW / "failure_log.jsonl").open(encoding="utf-8") as stream:
    for line in stream:
        item = json.loads(line)
        events[str(item["run_id"])].append(item)

checks = []
for row in grid.itertuples(index=False):
    if row.category != "B":
        continue
    prefix = set(json.loads(row.prefix_indices))
    relevant = sorted((e for e in events[str(row.run_id)]
                       if e["kind"] in ("safety_gap", "collision")
                       and int(e["vehicle"]) in prefix),
                      key=lambda e: (float(e["time_s"]), e["kind"] != "safety_gap"))
    first = relevant[0] if relevant else None
    first_collision = next((e for e in relevant if e["kind"] == "collision"), None)
    valid = (first is not None and first["kind"] == "safety_gap"
             and int(first["vehicle"]) == int(row.witness_vehicle)
             and np.isclose(float(first["time_s"]), row.witness_time_s)
             and np.isclose(float(first["value"]), row.witness_gap_m)
             and 0.0 < float(first["value"]) < 4.0
             and (first_collision is None or float(first["time_s"]) < float(first_collision["time_s"])))
    checks.append({"run_id": row.run_id, "case_id": row.case_id,
                   "layout": row.layout, "seed": row.seed,
                   "alpha_mps2": row.alpha_mps2,
                   "active_indices": row.active_indices,
                   "prefix_indices": row.prefix_indices,
                   "first_prefix_safety_vehicle": row.witness_vehicle,
                   "first_prefix_safety_time_s": row.witness_time_s,
                   "first_prefix_safety_gap_m": row.witness_gap_m,
                   "first_prefix_collision_time_s": None if first_collision is None else first_collision["time_s"],
                   "source": "results/nonlinear_revision/failure_log.jsonl",
                   "first_crossing_verified": valid})
out = pd.DataFrame(checks)
saved = pd.read_csv(DERIVED / "v6_prefix_witness_checks.csv")
assert len(out) == len(saved) == 198
assert out.first_crossing_verified.all()
assert set(out.run_id) == set(saved.run_id)
assert saved.first_crossing_verified.all()

traces = np.load(RAW / "representative_trajectories.npz")
time = traces["wave_time"]
speed = traces["wave_speed"][:, 1:]
collision_time = 19.3
pre = time <= collision_time
deficit = np.maximum(0.0, 20.0 - speed[pre]).sum(axis=1)
pre_collision_loss = float(np.trapz(deficit, time[pre]) / (40 * 20 * 60))
summary = {"category_counts": {layout: dict(Counter(group.category))
                                for layout, group in grid.groupby("layout")},
           "B_records": len(out), "B_first_crossings_verified": int(out.first_crossing_verified.sum()),
           "wave_pre_collision_normalized_loss": pre_collision_loss,
           "wave_loss_budget": 0.03,
           "wave_budget_exceeded_before_collision": bool(pre_collision_loss > 0.03),
           "wave_collision_time_s": collision_time}
archived_summary = json.loads((DERIVED / "v6_check_summary.json").read_text(encoding="utf-8"))
assert summary["B_records"] == archived_summary["B_records"] == 198
assert summary["B_first_crossings_verified"] == archived_summary["B_first_crossings_verified"] == 198
assert np.isclose(summary["wave_pre_collision_normalized_loss"],
                  archived_summary["wave_pre_collision_normalized_loss"], atol=1e-10)
print(json.dumps(summary, indent=2))
