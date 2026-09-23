"""Small, isolated OVM/FVD prefix check; never modifies the 1,848-run archive."""
from __future__ import annotations

import csv
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from sparse_recovery.model import simulate  # noqa: E402
from sparse_recovery.revision_experiment import _make_config, result_row  # noqa: E402


OUT = ROOT / "results" / "ovm_prefix_extension"
OUT.mkdir(parents=True, exist_ok=True)
CONFIG = json.loads((ROOT / "configs" / "revision.json").read_text(encoding="utf-8"))
SEEDS = tuple(int(x) for x in CONFIG["test_seeds"])
ALPHAS = (1.0, 1.5)
LAYOUTS = ("middle", "rear")


def write_csv(path: Path, rows: list[dict]) -> None:
    fields = sorted(set().union(*(row.keys() for row in rows))) if rows else []
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


raw_rows: list[dict] = []
class_rows: list[dict] = []
witness_rows: list[dict] = []
all_events: list[dict] = []
traces: dict[str, np.ndarray] = {}

for layout in LAYOUTS:
    for alpha in ALPHAS:
        for seed in SEEDS:
            spec = {
                "case_id": f"{layout}_ovm_leader_d02_v7",
                "N_followers": 40, "layout": layout, "hdv_model": "ovm",
                "disturbance_location": 0, "command_delay_s": 0.2,
                "controller": "margin", "alpha_mps2": alpha, "seed": seed,
                "studies": ["ovm_prefix_extension"],
            }
            identity = {k: v for k, v in spec.items() if k != "studies"}
            spec["run_id"] = hashlib.sha1(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:16]
            cfg, disturbance = _make_config(CONFIG, spec)
            result = simulate(cfg, disturbance, seed)
            raw_rows.append(result_row(result, spec))
            traces[f"gap_{spec['run_id']}"] = result.gap[:, 1:].copy()
            active = tuple(cfg.active_indices)
            prefix = set(range(1, min(active)))
            events = [{"run_id": spec["run_id"], **event} for event in result.failures]
            all_events.extend(events)
            safety = sorted((e for e in events if e["kind"] == "safety_gap"
                             and e["vehicle"] in prefix), key=lambda e: e["time_s"])
            prefix_collisions = sorted((e for e in events if e["kind"] == "collision"
                                        and e["vehicle"] in prefix), key=lambda e: e["time_s"])
            all_collisions = sorted((e for e in events if e["kind"] == "collision"),
                                    key=lambda e: e["time_s"])
            first_safety = safety[0] if safety else None
            first_prefix_collision = prefix_collisions[0] if prefix_collisions else None
            first_collision = all_collisions[0] if all_collisions else None
            before_prefix_collision = (first_safety is not None and
                                       (first_prefix_collision is None or
                                        first_safety["time_s"] < first_prefix_collision["time_s"]))
            before_any_collision = (first_safety is not None and
                                    (first_collision is None or
                                     first_safety["time_s"] < first_collision["time_s"]))
            # Use the stricter pre-any-collision condition for physical interpretation.
            witnessed = bool(before_prefix_collision and before_any_collision and
                             0 < first_safety["value"] < cfg.safety_gap)
            success = bool(result.metrics["success"])
            category = "A" if success else ("B" if witnessed else "C")
            record = {
                "run_id": spec["run_id"], "layout": layout, "hdv_model": "ovm",
                "controller": "margin", "alpha_mps2": alpha, "seed": seed,
                "active_indices": json.dumps(active), "prefix_indices": json.dumps(sorted(prefix)),
                "success": int(success), "category": category,
                "witness_vehicle": first_safety["vehicle"] if first_safety else "",
                "witness_time_s": first_safety["time_s"] if first_safety else "",
                "witness_gap_m": first_safety["value"] if first_safety else "",
                "first_prefix_collision_time_s": first_prefix_collision["time_s"] if first_prefix_collision else "",
                "first_any_collision_time_s": first_collision["time_s"] if first_collision else "",
                "first_prefix_safety_before_prefix_collision": int(before_prefix_collision),
                "first_prefix_safety_before_any_collision": int(before_any_collision),
            }
            class_rows.append(record)
            if category == "B":
                # Independent trace lookup: verify the earliest sampled positive gap < 4 m.
                gaps = result.gap[:, sorted(prefix)]
                mask = (gaps > 0) & (gaps < cfg.safety_gap)
                times, vehicles = np.where(mask)
                k = int(times[0])
                vehicle = sorted(prefix)[int(vehicles[0])]
                check = (np.isclose(result.time[k], first_safety["time_s"]) and
                         vehicle == first_safety["vehicle"] and
                         np.isclose(gaps[k, int(vehicles[0])], first_safety["value"]) and
                         before_any_collision)
                witness_rows.append({**record, "trace_first_time_s": float(result.time[k]),
                                     "trace_first_vehicle": vehicle,
                                     "trace_first_gap_m": float(gaps[k, int(vehicles[0])]),
                                     "validated": int(check)})

write_csv(OUT / "raw_added_runs.csv", raw_rows)
write_csv(OUT / "prefix_classification.csv", class_rows)
write_csv(OUT / "witness_validation.csv", witness_rows)
with (OUT / "failure_log.jsonl").open("w", encoding="utf-8") as stream:
    for event in all_events:
        stream.write(json.dumps(event, sort_keys=True) + "\n")
traces["time_s"] = result.time
np.savez_compressed(OUT / "prefix_gap_traces.npz", **traces)

idm = pd.read_csv(ROOT / "derived_results" / "nonlinear_directional_mechanism_runs.csv")
idem = idm[idm.layout.isin(LAYOUTS) & idm.seed.isin(SEEDS) & idm.alpha_mps2.isin(ALPHAS)]
summary = {
    "scope": "Two layouts, two requested amplitudes, six paired seeds; descriptive cells only",
    "n_new_runs": len(class_rows),
    "ovm_categories": {layout: dict(Counter(r["category"] for r in class_rows if r["layout"] == layout))
                       for layout in LAYOUTS},
    "matched_idm_categories": {layout: dict(Counter(idem[idem.layout == layout].category))
                               for layout in LAYOUTS},
    "ovm_witnesses_validated": sum(r["validated"] for r in witness_rows),
    "ovm_witnesses_total": len(witness_rows),
    "witness_rule": "positive first prefix gap below 4 m, before any collision in chain; matched to stored gap trace",
}
(OUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
print(json.dumps(summary, indent=2))
