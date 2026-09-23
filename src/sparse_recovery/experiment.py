from __future__ import annotations

import csv
import json
from dataclasses import asdict, replace
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import numpy as np

from .model import ChainConfig, Disturbance, simulate


def positions(n: int, count: int, layout: str) -> Sequence[int]:
    if count <= 0:
        return ()
    if layout == "front":
        return tuple(range(1, min(n, count + 1)))
    if layout == "rear":
        return tuple(range(max(1, n - count), n))
    if layout == "middle":
        start = max(1, n // 2 - count // 2)
        return tuple(range(start, min(n, start + count)))
    if layout == "even":
        return tuple(sorted(set(np.linspace(1, n - 1, count).round().astype(int).tolist())))
    raise ValueError(layout)


def _row(result, case: str, replicate: int) -> Dict[str, object]:
    row = dict(result.metadata)
    row.update(result.metrics)
    row.update({"case": case, "replicate": replicate})
    return row


def run_suite(config: Dict[str, object], output_dir: Path) -> List[Dict[str, object]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    seeds = [int(s) for s in config["seeds"]]
    alphas = [float(x) for x in config["alphas_mps2"]]
    ns = [int(x) for x in config["chain_lengths"]]
    rows: List[Dict[str, object]] = []
    failure_file = output_dir / "failure_log.jsonl"
    failure_lines = []

    # A: paired small/finite disturbances and N scaling.
    for n in ns:
        count = max(1, int(round(n * float(config["cav_fraction"]))))
        cavs = positions(n, count, "even")
        for controller in ("human", "acc", "wave"):
            for alpha in alphas:
                for rep, seed in enumerate(seeds):
                    cfg = ChainConfig(n=n, cav_indices=cavs, controller=controller)
                    res = simulate(cfg, Disturbance(alpha), seed)
                    rows.append(_row(res, "A_amplitude", rep))
                    for f in res.failures:
                        failure_lines.append({**res.metadata, **f, "case": "A_amplitude", "replicate": rep})

    # B: exact same vehicles/resources, change only placement.
    n = int(config["layout_n"])
    count = max(1, int(round(n * float(config["cav_fraction"]))))
    for layout in ("front", "middle", "rear", "even"):
        cavs = positions(n, count, layout)
        for alpha in alphas:
            for rep, seed in enumerate(seeds):
                cfg = ChainConfig(n=n, cav_indices=cavs, controller="wave")
                res = simulate(cfg, Disturbance(alpha), seed)
                row = _row(res, "B_layout", rep)
                row["layout"] = layout
                rows.append(row)

    # C: one-factor-at-a-time ablation at a representative amplitude.
    alpha = float(config["ablation_alpha_mps2"])
    cavs = positions(n, count, "even")
    variants = {
        "nominal": {},
        "no_delays": {"reaction_delay": 0.0, "info_delay": 0.0, "actuator_delay": 0.0},
        "no_heterogeneity": {"heterogeneity": 0.0},
        "loose_actuation": {"input_min": -9.0, "input_max": 4.0, "accel_max": 4.0,
                              "brake_max": 8.0, "jerk_max": 20.0},
    }
    for name, changes in variants.items():
        for rep, seed in enumerate(seeds):
            cfg = ChainConfig(n=n, cav_indices=cavs, controller="wave", **changes)
            res = simulate(cfg, Disturbance(alpha), seed)
            row = _row(res, "C_ablation", rep)
            row["ablation"] = name
            rows.append(row)

    # Cross-model check: do not fit either human model to remove wave growth.
    for model in ("idm", "ovm"):
        for alpha in (1.0, 3.0, 4.0):
            for rep, seed in enumerate(seeds):
                cfg = ChainConfig(n=n, cav_indices=cavs, controller="wave", hdv_model=model)
                res = simulate(cfg, Disturbance(alpha), seed)
                rows.append(_row(res, "E_cross_model", rep))

    fields = sorted(set().union(*(r.keys() for r in rows)))
    with (output_dir / "raw_runs.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    with failure_file.open("w", encoding="utf-8") as fh:
        for item in failure_lines:
            fh.write(json.dumps(item, sort_keys=True) + "\n")
    return rows


def summarize(rows: List[Dict[str, object]], output_dir: Path) -> List[Dict[str, object]]:
    keys = ["case", "n", "controller", "alpha_mps2", "layout", "ablation", "hdv_model"]
    groups: Dict[tuple, List[Dict[str, object]]] = {}
    for row in rows:
        key = tuple(row.get(k, "") for k in keys)
        groups.setdefault(key, []).append(row)
    out = []
    metrics = ["success", "safe", "recovered", "budget_ok", "tail_gain", "max_internal_gain",
               "min_gap_m", "throughput_loss_vehicle_m", "terminal_speed_error_mps", "terminal_gap_error_m",
               "terminal_acceleration_error_mps2"]
    for key, group in groups.items():
        row = dict(zip(keys, key))
        row["replicates"] = len(group)
        for metric in metrics:
            vals = np.asarray([float(g[metric]) for g in group])
            row[metric + "_mean"] = float(vals.mean())
            row[metric + "_ci95"] = float(1.96 * vals.std(ddof=1) / np.sqrt(len(vals))) if len(vals) > 1 else 0.0
        out.append(row)
    fields = sorted(set().union(*(r.keys() for r in out)))
    with (output_dir / "summary.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(out)
    return out


def empirical_boundaries(rows: List[Dict[str, object]], output_dir: Path) -> List[Dict[str, object]]:
    subset = [r for r in rows if r["case"] == "A_amplitude"]
    groups: Dict[tuple, List[Dict[str, object]]] = {}
    for r in subset:
        groups.setdefault((r["n"], r["controller"]), []).append(r)
    boundaries = []
    for (n, controller), rs in groups.items():
        rates = {}
        for alpha in sorted(set(float(r["alpha_mps2"]) for r in rs)):
            vals = [float(r["success"]) for r in rs if float(r["alpha_mps2"]) == alpha]
            rates[alpha] = float(np.mean(vals))
        successful = [a for a, rate in rates.items() if rate >= 0.5]
        boundaries.append({"n": n, "controller": controller,
                           "empirical_alpha_50_mps2": max(successful) if successful else 0.0,
                           "criterion": "paired-seed success rate >= 0.5; not a robust certificate"})
    with (output_dir / "empirical_boundaries.json").open("w", encoding="utf-8") as fh:
        json.dump(boundaries, fh, indent=2, sort_keys=True)
    return boundaries


def small_signal_diagnostic(rows: List[Dict[str, object]], boundaries: List[Dict[str, object]],
                            output_dir: Path) -> List[Dict[str, object]]:
    """LOO diagnostic of whether the measured small-amplitude gain predicts the boundary.

    This is a deliberately simple descriptive baseline, not an independently validated
    predictor. Each held-out (N, controller) group is predicted by OLS on the other groups.
    """
    points = []
    for b in boundaries:
        subset = [r for r in rows if r["case"] == "A_amplitude" and r["n"] == b["n"]
                  and r["controller"] == b["controller"] and float(r["alpha_mps2"]) == 0.05]
        points.append({**b, "small_signal_internal_gain": float(np.mean(
            [float(r["max_internal_gain"]) for r in subset]))})
    out = []
    for i, point in enumerate(points):
        train = [p for j, p in enumerate(points) if j != i]
        X = np.asarray([[1.0, p["small_signal_internal_gain"]] for p in train])
        y = np.asarray([p["empirical_alpha_50_mps2"] for p in train])
        beta = np.linalg.lstsq(X, y, rcond=None)[0]
        pred = float(max(0.0, beta.dot([1.0, point["small_signal_internal_gain"]])))
        relevant = [r for r in rows if r["case"] == "A_amplitude" and r["n"] == point["n"]
                    and r["controller"] == point["controller"] and float(r["alpha_mps2"]) > 0]
        declared_recoverable = [r for r in relevant if float(r["alpha_mps2"]) <= pred]
        false_positive = (float(np.mean([1.0 - float(r["success"]) for r in declared_recoverable]))
                          if declared_recoverable else 0.0)
        out.append({
            "n": point["n"], "controller": point["controller"],
            "small_signal_internal_gain": point["small_signal_internal_gain"],
            "observed_empirical_boundary_mps2": point["empirical_alpha_50_mps2"],
            "loo_predicted_boundary_mps2": pred,
            "absolute_prediction_error_mps2": abs(pred - point["empirical_alpha_50_mps2"]),
            "false_recoverable_fraction": false_positive,
            "scope": "LOO OLS descriptive baseline over nine (N, controller) groups; no external validation",
        })
    fields = list(out[0].keys())
    with (output_dir / "small_signal_boundary_diagnostic.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields); writer.writeheader(); writer.writerows(out)
    return out
