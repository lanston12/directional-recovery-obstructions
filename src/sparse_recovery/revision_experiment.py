from __future__ import annotations

import csv
import concurrent.futures
import hashlib
import json
import platform
import time
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np

from .model import ChainConfig, Disturbance, SimulationResult, simulate


CONTROLLERS = ("human", "acc", "wave", "predictive", "margin")


def positions(n_followers: int, count: int, layout: str) -> Tuple[int, ...]:
    count = max(0, min(int(count), int(n_followers)))
    if count == 0:
        return ()
    if layout == "front":
        return tuple(range(1, count + 1))
    if layout == "rear":
        return tuple(range(n_followers - count + 1, n_followers + 1))
    if layout == "middle":
        start = max(1, (n_followers - count) // 2 + 1)
        return tuple(range(start, start + count))
    if layout == "even":
        return tuple(sorted(set(np.linspace(1, n_followers, count).round().astype(int).tolist())))
    raise ValueError(layout)


def _case(case_id: str, n_followers: int, layout: str = "even", hdv_model: str = "idm",
          disturbance_location: int = 0, command_delay: float = 0.2) -> Dict[str, object]:
    return {"case_id": case_id, "N_followers": n_followers, "layout": layout,
            "hdv_model": hdv_model, "disturbance_location": disturbance_location,
            "command_delay_s": command_delay}


def declared_cases(config: Dict[str, object]) -> Dict[str, Dict[str, object]]:
    n = int(config["N_followers_core"])
    internal = int(config["internal_disturbance_follower"])
    cases = {
        "even_idm_leader_d02": _case("even_idm_leader_d02", n),
        "front_idm_leader_d02": _case("front_idm_leader_d02", n, "front"),
        "middle_idm_leader_d02": _case("middle_idm_leader_d02", n, "middle"),
        "rear_idm_leader_d02": _case("rear_idm_leader_d02", n, "rear"),
        "even_idm_internal8_d02": _case("even_idm_internal8_d02", n, disturbance_location=internal),
        "even_idm_leader_cmd0": _case("even_idm_leader_cmd0", n, command_delay=0.0),
        "even_idm_leader_cmd04": _case("even_idm_leader_cmd04", n, command_delay=0.4),
        "even_ovm_leader_d02": _case("even_ovm_leader_d02", n, hdv_model="ovm"),
    }
    for nx in config["N_followers_extensions"]:
        nx = int(nx)
        cases[f"even_idm_leader_N{nx}"] = _case(f"even_idm_leader_N{nx}", nx)
    return cases


def build_specs(config: Dict[str, object]) -> List[Dict[str, object]]:
    seeds = [int(x) for x in config["seeds"]]
    extension_seeds = [int(x) for x in config["test_seeds"]]
    core_a = [float(x) for x in config["core_alphas_mps2"]]
    ext_a = [float(x) for x in config["extension_alphas_mps2"]]
    high_a = [float(x) for x in config.get("high_alphas_mps2", [])]
    cases = declared_cases(config)
    unique: Dict[str, Dict[str, object]] = {}

    def add(case_id: str, controller: str, alpha: float, seed: int, study: str) -> None:
        case = cases[case_id]
        key_data = {**case, "controller": controller, "alpha_mps2": alpha, "seed": seed}
        run_id = hashlib.sha1(json.dumps(key_data, sort_keys=True).encode("utf-8")).hexdigest()[:16]
        if run_id not in unique:
            unique[run_id] = {**key_data, "run_id": run_id, "studies": []}
        if study not in unique[run_id]["studies"]:
            unique[run_id]["studies"].append(study)

    core = "even_idm_leader_d02"
    for controller in CONTROLLERS:
        for alpha in core_a:
            for seed in seeds:
                add(core, controller, alpha, seed, "controller_comparison")
    for case_id in ("front_idm_leader_d02", "middle_idm_leader_d02", "rear_idm_leader_d02", core):
        for alpha in core_a:
            for seed in seeds:
                add(case_id, "margin", alpha, seed, "layout_comparison")
    predictor_cases = (
        core, "front_idm_leader_d02", "even_idm_internal8_d02", "even_idm_leader_cmd0",
        "middle_idm_leader_d02", "rear_idm_leader_d02", "even_idm_leader_cmd04", "even_ovm_leader_d02",
    )
    for case_id in predictor_cases:
        for alpha in core_a:
            for seed in extension_seeds:
                add(case_id, "margin", alpha, seed, "prediction_scope")
    for nx in config["N_followers_extensions"]:
        case_id = f"even_idm_leader_N{int(nx)}"
        for controller in ("wave", "predictive", "margin"):
            for alpha in ext_a:
                for seed in extension_seeds:
                    add(case_id, controller, alpha, seed, "chain_length_scope")
    # Boundary-closing scan: only methods/cases that were right-censored at 2.5.
    for controller in ("predictive", "margin"):
        for alpha in high_a:
            for seed in seeds:
                add(core, controller, alpha, seed, "controller_boundary_extension")
    for case_id in ("front_idm_leader_d02", "middle_idm_leader_d02", "rear_idm_leader_d02"):
        for alpha in high_a:
            for seed in extension_seeds:
                add(case_id, "margin", alpha, seed, "layout_boundary_extension")
    for case_id in ("even_idm_internal8_d02", "even_idm_leader_cmd0",
                    "even_idm_leader_cmd04", "even_ovm_leader_d02"):
        for alpha in high_a:
            for seed in extension_seeds:
                add(case_id, "margin", alpha, seed, "prediction_boundary_extension")
    for nx in config["N_followers_extensions"]:
        case_id = f"even_idm_leader_N{int(nx)}"
        for controller in ("predictive", "margin"):
            for alpha in high_a:
                for seed in extension_seeds:
                    add(case_id, controller, alpha, seed, "chain_length_boundary_extension")
    return sorted(unique.values(), key=lambda r: r["run_id"])


def _make_config(base: Dict[str, object], spec: Dict[str, object]) -> Tuple[ChainConfig, Disturbance]:
    n = int(spec["N_followers"])
    count = max(1, int(round(n * float(base["active_fraction"]))))
    active = positions(n, count, str(spec["layout"]))
    controller = str(spec["controller"])
    if controller == "human":
        active = ()
    cfg = ChainConfig(
        n_followers=n, dt=float(base["dt_s"]), horizon=float(base["horizon_s"]),
        target_speed=float(base["target_speed_mps"]), base_gap=float(base["target_gap_m"]),
        safety_gap=float(base["safety_gap_m"]), loss_limit_normalized=float(base["normalized_loss_limit"]),
        reaction_delay=float(base["reaction_delay_s"]), info_delay=float(base["information_delay_s"]),
        actuator_delay=float(spec["command_delay_s"]), actuator_tau=float(base["actuator_tau_s"]),
        hdv_model=str(spec["hdv_model"]), equipped_indices=active, active_indices=active,
        controller=controller, predictor_horizon=float(base["predictor_horizon_s"]),
        predictor_grid_points=int(base["predictor_grid_points"]),
        margin_tracking_tolerance=float(base["margin_tracking_tolerance"]),
    )
    disturbance = Disturbance(
        float(spec["alpha_mps2"]), start=float(base["disturbance_start_s"]),
        half_duration=float(base["disturbance_half_duration_s"]),
        location=int(spec["disturbance_location"]),
    )
    return cfg, disturbance


def _json_cell(value: object) -> object:
    if isinstance(value, (list, tuple, dict)):
        return json.dumps(value, separators=(",", ":"), sort_keys=True)
    return value


def result_row(result: SimulationResult, spec: Dict[str, object]) -> Dict[str, object]:
    row = {k: _json_cell(v) for k, v in result.metadata.items()}
    row.update({k: _json_cell(v) for k, v in result.metrics.items()})
    row.update({k: _json_cell(v) for k, v in spec.items()})
    row["studies"] = ";".join(spec["studies"])
    row["layout"] = spec["layout"]
    row["case_id"] = spec["case_id"]
    return row


def _execute_one(payload: Tuple[Dict[str, object], Dict[str, object]]) -> Tuple[Dict[str, object], List[Dict[str, object]]]:
    config, spec = payload
    cfg, disturbance = _make_config(config, spec)
    result = simulate(cfg, disturbance, int(spec["seed"]))
    events = [{"run_id": spec["run_id"], **spec, **event} for event in result.failures]
    return result_row(result, spec), events


def _write_csv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    if not rows:
        return
    fields = sorted(set().union(*(row.keys() for row in rows)))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)


def _read_csv(path: Path) -> List[Dict[str, object]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def wilson(successes: int, trials: int, z: float = 1.96) -> Tuple[float, float]:
    if trials == 0:
        return float("nan"), float("nan")
    p = successes / trials
    den = 1.0 + z*z/trials
    centre = (p + z*z/(2*trials))/den
    half = z*np.sqrt(p*(1-p)/trials + z*z/(4*trials*trials))/den
    return float(max(0.0, centre-half)), float(min(1.0, centre+half))


def summarize(rows: Sequence[Dict[str, object]], out: Path) -> List[Dict[str, object]]:
    group_keys = ("case_id", "N_followers", "layout", "controller", "hdv_model",
                  "disturbance_location", "command_delay_s", "alpha_mps2")
    groups: Dict[Tuple[str, ...], List[Dict[str, object]]] = {}
    for row in rows:
        groups.setdefault(tuple(str(row[k]) for k in group_keys), []).append(row)
    summary: List[Dict[str, object]] = []
    metrics = ("normalized_loss", "throughput_loss_vehicle_m", "mean_loss_m_per_follower",
               "terminal_speed_error_mps", "terminal_gap_error_m", "min_gap_m",
               "peak_jerk_mps3", "pulse_internal_gain", "peak_constraint_usage")
    for key, group in groups.items():
        item: Dict[str, object] = dict(zip(group_keys, key))
        successes = sum(int(float(r["success"])) for r in group)
        lo, hi = wilson(successes, len(group))
        item.update({"trials": len(group), "successes": successes,
                     "success_rate": successes/len(group), "success_wilson_low": lo,
                     "success_wilson_high": hi})
        for metric in metrics:
            vals = np.asarray([float(r[metric]) for r in group])
            item[metric+"_median"] = float(np.median(vals))
            item[metric+"_q25"] = float(np.quantile(vals, 0.25))
            item[metric+"_q75"] = float(np.quantile(vals, 0.75))
        summary.append(item)
    _write_csv(out/"summary.csv", summary)
    return summary


def boundaries(rows: Sequence[Dict[str, object]], out: Path) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    keys = ("case_id", "N_followers", "layout", "controller", "hdv_model",
            "disturbance_location", "command_delay_s")
    groups: Dict[Tuple[str, ...], List[Dict[str, object]]] = {}
    for row in rows:
        groups.setdefault(tuple(str(row[k]) for k in keys), []).append(row)
    aggregate: List[Dict[str, object]] = []
    per_seed: List[Dict[str, object]] = []
    for key, group in groups.items():
        alphas = sorted({float(r["alpha_mps2"]) for r in group})
        rates = {a: np.mean([float(r["success"]) for r in group if float(r["alpha_mps2"]) == a]) for a in alphas}
        successful = [a for a in alphas if rates[a] >= 0.5]
        rebound = any(rates[alphas[j]] > rates[alphas[j-1]] + 1e-12 for j in range(1, len(alphas)))
        item: Dict[str, object] = dict(zip(keys, key))
        item.update({"empirical_alpha50_mps2": max(successful) if successful else 0.0,
                     "nonmonotone_success_rate": rebound, "amplitude_grid": json.dumps(alphas),
                     "success_rates": json.dumps([rates[a] for a in alphas]),
                     "scope": "finite paired-seed grid; not a robust certificate"})
        aggregate.append(item)
        for seed in sorted({int(float(r["seed"])) for r in group}):
            seed_rows = [r for r in group if int(float(r["seed"])) == seed]
            seed_alphas = sorted({float(r["alpha_mps2"]) for r in seed_rows})
            passed = sorted(float(r["alpha_mps2"]) for r in seed_rows if float(r["success"]) >= 0.5)
            pattern = [int(float(next(r["success"] for r in seed_rows if float(r["alpha_mps2"]) == a))) for a in seed_alphas]
            per_seed.append({**dict(zip(keys, key)), "seed": seed,
                             "empirical_boundary_mps2": max(passed) if passed else 0.0,
                             "nonmonotone": any(pattern[j] > pattern[j-1] for j in range(1, len(pattern))),
                             "success_pattern": json.dumps(pattern), "amplitude_grid": json.dumps(seed_alphas)})
    _write_csv(out/"boundaries.csv", aggregate)
    _write_csv(out/"boundaries_by_seed.csv", per_seed)
    return aggregate, per_seed


def prediction_analysis(rows: Sequence[Dict[str, object]], per_seed: Sequence[Dict[str, object]],
                        config: Dict[str, object], out: Path) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    dev_cases = {"even_idm_leader_d02", "front_idm_leader_d02", "even_idm_internal8_d02", "even_idm_leader_cmd0"}
    test_cases = {"middle_idm_leader_d02", "rear_idm_leader_d02", "even_idm_leader_cmd04", "even_ovm_leader_d02"}
    boundary_map = {(b["case_id"], int(b["seed"])): float(b["empirical_boundary_mps2"])
                    for b in per_seed if b["controller"] == "margin" and b["case_id"] in dev_cases | test_cases}
    records: List[Dict[str, object]] = []
    for case_id in sorted(dev_cases | test_cases):
        case_rows = [r for r in rows if r["case_id"] == case_id and r["controller"] == "margin"]
        for seed in sorted({int(float(r["seed"])) for r in case_rows}):
            probe = next((r for r in case_rows if int(float(r["seed"])) == seed and abs(float(r["alpha_mps2"])-0.25) < 1e-12), None)
            zero = next((r for r in case_rows if int(float(r["seed"])) == seed and abs(float(r["alpha_mps2"])) < 1e-12), None)
            if probe is None or zero is None or (case_id, seed) not in boundary_map:
                continue
            active = json.loads(probe["active_indices"])
            source = int(float(probe["disturbance_location"]))
            distance = min(abs(source-i) for i in active) if active else int(probe["N_followers"])
            m0 = float(zero["predicted_bottleneck_margin"]); mp = float(probe["predicted_bottleneck_margin"])
            slope = (mp-m0)/0.25
            reserve_proxy = (-m0/slope) if slope < -1e-9 else max(float(x) for x in config["core_alphas_mps2"])
            records.append({
                "case_id": case_id, "split": "development" if case_id in dev_cases else "test",
                "seed": seed, "observed_boundary_mps2": boundary_map[(case_id, seed)],
                "small_pulse_gain": float(probe["pulse_internal_gain"]),
                "directional_reserve_proxy_mps2": float(np.clip(reserve_proxy, 0.0, 5.0)),
                "predicted_margin_at_probe": mp, "distance_to_nearest_active_vehicle": distance,
                "average_time_headway_s": float(probe["target_gap_m"])/float(probe["target_speed_mps"]),
                "command_delay_s": float(probe["command_delay_s"]),
                "predicted_bottleneck_vehicle": int(float(probe["predicted_bottleneck_vehicle"])),
                "first_limit_vehicle_at_probe": int(float(probe["first_limit_vehicle"])),
                "predicted_bottleneck_constraint": probe["predicted_bottleneck_constraint"],
                "first_limit_kind_at_probe": probe["first_limit_kind"],
            })
    features = {
        "small_pulse_gain": ("small_pulse_gain",),
        "average_headway": ("average_time_headway_s",),
        "distance_to_control": ("distance_to_nearest_active_vehicle",),
        "directional_margin": ("directional_reserve_proxy_mps2", "small_pulse_gain",
                               "distance_to_nearest_active_vehicle", "command_delay_s"),
    }
    train = [r for r in records if r["split"] == "development"]
    test = [r for r in records if r["split"] == "test"]
    metrics: List[Dict[str, object]] = []
    for name, cols in features.items():
        x_train = np.asarray([[1.0]+[float(r[c]) for c in cols] for r in train])
        y_train = np.asarray([float(r["observed_boundary_mps2"]) for r in train])
        beta = np.linalg.lstsq(x_train, y_train, rcond=None)[0]
        errors = []; false = []
        for row in test:
            pred = float(np.clip(beta.dot([1.0]+[float(row[c]) for c in cols]), 0.0, 5.0))
            row["prediction_"+name] = pred
            observed = float(row["observed_boundary_mps2"])
            errors.append(abs(pred-observed)); false.append(float(pred > observed+1e-12))
        metrics.append({"predictor": name, "development_records": len(train), "test_records": len(test),
                        "test_MAE_mps2": float(np.mean(errors)),
                        "test_RMSE_mps2": float(np.sqrt(np.mean(np.square(errors)))),
                        "false_recoverable_fraction": float(np.mean(false)),
                        "coefficients": json.dumps(beta.tolist()),
                        "split_rule": "configuration-disjoint; all seeds within a case remain in one split"})
    failed_probe = [r for r in records if float(r["observed_boundary_mps2"]) < 0.25]
    hit_vehicle = np.mean([r["predicted_bottleneck_vehicle"] == r["first_limit_vehicle_at_probe"] for r in failed_probe]) if failed_probe else float("nan")
    hit_kind = np.mean([r["predicted_bottleneck_constraint"] == r["first_limit_kind_at_probe"] for r in failed_probe]) if failed_probe else float("nan")
    for item in metrics:
        item["failed_probe_records"] = len(failed_probe)
        item["bottleneck_vehicle_hit_fraction_failed_probe"] = hit_vehicle
        item["bottleneck_constraint_hit_fraction_failed_probe"] = hit_kind
    _write_csv(out/"prediction_records.csv", records)
    _write_csv(out/"prediction_metrics.csv", metrics)
    return records, metrics


def save_representative(rows: Sequence[Dict[str, object]], config: Dict[str, object], out: Path) -> Dict[str, object]:
    core = [r for r in rows if r["case_id"] == "even_idm_leader_d02"]
    alphas = sorted({float(r["alpha_mps2"]) for r in core if float(r["alpha_mps2"]) > 0})
    differences = []
    for alpha in alphas:
        margin_rate = np.mean([float(r["success"]) for r in core if r["controller"] == "margin" and float(r["alpha_mps2"]) == alpha])
        pred_rate = np.mean([float(r["success"]) for r in core if r["controller"] == "predictive" and float(r["alpha_mps2"]) == alpha])
        differences.append((margin_rate-pred_rate, alpha))
    best_difference = max(d[0] for d in differences)
    if best_difference > 0:
        alpha = min(d[1] for d in differences if d[0] == best_difference)
    else:
        # When the proposed and predictive methods tie, show the largest tested
        # amplitude at which at least one still has a majority of recoveries.
        viable = []
        for candidate in alphas:
            rates = [np.mean([float(r["success"]) for r in core
                              if r["controller"] == method and float(r["alpha_mps2"]) == candidate])
                     for method in ("predictive", "margin")]
            if max(rates) >= 0.5:
                viable.append(candidate)
        alpha = max(viable) if viable else min(alphas)
    candidates = [r for r in core if r["controller"] == "margin" and float(r["alpha_mps2"]) == alpha
                  and float(r["success"]) >= 0.5]
    contrasted = []
    for row in candidates:
        seed = int(float(row["seed"]))
        baseline = next(r for r in core if r["controller"] == "predictive"
                        and float(r["alpha_mps2"]) == alpha and int(float(r["seed"])) == seed)
        if float(baseline["success"]) < 0.5:
            contrasted.append(row)
    pool = contrasted if contrasted else (candidates if candidates else [r for r in core if r["controller"] == "margin" and float(r["alpha_mps2"]) == alpha])
    pool = sorted(pool, key=lambda r: float(r["normalized_loss"]))
    chosen = pool[(len(pool)-1)//2]
    seed = int(float(chosen["seed"]))
    case = declared_cases(config)["even_idm_leader_d02"]
    arrays: Dict[str, np.ndarray] = {}
    run_metrics: Dict[str, object] = {}
    for controller in CONTROLLERS:
        spec = {**case, "controller": controller, "alpha_mps2": alpha, "seed": seed}
        cfg, disturbance = _make_config(config, spec)
        result = simulate(cfg, disturbance, seed)
        prefix = controller+"_"
        arrays[prefix+"time"] = result.time
        arrays[prefix+"speed"] = result.speed
        arrays[prefix+"gap"] = result.gap
        arrays[prefix+"acceleration"] = result.acceleration
        arrays[prefix+"jerk"] = result.jerk
        arrays[prefix+"requested"] = result.command_requested
        arrays[prefix+"saturated"] = result.command_saturated
        arrays[prefix+"executed"] = result.command_executed
        run_metrics[controller] = result.metrics
    np.savez_compressed(out/"representative_trajectories.npz", **arrays)
    manifest = {"selection_rule": "positive advantage: amplitude maximizing margin-minus-predictive recovery rate; otherwise largest amplitude with majority recovery in either method; median margin normalized loss among contrasted seeds when available",
                "alpha_mps2": alpha, "seed": seed, "margin_minus_predictive_rate": best_difference,
                "contrasted_seed_pool": [int(float(r["seed"])) for r in contrasted], "metrics": run_metrics}
    (out/"representative_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return manifest


def run(config: Dict[str, object], out: Path) -> Dict[str, object]:
    out.mkdir(parents=True, exist_ok=True)
    specs = build_specs(config)
    checkpoint = out/"raw_runs_checkpoint.csv"
    rows = _read_csv(checkpoint)
    completed = {row["run_id"] for row in rows}
    failures_path = out/"failure_log.jsonl"
    start = time.time()
    pending = [spec for spec in specs if spec["run_id"] not in completed]
    workers = max(1, int(config.get("workers", 1)))
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
        payloads = ((config, spec) for spec in pending)
        for index, (row, events) in enumerate(executor.map(_execute_one, payloads, chunksize=1), start=1):
            rows.append(row)
            with failures_path.open("a", encoding="utf-8") as handle:
                for event in events:
                    handle.write(json.dumps(event, sort_keys=True)+"\n")
            if index % 20 == 0:
                _write_csv(checkpoint, rows)
                print("completed %d/%d unique simulations" % (len(rows), len(specs)), flush=True)
    _write_csv(out/"raw_runs.csv", rows)
    _write_csv(checkpoint, rows)
    summary = summarize(rows, out)
    aggregate, per_seed = boundaries(rows, out)
    prediction_records, prediction_metrics = prediction_analysis(rows, per_seed, config, out)
    representative = save_representative(rows, config, out)
    manifest = {
        "python": platform.python_version(), "numpy": np.__version__,
        "unique_simulations": len(rows), "declared_unique_specs": len(specs),
        "summary_rows": len(summary), "boundary_rows": len(aggregate),
        "prediction_records": len(prediction_records), "elapsed_s_this_invocation": time.time()-start,
        "config": config, "representative": representative,
        "notes": ["N_followers excludes one external leader.",
                  "All methods use the same closed pulse, physical limits and terminal definition within a case.",
                  "Grid thresholds and probabilities are empirical, not robust certificates.",
                  "A copied first-round data set and the 12-follower linear illustration remain separate provenance classes."],
    }
    (out/"run_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return manifest
