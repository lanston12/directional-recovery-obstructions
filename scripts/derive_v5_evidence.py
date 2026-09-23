from __future__ import annotations

import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT if (ROOT / "results" / "nonlinear_revision" / "raw_runs.csv").exists() else ROOT.parent
DATA = PROJECT / "results" / "nonlinear_revision"
OUT = ROOT / "derived_results"
OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(PROJECT / "src"))

from sparse_recovery.model import _delay_index, _parameters, simulate  # noqa: E402
from sparse_recovery.revision_experiment import (  # noqa: E402
    _make_config,
    declared_cases,
    positions,
)


CONFIG = json.loads((PROJECT / "configs" / "revision.json").read_text(encoding="utf-8"))
RAW = pd.read_csv(DATA / "raw_runs.csv")
PRED = pd.read_csv(DATA / "prediction_records.csv")
REP = json.loads((DATA / "representative_manifest.json").read_text(encoding="utf-8"))
TRACES = np.load(DATA / "representative_trajectories.npz")


def jdump(value: object) -> str:
    return json.dumps(value, separators=(",", ":"), sort_keys=True)


failures: dict[str, list[dict[str, object]]] = defaultdict(list)
with (DATA / "failure_log.jsonl").open(encoding="utf-8") as handle:
    for line in handle:
        event = json.loads(line)
        failures[str(event["run_id"])].append(event)


# Exact layout indices from the two implementations/configurations.
linear = json.loads((PROJECT / "results" / "revision_package" / "linear_demo.json").read_text(encoding="utf-8"))
layout_rows: list[dict[str, object]] = []
for name, active in linear["provenance"]["layouts_one_based"].items():
    layout_rows.append({
        "problem": "sampled-linear", "N_followers": 12, "layout": name,
        "active_count": len(active), "active_indices": jdump(active),
        "target_gap_m": 20.0, "target_speed_mps": 20.0,
        "horizon_s": 60.0, "terminal_window_s": 0.0,
        "disturbance": "four-coordinate homogeneous box; support on 0--4 s",
        "information_structure": "delayed predecessor-following affine rule",
    })
for n, layouts in ((40, ("front", "even", "middle", "rear")), (20, ("even",)), (80, ("even",))):
    count = max(1, int(round(n * float(CONFIG["active_fraction"]))))
    for name in layouts:
        active = list(positions(n, count, name))
        layout_rows.append({
            "problem": "nonlinear", "N_followers": n, "layout": name,
            "active_count": len(active), "active_indices": jdump(active),
            "target_gap_m": CONFIG["target_gap_m"], "target_speed_mps": CONFIG["target_speed_mps"],
            "horizon_s": CONFIG["horizon_s"], "terminal_window_s": 3.0,
            "disturbance": "requested leader pulse unless the case says follower 8",
            "information_structure": "0.2 s received-state delay; declared command delay; sparse active set",
        })
pd.DataFrame(layout_rows).to_csv(OUT / "layout_indices_and_config.csv", index=False)


# All-seed directional-mechanism classification for the existing layout comparison.
case_for_layout = {
    "front": "front_idm_leader_d02", "even": "even_idm_leader_d02",
    "middle": "middle_idm_leader_d02", "rear": "rear_idm_leader_d02",
}
all_alphas = sorted(set(map(float, CONFIG["core_alphas_mps2"] + CONFIG["high_alphas_mps2"])))
all_seeds = list(map(int, CONFIG["seeds"]))
mechanism_rows: list[dict[str, object]] = []
for layout, case_id in case_for_layout.items():
    subset = RAW[(RAW.case_id == case_id) & (RAW.controller == "margin")]
    lookup = {(float(r.alpha_mps2), int(r.seed)): r for r in subset.itertuples(index=False)}
    for alpha in all_alphas:
        for seed in all_seeds:
            row = lookup.get((alpha, seed))
            if row is None:
                mechanism_rows.append({
                    "layout": layout, "case_id": case_id, "alpha_mps2": alpha, "seed": seed,
                    "category": "D", "category_label": "not evaluated", "run_id": "",
                    "active_indices": "", "first_active": np.nan, "prefix_indices": "",
                    "success": np.nan, "witness_kind": "", "witness_vehicle": np.nan,
                    "witness_time_s": np.nan, "witness_gap_m": np.nan,
                    "witness_margin_m": np.nan,
                })
                continue
            active = list(map(int, json.loads(row.active_indices)))
            first_active = min(active) if active else math.inf
            prefix = list(range(1, int(first_active))) if math.isfinite(first_active) else list(range(1, int(row.N_followers) + 1))
            prefix_events = [e for e in failures.get(str(row.run_id), [])
                             if e["kind"] in ("safety_gap", "collision") and int(e["vehicle"]) in prefix]
            prefix_events.sort(key=lambda e: (float(e["time_s"]), 0 if e["kind"] == "safety_gap" else 1))
            witness = prefix_events[0] if prefix_events else None
            if float(row.success) >= 0.5:
                category, label = "A", "recovered"
            elif witness is not None and int(row.disturbance_location) == 0 and prefix:
                category, label = "B", "policy-independent prefix obstruction"
            else:
                category, label = "C", "controller failure without prefix obstruction"
            mechanism_rows.append({
                "layout": layout, "case_id": case_id, "alpha_mps2": alpha, "seed": seed,
                "category": category, "category_label": label, "run_id": row.run_id,
                "active_indices": row.active_indices, "first_active": first_active,
                "prefix_indices": jdump(prefix), "success": int(float(row.success)),
                "witness_kind": "" if witness is None else witness["kind"],
                "witness_vehicle": np.nan if witness is None else int(witness["vehicle"]),
                "witness_time_s": np.nan if witness is None else float(witness["time_s"]),
                "witness_gap_m": np.nan if witness is None else float(witness["value"]),
                "witness_margin_m": np.nan if witness is None else float(witness["value"]) - 4.0,
            })
mechanism = pd.DataFrame(mechanism_rows)
mechanism.to_csv(OUT / "nonlinear_directional_mechanism_runs.csv", index=False)
summary = (mechanism.groupby(["layout", "alpha_mps2", "category"]).size()
           .unstack(fill_value=0).reindex(columns=list("ABCD"), fill_value=0).reset_index())
summary["expected_seeds"] = 12
summary.to_csv(OUT / "nonlinear_directional_mechanism_counts.csv", index=False)


# Representative diagnostic replay provenance and collision/projection semantics.
rep_rows: list[dict[str, object]] = []
for method in ("wave", "predictive", "margin"):
    speed = TRACES[f"{method}_speed"]
    gap = TRACES[f"{method}_gap"]
    acc = TRACES[f"{method}_acceleration"]
    time = TRACES[f"{method}_time"]
    safety_idx = np.argwhere(gap[:, 1:] < 4.0)
    collision_idx = np.argwhere(gap[:, 1:] <= 0.0)
    first_safety = None if not len(safety_idx) else safety_idx[np.argmin(safety_idx[:, 0])]
    first_collision = None if not len(collision_idx) else collision_idx[np.argmin(collision_idx[:, 0])]
    raw_next = speed[:-1] + 0.1 * acc[1:]
    projected = (raw_next < -1e-12) | (raw_next > 35.0 + 1e-12)
    implied_acc = np.diff(speed, axis=0) / 0.1
    safety_k = len(time) if first_safety is None else int(first_safety[0])
    collision_k = len(time) if first_collision is None else int(first_collision[0])
    rep_rows.append({
        "method": method, "alpha_mps2": REP["alpha_mps2"], "seed": REP["seed"],
        "statistical_run_in_raw_table": int(((RAW.controller == method) &
                                               (RAW.case_id == "even_idm_leader_d02") &
                                               (RAW.alpha_mps2 == REP["alpha_mps2"]) &
                                               (RAW.seed == REP["seed"])).any()),
        "first_safety_time_s": np.nan if first_safety is None else float(time[first_safety[0]]),
        "first_safety_vehicle": np.nan if first_safety is None else int(first_safety[1] + 1),
        "first_safety_gap_m": np.nan if first_safety is None else float(gap[first_safety[0], first_safety[1] + 1]),
        "first_collision_time_s": np.nan if first_collision is None else float(time[first_collision[0]]),
        "first_collision_vehicle": np.nan if first_collision is None else int(first_collision[1] + 1),
        "first_collision_gap_m": np.nan if first_collision is None else float(gap[first_collision[0], first_collision[1] + 1]),
        "speed_projection_count": int(projected.sum()),
        "speed_projection_before_first_safety": int(projected[:safety_k].sum()),
        "speed_projection_before_first_collision": int(projected[:collision_k].sum()),
        "max_projection_acceleration_difference_mps2": float(np.max(np.abs(implied_acc - acc[1:]))),
        "minimum_gap_m_including_numerical_continuation": float(np.nanmin(gap[:, 1:])),
    })
pd.DataFrame(rep_rows).to_csv(OUT / "representative_provenance_and_events.csv", index=False)


def endpoint_approximation(cfg, gap, v, vp, a, ap, command):
    h = cfg.predictor_horizon
    rho = 1.0 - np.exp(-h / max(cfg.actuator_tau, cfg.dt))
    a1 = float(np.clip(a + rho * (command - a), -cfg.brake_max, cfg.accel_max))
    v1 = float(v + h * a1)
    g1 = float(gap + h * (vp - v) + 0.5 * h * h * (ap - a1))
    return a1, v1, g1


def exact_lag_reference(cfg, gap, v, vp, a, ap, command):
    h, tau = cfg.predictor_horizon, cfg.actuator_tau
    exp_term = np.exp(-h / tau)
    a1 = command + (a - command) * exp_term
    self_dx = v * h + 0.5 * command * h * h + (a - command) * (tau * h - tau * tau * (1.0 - exp_term))
    pred_dx = vp * h + 0.5 * ap * h * h
    v1 = v + command * h + (a - command) * tau * (1.0 - exp_term)
    return float(a1), float(v1), float(gap + pred_dx - self_dx)


def discrete_rollout(result, cfg, k: int, vehicle: int, command: float):
    steps = int(round(cfg.predictor_horizon / cfg.dt))
    x_ego = float(result.position[k, vehicle])
    v_ego = float(result.speed[k, vehicle])
    a_ego = float(result.acceleration[k, vehicle])
    projected = 0
    for offset in range(steps):
        kk = k + offset
        command_k = _delay_index(kk, cfg.actuator_delay, cfg.dt)
        delayed = float(result.command_saturated[command_k, vehicle]) if command_k < k else command
        da = (delayed - a_ego) / max(cfg.actuator_tau, cfg.dt)
        a_ego = float(np.clip(a_ego + cfg.dt * np.clip(da, -cfg.jerk_max, cfg.jerk_max),
                              -cfg.brake_max, cfg.accel_max))
        raw_v = v_ego + cfg.dt * a_ego
        if raw_v < 0.0 or raw_v > cfg.vmax:
            projected += 1
        v_ego = float(np.clip(raw_v, 0.0, cfg.vmax))
        x_ego += cfg.dt * v_ego
    pred_x = float(result.position[k + steps, vehicle - 1])
    return a_ego, v_ego, pred_x - x_ego - cfg.vehicle_length, projected


def projection_stats(result, cfg):
    raw_next = result.speed[:-1] + cfg.dt * result.acceleration[1:]
    mask = (raw_next < -1e-12) | (raw_next > cfg.vmax + 1e-12)
    implied = np.diff(result.speed, axis=0) / cfg.dt
    gap_hit = np.argwhere(result.gap[:, 1:] < cfg.safety_gap)
    first_safety_k = int(gap_hit[:, 0].min()) if len(gap_hit) else result.speed.shape[0]
    before = mask[:first_safety_k]
    return int(mask.sum()), int(before.sum()), float(np.max(np.abs(implied - result.acceleration[1:])))


# Focused diagnostic replays: the 24 paired high-amplitude predictive runs.
paired_rows: list[dict[str, object]] = []
decision_rows: list[dict[str, object]] = []
core_case = declared_cases(CONFIG)["even_idm_leader_d02"]
for controller in ("predictive", "margin"):
    for seed in all_seeds:
        spec = {**core_case, "controller": controller, "alpha_mps2": 4.5, "seed": seed}
        cfg, disturbance = _make_config(CONFIG, spec)
        result = simulate(cfg, disturbance, seed)
        pcount, pbefore, pmax = projection_stats(result, cfg)
        terminal_start = max(0, len(result.time) - int(np.ceil(cfg.recovery_window / cfg.dt)))
        paired_rows.append({
            "controller": controller, "seed": seed, "success": int(result.metrics["success"]),
            "endpoint_gap_error_m": float(np.nanmax(np.abs(result.gap[-1, 1:] - cfg.base_gap))),
            "window_max_gap_error_m": float(np.nanmax(np.abs(result.gap[terminal_start:, 1:] - cfg.base_gap))),
            "normalized_loss": float(result.metrics["normalized_loss"]),
            "speed_projection_count": pcount, "speed_projection_before_first_safety": pbefore,
            "max_projection_acceleration_difference_mps2": pmax,
        })
        candidates = np.linspace(cfg.input_min, cfg.input_max, cfg.predictor_grid_points)
        hsteps = int(round(cfg.predictor_horizon / cfg.dt))
        active = list(map(int, result.metadata["active_indices"]))
        for k in range(len(result.time) - hsteps):
            hi = _delay_index(k, cfg.info_delay, cfg.dt)
            for vehicle in active:
                gap0 = result.position[hi, vehicle - 1] - result.position[hi, vehicle] - cfg.vehicle_length
                v0, vp0 = result.speed[hi, vehicle], result.speed[hi, vehicle - 1]
                a0, ap0 = result.acceleration[hi, vehicle], result.acceleration[hi, vehicle - 1]
                rho = 1.0 - np.exp(-cfg.predictor_horizon / max(cfg.actuator_tau, cfg.dt))
                candidate_a = np.clip(a0 + rho * (candidates - a0), -cfg.brake_max, cfg.accel_max)
                candidate_v = v0 + cfg.predictor_horizon * candidate_a
                candidate_g = gap0 + cfg.predictor_horizon * (vp0 - v0) + 0.5 * cfg.predictor_horizon**2 * (ap0 - candidate_a)
                safe = (candidate_g >= cfg.safety_gap + cfg.predictor_gap_buffer) & (candidate_v >= 0) & (candidate_v <= cfg.vmax)
                selected = float(np.clip(result.command_requested[k, vehicle], cfg.input_min, cfg.input_max))
                _, old_v, old_g = endpoint_approximation(cfg, gap0, v0, vp0, a0, ap0, selected)
                _, exact_v, exact_g = exact_lag_reference(cfg, gap0, v0, vp0, a0, ap0, selected)
                _, disc_v, disc_g, rollout_projected = discrete_rollout(result, cfg, k, vehicle, selected)
                decision_rows.append({
                    "controller": controller, "seed": seed, "time_s": float(result.time[k]),
                    "vehicle": vehicle, "selected_command_mps2": selected,
                    "empty_feasible_pool": int(not safe.any()),
                    "old_endpoint_speed_mps": old_v, "exact_lag_speed_mps": exact_v,
                    "discrete_rollout_speed_mps": disc_v,
                    "old_endpoint_gap_m": old_g, "exact_lag_gap_m": exact_g,
                    "discrete_rollout_gap_m": disc_g,
                    "old_minus_exact_speed_mps": old_v - exact_v,
                    "old_minus_discrete_speed_mps": old_v - disc_v,
                    "old_minus_discrete_gap_m": old_g - disc_g,
                    "old_screen_feasible_selected": int(old_g >= 5.0 and 0 <= old_v <= 35.0),
                    "discrete_endpoint_feasible_selected": int(disc_g >= 5.0 and 0 <= disc_v <= 35.0),
                    "rollout_speed_projection_count": rollout_projected,
                })
paired = pd.DataFrame(paired_rows)
paired.to_csv(OUT / "paired_alpha45_replays.csv", index=False)
decisions = pd.DataFrame(decision_rows)
decisions.to_csv(OUT / "endpoint_predictor_decisions.csv", index=False)


wide = paired.pivot(index="seed", columns="controller")
paired_effects = pd.DataFrame({
    "seed": wide.index,
    "delta_endpoint_gap_margin_minus_predictive_m":
        wide["endpoint_gap_error_m"]["margin"] - wide["endpoint_gap_error_m"]["predictive"],
    "delta_window_gap_margin_minus_predictive_m":
        wide["window_max_gap_error_m"]["margin"] - wide["window_max_gap_error_m"]["predictive"],
    "delta_normalized_loss_margin_minus_predictive":
        wide["normalized_loss"]["margin"] - wide["normalized_loss"]["predictive"],
})
paired_effects.to_csv(OUT / "paired_alpha45_effects.csv", index=False)

rng = np.random.RandomState(20260922)
paired_summary: dict[str, object] = {}
for col in paired_effects.columns[1:]:
    values = paired_effects[col].to_numpy(float)
    boot = np.asarray([np.median(rng.choice(values, len(values), replace=True)) for _ in range(20000)])
    paired_summary[col] = {
        "median": float(np.median(values)), "mean": float(np.mean(values)),
        "bootstrap_median_95_interval": [float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))],
        "negative_count": int((values < 0).sum()), "zero_count": int(np.isclose(values, 0).sum()),
        "positive_count": int((values > 0).sum()), "n_pairs": int(len(values)),
    }

decision_summary = {
    "decisions": int(len(decisions)),
    "empty_feasible_pool_count": int(decisions.empty_feasible_pool.sum()),
    "empty_feasible_pool_fraction": float(decisions.empty_feasible_pool.mean()),
    "old_screen_feasible_but_discrete_endpoint_infeasible_count": int(
        ((decisions.old_screen_feasible_selected == 1) & (decisions.discrete_endpoint_feasible_selected == 0)).sum()),
    "median_abs_old_minus_exact_speed_mps": float(np.median(np.abs(decisions.old_minus_exact_speed_mps))),
    "q95_abs_old_minus_discrete_speed_mps": float(np.quantile(np.abs(decisions.old_minus_discrete_speed_mps), 0.95)),
    "q95_abs_old_minus_discrete_gap_m": float(np.quantile(np.abs(decisions.old_minus_discrete_gap_m), 0.95)),
}
diagnostic_subset = decisions[(np.abs(decisions.selected_command_mps2) > 0.1) |
                              ((decisions.time_s >= 5.0) & (decisions.time_s <= 15.0))]
decision_summary.update({
    "diagnostic_subset_definition": "abs(selected command)>0.1 or 5<=time<=15 s",
    "diagnostic_subset_decisions": int(len(diagnostic_subset)),
    "diagnostic_subset_mean_abs_old_minus_exact_speed_mps":
        float(np.mean(np.abs(diagnostic_subset.old_minus_exact_speed_mps))),
    "diagnostic_subset_max_abs_old_minus_exact_speed_mps":
        float(np.max(np.abs(diagnostic_subset.old_minus_exact_speed_mps))),
    "diagnostic_subset_mean_abs_old_minus_discrete_gap_m":
        float(np.mean(np.abs(diagnostic_subset.old_minus_discrete_gap_m))),
    "diagnostic_subset_max_abs_old_minus_discrete_gap_m":
        float(np.max(np.abs(diagnostic_subset.old_minus_discrete_gap_m))),
    "reference_a0_u1_old_speed_increment_mps":
        float(1.2 * (1.0 - np.exp(-1.2 / 0.35))),
    "reference_a0_u1_exact_speed_increment_mps":
        float(1.2 - 0.35 * (1.0 - np.exp(-1.2 / 0.35))),
})
(OUT / "paired_and_predictor_summary.json").write_text(
    json.dumps({"paired_effects": paired_summary, "predictor": decision_summary}, indent=2), encoding="utf-8")


# Smaller diagnostic replays for disturbance-source semantics and finite-horizon chain reach.
source_rows: list[dict[str, object]] = []
cases = declared_cases(CONFIG)
for case_id in ("even_idm_leader_d02", "even_idm_internal8_d02"):
    for seed in map(int, CONFIG["test_seeds"]):
        case = cases[case_id]
        spec = {**case, "controller": "margin", "alpha_mps2": 2.1, "seed": seed}
        cfg, disturbance = _make_config(CONFIG, spec)
        result = simulate(cfg, disturbance, seed)
        source = int(disturbance.location)
        neg_acc = np.maximum(-result.acceleration[:, source], 0.0)
        source_rows.append({
            "case_id": case_id, "seed": seed, "source_vehicle": source,
            "injection": "external leader request" if source == 0 else "additive HDV request before saturation",
            "speed_drop_mps": float(cfg.target_speed - np.min(result.speed[:, source])),
            "negative_acceleration_integral_mps": float(np.trapz(neg_acc, result.time)),
            "negative_acceleration_duration_s": float(cfg.dt * np.count_nonzero(result.acceleration[:, source] < -1e-9)),
            "pulse_window_negative_acceleration_integral_mps":
                float(np.trapz(neg_acc[(result.time >= 5.0) & (result.time <= 8.0)],
                                 result.time[(result.time >= 5.0) & (result.time <= 8.0)])),
            "pulse_window_negative_acceleration_duration_s":
                float(cfg.dt * np.count_nonzero(result.acceleration[(result.time >= 5.0) & (result.time <= 8.0), source] < -1e-9)),
            "success": int(result.metrics["success"]),
        })
pd.DataFrame(source_rows).to_csv(OUT / "source_protocol_replays_alpha21.csv", index=False)

reach_rows: list[dict[str, object]] = []
for n in (20, 40, 80):
    case_id = "even_idm_leader_d02" if n == 40 else f"even_idm_leader_N{n}"
    case = cases[case_id]
    spec = {**case, "controller": "margin", "alpha_mps2": 4.0, "seed": 97}
    cfg, disturbance = _make_config(CONFIG, spec)
    result = simulate(cfg, disturbance, 97)
    dev = np.abs(result.speed[:, 1:] - cfg.target_speed)
    affected = dev.max(axis=0) >= 1.0
    arrivals = []
    for j in range(n):
        hits = np.where(dev[:, j] >= 1.0)[0]
        if len(hits):
            arrivals.append(float(result.time[hits[0]]))
    reach_rows.append({
        "N_followers": n, "active_indices": jdump(result.metadata["active_indices"]),
        "affected_threshold_mps": 1.0, "affected_count": int(affected.sum()),
        "affected_fraction": float(affected.mean()),
        "last_arrival_time_s": np.nan if not arrivals else max(arrivals),
        "normalized_loss_full_N": float(result.metrics["normalized_loss"]),
        "success": int(result.metrics["success"]),
    })
pd.DataFrame(reach_rows).to_csv(OUT / "finite_horizon_chain_reach_alpha40_seed97.csv", index=False)


# Core common-random-number parameter table.
parameter_rows: list[dict[str, object]] = []
base_spec = {**core_case, "controller": "margin", "alpha_mps2": 0.0, "seed": 0}
base_cfg, _ = _make_config(CONFIG, base_spec)
for seed in all_seeds:
    pars = _parameters(base_cfg, np.random.RandomState(seed))
    for vehicle in range(1, base_cfg.total_vehicles):
        parameter_rows.append({"seed": seed, "vehicle": vehicle, **{key: float(value[vehicle]) for key, value in pars.items()}})
pd.DataFrame(parameter_rows).to_csv(OUT / "core_regenerated_parameter_draws.csv", index=False)


# Mean-parameter Jacobian diagnostic, verified by central finite differences after equilibrium calibration.
def mean_idm(gap, v, vp):
    A, B, s0, T = 1.25, 2.0, 2.0, 0.9
    desired0 = s0 + 20.0 * T
    v0 = 20.0 / (1.0 - (desired0 / 24.0) ** 2) ** 0.25
    desired = s0 + v * T + v * (v - vp) / (2 * math.sqrt(A * B))
    return A * (1 - (v / v0) ** 4 - (desired / gap) ** 2)


def mean_ovm(gap, v, vp):
    width, center, kov, kdv = 6.0, 18.0, 0.09, 0.55
    tanh0 = math.tanh(center / width)
    shape0 = math.tanh((24.0 - center) / width) + tanh0
    v0 = 20.0 * (1 + tanh0) / shape0
    vopt = v0 * (math.tanh((gap - center) / width) + tanh0) / (1 + tanh0)
    return kov * (vopt - v) + kdv * (vp - v)


derivative_rows = []
eps = 1e-5
for name, fun in (("IDM", mean_idm), ("OVM/FVD", mean_ovm)):
    base = [24.0, 20.0, 20.0]
    vals = []
    for q in range(3):
        hi, lo = base.copy(), base.copy()
        hi[q] += eps
        lo[q] -= eps
        vals.append((fun(*hi) - fun(*lo)) / (2 * eps))
    derivative_rows.append({"model": name, "df_ds_s-2": vals[0], "df_dv_s-1": vals[1], "df_dvpre_s-1": vals[2]})
pd.DataFrame(derivative_rows).to_csv(OUT / "mean_equilibrium_jacobians.csv", index=False)


# Configuration-wise prediction diagnostics and OLS conditioning.
pred_case_rows = []
for case_id, group in PRED[PRED.split == "test"].groupby("case_id"):
    observed = group.observed_boundary_mps2.to_numpy(float)
    predicted = group.prediction_directional_margin.to_numpy(float)
    pred_case_rows.append({
        "case_id": case_id, "records": len(group), "seeds": jdump(group.seed.astype(int).tolist()),
        "MAE_mps2": float(np.mean(np.abs(predicted - observed))),
        "RMSE_mps2": float(np.sqrt(np.mean((predicted - observed) ** 2))),
        "recorded_endpoint_exceedance_fraction": float(np.mean(predicted > observed + 1e-12)),
        "prediction_at_upper_clip_count": int(np.isclose(predicted, 5.0).sum()),
    })
pd.DataFrame(pred_case_rows).to_csv(OUT / "prediction_metrics_by_configuration.csv", index=False)
train = PRED[PRED.split == "development"]
X = np.column_stack([
    np.ones(len(train)), train.directional_reserve_proxy_mps2,
    train.small_pulse_gain, train.distance_to_nearest_active_vehicle, train.command_delay_s,
])
(OUT / "prediction_design_diagnostics.json").write_text(json.dumps({
    "rows": int(X.shape[0]), "columns": int(X.shape[1]), "rank": int(np.linalg.matrix_rank(X)),
    "condition_number_unstandardized": float(np.linalg.cond(X)),
    "reserve_proxy_at_lower_clip": int(np.isclose(train.directional_reserve_proxy_mps2, 0).sum()),
    "reserve_proxy_at_upper_clip": int(np.isclose(train.directional_reserve_proxy_mps2, 5).sum()),
    "definition": "m0 is the minimum normalized endpoint margin over vehicle/time/constraint stored at alpha=0; q=(m0.25-m0)/0.25; distance is absolute index distance from source to nearest active CAV",
}, indent=2), encoding="utf-8")


# Linear constraint candidates and delay-dependent limiting rows.
linear_rows = []
for layout, case in linear["cases"].items():
    for row in case["rows"]:
        linear_rows.append({"layout": layout, **{k: row.get(k) for k in ("constraint", "radius", "slack", "coefficient", "sample", "vehicle")}})
pd.DataFrame(linear_rows).to_csv(OUT / "linear_constraint_candidates.csv", index=False)
delay_rows = []
for item in linear["delay_scan"]:
    lim = item["limiting"]
    delay_rows.append({"layout": item["layout"], "delay_s": item["delay_s"], "radius": item["radius"],
                       "constraint": lim["constraint"], "vehicle": lim["vehicle"], "sample": lim["sample"]})
pd.DataFrame(delay_rows).to_csv(OUT / "linear_delay_limiting_rows.csv", index=False)


(OUT / "PROVENANCE.md").write_text(
    """# v5 derived evidence provenance\n\n"
    "All classification tables read the immutable 1,848-row `results/nonlinear_revision/raw_runs.csv` and its JSONL failure log. "
    "The representative wave/predictive/margin arrays are diagnostic replays created by the original `save_representative()` function and are not extra rows in the 1,848-run table. "
    "This script adds 24 paired alpha=4.5 replays, 12 source-protocol replays, and three chain-reach replays. "
    "They use the unchanged original model/controller functions and are diagnostic repetitions, not independent validation samples. "
    "No original CSV, JSON, NPZ, code, or v4 artifact is modified.\n""",
    encoding="utf-8",
)

print(json.dumps({
    "mechanism_rows": len(mechanism), "paired_replays": len(paired),
    "predictor_decisions": len(decisions), "source_replays": len(source_rows),
    "chain_replays": len(reach_rows), "output": str(OUT),
}, indent=2))
