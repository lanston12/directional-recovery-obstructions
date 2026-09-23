from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


@dataclass(frozen=True)
class Disturbance:
    """Closed acceleration-command pulse.

    ``alpha`` is the peak magnitude in m s^-2. The command is ``-alpha`` for
    ``half_duration`` seconds and ``+alpha`` for the same duration, so its
    commanded acceleration integral is zero. ``location=0`` denotes the
    external leader; positive indices denote followers.
    """

    alpha: float
    start: float = 5.0
    half_duration: float = 1.5
    location: int = 0

    @property
    def end(self) -> float:
        return self.start + 2.0 * self.half_duration

    def value(self, time_s: float) -> float:
        if self.start <= time_s < self.start + self.half_duration:
            return -float(self.alpha)
        if self.start + self.half_duration <= time_s < self.end:
            return float(self.alpha)
        return 0.0


@dataclass
class ChainConfig:
    """Configuration with an explicit follower count.

    ``n_followers`` excludes the external leader. The optional legacy ``n``
    field is interpreted as total vehicles and exists only so that the copied
    first-round smoke tests remain executable.
    """

    n_followers: int = 40
    n: Optional[int] = None
    dt: float = 0.1
    horizon: float = 60.0
    target_speed: float = 20.0
    vehicle_length: float = 5.0
    base_gap: float = 24.0
    min_gap: float = 2.0
    safety_gap: float = 4.0
    vmax: float = 35.0
    accel_max: float = 2.0
    brake_max: float = 4.5
    jerk_max: float = 5.0
    input_min: float = -5.0
    input_max: float = 2.5
    reaction_delay: float = 0.7
    info_delay: float = 0.2
    actuator_delay: float = 0.2
    actuator_tau: float = 0.35
    vehicle_mass_kg: float = 1500.0
    heavy_indices: Sequence[int] = field(default_factory=tuple)
    heavy_mass_kg: float = 30000.0
    traction_force_max_n: float = 60000.0
    brake_force_max_n: float = 135000.0
    traction_power_max_w: float = 300000.0
    hdv_model: str = "idm"
    heterogeneity: float = 0.12
    equipped_indices: Sequence[int] = field(default_factory=tuple)
    active_indices: Sequence[int] = field(default_factory=tuple)
    cav_indices: Sequence[int] = field(default_factory=tuple)  # legacy alias
    controller: str = "human"
    recovery_epsilon_v: float = 1.0
    recovery_epsilon_gap: float = 5.0
    recovery_epsilon_a: float = 0.5
    recovery_window: float = 3.0
    loss_limit_normalized: float = 0.03
    loss_budget: float = float("inf")
    predictor_horizon: float = 1.2
    predictor_gap_buffer: float = 1.0
    predictor_grid_points: int = 31
    margin_tracking_tolerance: float = 0.01

    @property
    def total_vehicles(self) -> int:
        return int(self.n) if self.n is not None else int(self.n_followers) + 1

    @property
    def followers(self) -> int:
        return self.total_vehicles - 1


@dataclass
class SimulationResult:
    time: np.ndarray
    position: np.ndarray
    speed: np.ndarray
    acceleration: np.ndarray
    jerk: np.ndarray
    command_requested: np.ndarray
    command_saturated: np.ndarray
    command_executed: np.ndarray
    gap: np.ndarray
    metrics: Dict[str, object]
    failures: List[Dict[str, object]]
    metadata: Dict[str, object]


def _delay_index(k: int, delay: float, dt: float) -> int:
    return max(0, k - int(np.ceil(max(0.0, delay) / dt)))


def _idm(gap: np.ndarray, v: np.ndarray, vp: np.ndarray, pars: Dict[str, np.ndarray]) -> np.ndarray:
    g = np.maximum(gap, 0.1)
    dv = v - vp
    root = 2.0 * np.sqrt(np.maximum(pars["a"] * pars["b"], 1e-6))
    desired = pars["s0"] + v * pars["T"] + v * dv / root
    return pars["a"] * (1.0 - (v / pars["v0"]) ** 4 - (desired / g) ** 2)


def _ovm(gap: np.ndarray, v: np.ndarray, vp: np.ndarray, pars: Dict[str, np.ndarray]) -> np.ndarray:
    """Matched-scale optimal/full-velocity-difference comparison family."""
    width = np.maximum(pars["ov_width"], 0.5)
    tanh0 = np.tanh(pars["ov_center"] / width)
    shape = np.tanh((gap - pars["ov_center"]) / width) + tanh0
    v_opt = pars["v0"] * shape / np.maximum(1.0 + tanh0, 1e-6)
    return pars["ov_k"] * (v_opt - v) + pars["dv_k"] * (vp - v)


def _parameters(cfg: ChainConfig, rng: np.random.RandomState) -> Dict[str, np.ndarray]:
    n = cfg.total_vehicles
    h = cfg.heterogeneity

    def positive(mean: float, floor: float) -> np.ndarray:
        return np.maximum(floor, mean * (1.0 + h * rng.uniform(-1.0, 1.0, size=n)))

    pars = {
        "a": positive(1.25, 0.4), "b": positive(2.0, 0.8),
        "s0": positive(2.0, 0.8), "T": positive(0.9, 0.45),
        "v0": np.empty(n), "ov_width": positive(6.0, 2.0),
        "ov_center": positive(18.0, 8.0),
        # Means chosen so the equilibrium Jacobian is on the IDM response scale.
        "ov_k": positive(0.09, 0.04), "dv_k": positive(0.55, 0.20),
    }
    pars["T"] = np.minimum(pars["T"], (0.95 * cfg.base_gap - pars["s0"]) / cfg.target_speed)
    desired = pars["s0"] + cfg.target_speed * pars["T"]
    free_term = np.maximum(1e-5, 1.0 - (desired / cfg.base_gap) ** 2)
    pars["v0"] = cfg.target_speed / free_term ** 0.25
    if cfg.hdv_model == "ovm":
        tanh0 = np.tanh(pars["ov_center"] / pars["ov_width"])
        shape = np.tanh((cfg.base_gap - pars["ov_center"]) / pars["ov_width"]) + tanh0
        pars["v0"] = cfg.target_speed * (1.0 + tanh0) / np.maximum(shape, 1e-4)
    return pars


def _active_sets(cfg: ChainConfig) -> Tuple[set, set]:
    n = cfg.total_vehicles
    active_source = cfg.active_indices if cfg.active_indices else cfg.cav_indices
    active = {int(i) for i in active_source if 0 < int(i) < n}
    equipped_source = cfg.equipped_indices if cfg.equipped_indices else tuple(active)
    equipped = {int(i) for i in equipped_source if 0 < int(i) < n}
    if not active.issubset(equipped):
        raise ValueError("active_indices must be a subset of equipped_indices")
    return equipped, active


def _acc_command(gap: float, v: float, vp: float) -> float:
    desired_gap = 3.0 + 1.05 * v
    return 0.18 * (gap - desired_gap) + 0.75 * (vp - v)


def _candidate_diagnostics(cfg: ChainConfig, gap: float, v: float, vp: float,
                           a: float, ap: float, command: float) -> Dict[str, float]:
    horizon = cfg.predictor_horizon
    response = 1.0 - np.exp(-horizon / max(cfg.actuator_tau, cfg.dt))
    a1 = float(np.clip(a + response * (command - a), -cfg.brake_max, cfg.accel_max))
    v1 = float(v + horizon * a1)
    g1 = float(gap + horizon * (vp - v) + 0.5 * horizon ** 2 * (ap - a1))
    scale_gap = max(cfg.base_gap - cfg.safety_gap, 1e-6)
    margins = {
        "gap_lower": (g1 - cfg.safety_gap) / scale_gap,
        "speed_lower": v1 / max(cfg.target_speed, 1e-6),
        "speed_upper": (cfg.vmax - v1) / max(cfg.vmax - cfg.target_speed, 1e-6),
        "acceleration_upper": (cfg.accel_max - a1) / max(cfg.accel_max, 1e-6),
        "acceleration_lower": (a1 + cfg.brake_max) / max(cfg.brake_max, 1e-6),
        "command_upper": (cfg.input_max - command) / max(cfg.input_max, 1e-6),
        "command_lower": (command - cfg.input_min) / max(abs(cfg.input_min), 1e-6),
    }
    limiting = min(margins, key=margins.get)
    return {"pred_gap": g1, "pred_speed": v1, "pred_acceleration": a1,
            "min_margin": float(margins[limiting]), "constraint": limiting}


def _controlled_command(kind: str, cfg: ChainConfig, gap: float, v: float, vp: float,
                        a: float, ap: float) -> Tuple[float, Dict[str, float]]:
    base = _acc_command(gap, v, vp)
    if kind == "acc":
        raw = base
        return raw, _candidate_diagnostics(cfg, gap, v, vp, a, ap, float(np.clip(raw, cfg.input_min, cfg.input_max)))
    if kind == "wave":
        raw = base + 0.35 * (cfg.target_speed - v)
        return raw, _candidate_diagnostics(cfg, gap, v, vp, a, ap, float(np.clip(raw, cfg.input_min, cfg.input_max)))
    if kind not in ("predictive", "margin"):
        raise ValueError("unknown controller: %s" % kind)
    candidates = np.linspace(cfg.input_min, cfg.input_max, cfg.predictor_grid_points)
    horizon = cfg.predictor_horizon
    response = 1.0 - np.exp(-horizon / max(cfg.actuator_tau, cfg.dt))
    a1 = np.clip(a + response * (candidates-a), -cfg.brake_max, cfg.accel_max)
    v1 = v + horizon*a1
    g1 = gap + horizon*(vp-v) + 0.5*horizon**2*(ap-a1)
    margin_rows = np.vstack((
        (g1-cfg.safety_gap)/max(cfg.base_gap-cfg.safety_gap, 1e-6),
        v1/max(cfg.target_speed, 1e-6),
        (cfg.vmax-v1)/max(cfg.vmax-cfg.target_speed, 1e-6),
        (cfg.accel_max-a1)/max(cfg.accel_max, 1e-6),
        (a1+cfg.brake_max)/max(cfg.brake_max, 1e-6),
        (cfg.input_max-candidates)/max(cfg.input_max, 1e-6),
        (candidates-cfg.input_min)/max(abs(cfg.input_min), 1e-6),
    ))
    constraint_names = ("gap_lower", "speed_lower", "speed_upper", "acceleration_upper",
                        "acceleration_lower", "command_upper", "command_lower")
    min_margins = margin_rows.min(axis=0)
    limiting_rows = margin_rows.argmin(axis=0)
    safe_mask = ((g1 >= cfg.safety_gap+cfg.predictor_gap_buffer) & (v1 >= 0.0) & (v1 <= cfg.vmax))
    pool = np.where(safe_mask)[0]
    if pool.size == 0:
        pool = np.arange(candidates.size)
    tracking = ((g1-cfg.base_gap)/8.0)**2 + ((v1-cfg.target_speed)/4.0)**2 + 0.025*candidates**2
    tracking_choice = int(pool[np.argmin(tracking[pool])])
    if kind == "predictive":
        chosen = tracking_choice
    else:
        # Maximise the worst physical reserve only among commands with bounded
        # degradation of the same tracking objective used by the baseline.
        # This prevents a large but unrecovered gap from being scored as success.
        cost_cap = tracking[tracking_choice] + cfg.margin_tracking_tolerance
        near_optimal = pool[tracking[pool] <= cost_cap]
        best_margin = np.max(min_margins[near_optimal])
        tied = near_optimal[np.isclose(min_margins[near_optimal], best_margin)]
        chosen = int(tied[np.argmin(tracking[tied])])
    diagnostics = {"pred_gap": float(g1[chosen]), "pred_speed": float(v1[chosen]),
                   "pred_acceleration": float(a1[chosen]),
                   "min_margin": float(min_margins[chosen]),
                   "constraint": constraint_names[int(limiting_rows[chosen])]}
    return float(candidates[chosen]), diagnostics


def _constraint_peak(cfg: ChainConfig, gap: np.ndarray, v: np.ndarray, a: np.ndarray,
                     jerk: np.ndarray, command: np.ndarray) -> Dict[str, object]:
    """Return the most occupied realized constraint row (one is its boundary)."""
    best = (-float("inf"), "none", 0, 1)
    for k in range(v.shape[0]):
        for i in range(1, v.shape[1]):
            rows = (
                ((cfg.base_gap-gap[k, i])/max(cfg.base_gap-cfg.safety_gap, 1e-9), "gap_lower"),
                ((cfg.target_speed-v[k, i])/max(cfg.target_speed, 1e-9), "speed_lower"),
                ((v[k, i]-cfg.target_speed)/max(cfg.vmax-cfg.target_speed, 1e-9), "speed_upper"),
                (a[k, i]/max(cfg.accel_max, 1e-9), "acceleration_upper"),
                (-a[k, i]/max(cfg.brake_max, 1e-9), "acceleration_lower"),
                (abs(jerk[k, i])/max(cfg.jerk_max, 1e-9), "jerk"),
                (command[k, i]/max(cfg.input_max, 1e-9), "command_upper"),
                (-command[k, i]/max(abs(cfg.input_min), 1e-9), "command_lower"),
            )
            value, kind = max(rows, key=lambda item: item[0])
            if value > best[0]:
                best = (float(value), kind, k, i)
    value, kind, k, i = best
    return {"value": value, "constraint": kind, "sample": int(k),
            "time_s": float(k * cfg.dt), "vehicle": int(i)}


def simulate(cfg: ChainConfig, disturbance: Disturbance, seed: int = 0) -> SimulationResult:
    """Simulate one open chain with separated information, command and actuator delay."""
    n = cfg.total_vehicles
    if n < 3 or cfg.dt <= 0 or cfg.horizon <= 0:
        raise ValueError("invalid discretization or chain size")
    if cfg.hdv_model not in ("idm", "ovm"):
        raise ValueError("hdv_model must be 'idm' or 'ovm'")
    if not (0 <= disturbance.location < n):
        raise ValueError("disturbance location is outside the chain")
    if cfg.recovery_epsilon_v >= cfg.target_speed:
        raise ValueError("terminal speed tolerance must exclude all-stop traffic")
    equipped, active = _active_sets(cfg)
    rng = np.random.RandomState(seed)
    pars = _parameters(cfg, rng)
    steps = int(round(cfg.horizon / cfg.dt)) + 1
    t = np.arange(steps) * cfg.dt
    x = np.zeros((steps, n)); v = np.zeros_like(x); a = np.zeros_like(x)
    requested = np.zeros_like(x); saturated = np.zeros_like(x); executed = np.zeros_like(x)
    gap = np.full_like(x, np.nan)
    v[0] = cfg.target_speed
    for i in range(1, n):
        x[0, i] = x[0, i - 1] - cfg.vehicle_length - cfg.base_gap
    gap[0, 1:] = cfg.base_gap
    heavy = {int(i) for i in cfg.heavy_indices if 0 <= int(i) < n}
    masses = np.full(n, cfg.vehicle_mass_kg)
    for i in heavy:
        masses[i] = cfg.heavy_mass_kg
    failures: List[Dict[str, object]] = []
    seen = set()
    predicted = {"margin": float("inf"), "constraint": "none", "vehicle": -1, "time_s": 0.0}

    def log(kind: str, k: int, i: int, value: float) -> None:
        key = (kind, i)
        if key not in seen:
            seen.add(key)
            failures.append({"kind": kind, "time_s": float(t[k]), "vehicle": int(i), "value": float(value)})

    for k in range(steps - 1):
        tk = t[k]
        external = disturbance.value(tk)
        if disturbance.location == 0 and tk < disturbance.end:
            leader_raw = external
        elif disturbance.location == 0:
            # The realised jerk-limited pulse need not have exactly zero velocity
            # integral, so an explicit common return rule closes the protocol.
            leader_raw = 0.8 * (cfg.target_speed - v[k, 0])
        else:
            leader_raw = 0.0
        requested[k, 0] = leader_raw
        saturated[k, 0] = np.clip(leader_raw, -cfg.brake_max, cfg.accel_max)
        executed[k, 0] = saturated[k, 0]
        da0 = (executed[k, 0] - a[k, 0]) / max(cfg.dt, 1e-9)
        if abs(da0) > cfg.jerk_max:
            log("jerk_saturation", k, 0, da0)
        a[k+1, 0] = np.clip(a[k, 0] + cfg.dt*np.clip(da0, -cfg.jerk_max, cfg.jerk_max),
                            -cfg.brake_max, cfg.accel_max)

        hd = _delay_index(k, cfg.reaction_delay, cfg.dt)
        local_gap_h = x[hd, :-1] - x[hd, 1:] - cfg.vehicle_length
        follower_pars = {q: z[1:] for q, z in pars.items()}
        hdv_a = (_idm(local_gap_h, v[hd, 1:], v[hd, :-1], follower_pars)
                 if cfg.hdv_model == "idm" else
                 _ovm(local_gap_h, v[hd, 1:], v[hd, :-1], follower_pars))

        for i in range(1, n):
            diagnostics = None
            if i in active and cfg.controller != "human":
                hi = _delay_index(k, cfg.info_delay, cfg.dt)
                sensed_gap = x[hi, i-1] - x[hi, i] - cfg.vehicle_length
                raw, diagnostics = _controlled_command(
                    cfg.controller, cfg, sensed_gap, v[hi, i], v[hi, i-1], a[hi, i], a[hi, i-1])
            else:
                raw = float(hdv_a[i-1])
            if disturbance.location == i:
                raw += external
            requested[k, i] = raw
            if raw < cfg.input_min or raw > cfg.input_max:
                log("input_saturation", k, i, raw)
            saturated[k, i] = np.clip(raw, cfg.input_min, cfg.input_max)
            if i in active and cfg.controller != "human":
                command_k = _delay_index(k, cfg.actuator_delay, cfg.dt)
                delayed = saturated[command_k, i] if command_k < k else saturated[k, i]
                executed[k, i] = delayed
                da = (executed[k, i] - a[k, i]) / max(cfg.actuator_tau, cfg.dt)
            else:
                executed[k, i] = saturated[k, i]
                da = (executed[k, i] - a[k, i]) / max(cfg.dt, 1e-9)
            if abs(da) > cfg.jerk_max:
                log("jerk_saturation", k, i, da)
            a[k+1, i] = a[k, i] + cfg.dt*np.clip(da, -cfg.jerk_max, cfg.jerk_max)
            if i in heavy:
                traction_cap = min(cfg.traction_force_max_n/masses[i],
                                   cfg.traction_power_max_w/(masses[i]*max(v[k, i], 1.0)))
                braking_cap = cfg.brake_force_max_n/masses[i]
                a[k+1, i] = np.clip(a[k+1, i], -braking_cap, traction_cap)
            a[k+1, i] = np.clip(a[k+1, i], -cfg.brake_max, cfg.accel_max)

            if diagnostics is None:
                hi = _delay_index(k, cfg.reaction_delay, cfg.dt)
                sensed_gap = x[hi, i-1] - x[hi, i] - cfg.vehicle_length
                diagnostics = _candidate_diagnostics(cfg, sensed_gap, v[hi, i], v[hi, i-1],
                                                     a[hi, i], a[hi, i-1], saturated[k, i])
            if diagnostics["min_margin"] < predicted["margin"]:
                predicted = {"margin": float(diagnostics["min_margin"]),
                             "constraint": str(diagnostics["constraint"]),
                             "vehicle": int(i), "time_s": float(tk)}

        v[k+1] = np.clip(v[k] + cfg.dt*a[k+1], 0.0, cfg.vmax)
        x[k+1] = x[k] + cfg.dt*v[k+1]
        gap[k+1, 1:] = x[k+1, :-1] - x[k+1, 1:] - cfg.vehicle_length
        for i in range(1, n):
            if gap[k+1, i] <= 0.0:
                log("collision", k+1, i, gap[k+1, i])
            elif gap[k+1, i] < cfg.safety_gap:
                log("safety_gap", k+1, i, gap[k+1, i])
        for i in np.where(v[k+1] >= cfg.vmax-1e-9)[0]:
            log("speed_upper", k+1, int(i), v[k+1, i])

    requested[-1] = requested[-2]; saturated[-1] = saturated[-2]; executed[-1] = executed[-2]
    jerk = np.vstack([np.zeros((1, n)), np.diff(a, axis=0)/cfg.dt])
    terminal_start = max(0, steps-int(np.ceil(cfg.recovery_window/cfg.dt)))
    v_err = float(np.max(np.abs(v[terminal_start:]-cfg.target_speed)))
    g_err = float(np.nanmax(np.abs(gap[terminal_start:, 1:]-cfg.base_gap)))
    a_err = float(np.max(np.abs(a[terminal_start:])))
    follower_deficit = np.maximum(0.0, cfg.target_speed-v[:, 1:])
    raw_loss = float(np.trapz(follower_deficit.sum(axis=1), t))
    mean_loss = raw_loss/cfg.followers
    norm_loss = raw_loss/(cfg.followers*cfg.target_speed*cfg.horizon)
    safe = not any(f["kind"] in ("collision", "safety_gap") for f in failures)
    recovered = v_err <= cfg.recovery_epsilon_v and g_err <= cfg.recovery_epsilon_gap and a_err <= cfg.recovery_epsilon_a
    budget_ok = norm_loss <= cfg.loss_limit_normalized and raw_loss <= cfg.loss_budget
    if not recovered:
        failures.append({"kind": "terminal_recovery", "time_s": float(t[terminal_start]),
                         "vehicle": int(np.argmax(np.max(np.abs(v[terminal_start:]-cfg.target_speed), axis=0))),
                         "value": max(v_err/cfg.recovery_epsilon_v, g_err/cfg.recovery_epsilon_gap,
                                      a_err/cfg.recovery_epsilon_a)})
    if not budget_ok:
        failures.append({"kind": "normalized_loss_budget", "time_s": float(t[-1]), "vehicle": -1,
                         "value": float(norm_loss/cfg.loss_limit_normalized)})
    peak = _constraint_peak(cfg, gap, v, a, jerk, saturated)
    first = min(failures, key=lambda item: item["time_s"]) if failures else {
        "kind": peak["constraint"], "time_s": peak["time_s"], "vehicle": peak["vehicle"],
        "value": peak["value"]}
    peak_dev = np.max(np.abs(v-cfg.target_speed), axis=0)
    velocity_input = max(float(disturbance.alpha*disturbance.half_duration), 1e-9)
    metrics: Dict[str, object] = {
        "safe": float(safe), "recovered": float(recovered), "budget_ok": float(budget_ok),
        "success": float(safe and recovered and budget_ok),
        "terminal_speed_error_mps": v_err, "terminal_gap_error_m": g_err,
        "terminal_acceleration_error_mps2": a_err,
        "throughput_loss_vehicle_m": raw_loss, "mean_loss_m_per_follower": mean_loss,
        "normalized_loss": norm_loss, "min_gap_m": float(np.nanmin(gap[:, 1:])),
        "peak_braking_mps2": float(-np.min(a)), "peak_jerk_mps3": float(np.max(np.abs(jerk))),
        "pulse_internal_gain": float(np.max(peak_dev[1:])/velocity_input),
        "pulse_tail_gain": float(peak_dev[-1]/velocity_input),
        "peak_constraint_usage": peak["value"], "peak_constraint_kind": peak["constraint"],
        "peak_constraint_vehicle": peak["vehicle"], "peak_constraint_time_s": peak["time_s"],
        "predicted_bottleneck_margin": predicted["margin"],
        "predicted_bottleneck_constraint": predicted["constraint"],
        "predicted_bottleneck_vehicle": predicted["vehicle"],
        "predicted_bottleneck_time_s": predicted["time_s"],
        "first_limit_kind": first["kind"], "first_limit_vehicle": first["vehicle"],
        "first_limit_time_s": first["time_s"], "first_limit_value": first["value"],
        "failure_count": float(len(failures)),
    }
    metadata: Dict[str, object] = {
        "seed": int(seed), "N_followers": int(cfg.followers), "total_vehicles": int(n),
        "hdv_model": cfg.hdv_model, "controller": cfg.controller,
        "equipped_indices": sorted(equipped), "active_indices": sorted(active),
        "equipped_count": len(equipped), "active_count": len(active),
        "heavy_indices": sorted(heavy), "target_gap_m": cfg.base_gap,
        "target_speed_mps": cfg.target_speed, "alpha_mps2": disturbance.alpha,
        "disturbance_location": int(disturbance.location), "disturbance_start_s": disturbance.start,
        "disturbance_half_duration_s": disturbance.half_duration, "disturbance_end_s": disturbance.end,
        "disturbance_waveform": "closed symmetric acceleration-command pulse",
        "reaction_delay_s": cfg.reaction_delay, "information_delay_s": cfg.info_delay,
        "command_delay_s": cfg.actuator_delay, "actuator_tau_s": cfg.actuator_tau,
        "dt_s": cfg.dt, "horizon_s": cfg.horizon,
    }
    return SimulationResult(t, x, v, a, jerk, requested, saturated, executed, gap,
                            metrics, failures, metadata)
