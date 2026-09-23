"""Small, auditable bound proxies; none is a nonlinear global recovery certificate."""

from __future__ import annotations

import itertools
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
from scipy.optimize import linprog

from .model import ChainConfig, Disturbance, simulate


CERT_INITIAL_GAP_M = 6.0
CERT_SAFETY_GAP_M = 4.0
CERT_LEADER_BRAKE_MAX_MPS2 = 4.5


def latency_safety_upper(gap0: float, safety_gap: float, total_latency: float) -> float:
    """Necessary pre-response safety bound for a constant-speed follower.

    During total_latency, a braking leader loses 0.5*alpha*latency^2 distance while
    a follower with no usable new information maintains its speed. Thus an alpha
    above this value violates the stated gap before any newly informed action can act.
    It only applies under those local assumptions.
    """
    if total_latency <= 0:
        return float("inf")
    return max(0.0, 2.0 * (gap0 - safety_gap) / total_latency ** 2)


def _linear_matrices(dt=0.25, tau=0.5, delay_steps=2, leader_recovery_gain=0.8):
    # x=[gap error, relative speed, leader v error, follower v error, follower accel, command queue]
    nx = 5 + delay_steps
    A = np.zeros((nx, nx)); B = np.zeros(nx); E = np.zeros(nx)
    A[0, 0] = 1.0; A[0, 1] = dt
    A[1, 1] = 1.0; A[1, 2] = -dt * leader_recovery_gain; A[1, 4] = -dt; E[1] = dt
    A[2, 2] = 1.0 - dt * leader_recovery_gain; E[2] = dt
    A[3, 3] = 1.0; A[3, 4] = dt
    A[4, 4] = 1.0 - dt / tau; A[4, 5] = dt / tau
    for j in range(delay_steps - 1): A[5 + j, 6 + j] = 1.0
    B[-1] = 1.0
    return A, B, E


def _fixed_gain_vertex_safe(alpha: float) -> bool:
    """Exact vertex check for the declared finite-dimensional linear disturbance box."""
    dt, steps, active = 0.25, 24, 4
    A, B, E = _linear_matrices(dt=dt)
    K = np.zeros(A.shape[0]); K[0] = 0.22; K[1] = 0.85; K[3] = -0.08
    for vertex in itertools.product((0.0, -alpha), repeat=active):
        x = np.zeros(A.shape[0]); last_a = 0.0
        loss_vehicle_m = 0.0
        for k in range(steps):
            q = float(K.dot(x)); w = vertex[k] if k < active else 0.0
            x_next = A.dot(x) + B * q + E * w
            if not (-5.0 <= q <= 2.5 and -4.5 <= x_next[4] <= 2.0): return False
            if abs((x_next[4] - last_a) / dt) > 5.0 + 1e-10: return False
            if (CERT_INITIAL_GAP_M + x_next[0] < CERT_SAFETY_GAP_M
                    or 20.0 + x_next[2] < 0 or 20.0 + x_next[3] < 0): return False
            loss_vehicle_m += dt * (max(0.0, -x_next[2]) + max(0.0, -x_next[3]))
            last_a = x_next[4]; x = x_next
        if abs(x[0]) > 5.0 or abs(x[2]) > 1.0 or abs(x[3]) > 1.0 or abs(x[4]) > 0.5: return False
        if loss_vehicle_m > 30.0: return False
    return True


def _clairvoyant_lp_feasible(alpha: float) -> bool:
    """Feasibility with noncausal open-loop commands for one specified braking pulse."""
    dt, steps, active = 0.25, 24, 4
    A, B, E = _linear_matrices(dt=dt); nx = A.shape[0]
    c = np.zeros(nx); M = np.zeros((nx, steps))
    aub, bub = [], []
    last_c = 0.0; last_M = np.zeros(steps)
    for k in range(steps):
        w = -alpha if k < active else 0.0
        c = A.dot(c) + E * w
        M = A.dot(M); M[:, k] += B
        # gap >= safety, nonnegative speeds, acceleration and jerk constraints.
        aub += [-M[0], -M[2], -M[3], M[4], -M[4], (M[4]-last_M)/dt, -(M[4]-last_M)/dt]
        gap_margin = CERT_INITIAL_GAP_M - CERT_SAFETY_GAP_M
        bub += [gap_margin + c[0], 20.0 + c[2], 20.0 + c[3], 2.0-c[4], 4.5+c[4],
                5.0-(c[4]-last_c)/dt, 5.0+(c[4]-last_c)/dt]
        last_c, last_M = c[4], M[4].copy()
    # Terminal recovery box. Omitting the throughput budget relaxes the problem;
    # infeasibility therefore remains a valid impossibility witness for this model.
    for idx, eps in ((0, 5.0), (2, 1.0), (3, 1.0)):
        aub += [M[idx], -M[idx]]; bub += [eps-c[idx], eps+c[idx]]
    sol = linprog(np.zeros(steps), A_ub=np.asarray(aub), b_ub=np.asarray(bub),
                  bounds=[(-5.0, 2.5)] * steps, method="highs")
    return bool(sol.success)


def grid_policy_proxy(output_dir: Path) -> List[Dict[str, object]]:
    """Compute D on an explicitly scoped delayed linear small model."""
    output_dir.mkdir(parents=True, exist_ok=True)
    alphas = np.arange(0.25, CERT_LEADER_BRAKE_MAX_MPS2 + 0.001, 0.25)
    lower_ok = [float(a) for a in alphas if _fixed_gain_vertex_safe(float(a))]
    first_infeasible = next((float(a) for a in alphas if not _clairvoyant_lp_feasible(float(a))), None)
    previous_grid = first_infeasible - 0.25 if first_infeasible is not None else None
    records = [{
        "linear_box_certified_lower_mps2": max(lower_ok) if lower_ok else 0.0,
        "linear_clairvoyant_infeasible_upper_mps2": first_infeasible,
        "leader_disturbance_bound_mps2": CERT_LEADER_BRAKE_MAX_MPS2,
        "initial_gap_m": CERT_INITIAL_GAP_M,
        "safety_gap_m": CERT_SAFETY_GAP_M,
        "amplitude_grid_step_mps2": 0.25,
        "lower_method": "all 2^4 vertices of w_k in [-alpha,0] for fixed causal K",
        "upper_method": "specific four-step constant braking pulse; clairvoyant open-loop LP infeasible",
        "lp_last_feasible_grid_mps2": previous_grid,
        "lp_last_feasible_grid_status": (_clairvoyant_lp_feasible(previous_grid)
                                          if previous_grid is not None else None),
        "lp_first_infeasible_grid_status": (_clairvoyant_lp_feasible(first_infeasible)
                                             if first_infeasible is not None else None),
        "lower_loss_budget_vehicle_m": 30.0,
        "scope": "Discrete N=2 relative linear model: initial gap 6 m, safety gap 4 m, target speed 20 m/s, leader disturbance 0<=alpha<=4.5 m/s^2 for four 0.25-s steps, total horizon 24 steps, leader speed recovery gain 0.8 s^-1, two-step command delay, first-order actuator tau=0.5 s. Command [-5,2.5] m/s^2, follower acceleration [-4.5,2] m/s^2, jerk magnitude <=5 m/s^3, nonnegative speeds, terminal |gap error|<=5 m and |speed errors|<=1 m/s. Lower additionally checks terminal |acceleration|<=0.5 m/s^2 and J_loss<=30 vehicle-m for all 2^4 disturbance vertices under fixed K. Upper relaxes terminal acceleration and J_loss and grants clairvoyant open-loop control; infeasibility is strategy-independent only for this discrete linear model. Neither bound transfers to the nonlinear chain without a remainder bound.",
    }]
    with (output_dir / "small_scale_bound_proxy.json").open("w", encoding="utf-8") as fh:
        json.dump(records, fh, indent=2, sort_keys=True)
    return records
