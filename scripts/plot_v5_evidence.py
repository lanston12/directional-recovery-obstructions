from __future__ import annotations

import json
import copy
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT if (ROOT / "results" / "nonlinear_revision" / "raw_runs.csv").exists() else ROOT.parent
RESULTS = PROJECT / "results" / "nonlinear_revision"
DERIVED = ROOT / "derived_results"
FIGURES = ROOT / "figures" / "results"
FIGURES.mkdir(parents=True, exist_ok=True)

COLORS = {
    "human": "#777777", "acc": "#4C78A8", "wave": "#F58518",
    "predictive": "#54A24B", "margin": "#B279A2",
}
LABELS = {
    "human": "Human", "acc": "ACC", "wave": "Wave rule",
    "predictive": "Endpoint", "margin": "Margin priority",
}
CATEGORY_COLORS = {"A": "#47769B", "B": "#B46E3E", "C": "#816696", "D": "#D6DBE0"}
CATEGORY_LABELS = {
    "A": "Recovered", "B": "Prefix obstruction", "C": "Failed; unresolved by prefix test", "D": "Not evaluated",
}
plt.rcParams.update({
    "font.size": 9, "axes.labelsize": 9, "axes.titlesize": 9.5,
    "legend.fontsize": 7.5, "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 120,
})


def save(fig: plt.Figure, name: str) -> None:
    if not fig.get_constrained_layout():
        fig.tight_layout()
    for ext in ("pdf", "svg", "png"):
        fig.savefig(FIGURES / f"{name}.{ext}", dpi=400, bbox_inches="tight")
    plt.close(fig)


raw = pd.read_csv(RESULTS / "raw_runs.csv")
bounds = pd.read_csv(RESULTS / "boundaries.csv")
pred_records = pd.read_csv(RESULTS / "prediction_records.csv")
pred_metrics = pd.read_csv(RESULTS / "prediction_metrics.csv")
representative = json.loads((RESULTS / "representative_manifest.json").read_text(encoding="utf-8"))
linear = json.loads((PROJECT / "results" / "revision_package" / "linear_demo.json").read_text(encoding="utf-8"))
traces = np.load(RESULTS / "representative_trajectories.npz")
rep_events = pd.read_csv(DERIVED / "representative_provenance_and_events.csv").set_index("method")


# Figure 1: representative diagnostic replay, with post-collision numerical continuation masked.
methods = ("wave", "predictive", "margin")
all_speed = np.concatenate([traces[f"{m}_speed"].ravel() for m in methods])
vmin, vmax = np.percentile(all_speed, [1, 99])
cmap = copy.copy(plt.get_cmap("viridis"))
cmap.set_bad("#eeeeee")
fig = plt.figure(figsize=(8.4, 6.5), constrained_layout=True)
grid = fig.add_gridspec(2, 3, width_ratios=(1, 1, 0.045))
axes = [fig.add_subplot(grid[0, 0]), fig.add_subplot(grid[0, 1]),
        fig.add_subplot(grid[1, 0]), fig.add_subplot(grid[1, 1])]
cax = fig.add_subplot(grid[:, 2])
for ax, method, panel in zip(axes[:3], methods, "abc"):
    speed = traces[f"{method}_speed"]
    time = traces[f"{method}_time"]
    displayed = np.ma.array(speed, mask=np.zeros_like(speed, dtype=bool))
    collision_time = rep_events.loc[method, "first_collision_time_s"]
    if np.isfinite(collision_time):
        displayed.mask[time >= collision_time, :] = True
    image = ax.pcolormesh(time, np.arange(speed.shape[1]), displayed.T, shading="auto",
                          cmap=cmap, vmin=vmin, vmax=vmax)
    if np.isfinite(collision_time):
        ax.axvline(collision_time, color="#B24C4C", lw=1.1, ls="--")
        title = f"Collision at {collision_time:.1f} s"
    else:
        title = "Recovered"
    ax.set(title=f"({panel}) {LABELS[method]}: {title}", xlabel="Time (s)", ylabel="Vehicle index")
cb = fig.colorbar(image, cax=cax)
cb.set_label("Speed (m s$^{-1}$)")
ax = axes[3]
for method in methods:
    time = traces[f"{method}_time"]
    min_gap = np.nanmin(traces[f"{method}_gap"], axis=1)
    collision_time = rep_events.loc[method, "first_collision_time_s"]
    if np.isfinite(collision_time):
        keep = time < collision_time
        ax.plot(time[keep], min_gap[keep], color=COLORS[method], label=LABELS[method], lw=1.5)
        ax.scatter(collision_time, 0, color="#B24C4C", marker="x", s=36, zorder=4)
    else:
        ax.plot(time, min_gap, color=COLORS[method], label=LABELS[method], lw=1.5)
ax.axhline(4, color="black", lw=.8, ls="--", label="Specified 4-m safety minimum")
ax.set(title="(d) Minimum net gap before any collision", xlabel="Time (s)", ylabel="Minimum net gap (m)")
ax.legend(frameon=True, loc="upper right", fontsize=6.8, facecolor="white", framealpha=.9)
fig.suptitle("Diagnostic replay: N=40, requested $\\alpha=4.5$ m s$^{-2}$, seed 3")
save(fig, "fig1_recovery_collision_aware")


# Figure 2: same-model directional mechanism across every existing layout seed.
mechanism = pd.read_csv(DERIVED / "nonlinear_directional_mechanism_runs.csv")
layout_cfg = pd.read_csv(DERIVED / "layout_indices_and_config.csv")
nonlinear40 = layout_cfg[(layout_cfg.problem == "nonlinear") & (layout_cfg.N_followers == 40)].set_index("layout")
layouts = ["front", "even", "middle", "rear"]
fig = plt.figure(figsize=(10.8, 7.8), constrained_layout=True)
grid = fig.add_gridspec(2, 3, width_ratios=(1.05, 1.25, 1.25))
schematic = fig.add_subplot(grid[:, 0])
mini_axes = [fig.add_subplot(grid[0, 1]), fig.add_subplot(grid[0, 2]),
             fig.add_subplot(grid[1, 1]), fig.add_subplot(grid[1, 2])]
for y, layout in enumerate(layouts[::-1]):
    active = json.loads(nonlinear40.loc[layout, "active_indices"])
    schematic.scatter(np.arange(1, 41), np.full(40, y), s=9, color="#c8c8c8")
    schematic.scatter(active, np.full(len(active), y), s=45, color="#345995", marker="s")
    if min(active) > 1:
        schematic.plot([1, min(active)-1], [y, y], color=CATEGORY_COLORS["B"], lw=4, alpha=.55)
schematic.set_yticks(range(4)); schematic.set_yticklabels([x.title() for x in layouts[::-1]])
schematic.set(xlim=(0, 41), xlabel="Follower index", title="(a) Active sets and uncontrolled prefixes")
schematic.text(.02, .015, "Blue squares: active CAVs; orange segment: prefix P", transform=schematic.transAxes, fontsize=7.5)

category_numbers = {cat: j for j, cat in enumerate("ABCD")}
for ax, layout, panel in zip(mini_axes, layouts, "bcde"):
    group = mechanism[mechanism.layout == layout]
    alphas = sorted(group.alpha_mps2.unique())
    seeds = sorted(group.seed.unique())
    lookup = group.set_index(["seed", "alpha_mps2"]).category
    image = np.array([[category_numbers[lookup.loc[(seed, alpha)]] for alpha in alphas]
                      for seed in seeds])
    ax.imshow(image, cmap=ListedColormap([CATEGORY_COLORS[c] for c in "ABCD"]),
              vmin=-.5, vmax=3.5, aspect="auto", interpolation="nearest")
    counts = group.category.value_counts()
    failures = int(counts.get("B", 0) + counts.get("C", 0))
    suffix = f"; B/fail={counts.get('B', 0)}/{failures}" if failures else ""
    ax.set_title(f"({panel}) {layout.title()}: " + "/".join(str(counts.get(c, 0)) for c in "ABCD") + suffix, fontsize=8)
    ax.set_xlabel("Requested amplitude (m s$^{-2}$); discrete categories")
    if layout in ("front", "middle"):
        ax.set_ylabel("Seed")
    ax.set_xticks(range(len(alphas))); ax.set_xticklabels([f"{x:g}" for x in alphas], rotation=55, fontsize=6.2)
    ax.set_yticks(range(0, len(seeds), 2)); ax.set_yticklabels([str(seeds[j]) for j in range(0, len(seeds), 2)], fontsize=6.2)
from matplotlib.patches import Patch
fig.legend([Patch(color=CATEGORY_COLORS[c], label=f"{c}: {CATEGORY_LABELS[c]}") for c in "ABCD"],
           [f"{c}: {CATEGORY_LABELS[c]}" for c in "ABCD"],
           loc="lower center", bbox_to_anchor=(0.53, -0.10), ncol=2, frameon=False)
save(fig, "fig2_directional_obstruction")


# Figure 3: constraint-level linear reserve and limiting-row changes with delay.
candidate = pd.read_csv(DERIVED / "linear_constraint_candidates.csv")
delay = pd.read_csv(DERIVED / "linear_delay_limiting_rows.csv")
keep_constraints = ["acceleration_upper", "command_upper", "jerk_upper", "leader_amplitude_domain", "normalized_loss"]
constraint_style = {
    "acceleration_upper": ("#B24C4C", "o", "Acceleration upper"),
    "command_upper": ("#345995", "s", "Command upper"),
    "jerk_upper": ("#E6A23C", "^", "Jerk upper"),
    "leader_amplitude_domain": ("#777777", "D", "Amplitude domain"),
    "normalized_loss": ("#4C956C", "v", "Distance-deficit budget"),
}
linear_layouts = ["Front", "Distributed", "Middle", "Rear"]
fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.8), constrained_layout=True)
for c in keep_constraints:
    sub = candidate[candidate.constraint == c].set_index("layout").loc[linear_layouts]
    color, marker, label = constraint_style[c]
    axes[0].plot(np.arange(4), sub.radius, color=color, marker=marker, lw=1.0, label=label)
for j, layout in enumerate(linear_layouts):
    radius = float(linear["cases"][layout]["radius"])
    axes[0].scatter(j, radius, s=95, facecolors="none", edgecolors="black", linewidths=1.2, zorder=5)
axes[0].set_xticks(np.arange(4)); axes[0].set_xticklabels(linear_layouts, rotation=15)
axes[0].set(title="(a) Candidate constraint radii", ylabel="Amplitude (m s$^{-2}$)", ylim=(1.9, 10.0))
fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center",
           bbox_to_anchor=(.29, -.18), ncol=3, frameon=False, fontsize=6.7)
for layout, color in (("Front", "#345995"), ("Rear", "#F58518")):
    sub = delay[delay.layout == layout]
    axes[1].plot(sub.delay_s, sub.radius, color=color, marker="o", label=layout)
    for row in sub.itertuples(index=False):
        if int(row.vehicle) != 1:
            axes[1].annotate(f"limiting follower {int(row.vehicle)}", (row.delay_s, row.radius),
                             xytext=(-68, 12), textcoords="offset points", fontsize=7,
                             arrowprops={"arrowstyle": "->", "lw": .7})
axes[1].set(title="(b) Delay changes the limiting row", xlabel="Common delay (s)", ylabel="Amplitude (m s$^{-2}$)")
axes[1].legend(frameon=False)
save(fig, "fig3_linear_constraint_rows")


# Figure 4: paired nonlinear trade-off from diagnostic replays of the same 12 seeds.
effects = pd.read_csv(DERIVED / "paired_alpha45_effects.csv")
summary_json = json.loads((DERIVED / "paired_and_predictor_summary.json").read_text(encoding="utf-8"))["paired_effects"]
fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.8), constrained_layout=True)
x = np.arange(len(effects))
axes[0].axhline(0, color="black", lw=.8)
axes[0].scatter(x-.13, effects.delta_endpoint_gap_margin_minus_predictive_m, marker="o", color="#345995",
             label="Endpoint gap error")
axes[0].scatter(x+.13, effects.delta_window_gap_margin_minus_predictive_m, marker="s", color="#B279A2",
             label="Final-3-s maximum gap error")
axes[0].set_xticks(x); axes[0].set_xticklabels(effects.seed.astype(int), rotation=45)
axes[0].set(title="(a) Paired gap-error differences", xlabel="Paired seed",
            ylabel="Margin minus endpoint-screened (m)")
axes[0].legend(frameon=False, loc="upper right")
axes[1].axhline(0, color="black", lw=.8)
axes[1].scatter(x, effects.delta_normalized_loss_margin_minus_predictive, color="#B279A2", marker="o")
axes[1].set_xticks(x); axes[1].set_xticklabels(effects.seed.astype(int), rotation=45)
axes[1].set(title="(b) Paired distance-deficit differences", xlabel="Paired seed",
            ylabel="Margin minus endpoint-screened")
save(fig, "fig4_paired_tradeoff")


# Figure 5: diagnostics that delimit source, model and finite-horizon comparisons.
source = pd.read_csv(DERIVED / "source_protocol_replays_alpha21.csv")
reach = pd.read_csv(DERIVED / "finite_horizon_chain_reach_alpha40_seed97.csv")
jac = pd.read_csv(DERIVED / "mean_equilibrium_jacobians.csv")
fig, axes = plt.subplots(1, 3, figsize=(10.3, 3.6), constrained_layout=True)
leader = source[source.case_id == "even_idm_leader_d02"].set_index("seed")
internal = source[source.case_id == "even_idm_internal8_d02"].set_index("seed")
common = leader.index.intersection(internal.index)
for seed in common:
    axes[0].plot([0, 1], [leader.loc[seed, "speed_drop_mps"], internal.loc[seed, "speed_drop_mps"]],
                 color="#aaaaaa", lw=.8)
axes[0].scatter(np.zeros(len(common)), leader.loc[common, "speed_drop_mps"], color="#345995", label="External leader")
axes[0].scatter(np.ones(len(common)), internal.loc[common, "speed_drop_mps"], color="#B279A2", label="Follower-8 HDV request")
axes[0].set_xticks([0, 1]); axes[0].set_xticklabels(["Leader", "Follower 8"])
axes[0].set(title="(a) Source response at requested $\\alpha=2.1$", ylabel="Actual source speed drop (m s$^{-1}$)")
axes[0].legend(frameon=False, fontsize=6.8)
axes[1].bar(reach.N_followers.astype(str), reach.affected_fraction, color="#72B7B2")
for j, row in enumerate(reach.itertuples(index=False)):
    axes[1].text(j, row.affected_fraction + .025, f"{int(row.affected_count)}/{int(row.N_followers)}\n$\\bar J$={row.normalized_loss_full_N:.3f}",
                 ha="center", fontsize=7)
axes[1].set_ylim(0, 1.12)
axes[1].set(title="(b) Disturbance reach by 60 s", xlabel="Follower count N", ylabel="Fraction with $|v-20|\\geq1$ m s$^{-1}$")
xj = np.arange(3)
width = .32
axes[2].bar(xj[1:]-width/2, jac.iloc[0, 2:].to_numpy(float), width, color="#4C78A8", label="IDM")
axes[2].bar(xj[1:]+width/2, jac.iloc[1, 2:].to_numpy(float), width, color="#F58518", label="OVM/FVD")
axes[2].axhline(0, color="black", lw=.7)
axes[2].set_xticks(xj); axes[2].set_xticklabels([r"$\partial f/\partial s$", r"$\partial f/\partial v$", r"$\partial f/\partial v_p$"])
axes[2].set(title="(c) Mean-equilibrium derivatives", ylabel="Speed derivative (s$^{-1}$)")
gap_axis = axes[2].twinx()
gap_axis.scatter([-width/2, width/2], jac.iloc[:, 1].to_numpy(float), marker="D", color=["#4C78A8", "#F58518"], s=36)
gap_axis.set_ylim(.069, .075); gap_axis.set_xlim(-.55, 2.55)
gap_axis.set_ylabel("Gap derivative (s$^{-2}$)")
axes[2].legend(frameon=False, loc="upper center", fontsize=6.6)
save(fig, "fig5_scope_diagnostics")


# Supplementary Figures S1-S2: independent sampled-linear archive.
response = np.load(PROJECT / "results" / "revision_package" / "linear_response_coefficients.npz")
fig, ax = plt.subplots(figsize=(7.2, 3.6))
radii = np.asarray([linear["cases"][name]["radius"] for name in linear_layouts])
vertices = np.asarray([linear["cases"][name]["limiting"]["vertex_radius"] for name in linear_layouts])
x = np.arange(len(linear_layouts))
ax.scatter(x, radii, s=55, facecolors="none", edgecolors="#1f77b4", label="Support-function expression")
ax.scatter(x, vertices, s=38, color="#1f77b4", marker="x", label="All 16 disturbance vertices")
ax.set_xticks(x); ax.set_xticklabels(linear_layouts)
ax.set(ylabel="Fixed-policy recovery amplitude (m s$^{-2}$)", ylim=(2.2, 2.9))
ax.legend(frameon=False)
save(fig, "figS1_vertex_check")

coordinate = -np.ones(4)
time = np.arange(response["front_speed"].shape[0]) * 0.1
front_tail = response["front_speed"][:, -1, :] @ coordinate
rear_tail = response["rear_speed"][:, -1, :] @ coordinate
fig, ax = plt.subplots(figsize=(7.2, 3.6))
ax.plot(time, front_tail, color="#1f77b4", lw=1.5, label="Front")
ax.plot(time, rear_tail, color="#ff7f0e", lw=1.4, ls="--", label="Rear")
ax.set(xlabel="Time (s)", ylabel="Tail-speed deviation (m s$^{-1}$)", xlim=(0, 60))
ax.legend(frameon=False)
save(fig, "figS2_endpoint_response")


# Supplementary Figure S3: predictor failure, with units and configuration clustering separated.
order = ["small_pulse_gain", "average_headway", "distance_to_control", "directional_margin"]
pm = pred_metrics.set_index("predictor").loc[order]
case_metrics = pd.read_csv(DERIVED / "prediction_metrics_by_configuration.csv")
fig, axes = plt.subplots(2, 2, figsize=(8.8, 6.5), constrained_layout=True)
names = ["Pulse gain", "Headway", "Distance", "Directional\nfeature"]
axes[0, 0].scatter(pm.test_MAE_mps2, np.arange(4), color="#4C78A8", s=45)
axes[0, 0].set_yticks(np.arange(4)); axes[0, 0].set_yticklabels(names)
axes[0, 0].set(title="(a) Held-out MAE", xlabel="MAE (m s$^{-2}$)")
axes[0, 1].scatter(pm.false_recoverable_fraction, np.arange(4), color="#E45756", s=45)
axes[0, 1].set_yticks(np.arange(4)); axes[0, 1].set_yticklabels(names)
axes[0, 1].set(title="(b) Recorded-endpoint exceedance", xlabel="Fraction of test records", xlim=(-.02, 1.02))
test = pred_records[pred_records.split == "test"].copy()
case_labels = {
    "middle_idm_leader_d02": "Middle", "rear_idm_leader_d02": "Rear",
    "even_idm_leader_cmd04": "Command delay 0.4 s", "even_ovm_leader_d02": "OVM/FVD",
}
for case_id, group in test.groupby("case_id"):
    grouped = group.groupby(["observed_boundary_mps2", "prediction_directional_margin"]).size().reset_index(name="count")
    axes[1, 0].scatter(grouped.observed_boundary_mps2, grouped.prediction_directional_margin,
                       s=25 + 18 * grouped["count"], alpha=.75, label=case_labels[case_id])
axes[1, 0].plot([0, 5], [0, 5], "k--", lw=.8)
axes[1, 0].set(xlim=(0, 5.15), ylim=(0, 5.15), title="(c) Directional predictor", xlabel="Recorded endpoint (m s$^{-2}$)", ylabel="Prediction after code clipping (m s$^{-2}$)")
axes[1, 0].legend(frameon=False, fontsize=6.6)
yy = np.arange(len(case_metrics))
axes[1, 1].barh(yy, case_metrics.MAE_mps2, color="#4C78A8")
axes[1, 1].set_yticks(yy); axes[1, 1].set_yticklabels([case_labels[x] for x in case_metrics.case_id])
axes[1, 1].set(title="(d) Directional MAE by configuration", xlabel="MAE (m s$^{-2}$)")
save(fig, "figS3_predictor_diagnostics")


# Supplementary Figure S4: endpoint-approximation audit.
decisions = pd.read_csv(DERIVED / "endpoint_predictor_decisions.csv")
audit = json.loads((DERIVED / "paired_and_predictor_summary.json").read_text(encoding="utf-8"))["predictor"]
subset = decisions[(np.abs(decisions.selected_command_mps2) > .1) |
                   ((decisions.time_s >= 5) & (decisions.time_s <= 15))]
fig, axes = plt.subplots(1, 3, figsize=(9.5, 3.5), constrained_layout=True)
axes[0].bar([0, 1], [audit["reference_a0_u1_old_speed_increment_mps"], audit["reference_a0_u1_exact_speed_increment_mps"]],
            color=["#B279A2", "#4C78A8"])
axes[0].set_xticks([0, 1]); axes[0].set_xticklabels(["Endpoint-accel.\napproximation", "Exact lag\nintegral"])
axes[0].set(title="(a) Reference: a=0, u=1", ylabel="Predicted speed increment (m s$^{-1}$)")
axes[1].boxplot([np.abs(subset.old_minus_exact_speed_mps), np.abs(subset.old_minus_discrete_speed_mps)],
                labels=["vs exact lag", "vs discrete rollout"], showfliers=False)
axes[1].set(title="(b) Selected-command speed error", ylabel="Absolute endpoint error (m s$^{-1}$)")
time_bins = pd.cut(decisions.time_s, bins=np.arange(0, 61, 5), right=False)
q95 = decisions.assign(bin=time_bins).groupby("bin", observed=True).old_minus_discrete_gap_m.apply(lambda x: np.quantile(np.abs(x), .95))
centres = np.arange(2.5, 60, 5)[:len(q95)]
axes[2].plot(centres, q95.to_numpy(), marker="o", color="#B24C4C")
axes[2].set(title="(c) Gap-error 95th percentile", xlabel="Decision time bin midpoint (s)", ylabel="Absolute gap error (m)")
axes[2].text(.03, .95, f"Empty feasible pools: {audit['empty_feasible_pool_count']}/{audit['decisions']}",
             transform=axes[2].transAxes, va="top", fontsize=7.2)
save(fig, "figS4_endpoint_approximation")


# Supplementary Figure S5: follower-only state/actuator use and separate event categories.
def state_use(speed, gap):
    return np.maximum.reduce(((24-gap[:, 1:])/20, (20-speed[:, 1:])/20, (speed[:, 1:]-20)/15))


def actuator_use(acc, jerk, command):
    return np.maximum.reduce((acc[:, 1:]/2, -acc[:, 1:]/4.5, np.abs(jerk[:, 1:])/5,
                              command[:, 1:]/2.5, -command[:, 1:]/5))


fig = plt.figure(figsize=(10.4, 6.0), constrained_layout=True)
grid = fig.add_gridspec(2, 4, width_ratios=(1, 1, .055, 1.22))
heat = [fig.add_subplot(grid[0, 0]), fig.add_subplot(grid[0, 1]),
        fig.add_subplot(grid[1, 0]), fig.add_subplot(grid[1, 1])]
cax = fig.add_subplot(grid[:, 2]); spatial = fig.add_subplot(grid[0, 3]); events_ax = fig.add_subplot(grid[1, 3])
states, actuators = {}, {}
for method in ("predictive", "margin"):
    states[method] = state_use(traces[f"{method}_speed"], traces[f"{method}_gap"])
    actuators[method] = actuator_use(traces[f"{method}_acceleration"], traces[f"{method}_jerk"], traces[f"{method}_saturated"])
for ax, method, panel in zip(heat[:2], ("predictive", "margin"), "ab"):
    image = ax.pcolormesh(traces[f"{method}_time"], np.arange(1, 41), states[method].T,
                          shading="auto", cmap="magma", vmin=0, vmax=1)
    ax.set(title=f"({panel}) {LABELS[method]}: state", xlabel="Time (s)", ylabel="Follower")
for ax, method, panel in zip(heat[2:], ("predictive", "margin"), "de"):
    image = ax.pcolormesh(traces[f"{method}_time"], np.arange(1, 41), actuators[method].T,
                          shading="auto", cmap="magma", vmin=0, vmax=1)
    ax.set(title=f"({panel}) {LABELS[method]}: actuator", xlabel="Time (s)", ylabel="Follower")
fig.colorbar(image, cax=cax, label="Realised follower utilisation")
followers = np.arange(1, 41)
for method in ("predictive", "margin"):
    spatial.plot(followers, np.max(states[method], axis=0), color=COLORS[method], label=f"{LABELS[method]} state")
    spatial.plot(followers, np.max(actuators[method], axis=0), color=COLORS[method], ls="--", label=f"{LABELS[method]} actuator")
spatial.axhline(1, color="black", ls=":", lw=.8)
spatial.set(title="(c) Spatial peak", xlabel="Follower", ylabel="Peak use")
spatial.legend(frameon=False, fontsize=6.2)
preloss_check = json.loads((DERIVED / "v6_check_summary.json").read_text(encoding="utf-8"))
assert not preloss_check["wave_budget_exceeded_before_collision"]
assert representative["metrics"]["wave"]["success"] == 0
assert all(representative["metrics"][m]["success"] == 1 for m in ("predictive", "margin"))
# Archived wave terminal/loss booleans use a post-collision numerical continuation.
# The budget was not exceeded in the physically interpretable pre-collision segment.
statuses = [["Fail", "N/A", "N/A", "N/A", "N/A"],
            ["Pass"] * 5, ["Pass"] * 5]
matrix = np.array([[{"Pass": 0, "Fail": 1, "N/A": 2}[value] for value in row]
                   for row in statuses])
events_ax.imshow(matrix, cmap=ListedColormap(["#d9ead3", "#e06666", "#dddddd"]),
                 vmin=-.5, vmax=2.5, aspect="auto")
events_ax.set_xticks(np.arange(5)); events_ax.set_xticklabels(["1", "2", "3", "4", "5"])
events_ax.set_yticks(np.arange(3)); events_ax.set_yticklabels(["Wave", "Endpoint", "Margin"])
for i in range(3):
    for j in range(5):
        events_ax.text(j, i, statuses[i][j], ha="center", va="center",
                       fontsize=6.5, color="white" if statuses[i][j] == "Fail" else "#274e13")
events_ax.set_title("(f) Collision-aware recovery criteria")
events_ax.set_xlabel("1 path; 2 speed; 3 gap; 4 accel.; 5 deficit")
events_ax.tick_params(length=0)
save(fig, "figS5_constraint_use")

print(f"Generated 5 main and 5 supplementary figure groups in {FIGURES}")
