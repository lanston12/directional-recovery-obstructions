"""Draw the submission Fig. 1 as native vector graphics (no simulated data)."""
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "figures" / "results"
OUT.mkdir(parents=True, exist_ok=True)

NAVY, SLATE, BLUE, ORANGE = "#273b50", "#596c7c", "#47769b", "#b46e3e"
LINE, GREY, PALE_BLUE, PALE_ORANGE = "#bec9d1", "#edf1f4", "#e6eef4", "#f7ede5"

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9,
                     "pdf.fonttype": 42, "svg.fonttype": "none",
                     "axes.unicode_minus": False})

# Designed near 180-mm final width. The 2x2 reading order retains the reference
# image's problem -> topology -> evidence -> synthesis sequence without forcing
# the mathematical panel into an illegibly narrow fourth column.
fig = plt.figure(figsize=(7.25, 6.0), facecolor="white")


def panel(bounds, letter, title):
    ax = fig.add_axes(bounds)
    ax.set(xlim=(0, 100), ylim=(0, 100))
    ax.axis("off")
    ax.add_patch(FancyBboxPatch(
        (0.5, 0.5), 99, 99, boxstyle="round,pad=0,rounding_size=1.6",
        facecolor="white", edgecolor=LINE, linewidth=0.8))
    ax.text(4.5, 94, f"({letter})", color=NAVY, weight="bold", fontsize=10.5,
            va="center")
    ax.text(15, 94, title, color=NAVY, weight="bold", fontsize=9.7,
            va="center")
    return ax


def arrow(ax, start, end, color=SLATE, width=0.9, dashed=False):
    ax.add_patch(FancyArrowPatch(
        start, end, arrowstyle="-|>", mutation_scale=8.5,
        linewidth=width, linestyle="dashed" if dashed else "solid", color=color))


def node(ax, x, y, name, active=False, width=11, height=9):
    ax.add_patch(FancyBboxPatch(
        (x-width/2, y-height/2), width, height,
        boxstyle="round,pad=0,rounding_size=1.1",
        edgecolor=BLUE if active else SLATE,
        facecolor=PALE_BLUE if active else GREY, linewidth=1.0))
    ax.text(x, y, name, ha="center", va="center", fontsize=8.1, color=NAVY,
            weight="bold" if active else "normal")


# (a) Recovery specification; all nonlinear runs start at the declared equilibrium.
a = panel([0.025, 0.525, 0.465, 0.455], "a", "Finite-amplitude recovery")
a.text(50, 85.5, "What requested disturbance amplitude", ha="center",
       fontsize=8.4, color=NAVY)
a.text(50, 79.5, "remains recoverable?", ha="center", fontsize=8.4, color=NAVY)
a.text(11, 75, "leader\npulse", ha="center", va="center", fontsize=8.1, color=ORANGE)
arrow(a, (11, 70), (11, 65), color=ORANGE)
for x, label, active in [(11, "0", False), (30, "1", False),
                         (49, "CAV", True), (68, "HDV", False),
                         (87, "CAV", True)]:
    node(a, x, 59, label, active, width=12)
for x in (18, 37, 56, 75):
    arrow(a, (x, 59), (x+6, 59))
a.text(50, 48.5, "Directed chain; sparse active vehicles",
       ha="center", fontsize=8.1, color=SLATE)
a.add_patch(FancyBboxPatch(
    (6, 17), 88, 27, boxstyle="round,pad=0,rounding_size=1.2",
    facecolor=GREY, edgecolor="none"))
a.text(10, 38, r"Path: $s_i(t)\geq s_{\min}$", fontsize=8.5, color=NAVY)
a.text(10, 31.5, r"Terminal: $v_i\to\bar v$, $s_i\to\bar s_i$, $a_i\to0$",
       fontsize=8.5, color=NAVY)
a.text(10, 25, r"Loss: $\bar J\leq\ell$", fontsize=8.5, color=NAVY)
a.text(57, 25, r"Deadline: $T_r$", fontsize=8.5, color=NAVY)
a.text(50, 8.5, r"System-level limit $\alpha_\star$", ha="center",
       fontsize=9.5, weight="bold", color=BLUE)


# (b) A required safety-bound crossing, deliberately distinct from collision.
b = panel([0.51, 0.525, 0.465, 0.455], "b", "Directed prefix obstruction")
b.add_patch(FancyBboxPatch(
    (21, 65), 47, 18, boxstyle="round,pad=0,rounding_size=1.4",
    facecolor=PALE_ORANGE, edgecolor=ORANGE, linewidth=0.65))
b.text(44.5, 79, r"Uncontrolled prefix $P$", ha="center",
       fontsize=8.2, weight="bold", color=ORANGE)
for x, label, active in [(9, "0", False), (27, "1", False),
                         (42, "2", False), (59, "j-1", False),
                         (77, "j", True), (92, "N", False)]:
    node(b, x, 68.5, label, active, width=10.5)
for start, end in [(15, 21), (33, 36), (48, 53), (65, 71), (83, 86)]:
    arrow(b, (start, 68.5), (end, 68.5), width=0.8)
b.text(43, 57.5, r"$P=\{1,\ldots,j-1\}$", ha="center", fontsize=8.7,
       color=ORANGE)
b.text(79, 57.5, r"$j=\min C$", ha="center", fontsize=8.4,
       color=BLUE)
b.text(43, 49.5, r"$s_i<s_{\min}$", ha="center", fontsize=8.8, color=ORANGE)
b.text(43, 44.5, "Required safety-bound violation", ha="center",
       fontsize=7.7, color=ORANGE)
arrow(b, (77, 40), (53, 40), color=BLUE, dashed=True)
b.text(59, 40, "×", ha="center", va="center", fontsize=14, weight="bold",
       color=ORANGE)
b.text(50, 33, "Downstream actuation cannot change", ha="center",
       fontsize=8.0, color=NAVY)
b.text(50, 28, "this prefix trajectory", ha="center", fontsize=8.0, color=NAVY)
b.text(50, 23, "Fixed leader path, delayed histories,", ha="center",
       fontsize=7.7, color=SLATE)
b.text(50, 18, "parameters, active set and directed topology", ha="center",
       fontsize=7.7, color=SLATE)
b.add_patch(FancyBboxPatch(
    (9, 2.5), 82, 12.5, boxstyle="round,pad=0,rounding_size=1.1",
    facecolor=PALE_ORANGE, edgecolor="none"))
b.text(50, 11, "Prefix safety violation →", ha="center", va="center",
       fontsize=8.0, weight="bold", color=ORANGE)
b.text(50, 5.8, "conditional policy-independent obstruction", ha="center",
       va="center", fontsize=7.8, weight="bold", color=ORANGE)


# (c) These statements are complementary, not one nonlinear certificate.
c = panel([0.025, 0.025, 0.465, 0.475], "c", "Complementary evidence")
c.add_patch(Rectangle((49.8, 19), .35, 66, facecolor=LINE, edgecolor="none"))
c.text(25, 85.5, "NONLINEAR", ha="center", fontsize=8.4,
       weight="bold", color=ORANGE)
c.text(25, 80, "SAME-MODEL WITNESS", ha="center", fontsize=7.6,
       weight="bold", color=ORANGE)
c.text(25, 73, "Fixed leader path/history", ha="center", fontsize=8.2, color=NAVY)
c.text(25, 63, "Heterogeneous nonlinear", ha="center", fontsize=8.2, color=NAVY)
c.text(25, 58, "dynamics", ha="center", fontsize=8.2, color=NAVY)
c.text(25, 48, "Logged positive-gap", ha="center", fontsize=8.2, color=NAVY)
c.text(25, 43, "prefix crossing", ha="center", fontsize=8.2, color=NAVY)
arrow(c, (25, 39), (25, 33), color=ORANGE)
c.text(25, 27, "Policy-independent", ha="center", fontsize=7.8,
       weight="bold", color=ORANGE)
c.text(25, 21, "obstruction (fixed run)", ha="center", fontsize=8.0,
       color=SLATE)
c.text(75, 85.5, "SAMPLED-LINEAR", ha="center", fontsize=8.4,
       weight="bold", color=BLUE)
c.text(75, 80, "RESERVE", ha="center", fontsize=8.1,
       weight="bold", color=BLUE)
c.text(75, 73, r"$Q=Q_0+\alpha L_\pi d$", ha="center", fontsize=9.2, color=NAVY)
c.text(75, 63, r"$m_r=b_r-c_r^TQ_0$", ha="center",
       fontsize=9.2, color=NAVY)
c.text(75, 53, r"$\gamma_r=h_D(L_\pi^Tc_r)$", ha="center",
       fontsize=9.2, color=NAVY)
c.text(75, 43, r"$\alpha_{\pi,aff}=\min\{\alpha_{max},$",
       ha="center", fontsize=9.3, color=BLUE)
c.text(75, 37, r"$\min_{r:\gamma_r>0}m_r/\gamma_r\}$",
       ha="center", fontsize=9.3, color=BLUE)
c.text(75, 27, "Limiting row:", ha="center", fontsize=8.0, color=NAVY)
c.text(75, 22, "vehicle × time × constraint", ha="center",
       fontsize=7.7, color=NAVY)
c.text(75, 16.5, "Conditional upper bound",
       ha="center", fontsize=7.7, color=SLATE)
c.add_patch(FancyBboxPatch(
    (7, 4.5), 86, 10, boxstyle="round,pad=0,rounding_size=1.1",
    facecolor=GREY, edgecolor="none"))
c.text(50, 9.5, "Distinct nonlinear and sampled-linear evidence",
       ha="center", va="center", fontsize=8.2, color=NAVY)


# (d) The four schematics represent archived active-set placement, not new data.
d = panel([0.51, 0.025, 0.465, 0.475], "d", "Spatial implication")
d.text(50, 83.5, r"Four active CAVs in each nonlinear $N=40$ layout",
       ha="center", fontsize=8.1, color=SLATE)
rows = [("Front", 71, [0, 1, 2, 3], None),
        ("Even", 56.5, [0, 2, 5, 7], None),
        ("Middle", 42, [2, 3, 4, 5], "93/102"),
        ("Rear", 27.5, [4, 5, 6, 7], "105/107")]
xs = [27, 33, 39, 45, 51, 57, 63, 69]
for name, y, actives, count in rows:
    d.text(6, y, name, va="center", fontsize=8.8, weight="bold", color=NAVY)
    first = min(actives)
    if first:
        d.add_patch(Rectangle((xs[0]-3, y-4.2), xs[first]-xs[0], 8.4,
                              facecolor=PALE_ORANGE, edgecolor="none"))
    d.plot([xs[0], xs[-1]], [y, y], color=SLATE, linewidth=0.75, zorder=1)
    for i, x in enumerate(xs):
        d.add_patch(Circle((x, y), radius=2.0,
                           facecolor=PALE_BLUE if i in actives else GREY,
                           edgecolor=BLUE if i in actives else SLATE,
                           linewidth=0.95, zorder=2))
    if count:
        d.text(81, y+2.5, count, fontsize=8.6, weight="bold", color=ORANGE)
        d.text(81, y-3, "witnesses", fontsize=7.4, color=SLATE)
    else:
        d.text(81, y, "empty $P$", fontsize=8.0, color=SLATE, va="center")
d.text(50, 13.5, "Equal tail transfer can coexist with different",
       ha="center", fontsize=8.0, color=NAVY)
d.text(50, 7.5, "local constraint reserves (Supplementary Fig. S2).",
       ha="center", fontsize=8.0, color=SLATE)

for extension in ("pdf", "svg", "png"):
    fig.savefig(OUT / f"fig1_framework.{extension}", dpi=400,
                bbox_inches="tight", pad_inches=.035)
plt.close(fig)
print("Generated native-vector Fig. 1 PDF/SVG and a raster QA preview")
