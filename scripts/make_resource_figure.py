"""Publication figure: error against physical qubit cost for every protection plan.

Six panels from one script:
  (a) MSE against physical qubits, mean and standard error over subjects,
      every subject as a faint dot, efficient frontier dotted
  (b) the same for Wasserstein
  (c) per subject change in MSE against no QEC, one dot per subject
  (d) the same for Wasserstein
  (e) MSE difference against the classical 50 shot average, mean and 95% CI
  (f) the same for Wasserstein

Reads outputs/controls_results.csv (run_controls.py) and
outputs/full_study_results_erroraware.csv (the main study). Seeds are
averaged inside each subject first, so every statistic is a subject level
statistic. Also writes outputs/figures/fig5_summary_table.csv.

Usage:
  python scripts/make_resource_figure.py
  python scripts/make_resource_figure.py --dpi 500 --svg
  python scripts/make_resource_figure.py --no_title
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = PROJECT_ROOT / "outputs"
FIG_DIR = OUT_DIR / "figures"
FIG_DIR.mkdir(exist_ok=True, parents=True)

# key: (short label, long label, colour, marker, x offset in panels a and b)
STYLE = {
    "classical": ("Classical\n50 shot", "Classical 50 shot average", "#4D4D4D", "X", 0.0),
    "no_qec": ("No QEC\n4 qubits", "No QEC", "#8C8C8C", "o", 0.0),
    "rep03": ("Repetition\n0 and 3, 8q", "Repetition on qubits 0 and 3 only", "#009E73", "s", 0.0),
    "shor2": ("Shor on\nqubit 2, 12q", "Shor on qubit 2 only", "#E69F00", "^", -0.22),
    "uniform": ("Uniform\nrepetition, 12q", "Uniform repetition", "#0072B2", "D", 0.22),
    "plan_A": ("Fixed plan A\n16q", "Fixed plan A, same for every subject", "#56B4E9", "P", -0.22),
    "rule": ("Error aware\nrule, 16q", "Error aware rule", "#D55E00", "*", 0.22),
    "q1rep": ("Repetition on\nqubit 1, 16q", "Negative control, repetition on qubit 1", "#B0413E", "v", 0.0),
}
PLAN_KEYS = {"plan_A": "plan_A", "rep_only_03": "rep03", "shor_only_2": "shor2", "plan_q1_rep": "q1rep"}
STUDY_KEYS = {
    "no_qec": "no_qec",
    "uniform_repetition_d3": "uniform",
    "application_aware": "rule",
    "classical_averaged_50_shots": "classical",
}
STUDY_QUBITS = {"no_qec": 4, "uniform": 12, "rule": 16, "classical": 0}
ORDER = ["no_qec", "rep03", "shor2", "uniform", "plan_A", "rule", "q1rep"]  # panels a, b, e, f
STRIP_ORDER = ["classical", "rep03", "shor2", "uniform", "plan_A", "rule", "q1rep"]  # panels c, d

GREEN, RED = "#1E8449", "#B03A2E"


def describe_subjects(subjects):
    """EPCTL01 to 07, 09, 11 and 12 style text, so skipped subjects are not hidden."""
    nums = sorted(int(s[-2:]) for s in subjects)
    runs, start = [], nums[0]
    for prev, cur in zip(nums, nums[1:] + [None]):
        if cur != prev + 1:
            runs.append((start, prev))
            start = cur
    parts = []
    for a, b in runs:
        if b - a >= 2:
            parts.append(f"{a:02d} to {b:02d}")
        else:
            parts.extend(f"{n:02d}" for n in range(a, b + 1))
    text = ", ".join(parts[:-1]) + " and " + parts[-1] if len(parts) > 1 else parts[0]
    return "EPCTL" + text


def set_style():
    plt.rcParams.update({
        "font.family": "DejaVu Serif",
        "font.size": 10,
        "axes.titlesize": 11.5,
        "axes.titleweight": "bold",
        "axes.labelsize": 10,
        "axes.edgecolor": "#333333",
        "axes.linewidth": 0.9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": "#D9D9D9",
        "grid.alpha": 0.7,
        "grid.linewidth": 0.6,
        "xtick.color": "#333333",
        "ytick.color": "#333333",
        "figure.facecolor": "white",
    })


def load_long():
    controls = pd.read_csv(OUT_DIR / "controls_results.csv")
    study = pd.read_csv(OUT_DIR / "full_study_results_erroraware.csv")
    parts, qubits = [], dict(STUDY_QUBITS)
    for method, key in STUDY_KEYS.items():
        g = study[study["method"] == method]
        parts.append(pd.DataFrame({
            "key": key, "subject": g["subject"], "seed": g["seed"],
            "mse": g["mse_vs_ideal"], "w": g["wasserstein_vs_ideal"],
        }))
    for plan, g in controls.groupby("plan"):
        key = PLAN_KEYS.get(plan)
        if key is None:
            continue
        qubits[key] = int(g["physical_qubits"].iloc[0])
        parts.append(pd.DataFrame({
            "key": key, "subject": g["subject"], "seed": g["seed"],
            "mse": g["mse_vs_ideal"], "w": g["wasserstein_vs_ideal"],
        }))
    long = pd.concat(parts, ignore_index=True)
    sub = long.groupby(["key", "subject"])[["mse", "w"]].mean().reset_index()
    return sub, qubits


def sem(x):
    return float(np.std(x, ddof=1) / np.sqrt(len(x)))


def frontier(points):
    """points: list of (qubits, error, key). Keeps points no worse than every cheaper point."""
    best, out = np.inf, []
    for q, e, k in sorted(points, key=lambda p: (p[0], p[1])):
        if e < best - 1e-12:
            out.append((q, e, k))
            best = e
    return out


def panel_label(ax, text):
    ax.text(-0.13, 1.07, text, transform=ax.transAxes, fontsize=15, fontweight="bold", va="bottom")


def pareto_panel(ax, sub, qubits, col, ylabel, title, keys, rng):
    ref = sub[sub["key"] == "no_qec"].set_index("subject")[col]
    pts = []
    for key in keys:
        g = sub[sub["key"] == key]
        label, _, color, marker, dx = STYLE[key]
        x = qubits[key] + dx
        jitter = rng.uniform(-0.07, 0.07, size=len(g))
        ax.scatter(np.full(len(g), x) + jitter, g[col], s=14, color=color, alpha=0.30, linewidths=0, zorder=2)
        m = g[col].mean()
        ax.errorbar(x, m, yerr=sem(g[col]), fmt=marker, color=color, ecolor=color, elinewidth=1.6, capsize=3,
                    markersize=13 if marker == "*" else 9, markeredgecolor="#1A1A1A", markeredgewidth=0.7, zorder=4)
        pts.append((qubits[key], m, key))
    fr = frontier(pts)
    ax.plot([p[0] for p in fr], [p[1] for p in fr], linestyle=(0, (2, 2)), color="#444444", linewidth=1.1,
            zorder=1, label="Efficient frontier")
    c = sub[sub["key"] == "classical"][col].mean()
    ax.axhline(c, color=STYLE["classical"][2], linestyle="--", linewidth=1.3, zorder=1)
    ax.text(16.9, c, " classical", color=STYLE["classical"][2], fontsize=8.5, va="bottom", ha="right")
    # callouts
    for key, off in [("rep03", (22, 26)), ("rule", (-78, 34))]:
        if key not in keys:
            continue
        g = sub[sub["key"] == key].set_index("subject")[col]
        pct = 100 * (g.mean() / ref.mean() - 1)
        label, _, color, marker, dx = STYLE[key]
        ax.annotate(f"{pct:+.0f}% vs no QEC", xy=(qubits[key] + dx, g.mean()), xytext=off,
                    textcoords="offset points", fontsize=8.5, color=color, fontweight="bold",
                    arrowprops=dict(arrowstyle="-", color=color, lw=0.9), ha="center")
    ax.set_xlabel("Physical qubits")
    ax.set_ylabel(ylabel)
    ax.set_title(title, loc="left")
    ax.set_xticks([4, 8, 12, 16])
    ax.set_xlim(2.8, 17.2)
    top = max(sub[sub["key"].isin(keys)][col].max(), c) * 1.08
    ax.set_ylim(0, top)


def strip_panel(ax, sub, col, ylabel, title, keys, rng):
    ref = sub[sub["key"] == "no_qec"].set_index("subject")[col]
    n = len(ref)
    ax.axhline(0, color="#1A1A1A", linewidth=1.1, zorder=1)
    ax.axhspan(-100, 0, color=GREEN, alpha=0.05, zorder=0)
    ax.axhspan(0, 100, color=RED, alpha=0.05, zorder=0)
    lo = 0
    for i, key in enumerate(keys):
        g = sub[sub["key"] == key].set_index("subject")[col]
        pct = 100 * (g / ref - 1)
        color = STYLE[key][2]
        x = np.full(len(pct), i) + rng.uniform(-0.17, 0.17, size=len(pct))
        ax.scatter(x, pct, s=34, color=color, alpha=0.85, edgecolor="#1A1A1A", linewidth=0.5, zorder=3)
        ax.hlines(pct.mean(), i - 0.33, i + 0.33, color="#1A1A1A", linewidth=2.4, zorder=4)
        better = int((pct < 0).sum())
        ax.text(i, 1.0, f"{better}/{n}", transform=ax.get_xaxis_transform(), ha="center", va="bottom",
                fontsize=8.5, color="#333333", fontweight="bold")
        ax.text(i + 0.36, pct.mean(), f"{pct.mean():+.0f}%", fontsize=8.3, va="center", color="#1A1A1A")
        lo = min(lo, pct.min())
    ymax = max(10, max(100 * (sub[sub["key"] == k].set_index("subject")[col] / ref - 1).max() for k in keys) + 6)
    ax.set_ylim(min(-80, lo - 6), ymax)
    ax.set_xlim(-0.6, len(keys) - 0.35)
    ax.set_xticks(range(len(keys)))
    ax.set_xticklabels([STYLE[k][0] for k in keys], fontsize=8)
    ax.set_ylabel(ylabel)
    ax.set_title(title, loc="left", pad=20)
    ax.grid(axis="x", visible=False)
    ax.text(0.995, 0.03, "better", transform=ax.transAxes, ha="right", va="bottom", fontsize=8, color=GREEN,
            style="italic")
    ax.text(0.995, 0.97, "worse", transform=ax.transAxes, ha="right", va="top", fontsize=8, color=RED, style="italic")


def forest_panel(ax, sub, col, xlabel, title, keys):
    cl = sub[sub["key"] == "classical"].set_index("subject")[col]
    n = len(cl)
    ys = np.arange(len(keys))[::-1]
    xmin, xmax = 0, 0
    rows = []
    for y, key in zip(ys, keys):
        g = sub[sub["key"] == key].set_index("subject")[col]
        pct = 100 * (g / cl - 1)
        m = pct.mean()
        lo, hi = stats.t.interval(0.95, n - 1, loc=m, scale=sem(pct))
        rows.append((y, key, m, lo, hi))
        xmin, xmax = min(xmin, lo), max(xmax, hi)
    pad = 0.12 * (xmax - xmin)
    ax.axvspan(xmin - 10 * pad, 0, color=GREEN, alpha=0.06, zorder=0)
    ax.axvspan(0, xmax + 10 * pad, color=RED, alpha=0.06, zorder=0)
    ax.axvline(0, color=STYLE["classical"][2], linestyle="--", linewidth=1.4, zorder=1)
    for y, key, m, lo, hi in rows:
        color, marker = STYLE[key][2], STYLE[key][3]
        ax.hlines(y, lo, hi, color=color, linewidth=2.6, zorder=3)
        ax.plot(m, y, marker, color=color, markersize=13 if marker == "*" else 9, markeredgecolor="#1A1A1A",
                markeredgewidth=0.7, zorder=4)
        ax.text(hi + 0.02 * (xmax - xmin + 2 * pad), y, f"{m:+.0f}%", va="center", fontsize=8.5, color="#1A1A1A",
                fontweight="bold")
    ax.set_yticks(ys)
    ax.set_yticklabels([STYLE[k][1] for k in keys], fontsize=8.3)
    ax.set_xlim(xmin - pad, xmax + 3 * pad)
    ax.set_ylim(-0.7, len(keys) - 0.3)
    ax.set_xlabel(xlabel)
    ax.set_title(title, loc="left")
    ax.grid(axis="y", visible=False)
    ax.text(0.01, 0.02, "better than classical", transform=ax.transAxes, fontsize=8, color=GREEN, style="italic",
            ha="left", va="bottom")
    ax.text(0.99, 0.02, "worse", transform=ax.transAxes, fontsize=8, color=RED, style="italic", ha="right",
            va="bottom")


def summary_table(sub, qubits, keys):
    ref_nq = sub[sub["key"] == "no_qec"].set_index("subject")
    ref_cl = sub[sub["key"] == "classical"].set_index("subject")
    rows = []
    for key in ["classical"] + keys:
        g = sub[sub["key"] == key].set_index("subject")
        rows.append({
            "plan": STYLE[key][1], "physical_qubits": qubits[key],
            "mse_mean": g["mse"].mean(), "mse_sem": sem(g["mse"]),
            "wasserstein_mean": g["w"].mean(), "wasserstein_sem": sem(g["w"]),
            "mse_vs_no_qec_pct": 100 * (g["mse"].mean() / ref_nq["mse"].mean() - 1),
            "wasserstein_vs_no_qec_pct": 100 * (g["w"].mean() / ref_nq["w"].mean() - 1),
            "mse_vs_classical_pct": 100 * (g["mse"].mean() / ref_cl["mse"].mean() - 1),
            "wasserstein_vs_classical_pct": 100 * (g["w"].mean() / ref_cl["w"].mean() - 1),
        })
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dpi", type=int, default=400)
    parser.add_argument("--svg", action="store_true", help="Also write an SVG file.")
    parser.add_argument("--no_title", action="store_true", help="Leave out the figure title, for journal figures.")
    args = parser.parse_args()

    set_style()
    rng = np.random.default_rng(7)
    sub, qubits = load_long()
    n_sub = sub["subject"].nunique()
    subjects = sorted(sub["subject"].unique())
    keys = [k for k in ORDER if k in set(sub["key"])]
    strip_keys = [k for k in STRIP_ORDER if k in set(sub["key"])]

    fig = plt.figure(figsize=(13.8, 15.2))
    gs = GridSpec(3, 2, figure=fig, height_ratios=[1.05, 1.0, 0.95], hspace=0.44, wspace=0.26,
                  left=0.075, right=0.975, top=0.925, bottom=0.135)
    ax_a, ax_b = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])
    ax_c, ax_d = fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])
    ax_e, ax_f = fig.add_subplot(gs[2, 0]), fig.add_subplot(gs[2, 1])

    pareto_panel(ax_a, sub, qubits, "mse", "MSE against the ideal circuit", "Error against qubit cost, MSE", keys, rng)
    pareto_panel(ax_b, sub, qubits, "w", "Wasserstein distance to the ideal", "Error against qubit cost, Wasserstein",
                 keys, rng)
    strip_panel(ax_c, sub, "mse", "Change in MSE against no QEC (%)", "Every subject, MSE", strip_keys, rng)
    strip_panel(ax_d, sub, "w", "Change in Wasserstein against no QEC (%)", "Every subject, Wasserstein",
                strip_keys, rng)
    forest_panel(ax_e, sub, "mse", "Difference in MSE against the classical average (%)",
                 "Against the classical baseline, MSE", keys)
    forest_panel(ax_f, sub, "w", "Difference in Wasserstein against the classical average (%)",
                 "Against the classical baseline, Wasserstein", keys)
    for ax, t in zip([ax_a, ax_b, ax_c, ax_d, ax_e, ax_f], "abcdef"):
        panel_label(ax, f"({t})")

    handles = [Line2D([0], [0], marker=STYLE[k][3], color="none", markerfacecolor=STYLE[k][2],
                      markeredgecolor="#1A1A1A", markersize=12 if STYLE[k][3] == "*" else 9, label=STYLE[k][1])
               for k in keys]
    handles.append(Line2D([0], [0], color=STYLE["classical"][2], linestyle="--", label=STYLE["classical"][1]))
    handles.append(Line2D([0], [0], color="#444444", linestyle=(0, (2, 2)), label="Efficient frontier"))
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=9, frameon=False, bbox_to_anchor=(0.5, 0.0))

    if not args.no_title:
        fig.suptitle("Application aware error correction: error against physical qubit cost", fontsize=15,
                     fontweight="bold", y=0.985)
        fig.text(0.5, 0.955,
                 f"{n_sub} subjects ({describe_subjects(subjects)}, three seeds each, averaged within subject), "
                 "depolarizing noise p = 0.02, 50 shots, 20 held out windows per run",
                 ha="center", fontsize=9.5, color="#444444")
    fig.text(0.5, 0.066,
             "(a, b) mean and standard error over subjects, faint dots are single subjects.   "
             "(c, d) one dot per subject, black bar is the mean, fractions count subjects better than no QEC.\n"
             "(e, f) mean and 95% confidence interval of the subject level difference against the classical average.",
             ha="center", fontsize=8, color="#555555", linespacing=1.5)

    base = FIG_DIR / "fig5_resource_overview"
    fig.savefig(f"{base}.png", dpi=args.dpi, bbox_inches="tight")
    fig.savefig(f"{base}.pdf", bbox_inches="tight")
    if args.svg:
        fig.savefig(f"{base}.svg", bbox_inches="tight")
    plt.close(fig)

    table = summary_table(sub, qubits, keys)
    table.to_csv(FIG_DIR / "fig5_summary_table.csv", index=False)
    pd.set_option("display.width", 220)
    print(table.round(6).to_string(index=False))
    print(f"\nSaved {base}.png, {base}.pdf" + (", .svg" if args.svg else "") + f" and {FIG_DIR / 'fig5_summary_table.csv'}")


if __name__ == "__main__":
    main()