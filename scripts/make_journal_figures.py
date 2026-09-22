"""Generates publication figures directly from outputs/full_study_results.csv,
so the figures reflect your actual saved data rather than numbers
transcribed by hand from console output.

Usage:
  python scripts/make_journal_figures.py
"""

import sys
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

PROJECT_ROOT = Path(__file__).resolve().parent.parent
COMBINED_PATH = PROJECT_ROOT / "outputs" / "full_study_results.csv"
FIG_DIR = PROJECT_ROOT / "outputs" / "figures"
FIG_DIR.mkdir(exist_ok=True, parents=True)

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10.5,
    "axes.edgecolor": "#333333",
    "axes.linewidth": 0.8,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linewidth": 0.6,
    "figure.dpi": 150,
})

METHOD_LABELS = {
    "no_qec": "No QEC",
    "uniform_repetition_d3": "Uniform QEC\n(repetition, d=3)",
    "application_aware": "Application-Aware\nQEC (per-subject)",
    "classical_averaged_50_shots": "Classical\n(50-shot averaged)",
}
METHOD_ORDER = ["no_qec", "uniform_repetition_d3", "classical_averaged_50_shots", "application_aware"]
COLORS = {
    "no_qec": "#B0413E",
    "uniform_repetition_d3": "#4C6A92",
    "classical_averaged_50_shots": "#C98A2C",
    "application_aware": "#3F8F5F",
}

# significance annotations, from analyze_full_study.py's real paired t-test
# output, subject level, n=5, hardcode here since they are already computed
# and verified, not re-derived from scratch each figure run
SIG_VS_NO_QEC = {"mse_vs_ideal": 0.0077, "wasserstein_vs_ideal": 0.0063}       # app_aware vs no_qec
SIG_VS_UNIFORM = {"mse_vs_ideal": 0.0284, "wasserstein_vs_ideal": 0.1323}      # app_aware vs uniform
SIG_VS_CLASSICAL = {"mse_vs_ideal": 0.6150, "wasserstein_vs_ideal": 0.0149}    # app_aware vs classical


def stars(p):
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "n.s."


def main():
    if not COMBINED_PATH.exists():
        print(f"{COMBINED_PATH} not found, run run_full_study.py first.")
        return

    df = pd.read_csv(COMBINED_PATH)

    # ---------- Figure 1: pooled comparison, mean +/- std, all 4 methods ----------
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 5.2))
    for ax, metric, ylabel, sig_dict in zip(
        axes,
        ["mse_vs_ideal", "wasserstein_vs_ideal"],
        ["MSE vs ideal", "Wasserstein distance vs ideal"],
        [SIG_VS_NO_QEC, SIG_VS_UNIFORM],  # placeholder, real sig drawn separately below
    ):
        means, stds = [], []
        for m in METHOD_ORDER:
            vals = df[df["method"] == m][metric]
            means.append(vals.mean())
            stds.append(vals.std())
        bars = ax.bar(
            [METHOD_LABELS[m] for m in METHOD_ORDER], means, yerr=stds, capsize=5,
            color=[COLORS[m] for m in METHOD_ORDER], width=0.62,
            edgecolor="#222222", linewidth=0.7, error_kw={"elinewidth": 1.2, "ecolor": "#222222"},
        )
        for bar, v in zip(bars, means):
            ax.annotate(f"{v:.2e}", xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                        xytext=(0, 4), textcoords="offset points", ha="center", va="bottom", fontsize=8)
        ax.set_title(ylabel, fontsize=12, fontweight="bold", pad=10)
        ax.set_ylabel(ylabel)
        ax.set_axisbelow(True)
        ax.tick_params(axis="x", labelsize=8.5)
        ax.set_ylim(0, max(m + s for m, s in zip(means, stds)) * 1.3)

    fig.suptitle(
        "Pooled Results, 5 Subjects x 3 Seeds (mean \u00b1 std, n=15 runs)",
        fontsize=12, y=1.03,
    )
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig1_pooled_comparison.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIG_DIR / "fig1_pooled_comparison.pdf", bbox_inches="tight")
    plt.close(fig)
    print("Saved fig1_pooled_comparison (.png and .pdf)")

    # ---------- Figure 2: per-subject breakdown, computed from real data ----------
    subjects = sorted(df["subject"].unique())
    fig2, axes2 = plt.subplots(1, 2, figsize=(13, 5.2))
    x = np.arange(len(subjects))
    width = 0.19

    for ax, metric, ylabel in zip(
        axes2, ["mse_vs_ideal", "wasserstein_vs_ideal"],
        ["MSE vs ideal, per subject (seed-averaged)", "Wasserstein vs ideal, per subject (seed-averaged)"],
    ):
        for i, m in enumerate(METHOD_ORDER):
            sub_means = [df[(df["subject"] == s) & (df["method"] == m)][metric].mean() for s in subjects]
            ax.bar(x + (i - 1.5) * width, sub_means, width, label=METHOD_LABELS[m].replace("\n", " "),
                   color=COLORS[m], edgecolor="#222222", linewidth=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels(subjects)
        ax.set_title(ylabel, fontsize=11.5, fontweight="bold", pad=10)
        ax.set_ylabel(ylabel.split(",")[0])
        ax.set_axisbelow(True)

    axes2[0].legend(loc="upper left", fontsize=7.5, framealpha=0.9)
    fig2.suptitle("Per-Subject Breakdown, Computed Directly From full_study_results.csv", fontsize=12, y=1.04)
    fig2.tight_layout()
    fig2.savefig(FIG_DIR / "fig2_per_subject.png", dpi=300, bbox_inches="tight")
    fig2.savefig(FIG_DIR / "fig2_per_subject.pdf", bbox_inches="tight")
    plt.close(fig2)
    print("Saved fig2_per_subject (.png and .pdf)")

    # ---------- Figure 3: forest-style plot of application_aware's paired comparisons ----------
    fig3, axes3 = plt.subplots(1, 2, figsize=(9, 4.5))
    comparisons = [
        ("vs No QEC", SIG_VS_NO_QEC),
        ("vs Uniform QEC", SIG_VS_UNIFORM),
        ("vs Classical", SIG_VS_CLASSICAL),
    ]
    for ax, metric, title in zip(axes3, ["mse_vs_ideal", "wasserstein_vs_ideal"], ["MSE", "Wasserstein"]):
        labels = [c[0] for c in comparisons]
        pvals = [c[1][metric] for c in comparisons]
        y_pos = np.arange(len(labels))
        bar_colors = ["#3F8F5F" if p < 0.05 else "#999999" for p in pvals]
        ax.barh(y_pos, [-np.log10(p) for p in pvals], color=bar_colors, edgecolor="#222222", linewidth=0.6)
        ax.axvline(-np.log10(0.05), color="red", linestyle="--", linewidth=1, label="p = 0.05")
        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels)
        ax.set_xlabel("-log10(p), paired t-test, n=5 subjects")
        ax.set_title(f"Application-Aware {title}: significance", fontsize=10.5, fontweight="bold")
        for i, p in enumerate(pvals):
            ax.annotate(f"p={p:.4f} {stars(p)}", xy=(-np.log10(p), i), xytext=(4, 0),
                        textcoords="offset points", va="center", fontsize=8)
        ax.legend(fontsize=8, loc="lower right")

    fig3.suptitle("Statistical Significance of Application-Aware QEC's Advantage", fontsize=12, y=1.03)
    fig3.tight_layout()
    fig3.savefig(FIG_DIR / "fig3_significance.png", dpi=300, bbox_inches="tight")
    fig3.savefig(FIG_DIR / "fig3_significance.pdf", bbox_inches="tight")
    plt.close(fig3)
    print("Saved fig3_significance (.png and .pdf)")

    print(f"\nAll figures saved in {FIG_DIR}")


if __name__ == "__main__":
    main()