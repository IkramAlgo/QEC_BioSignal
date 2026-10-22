"""Generates publication figures directly from outputs/full_study_results.csv,
so the figures reflect your actual saved data rather than numbers
transcribed by hand from console output.

Subject and seed counts, and all paired t-test p values, are now
computed from the csv on every run, nothing is hardcoded. Pairing is
at the subject level: seeds are averaged within each subject first,
then a paired t-test runs across subjects.

Usage:
  python scripts/make_journal_figures.py
"""

from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

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
METRICS = ["mse_vs_ideal", "wasserstein_vs_ideal"]


def paired_p(df, a, b, metric):
    """Subject level paired t-test, seeds averaged within each subject."""
    pa = df[df["method"] == a].groupby("subject")[metric].mean()
    pb = df[df["method"] == b].groupby("subject")[metric].mean()
    common = pa.index.intersection(pb.index)
    if len(common) < 2:
        return float("nan")
    return stats.ttest_rel(pa[common], pb[common]).pvalue


def fmt_p(p):
    """Readable p value for labels, avoids printing p=0.0000."""
    if np.isnan(p):
        return "p=n/a"
    return "p<0.0001" if p < 0.0001 else f"p={p:.4f}"


def stars(p):
    if np.isnan(p):
        return "n/a"
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

    n_sub = df["subject"].nunique()
    n_seed = df["seed"].nunique()

    sig_vs_no_qec = {m: paired_p(df, "application_aware", "no_qec", m) for m in METRICS}
    sig_vs_uniform = {m: paired_p(df, "application_aware", "uniform_repetition_d3", m) for m in METRICS}
    sig_vs_classical = {m: paired_p(df, "application_aware", "classical_averaged_50_shots", m) for m in METRICS}

    print(f"Subjects: {n_sub}, seeds: {n_seed}")
    print("Paired t-test p values, application_aware versus each method (subject level):")
    for name, d in [("no_qec", sig_vs_no_qec), ("uniform", sig_vs_uniform), ("classical", sig_vs_classical)]:
        print(f"  vs {name}: mse {fmt_p(d['mse_vs_ideal'])}, wasserstein {fmt_p(d['wasserstein_vs_ideal'])}")
    print()

    # ---------- Figure 1: pooled comparison, mean +/- std, all 4 methods ----------
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.4))
    for ax, metric, ylabel in zip(
        axes, METRICS, ["MSE vs ideal", "Wasserstein distance vs ideal"],
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
        # labels sit above the error bar, not on top of it, to avoid overlap
        for bar, v, s in zip(bars, means, stds):
            ax.annotate(f"{v:.2e}", xy=(bar.get_x() + bar.get_width() / 2, v + s),
                        xytext=(0, 5), textcoords="offset points", ha="center", va="bottom", fontsize=7.5)
        ax.set_title(ylabel, fontsize=12, fontweight="bold", pad=10)
        ax.set_ylabel(ylabel)
        ax.set_axisbelow(True)
        ax.tick_params(axis="x", labelsize=8.5)
        ax.set_ylim(0, max(m + s for m, s in zip(means, stds)) * 1.35)

    fig.suptitle(
        f"Pooled Results, {n_sub} Subjects x {n_seed} Seeds (mean \u00b1 std, n={n_sub * n_seed} runs)",
        fontsize=12, y=1.03,
    )
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig1_pooled_comparison.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIG_DIR / "fig1_pooled_comparison.pdf", bbox_inches="tight")
    plt.close(fig)
    print("Saved fig1_pooled_comparison (.png and .pdf)")

    # ---------- Figure 2: per-subject breakdown, computed from real data ----------
    subjects = sorted(df["subject"].unique())
    fig2, axes2 = plt.subplots(1, 2, figsize=(max(13, 1.5 * n_sub + 6), 5.4))
    x = np.arange(len(subjects))
    width = 0.19

    for ax, metric, ylabel in zip(
        axes2, METRICS,
        ["MSE vs ideal, per subject (seed-averaged)", "Wasserstein vs ideal, per subject (seed-averaged)"],
    ):
        for i, m in enumerate(METHOD_ORDER):
            sub_means = [df[(df["subject"] == s) & (df["method"] == m)][metric].mean() for s in subjects]
            ax.bar(x + (i - 1.5) * width, sub_means, width, label=METHOD_LABELS[m].replace("\n", " "),
                   color=COLORS[m], edgecolor="#222222", linewidth=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels(subjects, rotation=45 if n_sub > 6 else 0, ha="right" if n_sub > 6 else "center")
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

    # ---------- Figure 3: significance of application_aware's paired comparisons ----------
    fig3, axes3 = plt.subplots(1, 2, figsize=(9.5, 4.5))
    comparisons = [
        ("vs No QEC", sig_vs_no_qec),
        ("vs Uniform QEC", sig_vs_uniform),
        ("vs Classical", sig_vs_classical),
    ]
    for ax, metric, title in zip(axes3, METRICS, ["MSE", "Wasserstein"]):
        labels = [c[0] for c in comparisons]
        pvals = [c[1][metric] for c in comparisons]
        y_pos = np.arange(len(labels))
        bar_colors = ["#3F8F5F" if p < 0.05 else "#999999" for p in pvals]
        ax.barh(y_pos, [-np.log10(p) for p in pvals], color=bar_colors, edgecolor="#222222", linewidth=0.6)
        ax.axvline(-np.log10(0.05), color="red", linestyle="--", linewidth=1, label="p = 0.05")
        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels)
        ax.set_xlabel(f"-log10(p), paired t-test, n={n_sub} subjects")
        ax.set_title(f"Application-Aware {title}: significance", fontsize=10.5, fontweight="bold")
        for i, p in enumerate(pvals):
            ax.annotate(f"{fmt_p(p)} {stars(p)}", xy=(-np.log10(p), i), xytext=(4, 0),
                        textcoords="offset points", va="center", fontsize=8)
        ax.set_xlim(0, max(-np.log10(p) for p in pvals) * 1.4)
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