"""Diagnostic figure: does the tier given to logical qubit 0 predict whether
application aware beats no QEC?

For every subject and seed in the results csv, plots application aware
error divided by no QEC error (below 1 means application aware helped),
coloured by the tier that logical qubit 0 received. Reads the tiers column
that run_full_study.py now saves.

Usage:
  python scripts/make_tier_diagnostic_figure.py
  python scripts/make_tier_diagnostic_figure.py --csv full_study_results_erroraware.csv --tag erroraware
  python scripts/make_tier_diagnostic_figure.py --qubit 1
"""

import argparse
import ast
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = PROJECT_ROOT / "outputs"
FIG_DIR = OUT_DIR / "figures"
FIG_DIR.mkdir(exist_ok=True, parents=True)

TIER_COLORS = {"repetition": "#3F8F5F", "bare": "#B0413E", "shor": "#4C6A92"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="full_study_results.csv")
    parser.add_argument("--tag", default="")
    parser.add_argument("--qubit", type=int, default=0)
    args = parser.parse_args()

    df = pd.read_csv(OUT_DIR / args.csv)
    if "tiers" not in df.columns:
        print("This csv has no tiers column, it was made before run_full_study.py started saving tiers.")
        return

    aa = df[df["method"] == "application_aware"].set_index(["subject", "seed"])
    nq = df[df["method"] == "no_qec"].set_index(["subject", "seed"])
    runs = aa.index.intersection(nq.index)

    rows = []
    for key in runs:
        tiers = ast.literal_eval(str(aa.loc[key, "tiers"]))
        rows.append({
            "subject": key[0], "seed": key[1],
            "tier": tiers.get(args.qubit, "unknown"),
            "mse_ratio": aa.loc[key, "mse_vs_ideal"] / nq.loc[key, "mse_vs_ideal"],
            "wass_ratio": aa.loc[key, "wasserstein_vs_ideal"] / nq.loc[key, "wasserstein_vs_ideal"],
        })
    res = pd.DataFrame(rows).sort_values(["subject", "seed"]).reset_index(drop=True)

    print(f"Runs: {len(res)}.  Tier given to logical qubit {args.qubit}:")
    for tier, g in res.groupby("tier"):
        print(f"  {tier:11s} runs={len(g):2d}  application aware beat no QEC on MSE in "
              f"{int((g['mse_ratio'] < 1).sum())}, on Wasserstein in {int((g['wass_ratio'] < 1).sum())}, "
              f"median MSE ratio {g['mse_ratio'].median():.2f}")

    fig, axes = plt.subplots(2, 1, figsize=(max(9, 0.45 * len(res) + 3), 7.5), sharex=True)
    x = np.arange(len(res))
    labels = [f"{s.replace('EPCTL', '')}\ns{seed}" for s, seed in zip(res["subject"], res["seed"])]
    for ax, col, title in zip(
        axes, ["mse_ratio", "wass_ratio"],
        ["MSE, application aware divided by no QEC", "Wasserstein, application aware divided by no QEC"],
    ):
        ax.bar(x, res[col], color=[TIER_COLORS.get(t, "#999999") for t in res["tier"]],
               edgecolor="#222222", linewidth=0.5)
        ax.axhline(1.0, color="black", linestyle="--", linewidth=1)
        ax.set_ylabel("ratio (below 1 is better)")
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.grid(axis="y", alpha=0.25)
        ax.set_axisbelow(True)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, fontsize=7.5)
    axes[0].legend(
        handles=[Patch(color=c, label=f"qubit {args.qubit}: {t}") for t, c in TIER_COLORS.items()],
        fontsize=8, loc="upper right",
    )
    fig.suptitle(f"Per run outcome, coloured by the tier given to logical qubit {args.qubit}", fontsize=12, y=1.0)
    fig.tight_layout()

    suffix = f"_{args.tag}" if args.tag else ""
    png = FIG_DIR / f"fig4_tier_diagnostic{suffix}.png"
    pdf = FIG_DIR / f"fig4_tier_diagnostic{suffix}.pdf"
    fig.savefig(png, dpi=300, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    print(f"Saved {png}")
    print(f"Saved {pdf}")


if __name__ == "__main__":
    main()