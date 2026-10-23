"""Final figures and tables for the noise type study (publication style, version 4).

Changes from version 3
  * Loader fixed: depolarizing rows from sweep_results.csv are appended again, and the phase flip
    code rows at depolarizing p = 0.02 (3 seeds) are kept.
  * Phase flip noise is now shown at p = 0.05, 0.10 and 0.20 (regime map, level curves, tables).
  * Wording: the Holm family of 8 comparisons was chosen after an initial look at the data
    (post hoc, not preregistered). Everything else is exploratory.

Holm correction runs only over the post hoc PRIMARY family: the four bit/phase codes
at p = 0.05 under bit flip and phase flip noise (the settings with 3 seeds). Everything else is
exploratory: counts and intervals only, no stars, no filled markers.
With 10 subjects the smallest possible two sided Wilcoxon p is 0.002, so the best possible
Holm adjusted p over 8 tests is 0.0156.

Level curves (fig5, S1, S2) and the clean / raw reference lines use seed 0 only, so every point on a curve
has the same seed base. Matched seed pairs are still used for the p = 0.05 comparisons (fig1, fig2, tables).

Reads (from the outputs folder)
  v2_results.csv      depolarizing p = 0.02 (3 seeds) for no_qec, rep03, uniform, rule, plus clean and raw references
  sweep_results.csv   depolarizing p = 0.05, 0.10, 0.20 (all methods) and phase codes at p = 0.02 (3 seeds)
  sweep_pauli.csv     bit flip p = 0.05, 0.10, 0.20 and phase flip p = 0.05, 0.10, 0.20
  sweep_diag.csv      depolarizing p = 0.05 with separate Z and X errors (4 subjects)

Writes
  outputs/final_figures/*.png (300 dpi), *.pdf (vector) and *.svg (editable)
  outputs/final_summary.csv        mean and SE per noise type, level and method (all seeds, n_seeds column)
  outputs/final_summary_seed0.csv  the same, seed 0 only (what the level curves show)
  outputs/final_tables.csv         paired comparison of each method against no protection

Usage (from the project folder):
  python scripts\\make_final_figures.py
"""

import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyBboxPatch
import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs"
FIG = OUT / "final_figures"
FIG.mkdir(parents=True, exist_ok=True)

INK = "#1F2937"
SOFT = "#9CA3AF"
METHODS = ["no_qec", "rep03", "uniform", "phase03", "phase_uniform", "rule"]
COMPARED = ["rep03", "uniform", "phase03", "phase_uniform", "rule"]  # narrowed to what is in the data by main()
# Post hoc primary family for the Holm correction (chosen after an initial look at the data, not preregistered)
PRIMARY_SETTINGS = {("bitflip", 0.05), ("phaseflip", 0.05)}
PRIMARY_METHODS = {"rep03", "uniform", "phase03", "phase_uniform"}
METHOD_COLOR = {"no_qec": "#6B7280", "rep03": "#D97706", "uniform": "#1D4ED8", "phase03": "#22D3EE",
                "phase_uniform": "#0E7490", "rule": "#BE185D"}
NOISE_COLOR = {"bitflip": "#0F766E", "phaseflip": "#7C3AED", "depolarizing": "#C2410C"}
Z_COLOR, X_COLOR = "#0F766E", "#C2410C"
QUBITS = {"no_qec": 4, "rep03": 8, "uniform": 12, "phase03": 8, "phase_uniform": 12, "rule": 16}
SHORT = {"no_qec": "None", "rep03": "Bit code, qubits 0,3", "uniform": "Bit code, all 4",
         "phase03": "Phase code, qubits 0,3", "phase_uniform": "Phase code, all 4", "rule": "Error aware rule"}
FOREST = {"rep03": "Bit, qubits 0,3", "uniform": "Bit, all 4", "phase03": "Phase, qubits 0,3",
          "phase_uniform": "Phase, all 4", "rule": "Rule"}
TICK = {"no_qec": "None", "rep03": "Bit\n0,3", "uniform": "Bit\nall", "phase03": "Phase\n0,3",
        "phase_uniform": "Phase\nall", "rule": "Rule"}
LABEL = {
    "no_qec": "No protection (4 qubits)",
    "rep03": "Bit flip code on qubits 0 and 3 (8 qubits)",
    "uniform": "Bit flip code on all qubits (12 qubits)",
    "phase03": "Phase flip code on qubits 0 and 3 (8 qubits)",
    "phase_uniform": "Phase flip code on all qubits (12 qubits)",
    "rule": "Error aware rule (16 qubits)",
}
NOISE_TITLE = {"bitflip": "Bit flip noise", "phaseflip": "Phase flip noise", "depolarizing": "Depolarizing noise"}
ALL_SETTINGS = [("bitflip", 0.05), ("bitflip", 0.10), ("bitflip", 0.20),
                ("phaseflip", 0.05), ("phaseflip", 0.10), ("phaseflip", 0.20),
                ("depolarizing", 0.02), ("depolarizing", 0.05), ("depolarizing", 0.10), ("depolarizing", 0.20)]
COLS = ["subject", "seed", "noise_type", "noise_prob", "method", "mse_vs_ideal", "wasserstein_vs_ideal", "macro_f1"]

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 8.5, "axes.titlesize": 9.5, "axes.titleweight": "bold",
    "axes.labelsize": 8.5, "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
    "axes.edgecolor": INK, "axes.labelcolor": INK, "text.color": INK, "xtick.color": INK, "ytick.color": INK,
    "axes.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 120, "savefig.dpi": 300, "savefig.bbox": "tight", "pdf.fonttype": 42,
})


# ----------------------------------------------------------------------------- data
def read(name):
    path = OUT / name
    if not path.exists():
        print(f"  missing {name}, skipping whatever needs it")
        return None
    return pd.read_csv(path)


def load_all():
    parts, clean, raw = [], None, None
    v2 = read("v2_results.csv")
    if v2 is not None:
        v0 = v2[v2.seed == 0]                      # reference lines use seed 0 only, like the level curves
        ref = v0.groupby(["subject", "method"])["macro_f1"].mean().unstack("method")
        clean, raw = ref["ideal"], ref["raw_features"]
        d = v2[v2.method.isin(METHODS)].copy()
        d["noise_type"], d["noise_prob"] = "depolarizing", 0.02
        parts.append(d[COLS])
    dep = read("sweep_results.csv")
    if dep is not None:
        keep = (dep.noise_type == "depolarizing") & dep.method.isin(METHODS)
        # p = 0.02 baselines come from v2_results.csv; from sweep_results.csv keep only the phase code rows there
        keep &= ~np.isclose(dep.noise_prob, 0.02) | dep.method.isin(["phase03", "phase_uniform"])
        parts.append(dep[keep][COLS])
    pau = read("sweep_pauli.csv")
    if pau is not None:
        keep = pau.noise_type.isin(["bitflip", "phaseflip"]) & pau.method.isin(METHODS)
        parts.append(pau[keep][COLS])
    data = pd.concat(parts, ignore_index=True)
    data["noise_prob"] = data["noise_prob"].astype(float).round(4)
    data = data.drop_duplicates(subset=["subject", "seed", "noise_type", "noise_prob", "method"], keep="last")
    return data, clean, raw


def table(data, nt, p, metric):
    """One row per (subject, seed), one column per method."""
    d = data[(data.noise_type == nt) & np.isclose(data.noise_prob, p)]
    return d.groupby(["subject", "seed", "method"])[metric].mean().unstack("method")


def n_seeds(data, nt, p):
    d = data[(data.noise_type == nt) & np.isclose(data.noise_prob, p) & (data.method == "no_qec")]
    return int(d.seed.nunique())


def matched_pairs(data, nt, p, m, metric):
    t = table(data, nt, p, metric)
    if "no_qec" not in t.columns or m not in t.columns:
        return None
    both = t[["no_qec", m]].dropna()
    return both["no_qec"].values, both[m].values, int(both.index.get_level_values("seed").nunique())


def available(data):
    return [(nt, p) for nt, p in ALL_SETTINGS
            if ((data.noise_type == nt) & np.isclose(data.noise_prob, p)).any()]


def paired(data, nt, p, m, metric, min_n=5):
    """(no_protection_values, method_values) per subject, averaged over matched seeds only."""
    t = table(data, nt, p, metric)
    if "no_qec" not in t.columns or m not in t.columns:
        return None
    both = t[["no_qec", m]].dropna()
    per_subject = both.groupby(level="subject").mean()
    if len(per_subject) < min_n:
        return None
    return per_subject["no_qec"].values, per_subject[m].values


def wilcoxon_p(a, b=None):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            return stats.wilcoxon(a).pvalue if b is None else stats.wilcoxon(a, b).pvalue
        except ValueError:
            return np.nan


def boot_ci(x, n=5000, seed=1):
    x = np.asarray(x, dtype=float)
    idx = np.random.default_rng(seed).integers(0, len(x), (n, len(x)))
    return np.percentile(x[idx].mean(axis=1), [2.5, 97.5])


def fmt_ratio(r):
    if r < 0.01:
        return f"{r:.3f}x"
    return f"{r:.2f}x" if r < 10 else f"{r:.0f}x"


def stars(p):
    if not np.isfinite(p):
        return ""
    return "**" if p < 0.01 else ("*" if p < 0.05 else "")


def holm_adjust(pvals):
    p = np.nan_to_num(np.asarray(pvals, dtype=float), nan=1.0)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(p) - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj


def adjusted_pvalues(data):
    """Holm adjusted p values over the post hoc PRIMARY family only.
    Returns {(noise_type, p, method): (adjusted_p_error, adjusted_p_f1)}; other comparisons are absent."""
    keys, pe, pf = [], [], []
    for nt, p in available(data):
        if (nt, p) not in PRIMARY_SETTINGS:
            continue
        for m in COMPARED:
            if m not in PRIMARY_METHODS:
                continue
            e, f = paired(data, nt, p, m, "mse_vs_ideal"), paired(data, nt, p, m, "macro_f1")
            if e is None or f is None:
                continue
            keys.append((nt, p, m))
            pe.append(wilcoxon_p(*e))
            pf.append(wilcoxon_p(f[1] - f[0]))
    he, hf = holm_adjust(pe), holm_adjust(pf)
    print(f"  Holm family: {len(keys)} primary comparisons")
    return {k: (a, b) for k, a, b in zip(keys, he, hf)}


def log_ratio(a, b):
    return np.log(np.maximum(b, 1e-12) / np.maximum(a, 1e-12))


def save(fig, name):
    for ext in ("png", "pdf", "svg"):
        fig.savefig(FIG / f"{name}.{ext}")
    plt.close(fig)
    print(f"  saved {name}")


def panel(ax, letter, dx=-0.02, dy=1.04):
    ax.text(dx, dy, letter, transform=ax.transAxes, fontsize=11, fontweight="bold", ha="right", va="bottom")


# ----------------------------------------------------------------------------- fig 0
def _layout(ax, x0, y0, tiers, color):
    r, step, pad, gap = 0.75, 1.9, 0.6, 1.4
    x = x0
    for tier in tiers:
        rows = 3 if tier in ("rep", "shor") else 1
        cols = 3 if tier == "shor" else 1
        w = (cols - 1) * step + 2 * r + 2 * pad
        h = (rows - 1) * step + 2 * r + 2 * pad
        ax.add_patch(FancyBboxPatch((x, y0), w, h, boxstyle="round,pad=0,rounding_size=0.5",
                                    fc=color, ec=color, lw=0.8, alpha=0.10))
        ax.add_patch(FancyBboxPatch((x, y0), w, h, boxstyle="round,pad=0,rounding_size=0.5",
                                    fc="none", ec=color, lw=0.8))
        for i in range(rows):
            for j in range(cols):
                ax.add_patch(Circle((x + pad + r + j * step, y0 + pad + r + i * step), r,
                                    fc=color, ec="white", lw=0.3))
        x += w + gap
    return x - gap


def fig_schematic():
    fig = plt.figure(figsize=(7.2, 3.8))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 52.8)
    ax.axis("off")
    boxes = ["EEG night\n10 s windows\n10 subjects", "8 spectral\nfeatures\n(z scored)",
             "4 qubit\nquantum\ngenerator", "Pauli noise\n+ protection\ntier", "Frozen\nclassifier\n(5 stages)",
             "Outputs\nMSE vs clean\nmacro F1"]
    w, gap, x, y, h = 13.2, 3.7, 1.15, 38.5, 10.5
    for i, text in enumerate(boxes):
        accent = i == 3
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=1.2",
                                    fc="#FDE9DC" if accent else "#F3F4F6",
                                    ec="#C2410C" if accent else "#6B7280", lw=1.2 if accent else 0.9))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=7.6, linespacing=1.25)
        if i < len(boxes) - 1:
            ax.annotate("", xy=(x + w + gap - 0.3, y + h / 2), xytext=(x + w + 0.3, y + h / 2),
                        arrowprops=dict(arrowstyle="-|>", color=INK, lw=0.9, mutation_scale=9))
        x += w + gap
    ax.plot([1, 99], [34.0, 34.0], color="#D1D5DB", lw=0.8)
    ax.text(50, 30.2, "Protection tiers on the four logical qubits (each circle is one physical qubit)",
            ha="center", va="center", fontsize=8.6, fontweight="bold")
    layouts = [("no_qec", ["bare"] * 4, "No protection"),
               ("rep03", ["rep", "bare", "bare", "rep"], "Bit flip code on qubits 0 and 3"),
               ("uniform", ["rep"] * 4, "Bit flip code on all qubits"),
               ("rule", ["rep", "rep", "shor", "bare"], "Error aware rule (example)")]
    for (m, tiers, name), x0 in zip(layouts, [2.0, 26.0, 50.0, 74.0]):
        _layout(ax, x0, 15.5, tiers, METHOD_COLOR[m])
        ax.text(x0 + 8.5, 9.0, f"{name}\n{QUBITS[m]} physical qubits", ha="center", va="center",
                fontsize=7.6, color=METHOD_COLOR[m], fontweight="bold", linespacing=1.3)
    ax.text(50, 2.2, "1 circle = bare qubit,  3 circles = repetition code (the phase flip code has the same layout, in the Hadamard basis),  9 circles = Shor code",
            ha="center", va="center", fontsize=7.2, color="#4B5563", style="italic")
    save(fig, "fig0_study_design")


# ----------------------------------------------------------------------------- fig 1
def fig_regime_map(data, adj):
    methods = COMPARED
    cols = available(data)
    L = np.full((len(methods), len(cols)), np.nan)
    txt = [[None] * len(cols) for _ in methods]
    for j, (nt, p) in enumerate(cols):
        for i, m in enumerate(methods):
            ab = paired(data, nt, p, m, "mse_vs_ideal")
            if ab is None:
                continue
            a, b = ab
            lr = np.mean(np.log2(np.maximum(b, 1e-12) / np.maximum(a, 1e-12)))
            L[i, j] = lr
            txt[i][j] = (f"{fmt_ratio(2 ** lr)}\n{int((b < a).sum())}/{len(a)}"
                          f"{stars(adj.get((nt, p, m), (np.nan, np.nan))[0])}")
    fig, ax = plt.subplots(figsize=(9.0, 1.0 + 0.62 * len(methods)))
    cmap = plt.get_cmap("RdBu_r").copy()
    cmap.set_bad("#F3F4F6")
    im = ax.imshow(np.ma.masked_invalid(L), cmap=cmap, vmin=-4, vmax=4, aspect="auto")
    for i in range(len(methods)):
        for j in range(len(cols)):
            if txt[i][j] is None:
                ax.text(j, i, "not run", ha="center", va="center", fontsize=7, color="#9CA3AF", style="italic")
            else:
                ax.text(j, i, txt[i][j], ha="center", va="center", fontsize=7.8, linespacing=1.3,
                        color="white" if abs(L[i, j]) > 2.4 else INK, fontweight="bold")
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels([f"{p:g}\n({n_seeds(data, nt, p)} seed{'s' if n_seeds(data, nt, p) > 1 else ''})" for nt, p in cols], fontsize=7.4)
    ax.set_yticks(range(len(methods)))
    ax.set_yticklabels([f"{SHORT[m]}\n({QUBITS[m]} qubits)" for m in methods], fontsize=7.4)
    ax.set_xticks(np.arange(-0.5, len(cols), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(methods), 1), minor=True)
    ax.grid(which="minor", color="white", lw=2.2)
    ax.tick_params(which="both", length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xlim(-0.5, len(cols) - 0.5)
    ax.set_ylim(len(methods) - 0.5, -1.25)
    for nt in dict.fromkeys(c[0] for c in cols):
        idx = [j for j, c in enumerate(cols) if c[0] == nt]
        lo, hi = idx[0] - 0.42, idx[-1] + 0.42
        ax.plot([lo, hi], [-0.62, -0.62], color=NOISE_COLOR[nt], lw=2.4, clip_on=False, solid_capstyle="round")
        ax.text((lo + hi) / 2, -0.74, NOISE_TITLE[nt], ha="center", va="bottom", fontsize=9,
                fontweight="bold", color=NOISE_COLOR[nt])
    ax.set_xlabel("Noise probability p")
    cb = fig.colorbar(im, ax=ax, fraction=0.022, pad=0.015)
    cb.set_ticks([-4, -2, 0, 2, 4])
    cb.set_ticklabels(["1/16", "1/4", "1", "4", "16"])
    cb.set_label("Error vs no protection\n(blue lower, red higher)", fontsize=8)
    cb.outline.set_visible(False)
    fig.text(0.5, -0.09, "Cell text: geometric mean error ratio, then subjects with lower error out of 10. "
             "* Holm adjusted p < 0.05, only for the primary family, which was chosen after an initial look at the data\n"
             "(the four fixed codes at p = 0.05 under bit flip and phase flip noise); all other cells are exploratory.\n"
             "Colors saturate beyond 1/16 and 16. Per subject values are means over matched seeds (seed counts under each column).",
             ha="center", fontsize=7.0, color="#4B5563", style="italic")
    save(fig, "fig1_regime_map")


# ----------------------------------------------------------------------------- fig 2
def fig_paired_subjects(data, p=0.05):
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 3.6), sharey=True)
    for k, (ax, nt) in enumerate(zip(axes, ["bitflip", "phaseflip", "depolarizing"])):
        t = table(data, nt, p, "mse_vs_ideal")
        order = [m for m in ["no_qec", "rep03", "uniform", "phase03", "phase_uniform"] if m in t.columns]
        if len(order) < 2:
            ax.set_visible(False)
            continue
        t = t[order].dropna().groupby(level="subject").mean().clip(lower=1e-9)
        n = len(order)
        for _, row in t.iterrows():
            ax.plot(range(n), row.values, color=SOFT, lw=0.7, alpha=0.75, zorder=1)
            ax.scatter(range(n), row.values, s=8, color=SOFT, zorder=2)
        gm = np.exp(np.log(t.values).mean(axis=0))
        ax.plot(range(n), gm, color=NOISE_COLOR[nt], lw=2.4, zorder=3)
        ax.scatter(range(n), gm, s=46, color=[METHOD_COLOR[m] for m in order], edgecolor="white", lw=1.0, zorder=4)
        best = order[1 + int(np.argmin(gm[1:]))]
        a, b = t["no_qec"].values, t[best].values
        lower, higher = int((b < a).sum()), int((b > a).sum())
        verdict = (f"lower error in {lower}/{len(a)}" if lower >= higher else f"higher error in {higher}/{len(a)}")
        ax.set_xlabel(f"{'best code' if lower >= higher else 'least harmful code'}: {SHORT[best].lower()}\n{verdict}, p = {wilcoxon_p(a, b):.3f}",
                      fontsize=6.9, color=NOISE_COLOR[nt], labelpad=14)
        ax.set_yscale("log")
        ax.set_xticks(range(n))
        ax.set_xticklabels([TICK[m] for m in order], fontsize=7)
        ax.set_xlim(-0.4, n - 0.6)
        ax.set_title(NOISE_TITLE[nt], color=NOISE_COLOR[nt])
        ax.grid(axis="y", color="#E5E7EB", lw=0.6)
        ax.set_axisbelow(True)
        panel(ax, "abc"[k], dx=0.0)
    axes[0].set_ylabel("Error vs clean output (MSE, log scale)")
    fig.tight_layout()
    save(fig, "fig2_paired_subjects")


# ----------------------------------------------------------------------------- fig 3
def fig_staging_forest(data, adj):
    rows = []
    for nt, p in available(data):
        for m in COMPARED:
            ab = paired(data, nt, p, m, "macro_f1")
            if ab is None:
                continue
            diff = ab[1] - ab[0]
            lo, hi = boot_ci(diff)
            rows.append({"nt": nt, "p": p, "m": m, "mean": diff.mean(), "lo": lo, "hi": hi,
                         "sig": adj.get((nt, p, m), (np.nan, np.nan))[1] < 0.05})
    if not rows:
        return
    df = pd.DataFrame(rows)
    groups = list(dict.fromkeys(zip(df.nt, df.p)))
    fig, ax = plt.subplots(figsize=(7.2, max(4.5, 0.21 * len(df) + 1.9)))
    y, ticks, labels, centers = 0.0, [], [], []
    for gi, (nt, p) in enumerate(groups):
        sub = df[(df.nt == nt) & np.isclose(df.p, p)]
        top = y
        for _, r in sub.iterrows():
            c = METHOD_COLOR[r.m]
            ax.plot([r.lo, r.hi], [y, y], color=c, lw=1.8, solid_capstyle="round", zorder=2)
            ax.scatter(r["mean"], y, s=46, marker="o", facecolor=c if r.sig else "white",
                       edgecolor=c, lw=1.6, zorder=3)
            ticks.append(y)
            labels.append(FOREST[r.m])
            y += 1.0
        if gi % 2 == 0:
            ax.axhspan(top - 0.55, y - 0.45, color="#F3F4F6", zorder=0)
        centers.append(((top + y - 1.0) / 2, f"{NOISE_TITLE[nt]}\np = {p:g}", NOISE_COLOR[nt]))
        y += 0.8
    ax.axvline(0, color=INK, lw=0.9)
    ax.set_yticks(ticks)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    for yc, text, col in centers:
        ax.text(-0.34, yc, text, transform=ax.get_yaxis_transform(), ha="right", va="center",
                fontsize=8.2, fontweight="bold", color=col)
    ax.set_xlabel("Change in sleep staging macro F1 versus no protection\n"
                  "mean and 95% subject bootstrap interval; descriptive only\n"
                  "filled marker: Holm adjusted p < 0.05 (post hoc primary family only)", linespacing=1.4)
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    ax.grid(axis="x", color="#E5E7EB", lw=0.6)
    ax.set_axisbelow(True)
    fig.subplots_adjust(left=0.40)
    save(fig, "fig3_staging_forest")


# ----------------------------------------------------------------------------- fig 4
def fig_mechanism():
    diag, pau = read("sweep_diag.csv"), read("sweep_pauli.csv")
    if diag is None or pau is None:
        return
    subs = sorted(diag.subject.unique())
    dep = diag[(diag.noise_type == "depolarizing") & np.isclose(diag.noise_prob, 0.05)]
    pf = pau[pau.noise_type.isin(["bitflip", "phaseflip"]) & np.isclose(pau.noise_prob, 0.05)]
    d = pd.concat([dep, pf])
    d = d[d.subject.isin(subs)]
    order = [m for m in ["no_qec", "rep03", "uniform", "phase03", "phase_uniform"] if m in set(d.method)]
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 3.3), sharey=True)
    for k, (ax, nt) in enumerate(zip(axes, ["bitflip", "phaseflip", "depolarizing"])):
        g = d[d.noise_type == nt].groupby("method")[["mse_z", "mse_x"]].mean().reindex(order)
        xs = np.arange(len(order))
        for off, col, key, lab in [(-0.19, Z_COLOR, "mse_z", r"$\langle Z\rangle$ outputs (hit by bit flips)"),
                                   (0.19, X_COLOR, "mse_x", r"$\langle X\rangle$ outputs (hit by phase flips)")]:
            vals = g[key].values
            have = np.isfinite(vals)
            ok = have & (vals > 1e-8)
            ax.bar(xs[ok] + off, vals[ok], 0.36, color=col, label=lab if k == 0 else None, zorder=3)
            for xx in xs[have & ~ok]:
                ax.text(xx + off, 1.6e-7, "0", ha="center", fontsize=6.5, color=col)
        for xx in xs[~np.isfinite(g["mse_z"].values)]:
            ax.text(xx, 2.5e-7, "not run", rotation=90, fontsize=6.3, color="#9CA3AF", va="bottom", ha="center")
        ax.set_yscale("log")
        ax.set_ylim(1e-7, 1.0)
        ax.set_xticks(xs)
        ax.set_xticklabels([TICK[m] for m in order], fontsize=7)
        ax.set_title(NOISE_TITLE[nt], color=NOISE_COLOR[nt])
        ax.grid(axis="y", color="#E5E7EB", lw=0.6)
        ax.set_axisbelow(True)
        panel(ax, "abc"[k], dx=0.0)
    axes[0].set_ylabel("Mean squared error of the output group")
    axes[0].legend(frameon=False, loc="upper left", fontsize=6.6)
    fig.text(0.5, -0.05, f"Means over the {len(subs)} subjects run for all three noise types, p = 0.05. "
             "Bit and phase flip panels average all available seeds; the depolarizing panel is seed 0 only.\n"
             "A bare 0 means no error in that output group.", ha="center", fontsize=7.0, color="#4B5563",
             style="italic")
    fig.tight_layout()
    save(fig, "fig4_mechanism_zx")


# ----------------------------------------------------------------------------- fig 5
def fig_relative_sweep(data):
    """Level curves use seed 0 only so every point has the same seed base."""
    fig, axes = plt.subplots(1, 3, figsize=(9.6, 3.3), sharey=True)
    for k, (ax, nt) in enumerate(zip(axes, ["bitflip", "phaseflip", "depolarizing"])):
        ps = sorted(data[data.noise_type == nt].noise_prob.unique())
        ax.axhspan(1e-4, 1, color="#DBEAFE", alpha=0.55, zorder=0)
        ax.axhspan(1, 1e4, color="#FEE2E2", alpha=0.45, zorder=0)
        ax.axhline(1, color=INK, lw=0.9, zorder=1)
        for m in COMPARED:
            xs, mid, lo, hi = [], [], [], []
            for p in ps:
                ab = paired(data, nt, p, m, "mse_vs_ideal")
                if ab is None:
                    continue
                lr = log_ratio(*ab)
                ci = boot_ci(lr)
                xs.append(p)
                mid.append(np.exp(lr.mean()))
                lo.append(np.exp(ci[0]))
                hi.append(np.exp(ci[1]))
            if not xs:
                continue
            c = METHOD_COLOR[m]
            if len(xs) > 1:
                ax.fill_between(xs, lo, hi, color=c, alpha=0.20, lw=0, zorder=2)
                ax.plot(xs, mid, color=c, lw=2.0, marker="o", ms=5, mec="white", mew=0.8, zorder=3, label=LABEL[m])
            else:
                mid_a, lo_a, hi_a = np.array(mid), np.array(lo), np.array(hi)
                ax.errorbar(xs, mid_a, yerr=[mid_a - lo_a, hi_a - mid_a], fmt="o", color=c, ms=5.5, mec="white",
                            mew=0.8, elinewidth=1.6, capsize=2.5, zorder=3, label=LABEL[m])
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_ylim(0.001, 300)
        ax.set_xticks(ps)
        ax.set_xticklabels([f"{p:g}" for p in ps])
        ax.minorticks_off()
        ax.set_xlabel("Noise probability p")
        ax.set_title(NOISE_TITLE[nt], color=NOISE_COLOR[nt])
        ax.text(0.03, 0.05, "protection helps", transform=ax.transAxes, fontsize=7.5, color="#1E40AF", style="italic")
        ax.text(0.03, 0.93, "protection hurts", transform=ax.transAxes, fontsize=7.5, color="#B91C1C",
                style="italic", va="top")
        panel(ax, "abc"[k], dx=0.0)
    axes[0].set_ylabel("Error relative to no protection")
    handles, labels = [], []
    for ax in axes:
        for h, l in zip(*ax.get_legend_handles_labels()):
            if l not in labels:
                handles.append(h)
                labels.append(l)
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.17), fontsize=7.4)
    fig.text(0.5, -0.24, "Seed 0 only at every point. Bands: 95% subject bootstrap interval of the geometric mean ratio.",
             ha="center", fontsize=7.0, color="#4B5563", style="italic")
    fig.tight_layout()
    save(fig, "fig5_relative_sweep")


# ----------------------------------------------------------------------------- fig 6
def fig_cost_benefit(data):
    settings = [("bitflip", 0.05), ("phaseflip", 0.05)]
    marker = {"bitflip": "o", "phaseflip": "s"}
    dx = {"rep03": -0.3, "uniform": -0.3, "phase03": 0.3, "phase_uniform": 0.3}
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.3))
    for nt, p in settings:
        for m in ["no_qec"] + COMPARED:
            if m == "no_qec":
                ratio, gain = 1.0, 0.0
            else:
                e, f = paired(data, nt, p, m, "mse_vs_ideal"), paired(data, nt, p, m, "macro_f1")
                if e is None or f is None:
                    continue
                ratio, gain = float(np.exp(log_ratio(*e).mean())), float((f[1] - f[0]).mean())
            for ax, y in zip(axes, (ratio, gain)):
                ax.scatter(QUBITS[m] + dx.get(m, 0), y, s=62, marker=marker[nt], color=METHOD_COLOR[m],
                           edgecolor=NOISE_COLOR[nt], lw=1.8, zorder=3)
    axes[0].axhline(1, color=INK, lw=0.8, ls="--")
    axes[1].axhline(0, color=INK, lw=0.8, ls="--")
    axes[0].set_yscale("log")
    axes[0].set_ylabel("Error relative to no protection")
    axes[1].set_ylabel("Change in macro F1 versus no protection")
    for k, ax in enumerate(axes):
        ax.set_xticks([4, 8, 12, 16])
        ax.set_xlabel("Physical qubits spent")
        ax.grid(color="#E5E7EB", lw=0.6)
        ax.set_axisbelow(True)
        panel(ax, "ab"[k], dx=0.0)
    from matplotlib.lines import Line2D
    handles = [Line2D([0], [0], marker="o", ls="", mfc=METHOD_COLOR[m], mec="white", ms=7, label=SHORT[m])
               for m in ["no_qec"] + COMPARED]
    handles += [Line2D([0], [0], marker=marker[nt], ls="", mfc="white", mec=NOISE_COLOR[nt], mew=1.8, ms=7,
                       label=f"{NOISE_TITLE[nt]}, p = {p:g}") for nt, p in settings]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.2), fontsize=7)
    fig.tight_layout()
    save(fig, "fig6_cost_benefit")


# ----------------------------------------------------------------------------- supplementary
def _level_panel(ax, data, nt, metric):
    d = data[data.noise_type == nt]
    for m in METHODS:
        g = d[d.method == m].groupby(["subject", "noise_prob"])[metric].mean().reset_index()
        if g.empty:
            continue
        s = g.groupby("noise_prob")[metric].agg(["mean", "std", "count"])
        se = (s["std"] / np.sqrt(s["count"])).fillna(0)
        c = METHOD_COLOR[m]
        if len(s) > 1:
            ax.errorbar(s.index, s["mean"], yerr=se, color=c, marker="o", ms=4.5, lw=1.6, mec="white", mew=0.7,
                        elinewidth=1.0, capsize=2, label=LABEL[m])
        else:
            ax.errorbar(s.index, s["mean"], yerr=se, color=c, marker="o", ms=5, mec="white", mew=0.7,
                        elinewidth=1.4, capsize=2.5, ls="", label=LABEL[m])
    ps = sorted(d.noise_prob.unique())
    ax.set_xscale("log")
    ax.set_xticks(ps)
    ax.set_xticklabels([f"{p:g}" for p in ps])
    ax.minorticks_off()
    ax.set_xlabel("Noise probability p")
    ax.set_title(NOISE_TITLE[nt], color=NOISE_COLOR[nt])
    ax.grid(color="#E5E7EB", lw=0.6)
    ax.set_axisbelow(True)


def fig_supplementary(data0, clean, raw):
    """data0 is seed 0 only."""
    fig, axes = plt.subplots(1, 3, figsize=(9.6, 3.4))
    for ax, nt in zip(axes, ["bitflip", "phaseflip", "depolarizing"]):
        _level_panel(ax, data0, nt, "mse_vs_ideal")
        ax.set_yscale("log")
    axes[0].set_ylabel("Mean squared error vs clean output")
    handles, labels = [], []
    for ax in axes:
        for h, l in zip(*ax.get_legend_handles_labels()):
            if l not in labels:
                handles.append(h)
                labels.append(l)
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.24), fontsize=7.4)
    fig.text(0.5, -0.33, "Seed 0 only. Error bars: standard error over subjects.", ha="center", fontsize=7.0,
             color="#4B5563", style="italic")
    fig.tight_layout()
    save(fig, "figS1_absolute_error")

    fig, axes = plt.subplots(1, 3, figsize=(9.6, 3.4), sharey=True)
    for ax, nt in zip(axes, ["bitflip", "phaseflip", "depolarizing"]):
        _level_panel(ax, data0, nt, "macro_f1")
        if clean is not None:
            ax.axhline(clean.mean(), color=INK, ls="--", lw=1.0, label="Clean generator output")
        if raw is not None:
            ax.axhline(raw.mean(), color=SOFT, ls=":", lw=1.3, label="Raw features, no circuit")
    axes[0].set_ylabel("Sleep staging macro F1")
    handles, labels = [], []
    for ax in axes:
        for h, l in zip(*ax.get_legend_handles_labels()):
            if l not in labels:
                handles.append(h)
                labels.append(l)
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.26), fontsize=7.2)
    fig.text(0.5, -0.36, "Seed 0 only. Error bars: standard error over subjects. Descriptive.", ha="center",
             fontsize=7.0, color="#4B5563", style="italic")
    fig.tight_layout()
    save(fig, "figS2_staging_levels")


# ----------------------------------------------------------------------------- tables
def _summary(data):
    rows = []
    for (nt, p, m), g in data.groupby(["noise_type", "noise_prob", "method"]):
        s = g.groupby("subject")[["mse_vs_ideal", "wasserstein_vs_ideal", "macro_f1"]].mean()
        n = len(s)
        rows.append({"noise_type": nt, "noise_prob": p, "method": m, "n_subjects": n, "n_seeds": int(g["seed"].nunique()),
                     "mse": s.mse_vs_ideal.mean(), "mse_se": s.mse_vs_ideal.std() / np.sqrt(n),
                     "wasserstein": s.wasserstein_vs_ideal.mean(),
                     "macro_f1": s.macro_f1.mean(), "macro_f1_se": s.macro_f1.std() / np.sqrt(n)})
    return pd.DataFrame(rows)


def summary_and_tests(data, data0, adj):
    summ = _summary(data)
    summ.to_csv(OUT / "final_summary.csv", index=False)
    _summary(data0).to_csv(OUT / "final_summary_seed0.csv", index=False)

    out = []
    for nt, p in available(data):
        for m in COMPARED:
            e, f = paired(data, nt, p, m, "mse_vs_ideal"), paired(data, nt, p, m, "macro_f1")
            if e is None or f is None:
                continue
            a, b = e
            diff = f[1] - f[0]
            sp = matched_pairs(data, nt, p, m, "mse_vs_ideal")
            out.append({"noise_type": nt, "noise_prob": p, "method": m, "n_subjects": len(a),
                        "n_seeds_matched": sp[2],
                        "primary_family": (nt, p) in PRIMARY_SETTINGS and m in PRIMARY_METHODS,
                        "lower_error_in_subject_seed_pairs": f"{int((sp[1] < sp[0]).sum())}/{len(sp[0])}",
                        "error_ratio_vs_no_protection": float(np.exp(log_ratio(a, b).mean())),
                        "lower_error_in": f"{int((b < a).sum())}/{len(a)}", "wilcoxon_p_error": wilcoxon_p(a, b),
                        "f1_diff_vs_no_protection": float(diff.mean()),
                        "higher_f1_in": f"{int((diff > 0).sum())}/{len(diff)}", "wilcoxon_p_f1": wilcoxon_p(diff),
                        "holm_p_error_primary": adj.get((nt, p, m), (np.nan, np.nan))[0],
                        "holm_p_f1_primary": adj.get((nt, p, m), (np.nan, np.nan))[1]})
    tests = pd.DataFrame(out)
    tests.to_csv(OUT / "final_tables.csv", index=False)
    print("\nPaired comparison against no protection (ratio below 1 means lower error):")
    print(tests.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("\nSummary (all seeds):")
    print(summ.to_string(index=False, float_format=lambda x: f"{x:.6f}"))


def main():
    global COMPARED
    print("loading results ...")
    data, clean, raw = load_all()
    COMPARED = [m for m in COMPARED if m in set(data.method)]
    data0 = data[data.seed == 0]
    print(f"  {len(data)} rows ({len(data0)} seed 0), noise settings: {available(data)}")
    adj = adjusted_pvalues(data)
    print("making figures ...")
    fig_schematic()
    fig_regime_map(data, adj)
    fig_paired_subjects(data)
    fig_staging_forest(data, adj)
    fig_mechanism()
    fig_relative_sweep(data0)
    fig_cost_benefit(data)
    fig_supplementary(data0, clean, raw)
    summary_and_tests(data, data0, adj)
    print(f"\nDone. Figures are in {FIG}")


if __name__ == "__main__":
    main()