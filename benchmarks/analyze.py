"""Turn the raw JSON-lines results into the tables and figures used in the paper.

    python benchmarks/analyze.py            # writes paper/figures/*.svg and paper/tables.json
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
FIG = ROOT / "paper" / "figures"

# Reference categorical palette (fixed order; validated for CVD separation).
COLORS = {"EML": "#2a78d6", "MLP": "#eb6834", "GBM": "#1baf7a", "GPLearn": "#eda100",
          "RF": "#e87ba4", "Linear": "#8a8984"}
TEXT, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
METHOD_ORDER = ["EML", "GPLearn", "MLP", "GBM", "RF", "Linear"]


def _style():
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": MUTED,
        "axes.labelcolor": TEXT, "xtick.color": MUTED, "ytick.color": MUTED,
        "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
        "grid.color": GRID, "grid.linewidth": 0.6, "legend.frameon": False,
        "svg.fonttype": "none",
    })


def load(name: str) -> pd.DataFrame:
    path = RES / name
    if not path.exists():
        return pd.DataFrame()
    return pd.DataFrame([json.loads(l) for l in path.read_text().splitlines() if l.strip()])


def fmt(x, nd=3):
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return "–"
    return f"{x:.{nd}f}"


# --------------------------------------------------------------------- Feynman
def feynman_tables(df: pd.DataFrame) -> dict:
    out = {}
    df = df[~df.get("error", pd.Series([None] * len(df))).notna()].copy() if "error" in df else df.copy()
    df["nmse_ood_c"] = df["nmse_ood"].clip(upper=1e6)
    for noise, g in df.groupby("noise"):
        rows = []
        for m in [m for m in METHOD_ORDER if m in set(g.method)]:
            h = g[g.method == m]
            sym = h["symbolic"].mean() if "symbolic" in h and h["symbolic"].notna().any() else None
            rows.append(dict(
                method=m, n=len(h),
                symbolic=sym,
                id_r2_med=float(h.r2_id.median()),
                id_r2_999=float((h.r2_id > 0.999).mean()),
                ood_r2_med=float(h.r2_ood.median()),
                ood_r2_99=float((h.r2_ood > 0.99).mean()),
                ood_exact=float((h.nmse_ood < 1e-9).mean()),
                time_med=float(h.time.median()),
                size_med=float(h["size"].median()) if "size" in h and h["size"].notna().any() else None,
            ))
        out[f"noise_{noise}"] = rows
    # breakdown for EML: trig vs no trig, dev vs held-out
    e = df[(df.method == "EML") & (df.noise == 0.0)]
    if len(e):
        out["eml_breakdown"] = {
            k: dict(n=int(len(s)), symbolic=float(s.symbolic.mean()),
                    ood_r2_99=float((s.r2_ood > 0.99).mean()))
            for k, s in {"no trig": e[~e.has_trig], "trig": e[e.has_trig],
                         "dev": e[e.dev], "held-out": e[~e.dev],
                         "held-out, no trig": e[~e.dev & ~e.has_trig]}.items() if len(s)
        }
    return out


def feynman_profile_fig(df: pd.DataFrame, noise: float = 0.0):
    """Fraction of problems solved to within NMSE <= tau, in- and out-of-distribution."""
    _style()
    g = df[df.noise == noise]
    taus = np.logspace(-14, 0, 200)
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9), sharey=True)
    for ax, col, title in [(axes[0], "nmse_id", "In distribution"),
                           (axes[1], "nmse_ood", "Out of distribution (extrapolation)")]:
        for m in [m for m in ["EML", "GPLearn", "MLP", "GBM", "Linear"] if m in set(g.method)]:
            v = g[g.method == m][col].fillna(np.inf).to_numpy()
            frac = [(v <= t).mean() for t in taus]
            ax.plot(taus, frac, color=COLORS[m], lw=2, label=m)
            ax.annotate(m, (taus[-1], frac[-1]), xytext=(3, 0), textcoords="offset points",
                        color=TEXT, fontsize=8, va="center")
        ax.set_xscale("log")
        ax.set_xlim(1e-14, 1)
        ax.set_ylim(0, 1.02)
        ax.set_xlabel("NMSE threshold τ")
        ax.set_title(title, fontsize=9.5, color=TEXT, loc="left")
    axes[0].set_ylabel("fraction of problems with NMSE ≤ τ")
    axes[1].legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / f"feynman_profile_noise{noise}.svg")
    fig.savefig(FIG / f"feynman_profile_noise{noise}.png", dpi=200)
    plt.close(fig)


# ------------------------------------------------------------------ real world
def realworld_tables(df: pd.DataFrame) -> dict:
    if df.empty:
        return {}
    df = df[df.get("error").isna()] if "error" in df else df
    piv = df.pivot_table(index="dataset", columns="method", values="r2", aggfunc="first")
    methods = [m for m in METHOD_ORDER if m in piv.columns]
    piv = piv[methods]
    ranks = piv.rank(axis=1, ascending=False)
    summary = []
    for m in methods:
        summary.append(dict(method=m, median_r2=float(piv[m].median()),
                            mean_rank=float(ranks[m].mean()),
                            wins_vs_mlp=(int((piv[m] >= piv["MLP"]).sum()) if "MLP" in piv and m != "MLP" else None)))
    rows = []
    for d, r in piv.iterrows():
        e = df[(df.dataset == d) & (df.method == "EML")]
        g = df[(df.dataset == d) & (df.method == "GPLearn")]
        rows.append(dict(dataset=d, n=int(df[df.dataset == d].n.iloc[0]), d=int(df[df.dataset == d].d.iloc[0]),
                         **{m: (None if pd.isna(r[m]) else float(r[m])) for m in methods},
                         eml_size=(float(e.size_median.iloc[0]) if len(e) and "size_median" in e else None),
                         gp_size=(float(g.size_median.iloc[0]) if len(g) and "size_median" in g else None),
                         eml_formula=(e.formula_full.iloc[0] if len(e) and "formula_full" in e and isinstance(e.formula_full.iloc[0], str) else
                                      (e.fold_formulas.iloc[0][0] if len(e) and "fold_formulas" in e and isinstance(e.fold_formulas.iloc[0], list) else None))))
    return dict(summary=summary, rows=rows)


def tabular_scatter_fig(df: pd.DataFrame, other: str = "MLP"):
    _style()
    piv = df.pivot_table(index="dataset", columns="method", values="r2", aggfunc="first")
    if "EML" not in piv or other not in piv:
        return
    x = piv[other].clip(lower=-0.2)
    y = piv["EML"].clip(lower=-0.2)
    fig, ax = plt.subplots(figsize=(3.4, 3.2))
    ax.plot([-0.2, 1], [-0.2, 1], color=MUTED, lw=1, ls="--")
    ax.scatter(x, y, s=28, color=COLORS["EML"], edgecolor="white", linewidth=1.2, zorder=3)
    ax.set_xlim(-0.2, 1.02); ax.set_ylim(-0.2, 1.02)
    ax.set_xlabel(f"{other} cross-validated R²")
    ax.set_ylabel("EML (closed form) cross-validated R²")
    above = int((y >= x).sum())
    ax.text(0.02, 0.97, f"EML ≥ {other} on {above}/{len(x)} data sets", transform=ax.transAxes,
            color=TEXT, fontsize=8, va="top")
    fig.tight_layout()
    fig.savefig(FIG / f"tabular_eml_vs_{other.lower()}.svg")
    fig.savefig(FIG / f"tabular_eml_vs_{other.lower()}.png", dpi=200)
    plt.close(fig)


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    tables = {}
    fe = load("feynman.jsonl")
    if not fe.empty:
        tables["feynman"] = feynman_tables(fe)
        for nz in sorted(fe.noise.unique()):
            feynman_profile_fig(fe, nz)
    ab = load("ablation.jsonl")
    if not ab.empty:
        rows = []
        for m, h in ab.groupby("method"):
            rows.append(dict(method=m, n=len(h), symbolic=float(h.symbolic.mean()),
                             ood_r2_99=float((h.r2_ood > 0.99).mean()),
                             id_r2_999=float((h.r2_id > 0.999).mean()), time_med=float(h.time.median())))
        tables["ablation"] = rows
    for grp in ("physics", "tabular"):
        rw = load(f"realworld_{grp}.jsonl")
        if not rw.empty:
            tables[grp] = realworld_tables(rw)
            if grp == "tabular":
                tabular_scatter_fig(rw, "MLP")
                tabular_scatter_fig(rw, "GBM")
    (ROOT / "paper").mkdir(exist_ok=True)
    (ROOT / "paper" / "tables.json").write_text(json.dumps(tables, indent=1, default=str))
    print(json.dumps(tables, indent=1, default=str)[:6000])


if __name__ == "__main__":
    main()
