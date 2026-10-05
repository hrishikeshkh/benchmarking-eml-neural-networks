"""Compute every table, figure and claim verdict from the raw results.

    python benchmarks/analyze.py      # -> paper/tables.json, paper/figures/*.svg|png, RESULTS.md

Claim thresholds are the ones fixed in CLAIMS.md before the full results were in.
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
BASELINES = ["GPLearn", "MLP", "GBM", "RF", "Linear"]
METHODS = ["EML"] + BASELINES
# fixed categorical order (reference palette); Linear/RF are muted context series
COLORS = {"EML": "#2a78d6", "MLP": "#eb6834", "GBM": "#1baf7a", "GPLearn": "#eda100",
          "RF": "#e87ba4", "Linear": "#8a8984"}
TEXT, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def load(name: str) -> pd.DataFrame:
    path = RES / name
    if not path.exists() or not path.read_text().strip():
        return pd.DataFrame()
    df = pd.DataFrame([json.loads(l) for l in path.read_text().splitlines() if l.strip()])
    if "error" in df:
        bad = df["error"].notna()
        if bad.any():
            print(f"{name}: {int(bad.sum())} runs failed and are counted as failures")
    return df


def _f(x):
    return None if x is None or (isinstance(x, float) and not math.isfinite(x)) else float(x)


def verdict(value, s_ok, p_ok):
    return "S" if s_ok(value) else ("P" if p_ok is not None and p_ok(value) else "N")


# ----------------------------------------------------------------------------- Feynman
def feynman(df: pd.DataFrame) -> dict:
    df = df.copy()
    for col in ("r2_id", "r2_ood"):
        df[col] = df[col].fillna(-np.inf)
    df["nmse_ood"] = df["nmse_ood"].fillna(np.inf)
    out = {}
    for noise, g in df.groupby("noise"):
        # compare methods on the same problems (matters only while a sweep is incomplete)
        if "EML" in set(g.method):
            g = g[g.problem.isin(set(g[g.method == "EML"].problem))]
        rows = []
        for m in [m for m in METHODS if m in set(g.method)]:
            h = g[g.method == m]
            rec = h["symbolic"].astype(float).mean() if "symbolic" in h and h["symbolic"].notna().any() else None
            rows.append(dict(
                method=m, n=int(len(h)), recovery=_f(rec),
                id_r2_999=float((h.r2_id > 0.999).mean()), ood_r2_99=float((h.r2_ood > 0.99).mean()),
                ood_exact=float((h.nmse_ood < 1e-9).mean()),
                id_r2_median=float(h.r2_id.median()), ood_r2_median=float(h.r2_ood.median()),
                time_median=float(h.time.median()),
                size_median=_f(h["size"].median()) if "size" in h and h["size"].notna().any() else None))
        out[f"noise_{noise}"] = rows
    e = df[df.method == "EML"]
    subsets = {}
    for noise, g in e.groupby("noise"):
        for k, s in {"all": g, "no trig": g[~g.has_trig], "trig": g[g.has_trig], "dev": g[g.dev],
                     "held-out": g[~g.dev], "held-out, no trig": g[~g.dev & ~g.has_trig]}.items():
            if len(s):
                subsets[f"{k} @ {noise}"] = dict(n=int(len(s)), recovery=float(s.symbolic.astype(float).mean()),
                                                 ood_r2_99=float((s.r2_ood > 0.99).mean()))
    out["eml_subsets"] = subsets
    return out


def feynman_profile_fig(df: pd.DataFrame, noise: float) -> None:
    _style()
    g = df[df.noise == noise]
    taus = np.logspace(-14, 0, 300)
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 2.9), sharey=True)
    for ax, col, title in [(axes[0], "nmse_id", "In distribution"),
                           (axes[1], "nmse_ood", "Out of distribution")]:
        for m in [m for m in ["EML", "GPLearn", "MLP", "GBM"] if m in set(g.method)]:
            v = g[g.method == m][col].fillna(np.inf).to_numpy()
            frac = np.array([(v <= t).mean() for t in taus])
            ax.plot(taus, frac, color=COLORS[m], lw=2, label=m, solid_capstyle="round")
        ax.set_xscale("log")
        ax.set_xlim(1e-14, 1)
        ax.set_ylim(0, 1.02)
        ax.set_xticks([1e-12, 1e-9, 1e-6, 1e-3, 1])
        ax.set_xlabel("NMSE threshold τ")
        ax.set_title(title, fontsize=9.5, color=TEXT, loc="left")
    axes[0].set_ylabel("share of problems with NMSE ≤ τ")
    axes[0].legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    _save(fig, f"feynman_profile_noise{noise}")


# --------------------------------------------------------------------------- real world
def realworld(df: pd.DataFrame) -> dict:
    if df.empty:
        return {}
    df = df.copy()
    df["r2"] = df["r2"].fillna(-np.inf) if "r2" in df else -np.inf
    piv = df.pivot_table(index="dataset", columns="method", values="r2", aggfunc="first")
    methods = [m for m in METHODS if m in piv.columns]
    piv = piv[methods]
    rows = []
    for d, r in piv.iterrows():
        e = df[(df.dataset == d) & (df.method == "EML")]
        base = [m for m in BASELINES if m in piv.columns and pd.notna(r[m])]
        best_base = max(base, key=lambda m: r[m]) if base else None
        row = dict(dataset=d, n=int(df[df.dataset == d].n.iloc[0]), d=int(df[df.dataset == d].d.iloc[0]),
                   r2={m: _f(r[m]) for m in methods}, best_baseline=best_base)
        if len(e):
            e = e.iloc[0]
            row["eml_size"] = _f(e.get("size_median"))
            ff = e.get("formula_full")
            row["eml_formula"] = ff if isinstance(ff, str) else (
                e["fold_formulas"][0] if isinstance(e.get("fold_formulas"), list) and e["fold_formulas"] else None)
            row["eml_size_full"] = _f(e.get("size_full"))
        rows.append(row)
    summary = []
    ranks = piv.rank(axis=1, ascending=False)
    for m in methods:
        summary.append(dict(method=m, median_r2=float(piv[m].median()), mean_rank=float(ranks[m].mean())))
    return dict(rows=rows, summary=summary)


def tabular_scatter_fig(tab: dict, other: str) -> None:
    _style()
    pts = [(r["r2"].get(other), r["r2"].get("EML")) for r in tab.get("rows", [])
           if r["r2"].get(other) is not None and r["r2"].get("EML") is not None]
    if not pts:
        return
    x, y = (np.clip(np.array(v, dtype=float), -0.2, 1) for v in zip(*pts))
    fig, ax = plt.subplots(figsize=(3.5, 3.3))
    ax.fill_between([-0.2, 1], [-0.22, 0.98], [1.2, 1.2], color=COLORS["EML"], alpha=0.07, lw=0)
    ax.plot([-0.2, 1], [-0.2, 1], color=MUTED, lw=1)
    ax.plot([-0.2, 1], [-0.22, 0.98], color=MUTED, lw=0.8, ls=(0, (3, 3)))
    ax.scatter(x, y, s=30, color=COLORS["EML"], edgecolor="white", linewidth=1.2, zorder=3)
    ax.set_xlim(-0.2, 1.02); ax.set_ylim(-0.2, 1.02)
    ax.set_xlabel(f"{other}: 5-fold CV R²")
    ax.set_ylabel("EML formula: 5-fold CV R²")
    k = int((y >= x - 0.02).sum())
    ax.text(0.03, 0.97, f"{k} of {len(x)} data sets within 0.02 of {other} or better",
            transform=ax.transAxes, color=TEXT, fontsize=7.8, va="top")
    fig.tight_layout()
    _save(fig, f"tabular_eml_vs_{other.lower()}")


# ------------------------------------------------------------------------------ claims
def claims(T: dict, fe: pd.DataFrame, ab: pd.DataFrame) -> list:
    out = []
    F0 = {r["method"]: r for r in T["feynman"].get("noise_0.0", [])}
    F1 = {r["method"]: r for r in T["feynman"].get("noise_0.01", [])}
    sub = T["feynman"]["eml_subsets"]

    v = sub["held-out, no trig @ 0.0"]["recovery"]
    out.append(("C1", "EML recovers exact laws", f"{v:.0%} of held-out non-trig problems",
                verdict(v, lambda x: x >= .5, lambda x: x >= .25)))
    e, g = F0["EML"]["recovery"], F0.get("GPLearn", {}).get("recovery")
    out.append(("C2", "at least as good as GP symbolic regression", f"EML {e:.0%} vs GPLearn {g:.0%}",
                verdict(e - g, lambda x: x >= 0, lambda x: x >= -0.05)))
    e, m = F0["EML"]["id_r2_999"], F0["MLP"]["id_r2_999"]
    out.append(("C3", "matches an MLP in distribution", f"R²>0.999 on {e:.0%} (EML) vs {m:.0%} (MLP)",
                verdict(e - m, lambda x: x >= -.05, lambda x: x >= -.15)))
    e = F0["EML"]["ood_r2_99"]; b = max(F0["MLP"]["ood_r2_99"], F0["GBM"]["ood_r2_99"])
    out.append(("C4", "extrapolates far better than MLP / GBM",
                f"OOD R²>0.99 on {e:.0%} (EML) vs {F0['MLP']['ood_r2_99']:.0%} (MLP), {F0['GBM']['ood_r2_99']:.0%} (GBM)",
                verdict(e - b, lambda x: x >= .25, lambda x: x > 0)))
    if "EML" in F1:
        r = F1["EML"]["recovery"] / max(F0["EML"]["recovery"], 1e-9)
        out.append(("C5", "survives 1% label noise",
                    f"recovery {F1['EML']['recovery']:.0%} at 1% vs {F0['EML']['recovery']:.0%} noiseless",
                    verdict(r, lambda x: x >= .5, lambda x: x >= .25)))
    sizes_real = [r.get("eml_size") for k in ("physics", "tabular") for r in T.get(k, {}).get("rows", [])
                  if r.get("eml_size") is not None]
    sf, sr = F0["EML"]["size_median"], (float(np.median(sizes_real)) if sizes_real else math.nan)
    out.append(("C6", "models are readable", f"median size {sf:.0f} (Feynman), {sr:.0f} (real data) SymPy nodes",
                "S" if sf <= 30 and sr <= 40 else ("P" if sf <= 50 and sr <= 60 else "N")))
    rows = T.get("physics", {}).get("rows", []) + T.get("tabular", {}).get("rows", [])
    wins = [r["dataset"] for r in rows if r["r2"].get("EML") is not None and
            all(r["r2"]["EML"] >= (r["r2"].get(b) if r["r2"].get(b) is not None else -np.inf) for b in BASELINES)]
    out.append(("C7", "best model on some real data sets", f"{len(wins)} data sets: {', '.join(wins[:8])}",
                verdict(len(wins), lambda x: x >= 3, lambda x: x >= 1)))
    ph = T.get("physics", {}).get("rows", [])
    if ph:
        gaps = [max(r["r2"][b] for b in BASELINES if r["r2"].get(b) is not None) - r["r2"]["EML"]
                for r in ph if r["r2"].get("EML") is not None]
        gm = float(np.median(gaps))
        out.append(("C8", "accurate while explainable (physics data)", f"median gap to best baseline {gm:+.3f} R²",
                    verdict(gm, lambda x: x <= .02, lambda x: x <= .05)))
    tb = T.get("tabular", {}).get("rows", [])
    if tb:
        ok = [r for r in tb if r["r2"].get("EML") is not None and r["r2"].get("MLP") is not None]
        share = float(np.mean([r["r2"]["EML"] >= r["r2"]["MLP"] - 0.02 for r in ok]))
        out.append(("C9", "comparable to a NN (tabular data)", f"within 0.02 of the MLP or better on {share:.0%} of data sets",
                    verdict(share, lambda x: x >= 1 / 3, lambda x: x >= 1 / 5)))
    if not ab.empty:
        base = fe[(fe.method == "EML") & (fe.noise == 0.0) & fe.problem.isin(set(ab.problem))]
        br = base.symbolic.astype(float).mean()
        drops = {m: br - h.symbolic.astype(float).mean() for m, h in ab.groupby("method")}
        T["ablation_drops"] = dict(base=float(br), n=int(len(base)), drops={k: float(v) for k, v in drops.items()})
        worst = min(drops.values())
        out.append(("C10", "every training component matters",
                    "; ".join(f"{k.replace('EML-', '')} {-v * 100:+.0f} pts" for k, v in sorted(drops.items(), key=lambda kv: -kv[1])),
                    verdict(worst, lambda x: x >= .05, lambda x: x > 0)))
    cs = json.loads((ROOT / "paper" / "compiler_stats.json").read_text())
    err = max(r["err"] for r in cs["compile"])
    out.append(("C11", "compiler is exact", f"max error {err:.1e} over {len(cs['compile'])} expressions",
                "S" if err < 1e-10 else "N"))
    t = sub.get("trig @ 0.0", {}).get("recovery")
    if t is not None:
        out.append(("C12", "limitation: trigonometric laws", f"{t:.0%} recovered on trig problems",
                    "S" if t <= .10 else "N"))
    return out


# ------------------------------------------------------------------------------- utils
def _style():
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": TEXT,
        "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "legend.frameon": False,
        "svg.fonttype": "none"})


def _save(fig, name):
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / f"{name}.svg")
    fig.savefig(FIG / f"{name}.png", dpi=200)
    plt.close(fig)


def results_md(T: dict, C: list) -> str:
    lines = ["# Results", "", "Generated by `benchmarks/analyze.py` from `results/*.jsonl`. Thresholds: `CLAIMS.md`.", "",
             "| # | Claim | Evidence | Verdict |", "|---|---|---|---|"]
    word = {"S": "supported", "P": "partly", "N": "not supported"}
    lines += [f"| {c} | {t} | {e} | **{word[v]}** |" for c, t, e, v in C]
    for noise in ("0.0", "0.01"):
        rows = T["feynman"].get(f"noise_{noise}")
        if not rows:
            continue
        lines += ["", f"## Feynman (99 problems, noise {noise})", "",
                  "| method | recovery | R²_id>0.999 | R²_ood>0.99 | median size | median time (s) |", "|---|---|---|---|---|---|"]
        for r in rows:
            rec = "–" if r["recovery"] is None else f"{r['recovery']:.0%}"
            size = "–" if r["size_median"] is None else f"{r['size_median']:.0f}"
            lines.append(f"| {r['method']} | {rec} | {r['id_r2_999']:.0%} | {r['ood_r2_99']:.0%} | {size} | {r['time_median']:.0f} |")
    for k in ("physics", "tabular"):
        if T.get(k):
            ms = [s["method"] for s in T[k]["summary"]]
            lines += ["", f"## Real-world: {k} (5-fold CV R²)", "", "| data set | n | " + " | ".join(ms) + " | EML formula |",
                      "|---|---|" + "---|" * len(ms) + "---|"]
            for r in T[k]["rows"]:
                vals = " | ".join("–" if r["r2"].get(m) is None else f"{r['r2'][m]:.3f}" for m in ms)
                fm = (r.get("eml_formula") or "").replace("|", "\\|")
                lines.append(f"| {r['dataset']} | {r['n']} | {vals} | `{fm[:90]}` |")
    return "\n".join(lines) + "\n"


def main():
    T = {}
    fe = load("feynman.jsonl")
    if not fe.empty:
        T["feynman"] = feynman(fe)
        for nz in sorted(fe.noise.unique()):
            feynman_profile_fig(fe, nz)
    for k in ("physics", "tabular"):
        rw = load(f"realworld_{k}.jsonl")
        if not rw.empty:
            T[k] = realworld(rw)
    if T.get("tabular"):
        tabular_scatter_fig(T["tabular"], "MLP")
        tabular_scatter_fig(T["tabular"], "GBM")
    ab = load("ablation.jsonl")
    if not ab.empty:
        T["ablation"] = [dict(method=m, n=int(len(h)), recovery=float(h.symbolic.astype(float).mean()),
                              ood_r2_99=float((h.r2_ood.fillna(-np.inf) > 0.99).mean()),
                              time_median=float(h.time.median())) for m, h in ab.groupby("method")]
    C = claims(T, fe, ab) if "feynman" in T else []
    T["claims"] = [dict(id=c, claim=t, evidence=e, verdict=v) for c, t, e, v in C]
    (ROOT / "paper").mkdir(exist_ok=True)
    (ROOT / "paper" / "tables.json").write_text(json.dumps(T, indent=1, default=str))
    (ROOT / "RESULTS.md").write_text(results_md(T, C))
    for c, t, e, v in C:
        print(f"{c:4s} {v}  {t:45s} {e}")


if __name__ == "__main__":
    main()
