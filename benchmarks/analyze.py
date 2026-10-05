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
    # failed / timed-out runs of symbolic methods count as "not recovered"
    sym = df.method.isin(["EML", "GPLearn"])
    df.loc[sym, "symbolic"] = df.loc[sym, "symbolic"].fillna(False)
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
    axes[0].legend(loc="center left", bbox_to_anchor=(0.0, 0.45), fontsize=8)
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
    ax.text(0.04, 0.96, f"{k} of {len(x)} data sets\nwithin 0.02 of {other}\nor better",
            transform=ax.transAxes, color=TEXT, fontsize=7.8, va="top", linespacing=1.3)
    n_clip = int((y <= -0.2).sum())
    if n_clip:
        ax.text(0.98, 0.03, f"{n_clip} below −0.2 (shown at edge)", transform=ax.transAxes, color=MUTED,
                fontsize=7, ha="right", va="bottom")
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
    out.append(("C7", "best model on some real data sets",
                f"{len(wins)} data sets: {', '.join(w.replace('first_principles_', '') for w in wins[:8])}"
                + (", …" if len(wins) > 8 else ""),
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
    E = T.get("efficiency")
    if E and E.get("n"):
        for cid, kind, label in (("C13", "feynman", "same accuracy with far fewer parameters (Feynman)"),
                                 ("C14", "tabular", "same accuracy with fewer parameters (tabular)")):
            if kind in E:
                k = E[kind]
                out.append((cid, label, f"matching MLP needs {k['match_ratio_median']:.0f}× more parameters (median; "
                            f"no MLP matches EML on {k['no_mlp_matches']:.0%})",
                            verdict(k["match_ratio_median"], lambda x: x >= 10, lambda x: x >= 2)))
        a = E["all"]
        out.append(("C15", "cheaper inference", f"2×128 MLP is {a['latency_ratio_median']:.0f}× slower per prediction (median)",
                    verdict(a["latency_ratio_median"], lambda x: x >= 10, lambda x: x >= 2)))
        out.append(("C16", "beats a parameter-matched NN", f"EML ≥ parameter-matched MLP on {a['eml_beats_param_matched']:.0%} of tasks",
                    verdict(a["eml_beats_param_matched"], lambda x: x >= 2 / 3, lambda x: x >= 1 / 2)))
        out.append(("C17", "training cost comparable to an MLP", f"EML trains {a['train_ratio_median']:.0f}× slower (median)",
                    verdict(a["train_ratio_median"], lambda x: x <= 10, lambda x: x <= 100)))
        pr, pl = a["prune_reduction_median"], a["prune_r2_loss_median"]
        out.append(("C18", "pruning keeps accuracy", f"{pr:.0%} fewer parameters, validation R² change {-pl:+.4f} (medians)",
                    "S" if pr >= .8 and pl <= .01 else ("P" if pr >= .5 and pl <= .02 else "N")))
    t = sub.get("trig @ 0.0", {}).get("recovery")
    if t is not None:
        out.append(("C12", "limitation: trigonometric laws", f"{t:.0%} recovered on trig problems",
                    "S" if t <= .10 else "N"))
    return out


# -------------------------------------------------------------------------- efficiency
def efficiency(eff: pd.DataFrame) -> dict:
    """Per-task compute comparisons (definitions fixed in CLAIMS.md, C13-C18)."""
    rows = []
    for _, r in eff.iterrows():
        if isinstance(r.get("error"), str) or not isinstance(r.get("models"), list):
            continue
        ms = {m["model"]: m for m in r["models"]}
        e = ms.get("EML")
        mlps = sorted([m for k, m in ms.items() if k.startswith("MLP")], key=lambda m: m["n_params"])
        if e is None or not mlps:
            continue
        match = next((m for m in mlps if m["r2"] >= e["r2"] - 0.005), None)
        pm = next((m for m in mlps if m["n_params"] >= e["n_params"]), mlps[-1])
        big = ms.get("MLP128x128")
        rows.append(dict(
            kind=r["kind"], dataset=r["dataset"], eml_r2=e["r2"], eml_params=e["n_params"], eml_ops=e["ops"],
            match_ratio=(match["n_params"] / max(e["n_params"], 1)) if match else math.inf,
            match_model=match["model"] if match else None,
            pm_model=pm["model"], pm_r2=pm["r2"], eml_beats_pm=bool(e["r2"] >= pm["r2"]),
            mlp_r2=big["r2"] if big else None, mlp_params=big["n_params"] if big else None,
            latency_ratio=(big["latency_s_per_1M"] / max(e["latency_s_per_1M"], 1e-12)) if big else None,
            train_ratio=(e["train_s"] / max(big["train_s"], 1e-9)) if big else None,
            dense_params=e.get("dense_n_params"), prune_reduction=1 - e["n_params"] / max(e.get("dense_n_params") or 1, 1),
            prune_r2_loss=(e.get("dense_val_r2") or 0) - (e.get("val_r2") or 0),
            gbm_r2=ms["GBM"]["r2"] if "GBM" in ms else None, gbm_params=ms["GBM"]["n_params"] if "GBM" in ms else None,
            formula_r2=e.get("formula_r2")))
    df = pd.DataFrame(rows)
    out = {"n": int(len(df))}
    for kind, g in df.groupby("kind"):
        out[kind] = dict(n=int(len(g)), match_ratio_median=float(g.match_ratio.median()),
                         no_mlp_matches=float(np.isinf(g.match_ratio).mean()),
                         eml_beats_param_matched=float(g.eml_beats_pm.mean()),
                         eml_params_median=float(g.eml_params.median()), eml_r2_median=float(g.eml_r2.median()),
                         mlp_r2_median=float(g.mlp_r2.median()), latency_ratio_median=float(g.latency_ratio.median()),
                         train_ratio_median=float(g.train_ratio.median()),
                         prune_reduction_median=float(g.prune_reduction.median()),
                         prune_r2_loss_median=float(g.prune_r2_loss.median()))
    if "formula_r2" in df and df.formula_r2.notna().any():
        gap = (df.eml_r2 - df.formula_r2).abs()
        out["formula_mismatch_share"] = float((~(gap <= 0.01)).mean())
    out["all"] = dict(latency_ratio_median=float(df.latency_ratio.median()),
                      train_ratio_median=float(df.train_ratio.median()),
                      eml_beats_param_matched=float(df.eml_beats_pm.mean()),
                      prune_reduction_median=float(df.prune_reduction.median()),
                      prune_r2_loss_median=float(df.prune_r2_loss.median()))
    out["rows"] = rows
    return out


def efficiency_fig(eff: pd.DataFrame) -> None:
    """Median test R^2 against parameter count: the MLP family as a curve, EML and GBM as points."""
    _style()
    kinds = [k for k in ("feynman", "tabular") if k in set(eff.kind)]
    fig, axes = plt.subplots(1, len(kinds), figsize=(3.7 * len(kinds), 3.0), squeeze=False)
    for ax, kind in zip(axes[0], kinds):
        recs = [m | {"dataset": r["dataset"]} for _, r in eff[eff.kind == kind].iterrows()
                if isinstance(r.get("models"), list) for m in r["models"]]
        d = pd.DataFrame(recs)
        d["r2c"] = d.r2.clip(lower=-0.5)
        mlp = d[d.model.str.startswith("MLP")].groupby("model").agg(p=("n_params", "median"), r=("r2c", "median"))
        mlp = mlp.sort_values("p")
        ax.plot(mlp.p, mlp.r, color=COLORS["MLP"], lw=2, marker="o", ms=4, label="MLP (1 to 2×128 hidden)")
        for name, col in (("EML", COLORS["EML"]), ("GBM", COLORS["GBM"])):
            g = d[d.model == name]
            if len(g):
                ax.scatter([g.n_params.median()], [g.r2c.median()], s=60, color=col, edgecolor="white",
                           linewidth=1.5, zorder=4, label=name)
        ax.set_xscale("log")
        ax.set_xlabel("parameters (median over tasks)")
        ax.set_title("Feynman (1% noise)" if kind == "feynman" else "Tabular data", fontsize=9.5, loc="left", color=TEXT)
    axes[0][0].set_ylabel("median test R²")
    axes[0][-1].legend(loc="lower right", fontsize=7.5)
    fig.tight_layout()
    _save(fig, "efficiency_params_vs_r2")


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


GROUPS = [
    ("Does it perform very well?", ["C1", "C2", "C4", "C5", "C7"]),
    ("Is it explainable while staying accurate?", ["C6", "C8", "C18"]),
    ("Is it comparable to a regular neural network?", ["C3", "C9", "C16"]),
    ("Does it need less compute?", ["C13", "C14", "C15", "C17"]),
    ("Method and tooling", ["C10", "C11", "C12"]),
]


def readme_section(C: list) -> str:
    word = {"S": "✅ supported", "P": "🟡 partly", "N": "❌ not supported"}
    by = {c[0]: c for c in C}
    lines = []
    for title, ids in GROUPS:
        have = [by[i] for i in ids if i in by]
        if not have:
            continue
        lines += [f"**{title}**", "", "| # | Claim | Evidence | Verdict |", "|---|---|---|---|"]
        lines += [f"| {c} | {t} | {e} | {word[v]} |" for c, t, e, v in have]
        lines.append("")
    figs = [("feynman_profile_noise0.0.png", "Share of Feynman problems below each test-error threshold, in and out of distribution"),
            ("efficiency_params_vs_r2.png", "Accuracy against parameter count: EML formula vs. MLPs of every size and gradient boosting"),
            ("tabular_eml_vs_mlp.png", "EML formula vs. MLP, cross-validated R² on 34 tabular data sets")]
    for f, cap in figs:
        if (FIG / f).exists():
            lines += [f"<img src=\"paper/figures/{f}\" width=\"640\" alt=\"{cap}\">", "", f"*{cap}.*", ""]
    return "\n".join(lines)


def audit_notes(T: dict) -> str:
    """Caveats found while auditing the results; numbers are recomputed on every run."""
    tab = T.get("tabular", {}).get("rows", [])
    phys = T.get("physics", {}).get("rows", [])
    notes = []
    bad = [r["dataset"] for r in tab + phys if r["r2"].get("EML") is not None and r["r2"]["EML"] < -1]
    if bad:
        notes.append(f"EML fails catastrophically (CV R² < −1) on {len(bad)} data sets ({', '.join(bad)}): on small, noisy "
                     "data a learned exp(·) can blow up on held-out folds. Nothing clips the predictions, and the "
                     "pre-registered numbers include these failures.")
    ok = [r for r in tab if r["r2"].get("EML") is not None]
    if ok:
        best = [max(v for k, v in r["r2"].items() if k != "EML" and v is not None) for r in ok]
        share = np.mean([r["r2"]["EML"] >= b - 0.02 for r, b in zip(ok, best)])
        notes.append(f"C9 compares against the MLP, which is weak on the smallest data sets. Against the *best* baseline "
                     f"per data set, EML is within 0.02 R² or better on {share:.0%} of tabular data sets.")
    notes.append("`561_cpu`: the target equals the *estimated* relative performance (ERP) of the original 1987 study "
                 "(correlation 1.0 with the UCI ERP column), i.e. it is itself a regression formula. EML's R² = 1.000 there "
                 "means it recovered that formula; it is not a typical real-world result.")
    notes.append("Some C7 wins come from data sets so small that several baselines collapse (Bode n = 8, Kepler n = 6, "
                 "leave-one-out). Kepler is still informative: EML finds period ∝ a^1.5 with R² = 1.000 vs 0.865 for the MLP.")
    notes.append("Physics formulas are not always clean laws. Kepler comes out as 359.9·a^1.509 times near-1 factors (true: "
                 "365.25·a^1.5). On `ideal_gas` EML fails (R² 0.38): that target is ln P = ln n + ln R + ln T − ln V, a plain *sum* "
                 "of log-features, and the curriculum has no stage for that (every stage routes the output through exp/ln nodes). "
                 "Adding an additive-log stage is the obvious fix; we did not apply it, to keep the pre-registered protocol intact.")
    E = T.get("efficiency", {})
    if "formula_mismatch_share" in E:
        notes.append(f"The exported formula and `model.predict` disagree (|ΔR²| > 0.01) on {E['formula_mismatch_share']:.0%} of "
                     "compute-benchmark test sets. The network clamps exp(·) and guards ln|·|; the printed formula does not. All "
                     "reported accuracies use `model.predict`. An earlier version of the compute benchmark scored the raw "
                     "formula and is kept in `results/superseded/`.")
    notes.append("The SRBench-style recovery check is conservative. It rounds constants to 3 decimals, which can hide an "
                 "exact recovery (e.g. II.24.17 is recovered exactly but scored as a miss).")
    return "\n".join(["**Audit notes**", ""] + [f"- {n}" for n in notes]) + "\n"


def inject_readme(text: str) -> None:
    path = ROOT / "README.md"
    s = path.read_text()
    a, b = "<!-- RESULTS:START -->", "<!-- RESULTS:END -->"
    if a in s and b in s:
        s = s[: s.index(a) + len(a)] + "\n" + text + "\n" + s[s.index(b):]
        path.write_text(s)


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
        ab["symbolic"] = ab["symbolic"].fillna(False)
        T["ablation"] = [dict(method=m, n=int(len(h)), recovery=float(h.symbolic.astype(float).mean()),
                              ood_r2_99=float((h.r2_ood.fillna(-np.inf) > 0.99).mean()),
                              time_median=float(h.time.median())) for m, h in ab.groupby("method")]
    ef = load("efficiency.jsonl")
    if not ef.empty:
        T["efficiency"] = efficiency(ef)
        efficiency_fig(ef)
    C = claims(T, fe, ab) if "feynman" in T else []
    T["claims"] = [dict(id=c, claim=t, evidence=e, verdict=v) for c, t, e, v in C]
    (ROOT / "paper").mkdir(exist_ok=True)
    (ROOT / "paper" / "tables.json").write_text(json.dumps(T, indent=1, default=str))
    (ROOT / "RESULTS.md").write_text(results_md(T, C))
    if C:
        notes = audit_notes(T)
        inject_readme(readme_section(C) + "\n" + notes)
        with open(ROOT / "RESULTS.md", "a") as fh:
            fh.write("\n" + notes)
    for c, t, e, v in C:
        print(f"{c:4s} {v}  {t:45s} {e}")


if __name__ == "__main__":
    main()
