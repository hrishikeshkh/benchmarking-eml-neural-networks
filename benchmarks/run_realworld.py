"""Experiment 2: real-world regression (physics measurements and general tabular data).

Protocol: K-fold cross-validation (leave-one-out for n < 20), out-of-fold predictions are
pooled to compute R^2 (stable for tiny data sets), and per-fold R^2 is stored as well.
Every symbolic model also reports its formula and size; for the physics group we
additionally fit every symbolic method once on the full data set to report the law it finds.

    python benchmarks/run_realworld.py --group physics --jobs 6
    python benchmarks/run_realworld.py --group tabular --jobs 6
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "benchmarks"))


def folds(n: int, k: int, seed: int):
    rng = np.random.default_rng(seed)
    if n < 20:
        return [(np.setdiff1d(np.arange(n), [i]), np.array([i])) for i in range(n)]
    perm = rng.permutation(n)
    parts = np.array_split(perm, k)
    return [(np.concatenate([parts[j] for j in range(k) if j != i]), parts[i]) for i in range(k)]


def make(method: str, seed: int, names):
    from baselines import BASELINES
    from emlkit import EMLRegressor
    if method == "EML":
        return EMLRegressor(feature_names=names, random_state=seed)
    return BASELINES[method](seed)


def run_one(dataset: str, method: str, seed: int, k: int, full_fit: bool) -> dict:
    import torch
    torch.set_num_threads(1)
    from realworld_data import load
    X, y, names = load(dataset)
    out = dict(dataset=dataset, method=method, seed=seed, n=len(y), d=X.shape[1])
    t0 = time.time()
    try:
        oof = np.full(len(y), np.nan)
        fold_r2, sizes, formulas = [], [], []
        for tr, te in folds(len(y), k, seed):
            m = make(method, seed, names).fit(X[tr], y[tr])
            with np.errstate(all="ignore"):
                p = np.asarray(m.predict(X[te]), dtype=float)
            oof[te] = p
            if len(te) > 1:
                fold_r2.append(float(1 - np.mean((y[te] - p) ** 2) / np.var(y[te])))
            if method == "EML":
                sizes.append(m.complexity_); formulas.append(m.formula(4))
            elif method == "GPLearn":
                sizes.append(m.size); formulas.append(m.formula())
        ok = np.isfinite(oof)
        out["r2"] = float(1 - np.mean((y - np.where(ok, oof, np.mean(y))) ** 2) / np.var(y)) if ok.all() else -math.inf
        out["fold_r2"] = fold_r2
        out["time"] = time.time() - t0
        if sizes:
            out["size_median"] = float(np.median(sizes))
            out["fold_formulas"] = formulas[:10]
        if full_fit and method in ("EML", "GPLearn"):
            m = make(method, seed, names).fit(X, y)
            out["formula_full"] = m.formula(4) if method == "EML" else m.formula()
            out["size_full"] = m.complexity_ if method == "EML" else m.size
            with np.errstate(all="ignore"):
                p = np.asarray(m.predict(X), dtype=float)
            out["r2_full_train"] = float(1 - np.mean((y - p) ** 2) / np.var(y))
    except Exception as e:
        out.update(error=repr(e), trace=traceback.format_exc()[-2000:], time=time.time() - t0)
    return out


def main():
    from realworld_data import PHYSICS, TABULAR
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", default="physics", choices=["physics", "tabular"])
    ap.add_argument("--datasets", default=None)
    ap.add_argument("--methods", default="EML,Linear,MLP,GBM,RF,GPLearn")
    ap.add_argument("--seeds", default="0")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    names = args.datasets.split(",") if args.datasets else (PHYSICS if args.group == "physics" else TABULAR)
    out_path = Path(args.out or ROOT / "results" / f"realworld_{args.group}.jsonl")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out_path.exists():
        for line in out_path.read_text().splitlines():
            r = json.loads(line)
            done.add((r["dataset"], r["method"], r["seed"]))
    tasks = [(d, m, int(s)) for s in args.seeds.split(",") for m in args.methods.split(",") for d in names]
    tasks = [t for t in tasks if t not in done]
    order = {"EML": 0, "GPLearn": 1}
    tasks.sort(key=lambda t: order.get(t[1], 2))
    print(f"{len(tasks)} runs ({len(done)} done) with {args.jobs} workers", flush=True)
    full_fit = args.group == "physics"
    with ProcessPoolExecutor(max_workers=args.jobs) as ex, open(out_path, "a") as fh:
        futs = {ex.submit(run_one, d, m, s, args.k, full_fit): (d, m, s) for d, m, s in tasks}
        for i, fut in enumerate(as_completed(futs), 1):
            res = fut.result()
            fh.write(json.dumps(res) + "\n")
            fh.flush()
            msg = res.get("error") or f"R2 {res['r2']:.4f}" + (f"  {res.get('formula_full', '')[:120]}")
            print(f"[{i}/{len(tasks)}] {res['dataset']:30s} {res['method']:8s}: {msg} ({res['time']:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
