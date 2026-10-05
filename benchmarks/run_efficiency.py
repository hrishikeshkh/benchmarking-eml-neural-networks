"""Experiment 4: accuracy per unit of compute.

For one train/test split of a Feynman problem (1% label noise, ID test set) or a tabular data set
(75/25 split) we fit
  * EML (library defaults),
  * an MLP family of increasing size: one hidden layer of 1..128 units, plus the 2x128 baseline,
  * histogram gradient boosting,
and record test R^2, number of parameters, arithmetic operations per prediction, measured
inference latency (single core, NumPy, 100k rows) and training time. For EML we also record
the network size and validation error *before* pruning and snapping.

Run on Modal:  modal run benchmarks/modal_app.py --kind efficiency
"""
from __future__ import annotations

import time
import traceback
import warnings
import zlib

import numpy as np

MLP_SIZES = [(1,), (2,), (4,), (8,), (16,), (32,), (64,), (128,), (128, 128)]


def _r2(y, p):
    p = np.asarray(p, dtype=float)
    return float(1 - np.mean((y - p) ** 2) / np.var(y)) if np.all(np.isfinite(p)) else float("-inf")


def _latency(fn, X, reps: int = 3) -> float:
    """Seconds per 1M predictions (best of ``reps``)."""
    best = float("inf")
    for _ in range(reps):
        t = time.perf_counter()
        fn(X)
        best = min(best, time.perf_counter() - t)
    return best / len(X) * 1e6


def _mlp(sizes, seed):
    from sklearn.compose import TransformedTargetRegressor
    from sklearn.neural_network import MLPRegressor
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    mlp = MLPRegressor(hidden_layer_sizes=sizes, activation="relu", solver="adam", learning_rate_init=1e-3,
                       max_iter=3000, early_stopping=True, validation_fraction=0.15, n_iter_no_change=50,
                       tol=1e-7, random_state=seed)
    return TransformedTargetRegressor(regressor=make_pipeline(StandardScaler(), mlp), transformer=StandardScaler())


def _mlp_numpy(model):
    """Plain NumPy forward pass of a fitted MLP pipeline (for a fair latency measurement)."""
    pipe = model.regressor_
    sc, mlp = pipe[0], pipe[-1]
    ty = model.transformer_
    Ws, bs = mlp.coefs_, mlp.intercepts_
    mu, sd, ymu, ysd = sc.mean_, sc.scale_, float(ty.mean_[0]), float(ty.scale_[0])

    def f(X):
        h = (X - mu) / sd
        for W, b in zip(Ws[:-1], bs[:-1]):
            h = np.maximum(h @ W + b, 0)
        return (h @ Ws[-1] + bs[-1]).ravel() * ysd + ymu

    n_params = sum(W.size + b.size for W, b in zip(Ws, bs))
    ops = sum(2 * W.size + b.size for W, b in zip(Ws, bs)) + sum(b.size for b in bs[:-1])  # mul+add, bias, relu
    return f, n_params, ops


def _expr_ops(expr) -> int:
    import sympy as sp
    return int(sp.count_ops(expr, visual=False))


def _data(kind: str, name: str, seed: int):
    if kind == "feynman":
        from feynman_data import load_problems
        prob = {p.name: p for p in load_problems()}[name]
        rng = np.random.default_rng(zlib.crc32(f"eff-{name}-{seed}".encode()))
        X, y = prob.sample(1000, rng)
        y = y + rng.normal(0, 0.01 * np.std(y), len(y))
        Xt, yt = prob.sample(2000, rng)
        return X, y, Xt, yt, prob.variables
    from realworld_data import load
    X, y, names = load(name)
    rng = np.random.default_rng(zlib.crc32(f"eff-{name}-{seed}".encode()))
    perm = rng.permutation(len(y))
    k = int(round(0.75 * len(y)))
    return X[perm[:k]], y[perm[:k]], X[perm[k:]], y[perm[k:]], names


def run_one(kind: str, name: str, seed: int = 0) -> dict:
    import torch
    torch.set_num_threads(1)
    from sklearn.ensemble import HistGradientBoostingRegressor

    from emlkit import EMLRegressor

    X, y, Xt, yt, names = _data(kind, name, seed)
    Xbig = np.tile(Xt, (int(np.ceil(100_000 / len(Xt))), 1))[:100_000]
    out = dict(kind=kind, dataset=name, seed=seed, n=len(y), d=X.shape[1], models=[])
    try:
        t = time.time()
        m = EMLRegressor(feature_names=names, random_state=seed).fit(X, y)
        fit_s = time.time() - t
        f = m.lambdify()
        with np.errstate(all="ignore"):
            p = m.predict(Xt)          # scored exactly like every other experiment
            p_formula = f(Xt)          # the exported closed form, evaluated directly
            lat = _latency(f, Xbig)
        out["models"].append(dict(model="EML", r2=_r2(yt, p), formula_r2=_r2(yt, p_formula),
                                  n_params=m.n_params_, ops=_expr_ops(m.expr_),
                                  latency_s_per_1M=lat, train_s=fit_s, formula=m.formula(4),
                                  dense_n_params=m.dense_n_params_, dense_val_r2=1 - m.dense_val_nmse_,
                                  val_r2=1 - m.val_nmse_, arch=m.best_.arch))
        for sizes in MLP_SIZES:
            t = time.time()
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                mm = _mlp(sizes, seed).fit(X, y)
            fit_s = time.time() - t
            fn, n_params, ops = _mlp_numpy(mm)
            out["models"].append(dict(model="MLP" + "x".join(map(str, sizes)), r2=_r2(yt, fn(Xt)),
                                      n_params=int(n_params), ops=int(ops), latency_s_per_1M=_latency(fn, Xbig),
                                      train_s=fit_s))
        t = time.time()
        g = HistGradientBoostingRegressor(max_iter=1000, learning_rate=0.05, early_stopping="auto",
                                          min_samples_leaf=5, random_state=seed).fit(X, y)
        fit_s = time.time() - t
        nodes = sum(pred.nodes.shape[0] for preds in g._predictors for pred in preds)
        depth_ops = sum(int(np.log2(max(pred.nodes.shape[0], 2))) + 1 for preds in g._predictors for pred in preds)
        out["models"].append(dict(model="GBM", r2=_r2(yt, g.predict(Xt)), n_params=int(nodes), ops=int(depth_ops),
                                  latency_s_per_1M=_latency(g.predict, Xbig), train_s=fit_s))
    except Exception as e:  # record and move on
        out.update(error=repr(e), trace=traceback.format_exc()[-2000:])
    return out
