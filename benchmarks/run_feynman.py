"""Experiment 1: Feynman symbolic-regression benchmark with in- and out-of-distribution tests.

For every problem / method / seed / noise level we draw
  * a training set of ``--n-train`` points uniformly from the PMLB variable box,
  * an in-distribution test set (same box, noiseless),
  * an out-of-distribution test set (every variable in [hi, hi + (hi - lo)], noiseless),
and record R^2, NMSE, model size, wall time and (for symbolic methods) whether the
ground-truth formula was recovered.  Results are appended to a JSON-lines file so that
interrupted runs can be resumed.

    python benchmarks/run_feynman.py --methods EML,MLP,GBM,Linear,GPLearn --jobs 6
"""
from __future__ import annotations

import argparse
import json
import math
import os
import signal
import sys
import time
import traceback
import zlib
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "benchmarks"))

DEV_PROBLEMS = ["feynman_I_12_1", "feynman_I_6_2a", "feynman_II_13_23", "feynman_I_44_4", "feynman_I_9_18", "feynman_II_6_15a", "feynman_III_4_33",
                "feynman_I_41_16"]


# Ablations of the EML training recipe (see paper, Section 5.4).
from emlkit.regressor import DEFAULT_CURRICULUM as _CUR, LINEAR_OUTPUT_CURRICULUM as _LIN  # noqa: E402

EML_VARIANTS = {
    "EML": {},
    "EML-noLogVP": dict(curriculum=_LIN),                              # no log-space projection
    "EML-noVP": dict(varpro=False),                                   # read-out by gradient descent
    "EML-skip": dict(curriculum=tuple(a if a.endswith(("*", "+")) else a + "+" for a in _CUR)),
    "EML-wide": dict(curriculum=("E4-L2-E2+",)),                      # no curriculum
    "EML-untyped": dict(curriculum=tuple(a.replace("E", "M").replace("L", "M") for a in _CUR)),
    "EML-noQ": dict(quant=0.0),                                       # no annealed quantisation
    "EML-noLM": dict(lm_iters=0),                                     # Adam-only polishing
    "EML-noHalving": dict(halving=0),                                 # no successive halving
}

# Ablation subset: every other held-out problem without trigonometric functions (fixed list).
def ablation_problems():
    from feynman_data import load_problems
    names = [p.name for p in load_problems() if not p.has_trig and p.name not in DEV_PROBLEMS]
    return names[::2]


def r2(y, p):
    p = np.asarray(p, dtype=float)
    if not np.all(np.isfinite(p)):
        return -math.inf
    return float(1 - np.mean((y - p) ** 2) / np.var(y))


def nmse(y, p):
    p = np.asarray(p, dtype=float)
    if not np.all(np.isfinite(p)):
        return math.inf
    return float(np.mean((y - p) ** 2) / np.var(y))


class _Timeout(Exception):
    pass


def _alarm(signum, frame):
    raise _Timeout()


def symbolic_match(model_expr, true_expr, timeout: int = 20) -> bool:
    """SRBench-style check: after rounding constants to 3 decimals, model - truth or
    model / truth simplifies to a constant."""
    import sympy as sp
    from emlkit.symbolic import round_floats
    try:  # the alarm-based timeout only works in the main thread
        old = signal.signal(signal.SIGALRM, _alarm)
        signal.alarm(timeout)
    except ValueError:
        old = None
    try:
        names = {s.name: s for s in true_expr.free_symbols}
        m = model_expr.xreplace({s: names.get(s.name, s) for s in model_expr.free_symbols})
        m = round_floats(m, 3)
        diff = sp.simplify(m - true_expr)
        if diff == 0 or not diff.free_symbols:
            return True
        ratio = sp.simplify(m / true_expr)
        return (not ratio.free_symbols) and ratio != 0
    except _Timeout:
        return False
    except Exception:
        return False
    finally:
        if old is not None:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old)


def run_one(problem_name: str, method: str, seed: int, noise: float, n_train: int) -> dict:
    import torch
    torch.set_num_threads(1)
    from feynman_data import load_problems
    from baselines import BASELINES
    from emlkit import EMLRegressor

    prob = {p.name: p for p in load_problems()}[problem_name]
    rng = np.random.default_rng(zlib.crc32(f"{problem_name}-{seed}".encode()))
    X, y_clean = prob.sample(n_train, rng)
    y = y_clean + (rng.normal(0, noise * np.std(y_clean), len(y_clean)) if noise > 0 else 0.0)
    Xi, yi = prob.sample(2000, rng)
    Xo, yo = prob.sample(2000, rng, ood=True)

    out = dict(problem=problem_name, method=method, seed=seed, noise=noise, n_train=n_train,
               n_vars=prob.n_vars, has_trig=prob.has_trig, truth=str(prob.expr),
               dev=problem_name in DEV_PROBLEMS)
    t0 = time.time()
    try:
        if method.startswith("EML"):
            model = EMLRegressor(feature_names=prob.variables, random_state=seed,
                                 **EML_VARIANTS[method])
        else:
            model = BASELINES[method](seed)
        model.fit(X, y)
        out["time"] = time.time() - t0
        with np.errstate(all="ignore"):
            pi_, po_ = model.predict(Xi), model.predict(Xo)
        out.update(r2_id=r2(yi, pi_), r2_ood=r2(yo, po_), nmse_id=nmse(yi, pi_), nmse_ood=nmse(yo, po_))
        if method.startswith("EML"):
            out.update(formula=model.formula(6), size=model.complexity_, eml_size=model.size_,
                       arch=model.best_.arch, val_nmse=model.val_nmse_)
            out["symbolic"] = symbolic_match(model.expr_, prob.expr)
        elif method == "GPLearn":
            import sympy as sp
            from gplearn_sympy import gplearn_to_sympy
            expr = gplearn_to_sympy(model.model._program, prob.variables)
            out.update(formula=str(expr), size=model.size)
            out["symbolic"] = symbolic_match(expr, prob.expr)
    except Exception as e:  # keep going; record the failure
        out.update(error=repr(e), trace=traceback.format_exc()[-2000:], time=time.time() - t0)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--methods", default="EML,MLP,GBM,Linear,GPLearn")
    ap.add_argument("--problems", default="all", help="all | notrig | dev | comma list")
    ap.add_argument("--seeds", default="0")
    ap.add_argument("--noise", default="0.0")
    ap.add_argument("--n-train", type=int, default=1000)
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    ap.add_argument("--out", default=str(ROOT / "results" / "feynman.jsonl"))
    args = ap.parse_args()

    from feynman_data import load_problems
    probs = load_problems()
    if args.problems == "dev":
        names = DEV_PROBLEMS
    elif args.problems == "notrig":
        names = [p.name for p in probs if not p.has_trig]
    elif args.problems == "all":
        names = [p.name for p in probs]
    else:
        names = args.problems.split(",")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out_path.exists():
        for line in out_path.read_text().splitlines():
            r = json.loads(line)
            done.add((r["problem"], r["method"], r["seed"], r["noise"], r["n_train"]))

    tasks = [(p, m, int(s), float(nz), args.n_train)
             for nz in args.noise.split(",") for s in args.seeds.split(",")
             for m in args.methods.split(",") for p in names]
    tasks = [t for t in tasks if t not in done]
    # slow methods first for better load balancing
    order = {"GPLearn": 0, "EML": 1}
    tasks.sort(key=lambda t: order.get(t[1], 2))
    print(f"{len(tasks)} runs to do ({len(done)} already done) with {args.jobs} workers", flush=True)

    with ProcessPoolExecutor(max_workers=args.jobs) as ex, open(out_path, "a") as fh:
        futs = {ex.submit(run_one, *t): t for t in tasks}
        for i, fut in enumerate(as_completed(futs), 1):
            res = fut.result()
            fh.write(json.dumps(res) + "\n")
            fh.flush()
            msg = res.get("error") or (f"R2 id {res['r2_id']:.6f} ood {res['r2_ood']:.4g}"
                                       + (f" sym={res.get('symbolic')}" if "symbolic" in res else ""))
            print(f"[{i}/{len(tasks)}] {res['problem']:20s} {res['method']:8s} s{res['seed']} "
                  f"noise {res['noise']}: {msg}  ({res['time']:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
