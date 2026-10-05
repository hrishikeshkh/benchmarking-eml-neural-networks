"""Export a trained EMLNet restart to a closed-form SymPy expression."""
from __future__ import annotations

import math
from typing import Sequence

import numpy as np
import sympy as sp
import torch

from .net import EMLNet


def _coef(value: float, fixed: bool, digits: int = 15) -> sp.Expr:
    if fixed:
        return sp.nsimplify(value, rational=True)
    return sp.Float(value, digits)


@torch.no_grad()
def net_to_sympy(
    net: EMLNet,
    r: int,
    symbols: Sequence[sp.Symbol],
    x_scale: np.ndarray,
    y_scale: float,
    X_scaled: torch.Tensor | None = None,
) -> sp.Expr:
    """Closed form of restart ``r`` in the *original* (unscaled) units.

    ``X_scaled`` (training inputs, already divided by ``x_scale``) decides whether every
    log argument is positive on the data; if not, ln|c| is emitted, which is what the
    protected log computes.
    """
    live = net.live_units(r)
    xs = [s / sp.nsimplify(float(k)) for s, k in zip(symbols, x_scale)]
    pos_in = [True] * len(xs)
    c_pos = None
    if X_scaled is not None:
        pos_in = [bool((X_scaled[:, j] > 0).all()) for j in range(X_scaled.shape[1])]
        _, internals = net(X_scaled, return_internals=True)
        c_pos = [None if c is None else (c[r] > 0).all(dim=0) for _, c in internals]

    def log_(arg, positive):
        return sp.log(arg) if positive else sp.log(sp.Abs(arg))

    z = list(xs)
    if net.log_inputs:
        z += [log_(x, p) for x, p in zip(xs, pos_in)]

    def affine(w, b, fw, fb):
        e = _coef(float(b), bool(fb))
        for f in range(len(z)):
            if w[f] != 0:
                e += _coef(float(w[f]), bool(fw[f])) * z[f]
        return e

    for l, (typ, width) in enumerate(net.layers):
        new = []
        for u in range(width):
            if not live[l][u]:
                new.append(sp.Integer(0))  # placeholder, never referenced
                continue
            h = sp.Integer(0)
            if typ in ("E", "M"):
                a = affine(net.eff(f"wa{l}")[r, u], net.eff(f"ba{l}")[r, u],
                           net.fixed(f"wa{l}")[r, u], net.fixed(f"ba{l}")[r, u])
                h += sp.exp(a)
            if typ in ("L", "M"):
                c = affine(net.eff(f"wc{l}")[r, u], net.eff(f"bc{l}")[r, u],
                           net.fixed(f"wc{l}")[r, u], net.fixed(f"bc{l}")[r, u])
                lc = log_(c, c_pos is None or bool(c_pos[l][u]))
                h += lc if typ == "L" else -lc
            new.append(h)
        z += new

    wo, bo = net.eff("wo")[r], net.eff("bo")[r]
    out = sp.Float(float(bo), 15)
    for j in range(wo.shape[0]):
        if wo[j] != 0:
            out += sp.Float(float(wo[j]), 15) * z[j]
    if net.out_exp:
        out = sp.Integer(int(net.out_sign)) * sp.exp(out)
    return out * sp.nsimplify(float(y_scale))


def fold_numeric(expr: sp.Expr) -> sp.Expr:
    """Collapse numeric sub-products that contain a Float (e.g. 0.4082*sqrt(6)) into one Float."""
    if expr.is_Atom:
        return expr
    args = [fold_numeric(a) for a in expr.args]
    if expr.is_Mul:
        num = [a for a in args if not a.free_symbols]
        rest = [a for a in args if a.free_symbols]
        if len(num) > 1 and any(a.has(sp.Float) for a in num):
            return sp.Mul(sp.Float(sp.Mul(*num).evalf(15), 15), *rest)
    return expr.func(*args)


def simplify_expr(expr: sp.Expr, max_ops: int = 300) -> sp.Expr:
    """Cheap, robust simplification: split exp(a+b), fold exp(k*log x) into powers."""
    e = sp.expand(expr, power_exp=True, mul=False, multinomial=False, log=False, power_base=False)
    e = sp.powsimp(e, force=True)
    if sp.count_ops(e) <= max_ops:
        try:
            e2 = sp.simplify(e)
            if sp.count_ops(e2) <= sp.count_ops(e):
                e = e2
        except Exception:  # pragma: no cover - sympy edge cases
            pass
    return fold_numeric(e)


def clean_constants(expr: sp.Expr, zero_abs: float, rel: float = 1e-7,
                    max_den: int = 12) -> sp.Expr:
    """Replace Floats that are numerically 0 (|c| < zero_abs) by 0 and Floats within ``rel``
    of a rational p/q with q <= max_den by that rational.  Only meant for (near-)exact fits."""
    reps = {}
    for f in expr.atoms(sp.Float):
        v = float(f)
        if not math.isfinite(v):
            continue
        if abs(v) < zero_abs:
            reps[f] = sp.Integer(0)
            continue
        for q in range(1, max_den + 1):
            p = round(v * q)
            if p != 0 and abs(v - p / q) <= rel * abs(v):
                reps[f] = sp.Rational(p, q)
                break
    return expr.xreplace(reps)


def round_floats(expr: sp.Expr, sig: int = 4) -> sp.Expr:
    """Round every Float in ``expr`` to ``sig`` significant digits (for display)."""
    reps = {}
    for f in expr.atoms(sp.Float):
        v = float(f)
        if v != 0:
            reps[f] = sp.Float(float(f"{v:.{sig}g}"), sig)
    return expr.xreplace(reps)


def expr_complexity(expr: sp.Expr) -> int:
    """Number of nodes in the SymPy expression tree (SRBench-style model size)."""
    return sum(1 for _ in sp.preorder_traversal(expr))
