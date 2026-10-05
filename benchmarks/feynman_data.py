"""Feynman symbolic-regression problems (Udrescu & Tegmark 2020) as distributed by PMLB.

We read the ground-truth formula and the variable ranges from the PMLB metadata files
(``data/raw/feynman_meta/*.yaml``) and *re-sample* the data ourselves.  This lets us draw
an out-of-distribution test set from outside the training box, which the fixed PMLB
tables do not provide.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import sympy as sp
import yaml

META_DIR = Path(__file__).resolve().parents[1] / "data" / "raw" / "feynman_meta"
TRIG = (sp.sin, sp.cos, sp.tan, sp.asin, sp.acos, sp.atan, sp.tanh, sp.sinh, sp.cosh)


@dataclass
class FeynmanProblem:
    name: str
    target: str
    formula: str
    variables: List[str]
    ranges: List[Tuple[float, float]]
    expr: sp.Expr
    symbols: List[sp.Symbol]

    @property
    def n_vars(self) -> int:
        return len(self.variables)

    @property
    def has_trig(self) -> bool:
        return any(self.expr.has(f) for f in TRIG)

    def f(self, X: np.ndarray) -> np.ndarray:
        fn = sp.lambdify(self.symbols, self.expr, "numpy")
        with np.errstate(all="ignore"):
            out = fn(*[X[:, i].astype(complex) for i in range(X.shape[1])])
        out = np.broadcast_to(np.asarray(out, dtype=complex), (X.shape[0],))
        y = np.where(np.abs(out.imag) < 1e-9 * (1 + np.abs(out.real)), out.real, np.nan)
        return y

    def sample(self, n: int, rng: np.random.Generator, ood: bool = False,
               noise: float = 0.0, y_std: float | None = None) -> Tuple[np.ndarray, np.ndarray]:
        """Uniform samples inside the box (``ood=False``) or in [hi, hi + (hi - lo)]."""
        X_parts, y_parts, got = [], [], 0
        for _ in range(50):
            m = 2 * (n - got) + 16
            cols = []
            for lo, hi in self.ranges:
                a, b = (hi, 2 * hi - lo) if ood else (lo, hi)
                cols.append(rng.uniform(a, b, m))
            Xc = np.stack(cols, 1)
            yc = self.f(Xc)
            ok = np.isfinite(yc)
            X_parts.append(Xc[ok]); y_parts.append(yc[ok]); got += int(ok.sum())
            if got >= n:
                break
        X, y = np.concatenate(X_parts)[:n], np.concatenate(y_parts)[:n]
        if noise > 0:
            s = y_std if y_std is not None else np.std(y)
            y = y + rng.normal(0.0, noise * s, size=y.shape)
        return X, y


_RANGE_RE = re.compile(r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s+in\s+\[([^,]+),([^\]]+)\]")


def _parse(path: Path) -> FeynmanProblem:
    meta = yaml.safe_load(path.read_text())
    lines = [ln.strip() for ln in meta["description"].splitlines()]
    formula_line = next(ln for ln in lines if "=" in ln and " in [" not in ln)
    target, rhs = [s.strip() for s in formula_line.split("=", 1)]
    variables, ranges = [], []
    for ln in lines:
        m = _RANGE_RE.match(ln)
        if m:
            variables.append(m.group(1))
            ranges.append((float(m.group(2)), float(m.group(3))))
    feats = [f["name"] for f in meta["features"]]
    assert feats == variables, (path.name, feats, variables)
    symbols = [sp.Symbol(v, positive=True) for v in variables]
    local: Dict[str, object] = {v: s for v, s in zip(variables, symbols)}
    local.update({"pi": sp.pi, "ln": sp.log, "arcsin": sp.asin, "exp": sp.exp, "sqrt": sp.sqrt,
                  "sin": sp.sin, "cos": sp.cos, "tanh": sp.tanh})
    expr = sp.sympify(rhs, locals=local)
    return FeynmanProblem(meta["dataset"], target, rhs, variables, ranges, expr, symbols)


def load_problems() -> List[FeynmanProblem]:
    return sorted((_parse(p) for p in META_DIR.glob("feynman_*.yaml")), key=lambda p: p.name)


if __name__ == "__main__":
    probs = load_problems()
    print(len(probs), "problems;", sum(not p.has_trig for p in probs), "without trig")
    rng = np.random.default_rng(0)
    for p in probs:
        X, y = p.sample(200, rng)
        Xo, yo = p.sample(200, rng, ood=True)
        print(f"{p.name:20s} d={p.n_vars} trig={int(p.has_trig)} n={len(y)} n_ood={len(yo)} "
              f"y~[{np.min(y):.3g},{np.max(y):.3g}]  {p.expr}")
