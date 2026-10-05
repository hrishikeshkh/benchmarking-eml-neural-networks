"""Pure EML expression trees, following the grammar  S -> 1 | x | eml(S, S).

eml(a, b) = exp(a) - ln(b) with the principal branch of the complex logarithm
(Odrzywolek, 2026).  This module provides

* ``compile_expr``  - SymPy expression -> pure EML tree (only ``eml``, ``1`` and variables);
* ``evaluate``      - vectorised complex evaluation of a tree;
* ``verify``        - numerical check of a compiled tree against its source expression;
* ``search``        - exhaustive bottom-up search for the smallest tree matching a target.

The constructions used by the compiler are simple textbook identities, *not* the shortest
known ones; they exist to make EML programs easy to produce and check.  Intermediate
values are complex, ln(0) is handled with IEEE arithmetic (ln 0 = -inf, exp(-inf) = 0).
"""
from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import sympy as sp


# --------------------------------------------------------------------------- nodes
@dataclass(frozen=True)
class Node:
    """Base class of EML tree nodes (immutable, hashable => structural sharing)."""


@dataclass(frozen=True)
class One(Node):
    def __repr__(self) -> str:
        return "1"


@dataclass(frozen=True)
class Var(Node):
    name: str

    def __repr__(self) -> str:
        return self.name


@dataclass(frozen=True)
class EML(Node):
    left: Node
    right: Node
    _hash: int = field(init=False, repr=False, compare=False, default=0)

    def __post_init__(self):
        object.__setattr__(self, "_hash", hash((self.left, self.right)))

    def __hash__(self) -> int:
        return self._hash

    def __repr__(self) -> str:
        return f"eml({self.left!r}, {self.right!r})"


ONE = One()


def eml(a: Node, b: Node) -> Node:
    return EML(a, b)


# ------------------------------------------------------------------ inspection
def leaves(node: Node) -> int:
    """Number of leaves of the tree (``eml`` nodes = leaves - 1)."""
    return _leaves(node)


@lru_cache(maxsize=None)
def _leaves(node: Node) -> int:
    if isinstance(node, EML):
        return _leaves(node.left) + _leaves(node.right)
    return 1


@lru_cache(maxsize=None)
def depth(node: Node) -> int:
    if isinstance(node, EML):
        return 1 + max(depth(node.left), depth(node.right))
    return 0


def dag_size(node: Node) -> int:
    """Number of distinct ``eml`` nodes when identical subtrees are shared."""
    seen = set()
    stack = [node]
    while stack:
        n = stack.pop()
        if isinstance(n, EML) and n not in seen:
            seen.add(n)
            stack += [n.left, n.right]
    return len(seen)


def to_string(node: Node) -> str:
    return repr(node)


def parse(text: str) -> Node:
    """Inverse of ``to_string``: ``"eml(1, eml(x, 1))"`` -> tree."""
    tokens = text.replace("(", " ( ").replace(")", " ) ").replace(",", " , ").split()
    pos = 0

    def node() -> Node:
        nonlocal pos
        tok = tokens[pos]
        pos += 1
        if tok == "eml":
            assert tokens[pos] == "("; pos += 1
            a = node()
            assert tokens[pos] == ","; pos += 1
            b = node()
            assert tokens[pos] == ")"; pos += 1
            return EML(a, b)
        if tok == "1":
            return ONE
        return Var(tok)

    out = node()
    if pos != len(tokens):
        raise ValueError(f"trailing tokens in {text!r}")
    return out


# ------------------------------------------------------------------ evaluation
def evaluate(node: Node, env: Dict[str, Union[float, np.ndarray]]) -> np.ndarray:
    """Evaluate a tree on complex128 arrays (shared subtrees are computed once)."""
    memo: Dict[Node, np.ndarray] = {}
    shape = np.broadcast(*[np.asarray(v) for v in env.values()]).shape if env else ()

    def ev(n: Node) -> np.ndarray:
        if n in memo:
            return memo[n]
        if isinstance(n, One):
            v = np.ones(shape, dtype=complex)
        elif isinstance(n, Var):
            v = np.broadcast_to(np.asarray(env[n.name], dtype=complex), shape)
        else:
            a, b = ev(n.left), ev(n.right)
            with np.errstate(all="ignore"):
                v = np.exp(a) - np.log(b)
        memo[n] = v
        return v

    # iterative post-order to avoid recursion limits on deep trees
    stack: List[Tuple[Node, bool]] = [(node, False)]
    while stack:
        n, done = stack.pop()
        if n in memo:
            continue
        if isinstance(n, EML) and not done:
            stack.append((n, True))
            stack.append((n.right, False))
            stack.append((n.left, False))
        else:
            ev(n)
    return memo[node]


# ------------------------------------------------------------- constructions
# All identities hold for the principal branch on the domains noted.
@lru_cache(maxsize=None)
def exp_(a: Node) -> Node:              # e^a = eml(a, 1)
    return EML(a, ONE)


@lru_cache(maxsize=None)
def ln_(a: Node) -> Node:               # ln a = eml(1, eml(eml(1, a), 1))      (a not on R<0)
    return EML(ONE, EML(EML(ONE, a), ONE))


E_ = EML(ONE, ONE)                      # e = eml(1, 1)
ZERO = ln_(ONE)                         # 0 = ln 1


@lru_cache(maxsize=None)
def sub_(a: Node, b: Node) -> Node:     # a - b = eml(ln a, e^b)                (|Im b| < pi)
    return EML(ln_(a), exp_(b))


@lru_cache(maxsize=None)
def neg_(b: Node) -> Node:              # -b = eml(ln(1 - b), e) = (1 - b) - 1
    return EML(ln_(sub_(ONE, b)), E_)


@lru_cache(maxsize=None)
def add_(a: Node, b: Node) -> Node:     # a + b = a - (-b)
    return sub_(a, neg_(b))


@lru_cache(maxsize=None)
def mul_(a: Node, b: Node) -> Node:     # a * b = exp(ln a + ln b)
    return exp_(add_(ln_(a), ln_(b)))


@lru_cache(maxsize=None)
def div_(a: Node, b: Node) -> Node:     # a / b = exp(ln a - ln b)
    return exp_(sub_(ln_(a), ln_(b)))


@lru_cache(maxsize=None)
def pow_(a: Node, b: Node) -> Node:     # a ^ b = exp(b ln a)
    return exp_(mul_(b, ln_(a)))


@lru_cache(maxsize=None)
def integer_(n: int) -> Node:
    if n == 0:
        return ZERO
    if n < 0:
        return neg_(integer_(-n))
    if n == 1:
        return ONE
    half = integer_(n // 2)
    out = add_(half, half)
    return add_(out, ONE) if n % 2 else out


# ln(-1) = i*pi.  Which sign of i comes out depends on how signed zeros propagate through
# the branch cut (NumPy gives +i*pi); pi below is independent of that choice, and sin/cos
# are invariant under i -> -i, so only the constant I itself depends on the convention.
_L_NEG1 = ln_(neg_(ONE))                                 # i*pi
I_ = exp_(div_(_L_NEG1, integer_(2)))                    # exp(i*pi/2) = i
PI_ = div_(_L_NEG1, I_)                                  # (i*pi)/i = pi


def compile_expr(expr: Union[str, sp.Expr], variables: Optional[Sequence[str]] = None) -> Node:
    """Compile an elementary SymPy expression into a pure EML tree."""
    if isinstance(expr, str):
        expr = sp.sympify(expr)
    return _compile(sp.sympify(expr))


def _compile(e: sp.Expr) -> Node:
    if e.is_Symbol:
        return Var(e.name)
    if e.is_Integer:
        return integer_(int(e))
    if e.is_Rational:
        return div_(integer_(int(e.p)), integer_(int(e.q)))
    if e is sp.E:
        return E_
    if e is sp.pi:
        return PI_
    if e is sp.I:
        return I_
    if e.is_Float:
        frac = sp.nsimplify(e, rational=True)
        return _compile(frac)
    if isinstance(e, sp.Add):
        terms = list(e.args)
        out = _compile(terms[0])
        for t in terms[1:]:
            c, rest = t.as_coeff_Mul()
            if c < 0:
                out = sub_(out, _compile(-t))
            else:
                out = add_(out, _compile(t))
        return out
    if isinstance(e, sp.Mul):
        c, rest = e.as_coeff_Mul()
        if c == -1:
            return neg_(_compile(rest))
        num, den = sp.fraction(e)
        if den != 1:
            return div_(_compile(num), _compile(den))
        args = list(e.args)
        out = _compile(args[0])
        for a in args[1:]:
            out = mul_(out, _compile(a))
        return out
    if isinstance(e, sp.Pow):
        base, ex = e.args
        if ex == -1:
            return div_(ONE, _compile(base))
        if base is sp.E:
            return exp_(_compile(ex))
        return pow_(_compile(base), _compile(ex))
    if isinstance(e, sp.exp):
        return exp_(_compile(e.args[0]))
    if isinstance(e, sp.log):
        return ln_(_compile(e.args[0]))
    x = e.args[0] if e.args else None
    rewrites = {
        sp.sin: lambda x: (sp.exp(sp.I * x) - sp.exp(-sp.I * x)) / (2 * sp.I),
        sp.cos: lambda x: (sp.exp(sp.I * x) + sp.exp(-sp.I * x)) / 2,
        sp.tan: lambda x: -sp.I * (sp.exp(sp.I * x) - sp.exp(-sp.I * x)) / (sp.exp(sp.I * x) + sp.exp(-sp.I * x)),
        sp.sinh: lambda x: (sp.exp(x) - sp.exp(-x)) / 2,
        sp.cosh: lambda x: (sp.exp(x) + sp.exp(-x)) / 2,
        sp.tanh: lambda x: (sp.exp(x) - sp.exp(-x)) / (sp.exp(x) + sp.exp(-x)),
        sp.asin: lambda x: -sp.I * sp.log(sp.I * x + sp.sqrt(1 - x ** 2)),
        sp.acos: lambda x: sp.pi / 2 + sp.I * sp.log(sp.I * x + sp.sqrt(1 - x ** 2)),
        sp.atan: lambda x: sp.I / 2 * (sp.log(1 - sp.I * x) - sp.log(1 + sp.I * x)),
    }
    for f, rw in rewrites.items():
        if isinstance(e, f):
            return _compile(rw(x))
    raise NotImplementedError(f"cannot compile {e!r} ({type(e).__name__})")


def verify(node: Node, expr: Union[str, sp.Expr], n: int = 64, low: float = 0.5,
           high: float = 3.0, seed: int = 0) -> float:
    """Max relative error between a tree and ``expr`` at random positive real points."""
    expr = sp.sympify(expr)
    syms = sorted(expr.free_symbols, key=lambda s: s.name)
    rng = np.random.default_rng(seed)
    env = {s.name: rng.uniform(low, high, n) for s in syms}
    with np.errstate(all="ignore"):
        want = np.asarray(sp.lambdify(syms, expr, "numpy")(*[v.astype(complex) for v in env.values()]),
                          dtype=complex) * np.ones(n)
    got = evaluate(node, env) if env else evaluate(node, {"_": np.zeros(n)})
    return float(np.max(np.abs(got - want) / (1 + np.abs(want))))


# ----------------------------------------------------------------------- search
def search(
    target: Union[str, sp.Expr, Callable[..., np.ndarray]],
    variables: Sequence[str] = ("x",),
    max_leaves: int = 7,
    n_points: int = 12,
    tol: float = 1e-9,
    seed: int = 0,
    max_per_level: int = 200_000,
) -> Optional[Node]:
    """Smallest pure EML tree (fewest leaves) equal to ``target`` at random points.

    Bottom-up enumeration over leaves {1, variables}; trees whose values coincide at the
    sample points are deduplicated, so each level only keeps distinct functions.  Returns
    ``None`` if nothing is found within ``max_leaves`` leaves.
    """
    rng = np.random.default_rng(seed)
    # complex sample points avoid accidental coincidences on the real line
    env = {v: rng.uniform(0.6, 2.4, n_points) + 1j * rng.uniform(-0.3, 0.3, n_points) for v in variables}
    if callable(target) and not isinstance(target, sp.Basic):
        want = np.asarray(target(*env.values()), dtype=complex)
    else:
        expr = sp.sympify(target)
        syms = [sp.Symbol(v) for v in variables]
        want = np.asarray(sp.lambdify(syms, expr, "numpy")(*env.values()), dtype=complex) * np.ones(n_points)

    def key(vals: np.ndarray) -> Optional[bytes]:
        if not np.all(np.isfinite(vals)) or np.max(np.abs(vals)) > 1e12:
            return None
        return np.round(vals, 8).tobytes()

    levels: List[Tuple[List[Node], np.ndarray]] = [([], np.zeros((0, n_points), complex))]
    seen = set()
    base_nodes = [ONE] + [Var(v) for v in variables]
    base_vals = [np.ones(n_points, complex)] + [env[v] for v in variables]
    lvl_nodes, lvl_vals = [], []
    for nd, vv in zip(base_nodes, base_vals):
        k = key(vv)
        if k not in seen:
            seen.add(k); lvl_nodes.append(nd); lvl_vals.append(vv)
            if np.max(np.abs(vv - want) / (1 + np.abs(want))) < tol:
                return nd
    levels.append((lvl_nodes, np.array(lvl_vals)))

    for k in range(2, max_leaves + 1):
        new_nodes, new_vals = [], []
        for i in range(1, k):
            ln_nodes, ln_vals = levels[i]
            rn_nodes, rn_vals = levels[k - i]
            if not ln_nodes or not rn_nodes:
                continue
            with np.errstate(all="ignore"):
                log_r = np.log(rn_vals)                      # (m_r, P)
                exp_l = np.exp(ln_vals)                      # (m_l, P)
            for a_idx in range(len(ln_nodes)):
                with np.errstate(all="ignore"):
                    vals = exp_l[a_idx][None, :] - log_r    # (m_r, P)
                err = np.max(np.abs(vals - want) / (1 + np.abs(want)), axis=1)
                hit = np.flatnonzero(err < tol)
                if hit.size:
                    return EML(ln_nodes[a_idx], rn_nodes[hit[0]])
                for b_idx in range(len(rn_nodes)):
                    kk = key(vals[b_idx])
                    if kk is None or kk in seen:
                        continue
                    seen.add(kk)
                    new_nodes.append(EML(ln_nodes[a_idx], rn_nodes[b_idx]))
                    new_vals.append(vals[b_idx])
                    if len(new_nodes) >= max_per_level:
                        break
        levels.append((new_nodes, np.array(new_vals) if new_vals else np.zeros((0, n_points), complex)))
    return None
