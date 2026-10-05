"""Scikit-learn style symbolic regressor built on batched EML networks.

``EMLRegressor.fit`` runs an *Occam curriculum*: a sequence of increasingly deep EML
architectures (``"E1"``, ``"E2"``, ``"E1-L1-E1"``, ...).  For every architecture a batch of
restarts is trained with the pipeline below, and every restart becomes a candidate model.
The curriculum stops early once a candidate fits the validation data to numerical precision.
The final model is the candidate with the lowest Bayesian information criterion (BIC).

Per-architecture pipeline (all restarts in parallel):

1. dense training with Adam.  The affine read-out is not trained by gradient descent but
   solved exactly by ridge least squares at every step (variable projection), so gradients
   only shape the EML features.  In the second half an annealed penalty pulls argument
   weights towards multiples of ``grid`` (default 1/2: exponents such as 2, -1, 1/2);
2. pruning: argument weights below a magnitude threshold and read-out terms whose
   contribution is below a fraction of std(y) are removed, followed by fine-tuning;
3. snapping: argument weights close to a multiple of ``grid`` are frozen at that exact value,
   followed by fine-tuning;
4. polishing of the remaining free constants.

Each structural edit is accepted per restart only if the validation NMSE stays below
``ref * (1 + rel_tol) + abs_tol``, with ``ref`` the NMSE of that restart after dense training.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence

import numpy as np
import sympy as sp
import torch
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_is_fitted

from .lm import lm_polish
from .net import EMLNet
from .symbolic import clean_constants, expr_complexity, net_to_sympy, round_floats, simplify_expr

# Stages ending in '*' put an EML node exp(.) at the output and solve its whole argument by
# least squares on ln|y| (log-space variable projection); they require y of constant sign.
# Stages ending in '+' let the (linear) read-out see every feature.
DEFAULT_CURRICULUM = (
    "*",             # power law with exponential factors   C prod x^a exp(b . x)
    "E1*",           # ... times exp(monomial)               n0 exp(-m g x / k T)
    "L1*",           # ... times |affine|^p                  k (T2 - T1) A / d,  ln(V2/V1)
    "E2",            # sum of two monomials                  G m1 m2 (1/r2 - 1/r1)
    "E1-L1*",        # monomial times (1 + monomial)^p       m0 / sqrt(1 - v^2/c^2)
    "E3",            # sum of three monomials                x1 y1 + x2 y2 + x3 y3
    "E2-L1*",        # monomial times (mono + mono)^p        (u + v) / (1 + u v / c^2)
    "E3-L1*",        # e.g. sqrt(x^2 + y^2 + z^2)
    "E1-E1-L1*",     # e.g. 1 / (exp(h w / k T) - 1)
    "L3-E3-L1*",     # e.g. 1 / ((x2-x1)^2 + (y2-y1)^2 + (z2-z1)^2)
    "E4-L2-E2+",     # generic wide net, read-out over all features
)

# The same depth budget without log-space projection (used in the ablation study).
LINEAR_OUTPUT_CURRICULUM = (
    "E1", "E1-E1", "L1-E1", "E2", "E1-L1-E1", "E3", "E2-L1-E1", "E3-L1-E1", "E1-E1-L1-E1",
    "L3-E3-L1-E1", "E4-L2-E2+",
)

NMSE_FLOOR = 1e-14


def _pow10_scale(v: np.ndarray) -> np.ndarray:
    rms = np.sqrt(np.mean(np.asarray(v, dtype=float) ** 2, axis=0))
    rms = np.where(rms > 0, rms, 1.0)
    return 10.0 ** np.round(np.log10(rms))


@dataclass
class Candidate:
    stage: int
    arch: str
    restart: int
    val_nmse: float
    size: int
    bic: float
    expr: Optional[sp.Expr] = None


class EMLRegressor(BaseEstimator, RegressorMixin):
    """Interpretable regression with trainable Exp-Minus-Log networks.

    After ``fit``: ``expr_`` is the selected closed-form model (SymPy), ``formula()`` a
    readable string, ``candidates_`` every trained candidate and ``pareto_`` the
    accuracy/size front.
    """

    def __init__(
        self,
        curriculum: Sequence[str] = DEFAULT_CURRICULUM,
        n_restarts: int = 24,
        halving: int = 2,
        dense_steps: int = 600,
        lr: float = 0.03,
        quant: float = 1e-3,
        prune_thresholds: Sequence[float] = (0.05, 0.15, 0.3),
        readout_rel_thresholds: Sequence[float] = (1e-3, 5e-3, 3e-2),
        snap_deltas: Sequence[float] = (0.03, 0.1, 0.2),
        grid: float = 0.5,
        finetune_steps: int = 80,
        polish_steps: int = 200,
        lm_iters: int = 12,
        rel_tol: float = 0.1,
        abs_tol: float = 1e-8,
        exact_tol: float = 1e-12,
        ridge: float = 1e-10,
        varpro: bool = True,
        x_scale: str = "auto",
        val_fraction: float = 0.25,
        log_inputs: bool = True,
        exp_sees_raw: bool = False,
        init: str = "sparse",
        exp_clip: float = 30.0,
        max_train_samples: int = 512,
        max_val_samples: int = 1000,
        feature_names: Optional[Sequence[str]] = None,
        random_state: int = 0,
        verbose: int = 0,
    ):
        self.curriculum = curriculum
        self.n_restarts = n_restarts
        self.halving = halving
        self.dense_steps = dense_steps
        self.lr = lr
        self.quant = quant
        self.prune_thresholds = prune_thresholds
        self.readout_rel_thresholds = readout_rel_thresholds
        self.snap_deltas = snap_deltas
        self.grid = grid
        self.finetune_steps = finetune_steps
        self.polish_steps = polish_steps
        self.lm_iters = lm_iters
        self.rel_tol = rel_tol
        self.abs_tol = abs_tol
        self.exact_tol = exact_tol
        self.ridge = ridge
        self.varpro = varpro
        self.x_scale = x_scale
        self.val_fraction = val_fraction
        self.log_inputs = log_inputs
        self.exp_sees_raw = exp_sees_raw
        self.init = init
        self.exp_clip = exp_clip
        self.max_train_samples = max_train_samples
        self.max_val_samples = max_val_samples
        self.feature_names = feature_names
        self.random_state = random_state
        self.verbose = verbose

    # ------------------------------------------------------------- read-out
    def _solve_readout(self, z: torch.Tensor, y: torch.Tensor):
        """Ridge least-squares read-out for every restart.  z: (R, N, F) -> (wo, bo).

        Pruned read-out entries are 0 and snapped (fixed) entries keep their exact value; only
        the free entries and the bias are solved for.
        """
        net = self.net_
        R, N, F = z.shape
        z = torch.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0)
        active, fixed = net.active("wo"), net.fixed("wo")
        free = active & ~fixed
        w_fixed = net.value("wo") * (fixed & active)
        target = y.view(1, N) - torch.einsum("rnf,rf->rn", z, w_fixed)          # (R, N)
        zf = z * free[:, None, :]
        s = zf.pow(2).mean(1).sqrt().clamp_min(1e-12)                            # (R, F)
        Z = torch.cat([zf / s[:, None, :], torch.ones(R, N, 1, dtype=z.dtype)], dim=-1)
        A = Z.transpose(1, 2) @ Z / N + self.ridge * torch.eye(F + 1, dtype=z.dtype)
        b = (Z.transpose(1, 2) @ target.unsqueeze(-1)) / N
        try:
            sol = torch.linalg.solve(A, b).squeeze(-1)
        except RuntimeError:  # pragma: no cover - singular despite ridge
            sol = (torch.linalg.pinv(A) @ b).squeeze(-1)
        sol = torch.nan_to_num(sol, nan=0.0, posinf=0.0, neginf=0.0)
        wo = torch.where(fixed & active, w_fixed, sol[:, :F] / s * free)
        return wo, sol[:, F]

    def _target(self):
        """Training target of the read-out (ln|y| for log-space stages) and its variance."""
        if self.net_.out_exp:
            return self._lyt, self._lyvar
        return self._yt, self._yvar

    @torch.no_grad()
    def _refresh(self) -> torch.Tensor:
        """Store the least-squares read-out in the net; return per-restart validation NMSE
        (always measured on y itself, so all stages are comparable)."""
        net = self.net_
        if self.varpro:
            _, zt = net(self._Xt, return_features=True)
            wo, bo = self._solve_readout(zt, self._target()[0])
            net.wo.copy_(wo * net.active("wo"))
            net.bo.copy_(bo)
        pred = net(self._Xv)
        m = ((pred - self._yv) ** 2).mean(1) / self._yvar
        return torch.nan_to_num(m, nan=math.inf, posinf=math.inf)

    # ------------------------------------------------------------ optimizer
    def _optimize(self, steps: int, lr: float, quant: float = 0.0, keep_best: bool = True) -> None:
        net = self.net_
        if steps <= 0:
            return
        params = [net.param(n) for n in net.names if not (self.varpro and n in ("wo", "bo"))]
        if not params:  # e.g. the pure power-law stage: the read-out is the whole model
            self._refresh()
            return
        opt = torch.optim.Adam(params, lr=lr)
        best = self._refresh()
        best_state = net.get_state() if keep_best else None
        for t in range(steps):
            frac = t / steps
            for g in opt.param_groups:
                g["lr"] = lr * 0.5 * (1 + math.cos(math.pi * frac))
            rho = quant * max(0.0, 2 * frac - 1)  # annealed in the second half
            if self.varpro:
                tgt, var = self._target()
                _, z = net(self._Xt, return_features=True)
                wo, bo = self._solve_readout(z, tgt)
                z = torch.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0)
                pred = torch.einsum("rnf,rf->rn", z, wo) + bo[:, None]
            else:
                tgt, var = self._yt, self._yvar
                pred = net(self._Xt)
            loss = ((pred - tgt) ** 2).mean(1) / var
            if rho > 0:
                loss = loss + rho * net.quant_penalty(self.grid)
            loss = torch.where(torch.isfinite(loss), loss, torch.zeros_like(loss))
            opt.zero_grad(set_to_none=True)
            loss.sum().backward()
            net.clip_grads_(1.0)
            opt.step()
            net.project_()
            if keep_best and (t % 20 == 19 or t == steps - 1):
                v = self._refresh()
                better = v < best
                if better.any():
                    cur = net.get_state()
                    for k in best_state:
                        best_state[k][better] = cur[k][better]
                    best = torch.where(better, v, best)
        if keep_best:
            net.set_state(best_state)
        self._refresh()

    def _lm(self, iters: int) -> None:
        """Levenberg-Marquardt polish of all free parameters (keeps per-restart best)."""
        if iters <= 0:
            return
        snap = self.net_.get_state()
        v0 = self._refresh()
        tgt, var = self._target() if self.varpro else (self._yt, self._yvar)
        lm_polish(self.net_, self._Xt, tgt, var, iters=iters)
        v1 = self._refresh()
        worse = ~(v1 <= v0)
        if worse.any():
            self.net_.set_state(snap, worse)
            self._refresh()

    def _edit(self, edit: Callable[[], None], label: str) -> None:
        """Apply a structural edit, fine-tune, and roll back restarts that got worse."""
        snapshot = self.net_.get_state()
        edit()
        self._optimize(self.finetune_steps, self.lr * 0.5)
        self._lm(self.lm_iters)
        v = self._refresh()
        reject = ~(v <= self._ref * (1 + self.rel_tol) + self.abs_tol)
        if reject.any():
            self.net_.set_state(snapshot, reject)
        if self.verbose > 1:
            v = self._refresh()
            print(f"    [{label:10s}] accepted {int((~reject).sum()):2d}/{len(reject)}  "
                  f"best val nmse {float(v.min()):.3e}  median size "
                  f"{int(self.net_.complexity().median())}  ({time.time() - self._t0:.1f}s)")

    @torch.no_grad()
    def _prune(self, th: float, readout_rel: float) -> None:
        net = self.net_
        net.prune_(th, readout_threshold=0.0)
        _, z = net(self._Xt, return_features=True)
        contrib = torch.nan_to_num(net.eff("wo").abs() * z.std(dim=1), nan=0.0)
        scale = math.sqrt(self._lyvar if net.out_exp else self._yvar)
        net.active("wo")[contrib < readout_rel * scale] = False
        net.project_()
        net.remove_dead_()

    def _train_stage(self, arch: str, seed: int) -> torch.Tensor:
        d = self._Xt.shape[1]
        # successive halving: start with n_restarts * 2^halving restarts and keep the better
        # half after each short round; the survivors get the remaining (annealed) budget.
        R0 = self.n_restarts * 2 ** self.halving
        self.net_ = EMLNet(d, arch, n_restarts=R0, log_inputs=self.log_inputs,
                           exp_sees_raw=self.exp_sees_raw, exp_clip=self.exp_clip,
                           init=self.init, seed=seed)
        self.net_.out_sign = self._sign
        round_steps = self.dense_steps // 4
        for _ in range(self.halving):
            self._optimize(round_steps, self.lr, keep_best=False)
            v = self._refresh()
            keep = torch.argsort(v)[: self.net_.R // 2]
            self.net_.select_(keep)
        self._optimize(self.dense_steps - self.halving * round_steps if self.halving else self.dense_steps,
                       self.lr, quant=self.quant, keep_best=False)
        self._ref = self._refresh()
        # size and accuracy of every restart *before* pruning / snapping (for pruning statistics)
        self._dense_stats.append(dict(arch=arch, val_nmse=self._ref.clone(), n_params=self.net_.n_params()))
        for th, tro in zip(self.prune_thresholds, self.readout_rel_thresholds):
            self._edit(lambda th=th, tro=tro: self._prune(th, tro), f"prune {th}")
        # in log-space stages the read-out weights are exponents, so they are snapped too
        names = [n for n in self.net_.names if n.startswith(("wa", "wc", "bc"))]
        if self.net_.out_exp:
            names.append("wo")
        for dl in self.snap_deltas:
            self._edit(lambda dl=dl: self.net_.snap_(dl, self.grid, names=names), f"snap {dl}")
        self._edit(lambda: self._prune(1e-12, 1e-6), "cleanup")
        self._optimize(self.polish_steps, self.lr * 0.25)
        self._lm(2 * self.lm_iters)
        return self._refresh()

    # ------------------------------------------------------------------- API
    def fit(self, X, y):
        self._t0 = time.time()
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).ravel()
        n, d = X.shape
        rng = np.random.default_rng(self.random_state)
        torch.manual_seed(self.random_state)

        # Inputs are rescaled only by powers of ten (exponents are scale invariant and the
        # formula stays readable); 'auto' leaves features whose RMS is already O(1) alone.
        sx = _pow10_scale(X)
        if self.x_scale == "none":
            sx = np.ones(d)
        elif self.x_scale == "auto":
            rms = np.sqrt(np.mean(X ** 2, axis=0))
            sx = np.where((rms > 0.03) & (rms < 30), 1.0, sx)
        self.x_scale_, self.y_scale_ = sx, float(_pow10_scale(y[:, None])[0])
        Xs, ys = X / self.x_scale_, y / self.y_scale_

        perm = rng.permutation(n)
        n_val = int(round(self.val_fraction * n))
        if n_val >= 20:
            val_idx = perm[:n_val][: self.max_val_samples]
            tr_idx = perm[n_val:][: self.max_train_samples]
        else:  # tiny data set: no hold-out; selection falls back to BIC on the training data
            val_idx = tr_idx = perm[: self.max_train_samples]
        T = lambda a: torch.as_tensor(a, dtype=torch.float64)  # noqa: E731
        self._Xt, self._yt = T(Xs[tr_idx]), T(ys[tr_idx])
        self._Xv, self._yv = T(Xs[val_idx]), T(ys[val_idx])
        self._yvar = float(self._yt.var()) or 1.0
        n_v = len(val_idx)
        # log-space stages need a target of constant sign
        self._sign = 1.0 if (ys > 0).all() else (-1.0 if (ys < 0).all() else 0.0)
        if self._sign != 0.0:
            self._lyt, self._lyv = torch.log(self._yt.abs()), torch.log(self._yv.abs())
            self._lyvar = float(self._lyt.var()) or 1.0
        curriculum = [a for a in self.curriculum if self._sign != 0.0 or not a.endswith("*")]
        X_full_tr = T(Xs[perm[n_val:]] if n_val >= 20 else Xs[perm])
        y_full_tr = T(ys[perm[n_val:]] if n_val >= 20 else ys[perm])

        self.nets_, self.candidates_, self.stage_archs_ = [], [], []
        self._dense_stats = []
        for stage, arch in enumerate(curriculum):
            v = self._train_stage(arch, seed=self.random_state * 1000 + stage)
            cx = self.net_.complexity()
            self.nets_.append(self.net_)
            for r in range(self.n_restarts):
                nm = float(v[r])
                bic = n_v * math.log(nm + NMSE_FLOOR) + int(cx[r]) * math.log(n_v) if math.isfinite(nm) else math.inf
                self.candidates_.append(Candidate(stage, arch, r, nm, int(cx[r]), bic))
            if self.verbose:
                best = min((c for c in self.candidates_ if c.stage == stage), key=lambda c: c.bic)
                print(f"stage {stage} {arch:12s}: best val nmse {float(v.min()):.2e}; "
                      f"best-BIC size {best.size} nmse {best.val_nmse:.2e}  ({time.time() - self._t0:.1f}s)")
            if float(v.min()) < self.exact_tol:
                break

        best = min(self.candidates_, key=lambda c: (c.bic, c.size))
        self.best_ = best
        self.net_ = self.nets_[best.stage]
        # final polish of the selected network's constants on the full training split
        if len(y_full_tr) > len(self._yt):
            self._Xt, self._yt = X_full_tr[:20000], y_full_tr[:20000]
            if self._sign != 0.0:
                self._lyt = torch.log(self._yt.abs())
            v_before = self._refresh()
            snap = self.net_.get_state()
            self._optimize(self.polish_steps, self.lr * 0.25)
            self._lm(2 * self.lm_iters)
            v_after = self._refresh()
            if not v_after[best.restart] <= v_before[best.restart]:
                self.net_.set_state(snap)
                v_after = self._refresh()
            best.val_nmse = float(v_after[best.restart])
        self.restart_ = best.restart
        self.val_nmse_ = best.val_nmse
        self.size_ = best.size

        names = list(self.feature_names) if self.feature_names is not None else [f"x{i}" for i in range(d)]
        positive = bool((X > 0).all())
        self.symbols_ = [sp.Symbol(s, positive=True) if positive else sp.Symbol(s, real=True)
                         for s in names]
        self.expr_raw_ = self._export(best)
        self.expr_ = simplify_expr(self.expr_raw_)
        if best.val_nmse < 1e-9:
            try:
                self.expr_ = self._clean_exact(self.expr_)
            except (ArithmeticError, ValueError, TypeError):  # cosmetic step; never fail a fit
                pass
        best.expr = self.expr_
        self.complexity_ = expr_complexity(self.expr_)
        self.n_params_ = int(self.net_.n_params()[best.restart])
        ds = self._dense_stats[best.stage]
        self.dense_n_params_ = int(ds["n_params"][best.restart])
        self.dense_val_nmse_ = float(ds["val_nmse"][best.restart])
        self.fit_time_ = time.time() - self._t0
        if self.verbose:
            print(f"selected {best.arch} restart {best.restart}: val nmse {best.val_nmse:.3e}, "
                  f"size {best.size}: {self.formula()}  ({self.fit_time_:.1f}s)")
        return self

    def _expr_nmse(self, expr: sp.Expr) -> float:
        Xv = self._Xv.numpy() * self.x_scale_
        yv = self._yv.numpy() * self.y_scale_
        try:
            f = sp.lambdify(self.symbols_, expr, "numpy")
            with np.errstate(all="ignore"):
                pred = np.asarray(f(*Xv.T), dtype=complex) * np.ones(len(yv))
            if np.max(np.abs(pred.imag)) > 1e-9 * (1 + np.max(np.abs(pred.real))):
                return math.inf
            return float(np.mean((pred.real - yv) ** 2) / np.var(yv))
        except Exception:  # pragma: no cover
            return math.inf

    def _clean_exact(self, expr: sp.Expr) -> sp.Expr:
        """For (near-)exact fits: drop numerically-zero constants and rationalise constants
        such as 1.0000000002 or 0.4999999999; keep the result only if it still fits."""
        rms = float(np.sqrt(np.mean((self._yv.numpy() * self.y_scale_) ** 2)))
        before = self._expr_nmse(expr)
        cand = clean_constants(simplify_expr(clean_constants(self.expr_raw_, zero_abs=1e-6 * rms)),
                               zero_abs=1e-6 * rms)
        cand = sp.simplify(cand) if sp.count_ops(cand) < 80 else cand
        after = self._expr_nmse(cand)
        return cand if after <= max(10 * before, 1e-12) else expr

    def _export(self, c: Candidate) -> sp.Expr:
        return net_to_sympy(self.nets_[c.stage], c.restart, self.symbols_, self.x_scale_,
                            self.y_scale_, self._Xt)

    @property
    def pareto_(self) -> List[Candidate]:
        front, best = [], math.inf
        for c in sorted(self.candidates_, key=lambda c: (c.size, c.val_nmse)):
            if c.val_nmse < best:
                front.append(c)
                best = c.val_nmse
        return front

    def pareto_formulas(self, sig: int = 4) -> List[str]:
        """Readable formula for every point of the accuracy/size front."""
        out = []
        for c in self.pareto_:
            if c.expr is None:
                c.expr = simplify_expr(self._export(c))
            out.append(f"size {c.size:3d}  val_nmse {c.val_nmse:.2e}  [{c.arch}]  {round_floats(c.expr, sig)}")
        return out

    def predict(self, X) -> np.ndarray:
        check_is_fitted(self, "net_")
        Xs = torch.as_tensor(np.asarray(X, dtype=float) / self.x_scale_, dtype=torch.float64)
        with torch.no_grad():
            return self.net_(Xs)[self.restart_].numpy() * self.y_scale_

    def formula(self, sig: int = 4) -> str:
        """Human-readable formula with constants rounded to ``sig`` significant digits."""
        check_is_fitted(self, "expr_")
        return str(round_floats(self.expr_, sig))

    def latex(self, sig: int = 4) -> str:
        return sp.latex(round_floats(self.expr_, sig))

    def lambdify(self):
        """NumPy function evaluating the exported formula: f(X) with X of shape (n, d)."""
        # same guard as the network: ln|c| that stays finite at 0 (evalf avoids integer constants)
        guarded = {"log": lambda c: 0.5 * np.log(np.asarray(c, dtype=float) ** 2 + 1e-16)}
        f = sp.lambdify(self.symbols_, self.expr_.evalf(), [guarded, "numpy"])
        return lambda X: np.asarray(f(*np.asarray(X, dtype=float).T), dtype=float) * np.ones(len(X))
