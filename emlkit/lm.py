"""Batched Levenberg-Marquardt polishing for EMLNet.

Adam random-walks once the loss is tiny, so an exact formula never *looks* exact after
Adam fine-tuning.  Levenberg-Marquardt converges quadratically on the free parameters of a
fixed structure, which is what deciding "is this pruned/snapped structure exact?" needs.

Restarts are independent, so perturbing parameter slot k in *every* restart at once yields
column k of each restart's Jacobian in a single forward-mode pass: the cost per iteration is
(#parameter slots) forward passes, independent of the number of restarts.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import torch
from torch.func import functional_call, jvp

from .net import EMLNet


def _residual_fn(net: EMLNet, X: torch.Tensor, target: torch.Tensor, scale: float):
    def f(params: Dict[str, torch.Tensor]) -> torch.Tensor:
        pred = functional_call(net, params, (X,), {"linear_output": True})
        return (pred - target) / scale
    return f


@torch.no_grad()
def lm_polish(net: EMLNet, X: torch.Tensor, target: torch.Tensor, var: float,
              iters: int = 20, max_slots: int = 160, lam0: float = 1e-3,
              tol: float = 1e-30) -> torch.Tensor:
    """Polish the free parameters of every restart; returns the per-restart NMSE.

    ``target`` is the read-out target (ln|y| for log-space nets) and ``var`` its variance.
    Nets with more than ``max_slots`` free parameter positions are left unchanged.
    """
    R = net.R
    N = X.shape[0]
    names = list(net.names)
    params = {n: net.eff(n).detach().clone() for n in names}
    free = {n: (net.active(n) & ~net.fixed(n)) for n in names}
    slots: List[Tuple[str, int]] = []
    for n in names:
        fr = free[n].reshape(R, -1)
        for j in torch.nonzero(fr.any(0)).flatten().tolist():
            slots.append((n, j))
    scale = (var * N) ** 0.5
    f = _residual_fn(net, X, target, scale)

    def loss_of(p):
        r = torch.nan_to_num(f(p), nan=1e6, posinf=1e6, neginf=-1e6)
        return r, r.pow(2).sum(1)

    r, loss = loss_of(params)
    P = len(slots)
    if P == 0 or P > max_slots:
        return loss
    lam = torch.full((R,), lam0, dtype=net.dtype)
    for _ in range(iters):
        cols = []
        for n, j in slots:
            tang = {m: torch.zeros_like(params[m]) for m in names}
            t = tang[n].reshape(R, -1)
            t[:, j] = free[n].reshape(R, -1)[:, j].to(net.dtype)
            _, jc = jvp(f, (params,), (tang,))
            cols.append(torch.nan_to_num(jc, nan=0.0, posinf=0.0, neginf=0.0))
        J = torch.stack(cols, dim=-1)                                  # (R, N, P)
        A = J.transpose(1, 2) @ J                                      # (R, P, P)
        g = (J.transpose(1, 2) @ r.unsqueeze(-1)).squeeze(-1)          # (R, P)
        diag = torch.diagonal(A, dim1=1, dim2=2)
        dead = diag <= 0
        D = torch.where(dead, torch.ones_like(diag), diag)
        M = A + torch.diag_embed(lam[:, None] * D + dead.to(A.dtype))
        try:
            delta = -torch.linalg.solve(M, g.unsqueeze(-1)).squeeze(-1)
        except RuntimeError:
            delta = -(torch.linalg.pinv(M) @ g.unsqueeze(-1)).squeeze(-1)
        delta = torch.nan_to_num(delta, nan=0.0, posinf=0.0, neginf=0.0)
        new = {m: params[m].clone() for m in names}
        for k, (n, j) in enumerate(slots):
            v = new[n].reshape(R, -1)
            v[:, j] += delta[:, k] * free[n].reshape(R, -1)[:, j]
        r_new, loss_new = loss_of(new)
        better = loss_new < loss
        for m in names:
            sel = better.view(-1, *([1] * (params[m].dim() - 1)))
            params[m] = torch.where(sel, new[m], params[m])
        r = torch.where(better[:, None], r_new, r)
        loss = torch.where(better, loss_new, loss)
        lam = torch.where(better, lam / 3, lam * 4).clamp(1e-12, 1e8)
        if bool((loss < tol).all()):
            break
    for n in names:
        net.param(n).copy_(params[n])
    net.project_()
    return loss
