"""Differentiable networks built from the Exp-Minus-Log (EML) operator.

The EML operator eml(a, c) = exp(a) - ln(c) (Odrzywolek, 2026) together with the constant 1
generates every elementary function.  An ``EMLNet`` is a layered DAG of EML nodes whose
arguments are *affine* functions of everything computed before them:

    z_0   = [x, ln|x|]                       (ln x = 1 - eml(0, x): fixed input features)
    a, c  = W_a z + b_a,  W_c z + b_c         (affine arguments)
    unit  = one of
            'E'  eml(a, 1)  = exp(a)                     -- product / power / exponential
            'L'  1 - eml(0, c) = ln|c|                   -- log of a sum
            'M'  eml(a, c)  = exp(a) - ln|c|             -- the full operator
    y     = w_o . z + b_o                     (affine read-out over all features)

Typed units ('E' and 'L') are special cases of the full operator in which one argument is the
constant 1 (resp. 0 = ln 1).  They remove a redundant degree of freedom per unit and make the
loss landscape much friendlier: a monomial  C * prod_j x_j^{p_j}  is exactly one 'E' unit,
exp(sum_j p_j ln x_j + ln C).

All parameters carry a leading *restart* dimension ``R`` so that many independently initialised
networks are optimised in one batched computation.  Each parameter has an ``active`` mask
(pruned entries are 0) and a ``fixed`` mask with ``value`` (entries snapped to exact rationals
are frozen during fine-tuning).
"""
from __future__ import annotations

import math
import re
from typing import Dict, List, Sequence, Tuple

import torch
import torch.nn as nn

PROTECTED_LOG_EPS = 1e-8


def protected_log(c: torch.Tensor, eps: float = PROTECTED_LOG_EPS) -> torch.Tensor:
    """Smooth ln|c| that stays finite at c = 0 (equal to ln(c) for c >> eps > 0)."""
    return 0.5 * torch.log(c * c + eps * eps)


def parse_arch(arch: str | Sequence[Tuple[str, int]]) -> List[Tuple[str, int]]:
    """'E4-L2-E2' -> [('E', 4), ('L', 2), ('E', 2)] (trailing '+' / '*' flags are ignored here;
    '*' alone means no hidden layer)."""
    if not isinstance(arch, str):
        return [(t.upper(), int(w)) for t, w in arch]
    out = []
    body = arch.rstrip("+*")
    for tok in (body.split("-") if body else []):
        m = re.fullmatch(r"([ELM])(\d+)", tok.strip().upper())
        if not m:
            raise ValueError(f"bad layer spec {tok!r} in {arch!r}")
        out.append((m.group(1), int(m.group(2))))
    return out


class EMLNet(nn.Module):
    """A batch of ``n_restarts`` independent EML networks.

    Parameters
    ----------
    n_in : number of input features.
    arch : layer specification such as ``"E4-L2-E2"`` (type and width of each layer).
    n_restarts : number of independently initialised copies trained in parallel.
    log_inputs : append the fixed features ln|x_j| to the inputs.
    readout_skip : if False (default) the read-out only sees the units of the last layer,
        so the model must express the target through its EML structure; if True it sees
        every feature (inputs, logs and all units).  An ``arch`` ending in '+' sets it.
    out_exp : if True the output is itself an EML node,  y = s * eml(w . z + b, 1)
        = s * exp(w . z + b)  with a fixed sign s (``out_sign``), and the read-out sees every
        feature.  Products of factors become sums inside the exponent, so the whole
        argument can be solved by least squares on ln|y| ("log-space variable projection").
        An ``arch`` ending in '*' sets it.
    exp_clip : exp arguments are clamped at this value (overflow guard).
    """

    def __init__(
        self,
        n_in: int,
        arch: str | Sequence[Tuple[str, int]] = "E4-L2-E2",
        n_restarts: int = 16,
        log_inputs: bool = True,
        exp_sees_raw: bool = False,
        readout_skip: bool = False,
        out_exp: bool = False,
        exp_clip: float = 30.0,
        init: str = "sparse",
        seed: int = 0,
        dtype: torch.dtype = torch.float64,
    ):
        super().__init__()
        self.n_in = n_in
        self.layers = parse_arch(arch)
        self.out_exp = out_exp or (isinstance(arch, str) and arch.endswith("*"))
        self.out_sign = 1.0
        self.readout_skip = (readout_skip or self.out_exp
                             or (isinstance(arch, str) and arch.endswith("+")))
        self.R = n_restarts
        self.log_inputs = log_inputs
        self.exp_clip = exp_clip
        self.dtype = dtype
        g = torch.Generator().manual_seed(seed)
        R = n_restarts
        self.names: List[str] = []

        def sparse_init(shape, values, p_nonzero):
            """Random sparse 'integer-like' weights: a discrete starting point per restart."""
            vals = torch.tensor(values, dtype=dtype)
            pick = vals[torch.randint(len(vals), shape, generator=g)]
            on = torch.rand(shape, generator=g, dtype=dtype) < p_nonzero
            return pick * on + 0.05 * torch.randn(shape, generator=g, dtype=dtype)

        def unit_init(shape, prev_lo, prev_hi):
            """Structured init for units fed by earlier units: half of the restarts start each
            unit from +-1 times one randomly chosen earlier unit (all other weights 0), i.e.
            exp(+-u) or ln|1 +- u| -- the shapes that recur in physical laws."""
            Rr, W, F = shape
            out = torch.zeros(shape, dtype=dtype)
            k = torch.randint(prev_lo, prev_hi, (Rr, W), generator=g)
            sign = torch.where(torch.rand(Rr, W, generator=g) < 0.5, -1.0, 1.0).to(dtype)
            out.scatter_(2, k.unsqueeze(-1), sign.unsqueeze(-1))
            use = (torch.arange(Rr) % 2 == 1).view(-1, 1, 1)
            return use, out + 0.02 * torch.randn(shape, generator=g, dtype=dtype)

        fan = n_in * (2 if log_inputs else 1)
        self.n_base = fan
        for l, (typ, w) in enumerate(self.layers):
            p_on = min(0.5, 2.0 / fan)
            if typ in ("E", "M"):
                if init == "sparse":
                    wa = sparse_init((R, w, fan), [-2.0, -1.0, -0.5, 0.5, 1.0, 2.0], p_on)
                    if l > 0:
                        use, st = unit_init((R, w, fan), self.n_base, fan)
                        wa = torch.where(use, st, wa)
                else:
                    wa = torch.randn(R, w, fan, generator=g, dtype=dtype) * (0.5 / math.sqrt(fan))
                self._add(f"wa{l}", wa)
                self._add(f"ba{l}", torch.zeros(R, w, dtype=dtype))
                if not exp_sees_raw:  # exp arguments only see logs and earlier units
                    self.active(f"wa{l}")[:, :, :n_in] = False
            if typ in ("L", "M"):
                if init == "sparse":
                    wc = sparse_init((R, w, fan), [-1.0, 1.0], p_on)
                    if l > 0:
                        use, st = unit_init((R, w, fan), self.n_base, fan)
                        wc = torch.where(use, st, wc)
                else:
                    wc = torch.randn(R, w, fan, generator=g, dtype=dtype) * (0.5 / math.sqrt(fan))
                self._add(f"wc{l}", wc)
                self._add(f"bc{l}", torch.ones(R, w, dtype=dtype))
            fan += w
        self.n_feat = fan
        self._add("wo", torch.randn(R, fan, generator=g, dtype=dtype) * 0.1)
        self._add("bo", torch.zeros(R, dtype=dtype))
        if not self.readout_skip and self.layers:
            self.active("wo")[:, : self.layer_offset(self.n_layers - 1)] = False
        self.project_()

    # ------------------------------------------------------------------ params
    def _add(self, name: str, value: torch.Tensor) -> None:
        self.names.append(name)
        self.register_parameter(name, nn.Parameter(value))
        self.register_buffer(f"{name}_active", torch.ones_like(value, dtype=torch.bool))
        self.register_buffer(f"{name}_fixed", torch.zeros_like(value, dtype=torch.bool))
        self.register_buffer(f"{name}_value", torch.zeros_like(value))

    def has(self, name: str) -> bool:
        return name in self.names

    def param(self, name: str) -> torch.Tensor:
        return getattr(self, name)

    def active(self, name: str) -> torch.Tensor:
        return getattr(self, f"{name}_active")

    def fixed(self, name: str) -> torch.Tensor:
        return getattr(self, f"{name}_fixed")

    def value(self, name: str) -> torch.Tensor:
        return getattr(self, f"{name}_value")

    def eff(self, name: str) -> torch.Tensor:
        """Effective parameter: snapped entries use their exact value, pruned entries are 0."""
        p = torch.where(self.fixed(name), self.value(name), self.param(name))
        return p * self.active(name)

    @property
    def n_layers(self) -> int:
        return len(self.layers)

    def weight_names(self) -> List[str]:
        return [n for n in self.names if n.startswith("w")]

    def layer_offset(self, l: int) -> int:
        return self.n_base + sum(w for _, w in self.layers[:l])

    # ----------------------------------------------------------------- forward
    def base_features(self, X: torch.Tensor) -> torch.Tensor:
        X = X.to(self.dtype)
        return torch.cat([X, protected_log(X)], dim=-1) if self.log_inputs else X

    def forward(self, X: torch.Tensor, return_internals: bool = False, return_features: bool = False,
                linear_output: bool = False):
        """X: (N, n_in) -> predictions (R, N).

        ``return_internals`` additionally returns the (a, c) arguments of every layer and
        ``return_features`` the full feature tensor z of shape (R, N, n_feat).
        """
        z = self.base_features(X).unsqueeze(0).expand(self.R, -1, -1)
        internals = []
        for l, (typ, _) in enumerate(self.layers):
            a = c = None
            if typ in ("E", "M"):
                a = torch.einsum("rnf,rwf->rnw", z, self.eff(f"wa{l}")) + self.eff(f"ba{l}")[:, None, :]
            if typ in ("L", "M"):
                c = torch.einsum("rnf,rwf->rnw", z, self.eff(f"wc{l}")) + self.eff(f"bc{l}")[:, None, :]
            if typ == "E":
                h = torch.exp(torch.clamp(a, max=self.exp_clip))
            elif typ == "L":
                h = protected_log(c)
            else:
                h = torch.exp(torch.clamp(a, max=self.exp_clip)) - protected_log(c)
            if return_internals:
                internals.append((a, c))
            z = torch.cat([z, h], dim=-1)
        y = torch.einsum("rnf,rf->rn", z, self.eff("wo")) + self.eff("bo")[:, None]
        if self.out_exp and not linear_output:
            y = self.out_sign * torch.exp(torch.clamp(y, max=self.exp_clip))
        if return_features:
            return y, z
        return (y, internals) if return_internals else y

    # ---------------------------------------------------------- regularisation
    def l1(self) -> torch.Tensor:
        """Per-restart L1 norm of the free (not snapped) weights, shape (R,)."""
        tot = 0.0
        for n in self.weight_names():
            free = self.active(n) & ~self.fixed(n)
            tot = tot + (self.param(n).abs() * free).flatten(1).sum(1)
        return tot

    def quant_penalty(self, grid: float = 0.5) -> torch.Tensor:
        """Per-restart squared distance of free argument weights to the nearest multiple of
        ``grid`` (annealed during training it pulls exponents towards exact rationals)."""
        tot = 0.0
        for n in self.names:
            if n.startswith(("wa", "wc", "bc")):
                p = self.param(n)
                free = self.active(n) & ~self.fixed(n)
                d = p - torch.round(p.detach() / grid) * grid
                tot = tot + (d.pow(2) * free).flatten(1).sum(1)
        return tot

    @torch.no_grad()
    def clip_grads_(self, max_norm: float = 1.0) -> None:
        """Gradient clipping computed separately for each restart."""
        sq = torch.zeros(self.R, dtype=self.dtype)
        for n in self.names:
            p = self.param(n)
            if p.grad is not None:
                p.grad.nan_to_num_(0.0, 0.0, 0.0)
                p.grad.mul_(self.active(n) & ~self.fixed(n))
                sq += p.grad.pow(2).reshape(self.R, -1).sum(1)
        scale = torch.clamp(max_norm / (sq.sqrt() + 1e-12), max=1.0)
        for n in self.names:
            p = self.param(n)
            if p.grad is not None:
                p.grad.mul_(scale.view(-1, *([1] * (p.dim() - 1))))

    @torch.no_grad()
    def project_(self) -> None:
        """Write masks / snapped values into the raw parameters."""
        for n in self.names:
            self.param(n).copy_(self.eff(n))

    # ------------------------------------------------------- structure editing
    def _sel(self, restarts, ref):
        return restarts.view(-1, *([1] * (ref.dim() - 1)))

    @torch.no_grad()
    def prune_(self, threshold: float, readout_threshold: float | None = None,
               restarts: torch.Tensor | None = None) -> None:
        """Deactivate free weights with |w| < threshold (biases are never pruned)."""
        for n in self.weight_names():
            th = readout_threshold if (n == "wo" and readout_threshold is not None) else threshold
            p = self.eff(n)
            kill = (p.abs() < th) & ~self.fixed(n)
            if restarts is not None:
                kill &= self._sel(restarts, p)
            self.active(n)[kill] = False
        self.project_()

    @torch.no_grad()
    def snap_(self, delta: float, grid: float = 0.5, names: Sequence[str] | None = None,
              restarts: torch.Tensor | None = None) -> int:
        """Freeze free parameters within ``delta`` of a multiple of ``grid`` (0 => pruned)."""
        if names is None:
            names = [n for n in self.names if n.startswith(("wa", "wc", "bc"))]
        count = 0
        for n in names:
            p = self.eff(n)
            target = torch.round(p / grid) * grid
            close = ((p - target).abs() < delta) & self.active(n) & ~self.fixed(n)
            if restarts is not None:
                close &= self._sel(restarts, p)
            self.fixed(n)[close] = True
            self.value(n)[close] = target[close]
            self.active(n)[close & (target == 0)] = False
            count += int(close.sum())
        self.project_()
        return count

    @torch.no_grad()
    def remove_dead_(self) -> None:
        """Deactivate every parameter of units that do not influence the output."""
        for r in range(self.R):
            live = self.live_units(r)
            for l, (typ, _) in enumerate(self.layers):
                dead = ~live[l]
                for n in (f"wa{l}", f"ba{l}", f"wc{l}", f"bc{l}"):
                    if self.has(n):
                        self.active(n)[r][dead] = False
        self.project_()

    # --------------------------------------------------------- state handling
    def get_state(self) -> Dict[str, torch.Tensor]:
        st = {}
        for n in self.names:
            st[n] = self.param(n).detach().clone()
            for suf in ("active", "fixed", "value"):
                st[f"{n}_{suf}"] = getattr(self, f"{n}_{suf}").clone()
        return st

    @torch.no_grad()
    def set_state(self, st: Dict[str, torch.Tensor], restarts: torch.Tensor | None = None) -> None:
        """Restore ``st`` (optionally only for the restarts selected by a bool mask)."""
        for key, val in st.items():
            tgt = getattr(self, key)
            if restarts is None:
                tgt.copy_(val)
            else:
                tgt[restarts] = val[restarts]

    @torch.no_grad()
    def select_(self, idx: torch.Tensor) -> None:
        """Keep only the restarts in ``idx`` (used for successive halving)."""
        for n in self.names:
            setattr(self, n, nn.Parameter(self.param(n).detach()[idx].clone()))
            for suf in ("active", "fixed", "value"):
                key = f"{n}_{suf}"
                self._buffers[key] = getattr(self, key)[idx].clone()
        self.R = len(idx)

    # ------------------------------------------------------------- inspection
    @torch.no_grad()
    def live_units(self, r: int) -> List[torch.Tensor]:
        """Bool mask per layer of units that influence the output of restart ``r``."""
        wo = self.eff("wo")[r]
        live = [torch.zeros(w, dtype=torch.bool) for _, w in self.layers]
        for l in reversed(range(self.n_layers)):
            off = self.layer_offset(l)
            for u in range(self.layers[l][1]):
                zi = off + u
                used = bool(wo[zi] != 0)
                for l2 in range(l + 1, self.n_layers):
                    for n in (f"wa{l2}", f"wc{l2}"):
                        if self.has(n):
                            used |= bool((self.eff(n)[r][:, zi] != 0)[live[l2]].any())
                live[l][u] = used
        return live

    @torch.no_grad()
    def n_params(self) -> torch.Tensor:
        """Per-restart number of nonzero parameters in units that reach the output
        (argument weights and biases of live units, read-out weights and bias)."""
        out = torch.zeros(self.R, dtype=torch.long)
        for r in range(self.R):
            live = self.live_units(r)
            tot = int((self.eff("wo")[r] != 0).sum()) + int(self.eff("bo")[r] != 0)
            for l in range(self.n_layers):
                m = live[l]
                for n in (f"wa{l}", f"ba{l}", f"wc{l}", f"bc{l}"):
                    if self.has(n) and m.any():
                        tot += int((self.eff(n)[r][m] != 0).sum())
            out[r] = tot
        return out

    @torch.no_grad()
    def complexity(self) -> torch.Tensor:
        """Per-restart model size: live EML nodes + nonzero weights/biases feeding them
        + nonzero read-out weights."""
        out = torch.zeros(self.R, dtype=torch.long)
        for r in range(self.R):
            live = self.live_units(r)
            tot = int((self.eff("wo")[r] != 0).sum()) + int(self.out_exp)
            for l in range(self.n_layers):
                m = live[l]
                if not m.any():
                    continue
                tot += int(m.sum())
                for n in (f"wa{l}", f"wc{l}"):
                    if self.has(n):
                        tot += int((self.eff(n)[r][m] != 0).sum())
                for n in (f"ba{l}", f"bc{l}"):
                    if self.has(n):
                        tot += int((self.eff(n)[r][m] != 0).sum())
            out[r] = tot
        return out
