# emlkit

**Trainable Exp-Minus-Log networks for interpretable regression, plus a pure-EML compiler.**

Odrzywołek (2026, [arXiv:2603.21852](https://arxiv.org/abs/2603.21852)) showed that one binary
operator,

$$\mathrm{eml}(x, y) = e^{x} - \ln y ,$$

together with the constant 1, generates every elementary function — the continuous-math analogue of
the NAND gate. `emlkit` turns that observation into practical tools:

| Module | What it does |
|---|---|
| `emlkit.EMLRegressor` | scikit-learn regressor that **learns a closed-form formula** by gradient descent on a network of EML nodes, then prunes and snaps it to exact exponents. `fit / predict / formula() / latex() / lambdify()`. |
| `emlkit.EMLNet` | the underlying PyTorch module (batched over many random restarts) — usable as an interpretable head on top of any network. |
| `emlkit.tree` | pure EML trees (`S -> 1 | x | eml(S, S)`): **compiler** from SymPy, complex evaluator, verifier, parser and an exhaustive **shortest-tree search**. |

> Status: research code accompanying the paper in [`paper/`](paper/). Results, limitations and the
> full experimental protocol are reported there; everything is reproducible with the scripts in
> [`benchmarks/`](benchmarks/).

## Install

```bash
pip install -e .            # library only: numpy, torch, sympy, scikit-learn
pip install -e ".[bench]"   # + pandas, matplotlib, pyyaml, gplearn for the experiments
```

## Quick start

```python
import numpy as np
from emlkit import EMLRegressor

rng = np.random.default_rng(0)
X = rng.uniform(1, 5, size=(1000, 3))            # q1, q2, r
y = X[:, 0] * X[:, 1] / (4 * np.pi * X[:, 2] ** 2)   # Coulomb's law (epsilon = 1)

m = EMLRegressor(feature_names=["q1", "q2", "r"]).fit(X, y)
print(m.formula())          # 0.07958*q1*q2/r**2
print(m.latex())            # \frac{0.07958 q_{1} q_{2}}{r^{2}}
m.predict(X[:5])            # behaves like any scikit-learn regressor
```

`m.pareto_formulas()` lists the accuracy/size trade-off found across all restarts and architectures.

## How it works (short version)

An EML network is a layered DAG of EML nodes whose arguments are *affine* in everything computed before:

```
z0   = [x, ln|x|]                                 fixed input features (ln x = 1 - eml(0, x))
E    = eml(a, 1)   = exp(a)       a = W z + b     products, powers, exponentials
L    = 1 - eml(0, c) = ln|c|      c = W z + b     logs of sums
y    = w . z + b                                  affine read-out over all features
```

so a monomial `C x^a y^b` is a single `E` node, `m0 / sqrt(1 - v^2/c^2)` is `E -> L -> E`, and so on.
Training combines

1. **variable projection** – the read-out is solved by least squares at every step, gradients only
   shape the EML features;
2. **annealed quantisation** – exponents are pulled towards multiples of 1/2;
3. **prune → snap → polish** with per-restart roll-back if validation error degrades;
4. an **Occam curriculum** over architectures (`E1`, `E2`, `E1-L1-E1`, …) with early stopping on an
   exact fit and **BIC** model selection.

## Pure EML trees

```python
from emlkit.tree import compile_expr, verify, search, leaves

t = compile_expr("x*y")           # only eml(., .), 1 and variables
print(leaves(t), verify(t, "x*y"))     # 23 leaves, max error ~1e-16
search("log(x)")                  # eml(1, eml(eml(1, x), 1))  -- shortest tree, found by enumeration
```

## Reproducing the paper

```bash
python benchmarks/run_feynman.py  --methods EML,MLP,GBM,RF,Linear,GPLearn --noise 0.0,0.01
python benchmarks/run_realworld.py --group physics
python benchmarks/run_realworld.py --group tabular
python benchmarks/analyze.py      # tables + figures into paper/
```

## Citation

If you use this code, please cite the EML paper:

```bibtex
@article{odrzywolek2026eml,
  title   = {All elementary functions from a single binary operator},
  author  = {Odrzywo{\l}ek, Andrzej},
  journal = {arXiv preprint arXiv:2603.21852},
  year    = {2026}
}
```

## License

MIT
