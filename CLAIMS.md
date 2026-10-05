# Claims and pre-declared tests

Written before the full results were in. Verdicts are computed by `benchmarks/analyze.py` using exactly
these thresholds. **S** = supported, **P** = partly supported, **N** = not supported.

Notation: *recovery* means an SRBench-style symbolic match with the ground truth. *OOD* means every
input is drawn from [hi, 2·hi − lo], outside the training box. *Held-out* means the Feynman problems not
used during method development (the 8 dev problems are listed in `benchmarks/run_feynman.py`).

| # | Claim | Test | S | P |
|---|---|---|---|---|
| C1 | EML recovers exact laws | recovery rate, held-out non-trig Feynman, no noise | ≥ 50% | ≥ 25% |
| C2 | at least as good as GP symbolic regression | EML recovery ≥ GPLearn recovery (all Feynman, no noise) | ≥ | within 5 pts |
| C3 | matches an MLP in distribution | share of problems with R²_id > 0.999, EML vs MLP | ≥ MLP − 5 pts | ≥ MLP − 15 pts |
| C4 | extrapolates far better than MLP / GBM | share with R²_ood > 0.99 | ≥ max(MLP, GBM) + 25 pts | > max(MLP, GBM) |
| C5 | survives 1% label noise | recovery at 1% noise ÷ recovery at 0% | ≥ 0.5 | ≥ 0.25 |
| C6 | models are readable | median formula size (SymPy nodes), Feynman / real data | ≤ 30 / ≤ 40 | ≤ 50 / ≤ 60 |
| C7 | best model on some real data sets | # real data sets where EML CV R² ≥ every baseline | ≥ 3 | ≥ 1 |
| C8 | accurate while explainable (physics data) | median (best baseline R² − EML R²) on physics data | ≤ 0.02 | ≤ 0.05 |
| C9 | comparable to a NN (tabular data) | share of tabular data sets with EML R² ≥ MLP R² − 0.02 | ≥ 1/3 | ≥ 1/5 |
| C10 | every training component matters | recovery drop when the component is removed (ablation subset) | ≥ 5 pts | > 0 pts |
| C11 | compiler is exact | max error of compiled pure-EML trees on the test expressions | < 1e-10 | – |
| C12 | known limitation: trigonometric laws | recovery on trig Feynman problems (no noise) | ≤ 10% (limitation confirmed) | – |

Protocol: Feynman uses n = 1000 training points, 2000 ID test points and 2000 OOD test points, seed 0,
noise ∈ {0, 1% of std(y)}. Real-world data uses 5-fold CV (leave-one-out for n < 20) with pooled
out-of-fold R². Baselines: Ridge, MLP (2×128 ReLU, early stopping), histogram GBM, random forest,
GPLearn. Every method runs on one CPU core with its default hyper-parameters, and nothing is tuned on
test data.
