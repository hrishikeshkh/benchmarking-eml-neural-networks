"""Kepler's third law from the six planets Kepler had in 1618 (PMLB first_principles_kepler)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks"))
from realworld_data import load  # noqa: E402

from emlkit import EMLRegressor  # noqa: E402

X, y, names = load("first_principles_kepler")              # a [AU] -> period [days]
model = EMLRegressor(feature_names=names).fit(X, y)
print("period =", model.formula())
for c in model.pareto_formulas()[:5]:
    print("  ", c)
