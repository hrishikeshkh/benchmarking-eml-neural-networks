"""Baseline regressors used in the experiments (all scikit-learn compatible)."""
from __future__ import annotations

import warnings

import numpy as np
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import RidgeCV
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler


def make_linear(seed: int = 0):
    return make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-6, 3, 19)))


def make_poly2(seed: int = 0):
    return make_pipeline(StandardScaler(), PolynomialFeatures(2), RidgeCV(alphas=np.logspace(-6, 3, 19)))


class MLP:
    """Two hidden layers of 128 ReLU units, standardised inputs and target, Adam with early
    stopping (disabled for tiny datasets, where a held-out split is not possible)."""

    def __init__(self, seed: int = 0):
        self.seed = seed

    def fit(self, X, y):
        es = len(y) >= 60
        mlp = MLPRegressor(hidden_layer_sizes=(128, 128), activation="relu", solver="adam",
                           learning_rate_init=1e-3, max_iter=3000, early_stopping=es,
                           validation_fraction=0.15, n_iter_no_change=50, tol=1e-7,
                           random_state=self.seed)
        self.model = TransformedTargetRegressor(regressor=make_pipeline(StandardScaler(), mlp),
                                                transformer=StandardScaler())
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.model.fit(X, y)
        return self

    def predict(self, X):
        return self.model.predict(X)


def make_mlp(seed: int = 0):
    return MLP(seed)


def make_gbm(seed: int = 0):
    return HistGradientBoostingRegressor(max_iter=1000, learning_rate=0.05, early_stopping="auto",
                                         min_samples_leaf=5, random_state=seed)


def make_rf(seed: int = 0):
    return RandomForestRegressor(n_estimators=300, min_samples_leaf=1, n_jobs=1, random_state=seed)


class GPLearnSR:
    """gplearn genetic-programming symbolic regression (Koza-style GP) with a protected
    elementary function set.  Exposes ``formula()`` like EMLRegressor."""

    def __init__(self, seed: int = 0, population_size: int = 2000, generations: int = 30):
        from gplearn.genetic import SymbolicRegressor
        self.model = SymbolicRegressor(
            population_size=population_size, generations=generations, tournament_size=20,
            function_set=("add", "sub", "mul", "div", "sqrt", "log", "neg", "inv", "sin", "cos"),
            const_range=(-5.0, 5.0), init_depth=(2, 6), parsimony_coefficient=0.001,
            p_crossover=0.7, p_subtree_mutation=0.1, p_hoist_mutation=0.05, p_point_mutation=0.1,
            max_samples=1.0, n_jobs=1, random_state=seed)

    def fit(self, X, y):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.model.fit(X, y)
        return self

    def predict(self, X):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return self.model.predict(X)

    def formula(self) -> str:
        return str(self.model._program)

    @property
    def size(self) -> int:
        return int(self.model._program.length_)


BASELINES = {
    "Linear": make_linear,
    "Poly2": make_poly2,
    "MLP": make_mlp,
    "GBM": make_gbm,
    "RF": make_rf,
    "GPLearn": lambda seed=0: GPLearnSR(seed),
}
