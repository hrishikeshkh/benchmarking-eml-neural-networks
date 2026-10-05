import numpy as np
import sympy as sp
import torch

from emlkit import EMLNet, EMLRegressor


def test_net_shapes_and_monomial_unit():
    net = EMLNet(2, "E1", n_restarts=3)
    X = torch.rand(10, 2, dtype=torch.float64) + 1
    assert net(X).shape == (3, 10)
    # set restart 0 to exactly x0 * x1^2: exp(1*ln x0 + 2*ln x1), read-out weight 1
    with torch.no_grad():
        net.wa0.zero_(); net.wa0[0, 0, 2] = 1.0; net.wa0[0, 0, 3] = 2.0
        net.wo.zero_(); net.wo[0, 4] = 1.0; net.bo.zero_()
    assert torch.allclose(net(X)[0], X[:, 0] * X[:, 1] ** 2)


def test_recovers_product_exactly():
    rng = np.random.default_rng(0)
    X = rng.uniform(1, 5, size=(400, 2))
    y = 3 * X[:, 0] * X[:, 1] ** 2
    m = EMLRegressor(curriculum=("E1", "E2"), n_restarts=16, dense_steps=400,
                     feature_names=["a", "b"], random_state=0).fit(X, y)
    a, b = sp.symbols("a b", positive=True)
    assert sp.simplify(m.expr_ - 3 * a * b ** 2) == 0
    Xo = rng.uniform(5, 10, size=(100, 2))
    assert np.allclose(m.predict(Xo), 3 * Xo[:, 0] * Xo[:, 1] ** 2, rtol=1e-8)


def test_formula_and_lambdify_agree_with_predict():
    rng = np.random.default_rng(1)
    X = rng.uniform(1, 3, size=(300, 1))
    y = np.exp(-X[:, 0] ** 2 / 2) + 0.01 * rng.normal(size=300)
    m = EMLRegressor(curriculum=("E1-E1",), n_restarts=8, dense_steps=300, random_state=0).fit(X, y)
    f = m.lambdify()
    assert np.allclose(f(X), m.predict(X), rtol=1e-6, atol=1e-8)
    assert isinstance(m.formula(), str)
