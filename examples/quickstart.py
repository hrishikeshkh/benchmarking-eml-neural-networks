"""Recover Coulomb's law from 1000 noisy-free samples and check that it extrapolates."""
import numpy as np

from emlkit import EMLRegressor

rng = np.random.default_rng(0)
X = rng.uniform(1, 5, size=(1000, 3))                      # q1, q2, r
y = X[:, 0] * X[:, 1] / (4 * np.pi * X[:, 2] ** 2)

model = EMLRegressor(feature_names=["q1", "q2", "r"]).fit(X, y)
print("formula :", model.formula())                       # 0.07958*q1*q2/r**2
print("LaTeX   :", model.latex())

X_far = rng.uniform(5, 50, size=(1000, 3))                  # 10x outside the training box
y_far = X_far[:, 0] * X_far[:, 1] / (4 * np.pi * X_far[:, 2] ** 2)
err = np.max(np.abs(model.predict(X_far) - y_far) / np.abs(y_far))
print(f"max relative error far outside the training range: {err:.1e}")
