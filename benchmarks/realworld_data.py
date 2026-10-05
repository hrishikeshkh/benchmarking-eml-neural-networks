"""Real-world regression datasets: physics measurements and general tabular data.

PMLB datasets are downloaded on first use from the PMLB GitHub repository and cached in
``data/raw/pmlb``; UCI files are expected in ``data/raw/uci`` (see ``data/README.md``).
"""
from __future__ import annotations

import urllib.request
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

RAW = Path(__file__).resolve().parents[1] / "data" / "raw"
PMLB_URL = "https://github.com/EpistasisLab/pmlb/raw/master/datasets/{0}/{0}.tsv.gz"

# Measured / empirical physical laws (Cranmer 2023 "EmpiricalBench" + Nikuradse 1933).
PHYSICS = [
    "first_principles_absorption", "first_principles_bode", "first_principles_hubble",
    "first_principles_ideal_gas", "first_principles_kepler", "first_principles_leavitt",
    "first_principles_newton", "first_principles_planck", "first_principles_rydberg",
    "first_principles_schechter", "first_principles_supernovae_zg",
    "first_principles_supernovae_zr", "first_principles_tully_fisher",
    "nikuradse_1", "nikuradse_2",
]

# General tabular regression (UCI classics + small real-world PMLB sets, no synthetic data).
TABULAR_UCI = ["uci_yacht", "uci_airfoil", "uci_concrete", "uci_energy_heating", "uci_auto_mpg"]
TABULAR_PMLB = [
    "1027_ESL", "1028_SWD", "1029_LEV", "1030_ERA", "1096_FacultySalaries", "192_vineyard",
    "210_cloud", "228_elusage", "230_machine_cpu", "485_analcatdata_vehicle", "519_vinnie",
    "522_pm10", "523_analcatdata_neavote", "529_pollen", "547_no2",
    "556_analcatdata_apnea2", "557_analcatdata_apnea1", "561_cpu", "659_sleuth_ex1714",
    "663_rabe_266", "665_sleuth_case2002", "666_rmftsa_ladata",
    "678_visualizing_environmental", "687_sleuth_ex1605", "690_visualizing_galaxy",
    "695_chatfield_4", "706_sleuth_case1202", "712_chscase_geyser1", "solar_flare",
]
TABULAR = TABULAR_UCI + TABULAR_PMLB


def _pmlb(name: str) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    path = RAW / "pmlb" / f"{name}.tsv.gz"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(PMLB_URL.format(name), path)
    df = pd.read_csv(path, sep="\t", compression="gzip")
    y = df.pop("target").to_numpy(float)
    return df.to_numpy(float), y, [str(c) for c in df.columns]


def _uci(name: str) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    d = RAW / "uci"
    if name == "uci_yacht":
        cols = ["lcb", "cp", "ld", "bd", "lb", "Fr", "resistance"]
        df = pd.read_csv(d / "yacht_hydrodynamics.data", sep=r"\s+", header=None, names=cols)
    elif name == "uci_airfoil":
        cols = ["freq", "angle", "chord", "velocity", "thickness", "sound"]
        df = pd.read_csv(d / "airfoil_self_noise.dat", sep="\t", header=None, names=cols)
    elif name == "uci_concrete":
        df = pd.read_excel(d / "Concrete_Data.xls")
        df.columns = ["cement", "slag", "ash", "water", "superplast", "coarse", "fine", "age", "strength"]
    elif name == "uci_energy_heating":
        df = pd.read_excel(d / "ENB2012_data.xlsx").iloc[:, :9].dropna()
        df.columns = ["compact", "surface", "wall", "roof", "height", "orient", "glazing", "glaz_dist", "heating"]
    elif name == "uci_auto_mpg":
        cols = ["mpg", "cyl", "displ", "hp", "weight", "accel", "year", "origin", "name"]
        df = pd.read_csv(d / "auto-mpg.data", sep=r"\s+", header=None, names=cols, na_values="?")
        df = df.drop(columns="name").dropna()
        df = df[["cyl", "displ", "hp", "weight", "accel", "year", "origin", "mpg"]]
    else:
        raise KeyError(name)
    y = df.iloc[:, -1].to_numpy(float)
    X = df.iloc[:, :-1]
    return X.to_numpy(float), y, [str(c) for c in X.columns]


def load(name: str) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    X, y, names = _uci(name) if name.startswith("uci_") else _pmlb(name)
    names = [n if n.isidentifier() else f"x{i}" for i, n in enumerate(names)]
    return X, y, names


if __name__ == "__main__":
    for nm in PHYSICS + TABULAR:
        X, y, cols = load(nm)
        print(f"{nm:32s} n={len(y):5d} d={X.shape[1]:2d} {cols[:8]}")
