# Data

| Path | Source | License |
|---|---|---|
| `raw/feynman_meta/*.yaml` | Feynman equations (Udrescu & Tegmark 2020) as packaged by [PMLB](https://github.com/EpistasisLab/pmlb). We use the formula and variable ranges and re-sample the data ourselves. | MIT (PMLB) |
| `raw/pmlb/*.tsv.gz` | Real-world regression data sets from PMLB, including Cranmer (2023) *EmpiricalBench* physics data and Nikuradse (1933) pipe flow | MIT (PMLB); see each data set's PMLB metadata |
| `raw/uci/*` | UCI ML Repository: yacht hydrodynamics, airfoil self-noise, concrete strength, energy efficiency, auto MPG | CC BY 4.0 |
| `raw/pmlb_summary.tsv` | PMLB data set index | MIT |

Missing PMLB files are downloaded automatically by `benchmarks/realworld_data.py`.
