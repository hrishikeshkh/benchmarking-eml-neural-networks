"""Run the experiments on Modal (https://modal.com): one single-CPU container per run.

    modal run benchmarks/modal_app.py --kind feynman --methods EML --noise 0.0,0.01
    modal run benchmarks/modal_app.py --kind realworld --group physics --methods EML,MLP
    modal run benchmarks/modal_app.py --kind feynman --methods EML --problems dev --out results/dev.jsonl

Results are streamed back and appended to the same JSON-lines files the local runners use,
so ``analyze.py`` works unchanged and interrupted sweeps resume where they stopped.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parents[1]

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch", index_url="https://download.pytorch.org/whl/cpu")
    .pip_install("numpy", "scipy", "scikit-learn", "sympy", "pandas", "pyyaml", "gplearn",
                 "xlrd", "openpyxl")
    .env({"OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"})
    .add_local_dir(ROOT / "emlkit", "/root/emlkit")
    .add_local_dir(ROOT / "benchmarks", "/root/benchmarks")
    .add_local_dir(ROOT / "data", "/root/data")
)
app = modal.App("emlkit-bench", image=image)


def _setup():
    for p in ("/root", "/root/benchmarks"):
        if p not in sys.path:
            sys.path.insert(0, p)
    import torch
    torch.set_num_threads(1)


@app.function(cpu=1.0, memory=2048, timeout=2 * 3600, retries=1)
def feynman_task(args: tuple) -> dict:
    _setup()
    from run_feynman import run_one
    return run_one(*args)


@app.function(cpu=1.0, memory=3072, timeout=6 * 3600, retries=1)
def realworld_task(args: tuple) -> dict:
    _setup()
    from run_realworld import run_one
    return run_one(*args)


@app.function(cpu=1.0, memory=1024, timeout=600)
def versions() -> str:
    import importlib.metadata as md
    return " ".join(f"{p}=={md.version(p)}" for p in
                    ("torch", "numpy", "scipy", "scikit-learn", "sympy", "gplearn"))


def _done(path: Path, keys) -> set:
    if not path.exists():
        return set()
    out = set()
    for line in path.read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            out.add(tuple(r[k] for k in keys))
    return out


@app.local_entrypoint()
def main(kind: str = "feynman", methods: str = "EML", problems: str = "all", noise: str = "0.0",
         seeds: str = "0", n_train: int = 1000, group: str = "physics", datasets: str = "",
         k: int = 5, out: str = ""):
    sys.path.insert(0, str(ROOT / "benchmarks"))
    print("remote package versions:", versions.remote())
    if kind == "feynman":
        from feynman_data import load_problems
        from run_feynman import DEV_PROBLEMS, ablation_problems
        probs = load_problems()
        names = {"dev": DEV_PROBLEMS, "all": [p.name for p in probs],
                 "notrig": [p.name for p in probs if not p.has_trig],
                 "ablation": ablation_problems()}.get(problems, problems.split(","))
        out_path = Path(out or ROOT / "results" / "feynman.jsonl")
        done = _done(out_path, ("problem", "method", "seed", "noise", "n_train"))
        tasks = [(p, m, int(s), float(nz), n_train) for nz in noise.split(",") for s in seeds.split(",")
                 for m in methods.split(",") for p in names]
        fn = feynman_task
    else:
        from realworld_data import PHYSICS, TABULAR
        names = datasets.split(",") if datasets else (PHYSICS if group == "physics" else TABULAR)
        out_path = Path(out or ROOT / "results" / f"realworld_{group}.jsonl")
        done = {(d, m, s) for d, m, s in _done(out_path, ("dataset", "method", "seed"))}
        tasks = [(d, m, int(s), k, group == "physics") for s in seeds.split(",")
                 for m in methods.split(",") for d in names]
        done_keys = done
        tasks = [t for t in tasks if (t[0], t[1], t[2]) not in done_keys]
        fn = realworld_task
    if kind == "feynman":
        tasks = [t for t in tasks if t not in done]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"{len(tasks)} runs to do ({len(done)} already in {out_path.name})", flush=True)
    with open(out_path, "a") as fh:
        for i, res in enumerate(fn.map(tasks, order_outputs=False, return_exceptions=True), 1):
            if isinstance(res, Exception):
                print(f"[{i}/{len(tasks)}] remote failure: {res!r}", flush=True)
                continue
            fh.write(json.dumps(res) + "\n")
            fh.flush()
            name = res.get("problem") or res.get("dataset")
            if "error" in res:
                msg = res["error"]
            elif kind == "feynman":
                msg = (f"R2 id {res['r2_id']:.6f} ood {res['r2_ood']:.4g}"
                       + (f" sym={res['symbolic']}" if "symbolic" in res else "")
                       + (f"  {res.get('formula', '')[:100]}" if "formula" in res else ""))
            else:
                msg = f"R2 {res['r2']:.4f}  {res.get('formula_full', '')[:100]}"
            print(f"[{i}/{len(tasks)}] {name:28s} {res['method']:8s} s{res['seed']}: {msg} "
                  f"({res.get('time', 0):.0f}s)", flush=True)
