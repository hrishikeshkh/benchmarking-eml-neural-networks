"""Assemble paper/index.html from paper/template.html, the figures and paper/tables.json.

    python benchmarks/analyze.py && python benchmarks/build_paper.py

Every number in the results sections is read from tables.json, which analyze.py computes
from the raw results in results/*.jsonl.
"""
from __future__ import annotations

import datetime as dt
import html
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
P = ROOT / "paper"


def pct(x):
    return "–" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{100 * x:.0f}%"


def num(x, nd=3):
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return "–"
    return f"{x:.{nd}f}"


def table(headers, rows, ours=None, cls=""):
    th = "".join(f"<th>{h}</th>" for h in headers)
    trs = []
    for r in rows:
        tr_cls = ' class="ours"' if ours is not None and r[0] == ours else ""
        tds = "".join(c if c.startswith("<td") else f"<td>{c}</td>" for c in r)
        trs.append(f"<tr{tr_cls}>{tds}</tr>")
    return f'<div class="table-wrap {cls}"><table><thead><tr>{th}</tr></thead><tbody>{"".join(trs)}</tbody></table></div>'


def main():
    T = json.loads((P / "tables.json").read_text())
    cs = json.loads((P / "compiler_stats.json").read_text())
    fey = T.get("feynman", {})
    f0 = {r["method"]: r for r in fey.get("noise_0.0", [])}
    f1 = {r["method"]: r for r in fey.get("noise_0.01", [])}
    br = fey.get("eml_breakdown", {})
    phys, tab, abl = T.get("physics", {}), T.get("tabular", {}), T.get("ablation", [])
    ctx = dict(f0=f0, f1=f1, br=br, phys=phys, tab=tab, abl=abl, cs=cs, T=T)
    from paper_text import abstract, body  # prose lives next to this script
    page = (P / "template.html").read_text()
    page = page.replace("{{DATE}}", dt.date.today().strftime("%B %Y"))
    page = page.replace("{{AUTHORS}}", "Hrishikesh K Haritas · Carnegie Mellon University")
    page = page.replace("{{ABSTRACT}}", abstract(ctx))
    page = page.replace("{{BODY}}", body(ctx, dict(pct=pct, num=num, table=table, esc=html.escape),
                                         (P / "fig_architecture.svg.html").read_text()))
    (P / "index.html").write_text(page)
    print("wrote", P / "index.html")


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    main()
