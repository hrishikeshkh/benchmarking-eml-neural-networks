"""Prose of the paper.  Numbers come from ``ctx`` (built from paper/tables.json)."""
from __future__ import annotations

VERDICT = {"S": "Supported", "P": "Partly", "N": "Not supported"}
VCLASS = {"S": "best", "P": "", "N": ""}


def abstract(ctx) -> str:
    f0, sub = ctx["f0"], ctx["T"]["feynman"]["eml_subsets"]
    e, m, g = f0["EML"], f0["MLP"], f0["GBM"]
    ho = sub["held-out, no trig @ 0.0"]
    return f"""
<p>Odrzywołek (2026) showed that the single binary operator eml(x, y) = e<sup>x</sup> − ln y, together
with the constant 1, generates every elementary function. We ask whether that operator can serve as the only
primitive of a <em>trainable, interpretable</em> regression model. EML networks are layered graphs of EML nodes
with affine arguments. They are trained by gradient descent, then pruned and snapped to exact exponents. Training
needs four ingredients: least-squares solution of the output node's argument in log space, successive halving
over restarts, an annealed pull of exponents towards multiples of ½, and Levenberg–Marquardt verification of each
pruned or snapped structure.</p>
<p>On the 99 Feynman equations, EML networks recover the exact law for {ho['recovery']:.0%} of the held-out problems
without trigonometric functions ({e['recovery']:.0%} of all problems, against {f0['GPLearn']['recovery']:.0%} for genetic
programming). In distribution they match an MLP: R² &gt; 0.999 on {e['id_r2_999']:.0%} of problems against {m['id_r2_999']:.0%}.
Out of distribution they keep R² &gt; 0.99 on {e['ood_r2_99']:.0%} of problems, against {m['ood_r2_99']:.0%} for the MLP and
{g['ood_r2_99']:.0%} for gradient boosting. {_compute_sentence(ctx)} We also report where the method fails: trigonometric laws,
sign-changing products, exponential blow-ups on small noisy data, and data sets without a compact law. Every claim was
tested against thresholds fixed before the results. We release <span class="mono">emlkit</span>, a scikit-learn
estimator plus a compiler that turns any elementary expression into a pure EML tree.</p>"""


def _compute_sentence(ctx) -> str:
    E = ctx["T"].get("efficiency")
    if not E:
        return ""
    a = E["all"]
    f = E.get("feynman", {})
    return (f"The learned formulas are cheap to run: a median {a['latency_ratio_median']:.0f}× faster than a 2×128 MLP, and "
            f"an MLP needs a median {f.get('match_ratio_median', float('nan')):.0f}× more parameters to match them on the "
            f"Feynman problems. Training, however, is a median {a['train_ratio_median']:.0f}× slower.")


def body(ctx, h, fig1) -> str:
    pct, num, table = h["pct"], h["num"], h["table"]
    f0, f1, T = ctx["f0"], ctx["f1"], ctx["T"]
    sub = T["feynman"]["eml_subsets"]
    claims = T.get("claims", [])
    parts = []

    # ------------------------------------------------------------------ 1
    parts.append("""
<h2><span class="num">1</span>Introduction</h2>
<p>Symbolic regression looks for a formula, not just a predictor. A formula can be read, checked against
theory, and trusted outside the range of the data. Most practical systems search over expression trees built from a
hand-picked operator set (+, −, ×, ÷, exp, log, sin, …) with genetic programming. Odrzywołek's result offers a
different starting point. A single operator, eml(x, y) = e<sup>x</sup> − ln y, is complete for elementary functions
in the way NAND is complete for Boolean logic. A model built from one differentiable primitive can be trained
end to end by gradient descent, and every trained model is still a formula.</p>
<p>Pure EML trees are deep, though. Our compiler needs 23 leaves for x·y and 264 for sin x, and gradients through
nested exp/ln are badly conditioned. This paper makes the idea practical and tests it against concrete,
pre-declared claims (<a href="#claims">Table 1</a>). We contribute:</p>
<ul>
  <li><b>EML networks</b>: EML nodes with affine arguments and typed units, so that products, powers, radicals and
  exponentials of monomials need one to three nodes instead of dozens (§2).</li>
  <li><b>A training recipe</b> that turns near-misses into exact formulas: log-space variable projection, successive
  halving, annealed quantisation, Levenberg–Marquardt verification of structural edits, and an Occam curriculum
  with BIC selection (§3).</li>
  <li><b>An evaluation</b> on 99 Feynman equations (with out-of-distribution tests), 15 real physics data sets and
  34 tabular data sets. It covers accuracy, explainability, comparison with neural networks, compute, and ablations, with a
  scorecard of 18 pre-declared claims (§5).</li>
  <li><b><span class="mono">emlkit</span></b>, an open-source scikit-learn estimator and a pure-EML compiler,
  verifier and shortest-tree search.</li>
</ul>""")

    # ------------------------------------------------------------------ 2
    parts.append(r"""
<h2><span class="num">2</span>EML networks</h2>
<p>An EML network is a layered directed acyclic graph. Its input features are \(z_0 = [x,\ \ln|x|]\); the fixed
log-features are themselves EML nodes, since \(\ln x = 1 - \mathrm{eml}(0, x)\). Every unit sees all features computed
before it through affine maps \(a = W_a z + b_a\) and \(c = W_c z + b_c\), and is one of three types:</p>
\[ E:\ \mathrm{eml}(a, 1) = e^{a}, \qquad L:\ 1 - \mathrm{eml}(0, c) = \ln|c|, \qquad M:\ \mathrm{eml}(a, c) = e^{a} - \ln|c|. \]
<p>E and L are the full operator with one argument fixed to the constant 1, or to 0 = ln 1. Fixing an argument
removes a redundant degree of freedom. Because ln|x| is an input feature, a monomial \(C\prod_j x_j^{p_j}\) is a single E unit,
\(\exp(\sum_j p_j \ln x_j + \ln C)\). The output is either an affine read-out of the last layer, or itself an E node
\(y = s\,\exp(w^\top z + b)\) with a fixed sign \(s\). In the second case the network models
\(\ln|y|\) as a <em>linear</em> function of its features, so products of factors become sums (Figure 1). Pruned
weights are exactly zero and snapped weights are exact rationals, so a trained network is a closed-form expression
that SymPy simplifies directly.</p>
""" + fig1 + r"""
<p>The table below lists the shallowest architecture that represents each family of laws exactly. A star marks a
log-space output node.</p>
""" + table(["Family", "Example (Feynman)", "Architecture"], [
        ["power law ⋅ exp(linear)", "q₁q₂/(4πεr²)", "<code>*</code> (no hidden unit)"],
        ["⋅ exp(monomial)", "n₀ e<sup>−mgx/kT</sup>", "<code>E1*</code>"],
        ["⋅ |affine|<sup>p</sup>", "σ/(ε(1+χ))", "<code>L1*</code>"],
        ["sum of monomials", "Gm₁m₂(1/r₂ − 1/r₁)", "<code>E2</code>, <code>E3</code>"],
        ["⋅ (1 ± monomial)<sup>p</sup>", "m₀/√(1 − v²/c²)", "<code>E1-L1*</code>"],
        ["⋅ (Σ monomials)<sup>p</sup>", "√(x² + y² + z²)", "<code>E3-L1*</code>"],
        ["⋅ (e<sup>monomial</sup> − 1)<sup>p</sup>", "ħω/(e<sup>ħω/kT</sup> − 1)", "<code>E1-E1-L1*</code>"],
        ["⋅ (Σ (difference)²)<sup>p</sup>", "Gm₁m₂/Σ(x₂−x₁)²", "<code>L3-E3-L1*</code>"],
    ]))

    # ------------------------------------------------------------------ 3
    parts.append(r"""
<h2><span class="num">3</span>Training</h2>
<p>All restarts of an architecture are trained in one batched tensor computation. Each parameter carries a
pruning mask and a snapping mask.</p>
<ol>
  <li><b>Variable projection.</b> The read-out is never trained by gradient descent. At every step it is
  solved by ridge least squares, on y for linear outputs and on ln|y| for log-space outputs, with snapped
  entries held fixed. Gradients only shape the EML features (Golub &amp; Pereyra, 1973). For a log-space output this
  also solves every exponent attached to the input logs.</li>
  <li><b>Structured initialisation and successive halving.</b> E units start from sparse integer exponents. In
  half the restarts, units fed by earlier units start as exp(±u) or ln|1 ± u|. Training starts with 4× the target
  number of restarts and keeps the better half twice.</li>
  <li><b>Annealed quantisation.</b> In the second half of dense training a penalty
  \(\rho(t)\sum (w - \tfrac12\mathrm{round}(2w))^2\) pulls argument weights towards multiples of ½.</li>
  <li><b>Prune, snap, verify.</b> Weights below a magnitude threshold, and read-out terms that contribute less than
  a fraction of std(y), are removed. Weights near a multiple of ½ are frozen there. After each edit the free
  parameters are refit with Adam, then with Levenberg–Marquardt, and a restart is rolled back if its validation
  error rose by more than 10%. Levenberg–Marquardt is what makes an exact structure <em>look</em> exact: started at
  the true solution of II.13.23, Adam fine-tuning drifts to NMSE ≈ 10<sup>−7</sup>, while LM returns to 10<sup>−19</sup>.
  Restarts are independent, so perturbing parameter slot <i>k</i> in every restart at once yields column <i>k</i> of
  every Jacobian in one forward-mode pass.</li>
  <li><b>Occam curriculum.</b> Eleven architectures, from the plain power law <code>*</code> to a wide
  <code>E4-L2-E2</code> with skip read-out, are tried in order of complexity. Log-space stages are used only when
  y has a constant sign. The search stops early when a candidate fits the validation data to NMSE &lt; 10<sup>−12</sup>.
  The final model minimises BIC = n ln(NMSE + 10<sup>−14</sup>) + k ln n over all candidates.</li>
</ol>""")

    # ------------------------------------------------------------------ 4
    parts.append("""
<h2><span class="num">4</span>Experimental setup</h2>
<p><b>Feynman.</b> All 99 equations of the AI Feynman database (Udrescu &amp; Tegmark, 2020), with formulas and
variable ranges from PMLB. For each problem we draw 1,000 training points from the original box, 2,000 test points
from the same box (ID), and 2,000 points with every variable in [hi, 2·hi − lo] (OOD, pure extrapolation). Label noise
is 0 or 1% of std(y); test sets are noiseless. Recovery is the SRBench criterion: after rounding constants to three
decimals, model − truth or model / truth simplifies to a constant. Eight problems were used to develop the method and
are reported separately as the dev set.</p>
<p><b>Real data.</b> Fifteen physics data sets of measured or simulated laws, from PMLB's <i>first principles</i>
collection (Cranmer, 2023) and Nikuradse's pipe-flow measurements. Thirty-four general tabular sets: five UCI
classics (yacht, airfoil, concrete, energy, auto-MPG) and every small real-world PMLB regression set with ≤ 12
features and ≤ 5,000 rows, excluding synthetic generators. Metric: pooled out-of-fold R² from 5-fold CV
(leave-one-out for n &lt; 20).</p>
<p><b>Baselines</b> (default settings, no test-set tuning): Ridge regression; an MLP with two hidden layers of 128 ReLU
units, standardised inputs and target, and early stopping; histogram gradient boosting; random forest; GPLearn genetic
programming (population 2,000, 30 generations). EML uses the library defaults (24 restarts per stage, 1,000-step
budget per stage). Every run uses one CPU core on Modal with identical package versions.</p>
<p><b>Compute.</b> On one split per problem (Feynman with 1% noise; tabular with a 75/25 split) we compare EML with
MLPs of nine sizes and with gradient boosting in parameters, inference latency and training time (§5.4).</p>""")

    # ------------------------------------------------------------------ 5
    rows = [[c["id"], c["claim"], c["evidence"],
             f'<td class="{VCLASS[c["verdict"]]}">{VERDICT[c["verdict"]]}</td>'] for c in claims]
    parts.append(f"""
<h2 id="claims"><span class="num">5</span>Results</h2>
<p>Table 1 states each claim, the evidence and the verdict, computed mechanically from the thresholds declared in
<span class="mono">CLAIMS.md</span> before the full results existed.</p>
<p class="tcap"><b>Table 1.</b> Claims scorecard.</p>
{table(["#", "Claim", "Evidence", "Verdict"], rows, cls="claims")}""")

    def frow(r, noise_rows):
        return [r["method"], pct(r["recovery"]) if r["recovery"] is not None else "–",
                pct(r["id_r2_999"]), pct(r["ood_r2_99"]), pct(r["ood_exact"]),
                num(r["size_median"], 0) if r["size_median"] is not None else "–", num(r["time_median"], 0)]

    hdr = ["Method", "Exact recovery", "R²<sub>ID</sub> &gt; 0.999", "R²<sub>OOD</sub> &gt; 0.99", "OOD NMSE &lt; 10⁻⁹",
           "Median size", "Median s"]
    parts.append(f"""
<h3><span class="num">5.1</span>Feynman equations</h3>
<p class="tcap"><b>Table 2.</b> 99 Feynman problems without noise (top) and with 1% label noise (bottom).</p>
{table(hdr, [frow(r, f0) for r in f0.values()], ours="EML")}
{table(hdr, [frow(r, f1) for r in f1.values()], ours="EML") if f1 else ""}
<figure>
  <img src="figures/feynman_profile_noise0.0.svg" alt="Share of Feynman problems solved below each NMSE threshold, in and out of distribution, for EML, GPLearn, MLP and GBM.">
  <figcaption><b>Figure 2.</b> Share of the 99 problems whose test NMSE is below τ. In distribution the MLP is close to EML
  at moderate thresholds, but only exact formulas reach the left of the plot. Out of distribution, the gap between laws
  and fitted surfaces opens over the whole range.</figcaption>
</figure>
<p>{_feynman_text(ctx, h)}</p>""")

    parts.append(_realworld_section(ctx, h))
    parts.append(_ablation_section(ctx, h))
    parts.append(_compute_section(ctx, h))
    parts.append(_compiler_section(ctx, h))

    # ------------------------------------------------------------------ 6-8
    parts.append(f"""
<h2><span class="num">6</span>Limitations</h2>
<ul>
  <li><b>Trigonometric laws.</b> Real-valued EML networks cannot express sin or cos compactly (§5.5). EML recovers
  {pct(sub.get('trig @ 0.0', {}).get('recovery'))} of the trigonometric Feynman problems. Complex-valued units, where
  ln(−1) = iπ, are the natural extension.</li>
  <li><b>Sign-changing products.</b> Log-space stages need y of constant sign, and ln|·| loses the sign of a factor.
  A law such as n k T ln(V₂/V₁) is therefore only approximated.</li>
  <li><b>Training cost.</b> A fit takes seconds when an early curriculum stage is exact and minutes otherwise, typically
  two orders of magnitude more than an MLP (C17).</li>
  <li><b>Blow-ups on small noisy data.</b> On a few small tabular sets a learned exp(·) explodes on held-out folds
  (CV R² &lt; −1). Clipping predictions to the training range would hide this, so we do not.</li>
  <li><b>A missing stage.</b> A target that is a plain sum of logarithms (the <i>ideal_gas</i> set stores ln P) is
  not represented by any strict curriculum stage, and EML fails on it.</li>
  <li><b>Scope.</b> One seed per configuration, default hyper-parameters, and GPLearn as the only GP baseline.
  Stronger GP systems (PySR, Operon) would make a harder comparison.</li>
</ul>

<h2><span class="num">7</span>Related work</h2>
<p>Gradient-trained symbolic layers go back to product units (Durbin &amp; Rumelhart, 1989) and the Equation Learner
(Martius &amp; Lampert, 2016; Sahoo et al., 2018). Kolmogorov–Arnold networks (Liu et al., 2024) learn univariate
splines that are symbolified afterwards. Genetic programming dominates practice (Koza, 1992; Cranmer, 2023), and
SRBench (La Cava et al., 2021) standardised its evaluation. AI Feynman (Udrescu &amp; Tegmark, 2020) introduced
the benchmark used here. EML networks differ in using one operator for every node and in solving the outer formula
exactly by log-space variable projection.</p>

<h2><span class="num">8</span>Conclusion</h2>
<p>A single operator is enough to build a trainable model whose fitted parameters <em>are</em> a formula. With the
right optimisation scaffolding, gradient descent on EML networks recovers many physical laws exactly. Those laws keep
their accuracy far outside the training range, where neural networks and tree ensembles fail. On data without a
compact law the formulas remain readable, at a measured cost in accuracy. Code, data, raw results and this paper
are reproducible with <span class="mono">emlkit</span>.</p>

<h2>References</h2>
<ul class="refs">
  <li>Cranmer, M. (2023). Interpretable machine learning for science with PySR and SymbolicRegression.jl. arXiv:2305.01582.</li>
  <li>Durbin, R. &amp; Rumelhart, D. (1989). Product units: a computationally powerful and biologically plausible extension to backpropagation networks. <i>Neural Computation</i> 1(1).</li>
  <li>Golub, G. &amp; Pereyra, V. (1973). The differentiation of pseudo-inverses and nonlinear least squares problems whose variables separate. <i>SIAM J. Numer. Anal.</i> 10(2).</li>
  <li>Koza, J. (1992). <i>Genetic Programming</i>. MIT Press.</li>
  <li>La Cava, W. et al. (2021). Contemporary symbolic regression methods and their relative performance. NeurIPS Datasets and Benchmarks.</li>
  <li>Liu, Z. et al. (2024). KAN: Kolmogorov–Arnold Networks. arXiv:2404.19756.</li>
  <li>Martius, G. &amp; Lampert, C. (2016). Extrapolation and learning equations. arXiv:1610.02995.</li>
  <li>Nikuradse, J. (1933). Strömungsgesetze in rauhen Rohren. VDI-Forschungsheft 361.</li>
  <li>Odrzywołek, A. (2026). All elementary functions from a single binary operator. arXiv:2603.21852.</li>
  <li>Sahoo, S., Lampert, C. &amp; Martius, G. (2018). Learning equations for extrapolation and control. ICML.</li>
  <li>Udrescu, S.-M. &amp; Tegmark, M. (2020). AI Feynman: a physics-inspired method for symbolic regression. <i>Science Advances</i> 6(16).</li>
</ul>""")
    return "\n".join(parts)


def _feynman_text(ctx, h) -> str:
    pct = h["pct"]
    f0, f1, sub = ctx["f0"], ctx["f1"], ctx["T"]["feynman"]["eml_subsets"]
    e = f0["EML"]
    s = (f"EML recovers {pct(e['recovery'])} of all problems exactly ({pct(sub['held-out @ 0.0']['recovery'])} of the held-out "
         f"set, {pct(sub['dev @ 0.0']['recovery'])} of the dev set, and {pct(sub['held-out, no trig @ 0.0']['recovery'])} of "
         f"held-out problems without trigonometric functions). GPLearn recovers {pct(f0['GPLearn']['recovery'])}. ")
    s += (f"Every method that fits well in distribution looks similar there, but extrapolation separates them. "
          f"{pct(e['ood_exact'])} of EML models stay exact (NMSE &lt; 10⁻⁹) far outside the training box, "
          f"against {pct(f0['MLP']['ood_exact'])} for the MLP and {pct(f0['GBM']['ood_exact'])} for gradient boosting. ")
    if "EML" in f1:
        s += (f"With 1% label noise EML still recovers {pct(f1['EML']['recovery'])} of the laws, and keeps OOD R² &gt; 0.99 on "
              f"{pct(f1['EML']['ood_r2_99'])} of problems.")
    return s


def _realworld_section(ctx, h) -> str:
    pct, num, table, esc = h["pct"], h["num"], h["table"], h["esc"]
    phys, tab = ctx["phys"], ctx["tab"]
    if not phys and not tab:
        return ""
    out = ['<h3><span class="num">5.2</span>Real data</h3>']
    if phys:
        ms = [s["method"] for s in phys["summary"]]
        rows = []
        for r in phys["rows"]:
            vals = []
            best = max((v for v in r["r2"].values() if v is not None), default=None)
            for m in ms:
                v = r["r2"].get(m)
                cls = ' class="best"' if v is not None and best is not None and v >= best - 1e-9 else ""
                vals.append(f"<td{cls}>{num(v)}</td>")
            rows.append([r["dataset"].replace("first_principles_", ""), str(r["n"]), *vals,
                         f'<td class="formula">{esc((r.get("eml_formula") or "")[:110])}</td>'])
        out.append(f"""<p class="tcap"><b>Table 3.</b> Physics data: 5-fold CV R² (leave-one-out for n &lt; 20); best value per row
in green. The last column is the EML formula fit to all data.</p>
{table(["Data set", "n", *ms, "EML formula"], rows)}""")
    if tab:
        s = {x["method"]: x for x in tab["summary"]}
        rows = [[m, num(s[m]["median_r2"]), num(s[m]["mean_rank"], 2)] for m in s]
        out.append(f"""<p class="tcap"><b>Table 4.</b> 34 general tabular data sets: median 5-fold CV R² and mean rank (1 = best).</p>
{table(["Method", "Median R²", "Mean rank"], rows, ours="EML")}
<figure>
  <img src="figures/tabular_eml_vs_mlp.svg" alt="Scatter of EML versus MLP cross-validated R squared on 34 tabular data sets.">
  <figcaption><b>Figure 3.</b> EML closed-form models against the MLP on general tabular data. Points above the dashed
  line are within 0.02 R² of the MLP or better.</figcaption>
</figure>""")
    return "\n".join(out)


def _ablation_section(ctx, h) -> str:
    pct, table = h["pct"], h["table"]
    d = ctx["T"].get("ablation_drops")
    if not d:
        return ""
    names = {"EML-noLogVP": "no log-space projection (linear output only)", "EML-noVP": "no variable projection",
             "EML-skip": "read-out sees all features", "EML-wide": "no curriculum (one wide net)",
             "EML-untyped": "untyped M units only", "EML-noQ": "no annealed quantisation",
             "EML-noLM": "no Levenberg–Marquardt", "EML-noHalving": "no successive halving"}
    rows = [["full method", pct(d["base"]), "–"]]
    for k, v in sorted(d["drops"].items(), key=lambda kv: -kv[1]):
        rows.append([names.get(k, k), pct(d["base"] - v), f"{-100 * v:+.0f} pts"])
    return f"""<h3><span class="num">5.3</span>Ablations</h3>
<p class="tcap"><b>Table 5.</b> Exact recovery on {d['n']} held-out non-trigonometric Feynman problems (every other one,
fixed list) when one component is removed.</p>
{table(["Variant", "Recovery", "Change"], rows, ours="full method")}"""


def _compute_section(ctx, h) -> str:
    num, table, pct = h["num"], h["table"], h["pct"]
    E = ctx["T"].get("efficiency")
    if not E:
        return ""
    rows = []
    for kind, label in (("feynman", "Feynman, 1% noise"), ("tabular", "Tabular")):
        k = E.get(kind)
        if not k:
            continue
        rows.append([label, str(k["n"]), num(k["eml_r2_median"]), num(k["mlp_r2_median"]), num(k["eml_params_median"], 0),
                     f"{k['match_ratio_median']:.0f}×", pct(k["eml_beats_param_matched"]),
                     f"{k['latency_ratio_median']:.0f}×", f"{k['train_ratio_median']:.0f}×"])
    return f"""<h3><span class="num">5.4</span>Compute</h3>
<p>For each problem we also fit MLPs with one hidden layer of 1 to 128 units, plus the 2×128 baseline, and measured
parameters, single-core NumPy inference time on 100k rows, and training time. EML formulas are tiny and fast to
evaluate. The price is paid during training.</p>
<p class="tcap"><b>Table 6.</b> Medians per task. <i>Params ratio</i>: parameters of the smallest MLP that matches EML's
test R² within 0.005, divided by EML's (EML wins outright where no MLP matches). <i>vs. same-size MLP</i>: share of tasks
where EML ≥ the smallest MLP with at least as many parameters.</p>
{table(["Tasks", "n", "EML R²", "2×128 MLP R²", "EML params", "Params ratio", "vs. same-size MLP", "Inference speed-up", "Training slow-down"], rows)}
<figure>
  <img src="figures/efficiency_params_vs_r2.svg" alt="Median test R squared against parameter count for MLPs of every size, EML and gradient boosting.">
  <figcaption><b>Figure 4.</b> Median test R² against parameter count. The MLP curve has to grow by orders of magnitude to
  reach the accuracy of the EML formula on the Feynman problems; on tabular data the gap is smaller.</figcaption>
</figure>"""


def _compiler_section(ctx, h) -> str:
    table = h["table"]
    cs = ctx["cs"]
    rows = [[f"<code>{r['expr']}</code>", str(r["leaves"]), str(r["depth"]), str(r["dag"]), f"{r['err']:.0e}"]
            for r in cs["compile"]]
    return f"""<h3><span class="num">5.5</span>The pure EML compiler</h3>
<p><span class="mono">emlkit.tree</span> compiles any elementary SymPy expression into a tree over {{eml, 1, x}} using
textbook identities, for example ln x = eml(1, eml(eml(1, x), 1)) and a − b = eml(ln a, e<sup>b</sup>). It
evaluates the tree with the principal complex logarithm and checks it against the source expression. The constructions
are correct, not minimal. Exhaustive search confirms the shortest trees for exp and ln, with 2 and 4 leaves. The sizes
explain why we train shallow typed networks rather than pure trees: x·y already needs 23 leaves and sin x needs 264.
Fractional powers of negative numbers can land on the other branch of the logarithm. We verify on domains where the
source expression is real.</p>
<p class="tcap"><b>Table 7.</b> Pure EML trees produced by the compiler (leaves, depth, distinct nodes when identical
subtrees are shared) and maximum relative error at 64 random points.</p>
{table(["Expression", "Leaves", "Depth", "Shared nodes", "Max error"], rows)}"""
