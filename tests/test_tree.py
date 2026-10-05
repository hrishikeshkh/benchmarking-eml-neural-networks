import numpy as np
import pytest

from emlkit.tree import (ONE, Var, compile_expr, depth, eml, evaluate, leaves, parse,
                         search, to_string, verify)

EXPRS = ["exp(x)", "log(x)", "E", "x - y", "-x", "x + y", "x*y", "x/y", "x**y", "2", "1/2",
         "pi", "I", "sqrt(x)", "x**2 + 1", "sin(x)", "cos(x)", "tanh(x)",
         "exp(-x**2/2)/sqrt(2*pi)", "asin(x/4)", "atan(x)", "log(x)*y - 3*x/y"]


@pytest.mark.parametrize("expr", EXPRS)
def test_compile_matches_sympy(expr):
    assert verify(compile_expr(expr), expr) < 1e-10


def test_known_identities():
    x = np.linspace(0.5, 3, 7)
    assert np.allclose(evaluate(eml(Var("x"), ONE), {"x": x}), np.exp(x))
    ln_tree = parse("eml(1, eml(eml(1, x), 1))")
    assert np.allclose(evaluate(ln_tree, {"x": x}), np.log(x))


def test_parse_roundtrip():
    t = compile_expr("x*y + 1")
    assert parse(to_string(t)) == t


def test_sizes():
    t = compile_expr("log(x)")
    assert leaves(t) == 4 and depth(t) == 3


def test_search_finds_minimal_trees():
    assert to_string(search("exp(x)")) == "eml(x, 1)"
    assert leaves(search("log(x)")) == 4
    assert leaves(search("exp(x) - 1")) == 3
