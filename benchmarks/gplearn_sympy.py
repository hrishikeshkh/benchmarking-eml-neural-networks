"""Convert a fitted gplearn program into a SymPy expression (protected ops as plain ops)."""
from __future__ import annotations

from typing import Sequence

import sympy as sp

_OPS = {
    "add": lambda a, b: a + b,
    "sub": lambda a, b: a - b,
    "mul": lambda a, b: a * b,
    "div": lambda a, b: a / b,
    "sqrt": lambda a: sp.sqrt(sp.Abs(a)),
    "log": lambda a: sp.log(sp.Abs(a)),
    "neg": lambda a: -a,
    "inv": lambda a: 1 / a,
    "sin": sp.sin,
    "cos": sp.cos,
    "abs": sp.Abs,
    "max": sp.Max,
    "min": sp.Min,
    "tan": sp.tan,
}


def gplearn_to_sympy(program, names: Sequence[str]) -> sp.Expr:
    syms = [sp.Symbol(n, positive=True) for n in names]
    stack = []
    for node in program.program:
        if hasattr(node, "arity"):
            stack.append([node.name, node.arity, []])
        else:
            val = syms[node] if isinstance(node, int) else sp.Float(node)
            stack[-1][2].append(val) if stack else stack.append(["const", 0, [val]])
        while stack and len(stack[-1][2]) == stack[-1][1]:
            name, _, args = stack.pop()
            res = _OPS[name](*args)
            if stack:
                stack[-1][2].append(res)
            else:
                return res
    if len(stack) == 1 and stack[0][0] == "const":
        return stack[0][2][0]
    raise ValueError("malformed program")
