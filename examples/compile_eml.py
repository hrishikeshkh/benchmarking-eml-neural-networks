"""Pure EML trees: compile, verify and search."""
from emlkit.tree import compile_expr, depth, leaves, search, to_string, verify

for expr in ["log(x)", "x*y", "sqrt(x)", "sin(x)", "exp(-x**2/2)/sqrt(2*pi)"]:
    t = compile_expr(expr)
    print(f"{expr:26s} leaves={leaves(t):4d} depth={depth(t):3d} max error={verify(t, expr):.1e}")

print(to_string(compile_expr("x - y")))
print("shortest tree for e - ln(x):", to_string(search("E - log(x)")))
