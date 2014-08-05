from dolfin import *
from dolfin_adjoint import *

mesh = UnitSquareMesh(4,4)
V = FunctionSpace(mesh, "CG", 1)

class SourceExpression(Expression):
    def __init__(self, c):
        self.c = c

    def eval(self, value, x):
        value[0] = float(self.c)

c = Constant(1)
source = SourceExpression(c)

s = project(source, V, annotate=True)

assert max(abs(s.vector().array() - 1)) < 1e-5

adj_html("forward.html", "forward")
