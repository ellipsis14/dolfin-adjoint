from dolfin import *
from dolfin_adjoint import *

mesh = UnitSquareMesh(2, 2)
V = FunctionSpace(mesh, "CG", 1)
u = TrialFunction(V)
v = TestFunction(V)
m = Function(V)

a = u*v*dx
L = (m + 1)*v*dx

u = Function(V)
solve(a == L, u)

J = Functional(u*u*dx)

Jtilde = ReducedFunctional(J, SteadyParameter(m))
opt_problem = OptimizationProblem(Jtilde)

