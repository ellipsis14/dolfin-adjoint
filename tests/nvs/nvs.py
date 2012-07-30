from dolfin import *
from dolfin_adjoint import *
import sys

dolfin.parameters["adjoint"]["record_all"] = True

mesh = UnitSquare(4, 4)
V = FunctionSpace(mesh, "CG", 1)
v = TestFunction(V)
f = interpolate(Expression("sin(pi*x[0])"), V)

def main(f, annotate=False):
  u = Function(V)
  a = u*v*dx
  L = f*v*dx
  F = a - L

  bcs = None

  problem = NonlinearVariationalProblem(F, u, bcs, J=derivative(F, u))
  solver = NonlinearVariationalSolver(problem)
  solver.solve(annotate=annotate)

  return u

u = main(f, annotate=True)
replay_dolfin()
grad = compute_gradient(Functional(u*u*dx*dt[FINISH_TIME] + f*f*dx*dt[START_TIME]), InitialConditionParameter(f))

def J(f):
  u = main(f, annotate=False)
  return assemble(u*u*dx + f*f*dx)

minconv = test_initial_condition_adjoint(J, f, grad)
if minconv < 1.9:
  sys.exit(1)
