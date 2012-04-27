import random

from dolfin import *
from dolfin_adjoint import *
import sys

mesh = UnitSquare(4, 4)
V = FunctionSpace(mesh, "CG", 1)
dolfin.parameters["adjoint"]["record_all"] = True

def main(ic, annotate=False):
  u = TrialFunction(V)
  v = TestFunction(V)

  mass = inner(u, v)*dx
  soln = Function(V)

  solve(mass == action(mass, ic), soln, annotate=annotate)
  return soln

if __name__ == "__main__":

  ic = project(Expression("x[0]*(x[0]-1)*x[1]*(x[1]-1)"), V)
  soln = main(ic, annotate=True)

  svd = adj_compute_propagator_svd(ic, soln, 1)
  (sigma, error) = svd.get_svd(0, return_error=True)

  print "Maximal singular value: ", (sigma, error)

  if (abs(sigma - 1.0) > 1.0e-15):
    sys.exit(1)

