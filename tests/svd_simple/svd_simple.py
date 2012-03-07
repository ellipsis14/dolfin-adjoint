import random

from dolfin import *
from dolfin_adjoint import *
import sys

mesh = UnitSquare(4, 4)
V = FunctionSpace(mesh, "CG", 3)
debugging["record_all"] = True

def main(ic, annotate=False):
  u = TrialFunction(V)
  v = TestFunction(V)

  bc = DirichletBC(V, "-1.0", "on_boundary")

  mass = inner(u, v)*dx
  soln = Function(V)

  solve(mass == action(mass, ic), soln, bc, annotate=annotate)
  return soln

if __name__ == "__main__":

  ic = project(Expression("x[0]*(x[0]-1)*x[1]*(x[1]-1)"), V)
  soln = main(ic, annotate=True)

  svd = adj_compute_tlm_svd(ic, soln, 1)
  print "Maximal singular value: ", svd.get_svd(0, return_error=True)

