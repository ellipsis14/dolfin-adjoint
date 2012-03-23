import random

from dolfin import *
from dolfin_adjoint import *
import sys

mesh = UnitSquare(4, 4)
V3 = FunctionSpace(mesh, "CG", 3)
V2 = FunctionSpace(mesh, "CG", 2)
debugging["record_all"] = True

def main(ic, annotate=False):
  bc = DirichletBC(V2, "-1.0", "on_boundary")
  soln = project(ic, V2, bcs=bc, annotate=annotate)
  return soln

if __name__ == "__main__":

  ic = project(Expression("x[0]*(x[0]-1)*x[1]*(x[1]-1)"), V3)
  soln = main(ic, annotate=True)

  adj_html("projection_forward.html", "forward")
  replay_dolfin()

  J = FinalFunctional(soln*soln*dx)
  for (adjoint, var) in compute_adjoint(J):
    pass

  def J(ic):
    soln = main(ic, annotate=False)
    return assemble(soln*soln*dx)

  minconv = test_initial_condition_adjoint(J, ic, adjoint)
  if minconv < 1.9:
    sys.exit(1)

