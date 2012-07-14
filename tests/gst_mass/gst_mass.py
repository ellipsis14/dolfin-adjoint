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

  svd = compute_gst(ic, soln, 1, ic_norm=None, final_norm="mass")
  (sigma, u, v, error) = svd.get_gst(0, return_vectors=True, return_error=True)

  print "Maximal singular value: ", (sigma, error)
  u_l2 = assemble(inner(u, u)*dx)
  print "L2 norm of u: ", u_l2
  assert near(u_l2, 1.0)

