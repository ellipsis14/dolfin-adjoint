import random

from dolfin import *
from dolfin_adjoint import *
import sys

mesh = UnitSquare(4, 4)
V = FunctionSpace(mesh, "CG", 3)
debugging["record_all"] = True
a = Constant(2.0)

def main(ic, annotate=False):
  u = TrialFunction(V)
  v = TestFunction(V)

  bc = DirichletBC(V, "-1.0", "on_boundary")

  mass = inner(u, v)*dx
  rhs = a*action(mass, ic)
  soln = Function(V)
  da = Function(V)

  print "assemble(rhs): ", assemble(rhs)
  print "assemble(derivative(rhs, a, da)): ", assemble(derivative(rhs, a, da))

  solve(mass == rhs, soln, bc, annotate=annotate)
  return soln

if __name__ == "__main__":

  ic = project(Expression("x[0]*(x[0]-1)*x[1]*(x[1]-1)"), V)
  soln = main(ic, annotate=True)

  J = FinalFunctional(soln*soln*dx)
  dJda = compute_gradient(J, a)

  print "dJda: ", dJda

