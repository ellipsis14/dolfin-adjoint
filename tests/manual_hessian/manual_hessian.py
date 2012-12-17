# This test codes the tangent linear, first-order adjoint
# and second-order adjoints *by hand*.
# It was developed as part of the development process of the Hessian
# functionality, to build intuition.

# We're going to solve the steady Burgers' equation
# u . grad(u) - grad^2 u - f = 0
# and differentiate a functional of the solution u with respect to the
# parameter f.

from dolfin import *
from dolfin_adjoint import *

parameters["adjoint"]["stop_annotating"] = True

mesh = UnitSquare(10, 10)
Vu = VectorFunctionSpace(mesh, "CG", 2)
Vm = VectorFunctionSpace(mesh, "CG", 1)
u = Function(Vu, name="Velocity")
bcs = [DirichletBC(Vu, (1.0, 1.0), "on_boundary")]
hbcs = [homogenize(bc) for bc in bcs]

def F(m):
  u_test = TestFunction(Vu)

  F = (inner(dot(grad(u), u), u_test)*dx +
       inner(grad(u), grad(u_test))*dx +
      -inner(m, u_test)*dx)

  return F

def main(m):
  Fm = F(m)
  solve(Fm == 0, u, J=derivative(Fm, u), bcs=bcs)
  return u

def J(u, m):
  return inner(u, u)*dx + 0.5*inner(m, m)*dx

def tlm(u, m, mdot):
  Fm = F(m)
  dFmdu = derivative(Fm, u)
  dFmdm = derivative(Fm, m, mdot)
  udot = Function(Vu)

  solve(action(dFmdu, udot) + dFmdm == 0, udot, bcs=hbcs)
  return udot

if __name__ == "__main__":
  m = interpolate(Expression(("sin(x[0])", "cos(x[1])")), Vm)
  u = main(m)
  Jm = J(u, m)

  mdot = interpolate(Constant((1.0, 1.0)), Vm)

  u_tlm = tlm(u, m, mdot)
