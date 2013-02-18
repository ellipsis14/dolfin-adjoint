from dolfin import *
from dolfin_adjoint import *

mesh = UnitSquareMesh(3, 3)
V = FunctionSpace(mesh, "R", 0)

test = TestFunction(V)
trial = TrialFunction(V)

def main(m):
  u = interpolate(Constant(0.1), V, name="Solution")

  F = inner(u*u, test)*dx - inner(m, test)*dx
  solve(F == 0, u)
  F = inner(sin(u)*u*u*trial, test)*dx - inner(u**4, test)*dx
  solve(lhs(F) == rhs(F), u)

  return u

if __name__ == "__main__":
  m = interpolate(Constant(1), V, name="Parameter")
  u = main(m)

  parameters["adjoint"]["stop_annotating"] = True

  J = Functional((inner(u, u))**6*dx, name="NormSquared")
  Jm = assemble(inner(u, u)**6*dx)
  dJdm = compute_gradient(J, TimeConstantParameter(m), forget=None)
  HJm  = hessian(J, TimeConstantParameter(m), policy="default")

  def Jhat(m):
    u = main(m)
    return assemble(inner(u, u)**6*dx)

  minconv = taylor_test(Jhat, TimeConstantParameter(m), Jm, dJdm, HJm=HJm, perturbation_direction=interpolate(Constant(0.1), V))
  assert minconv > 2.9
