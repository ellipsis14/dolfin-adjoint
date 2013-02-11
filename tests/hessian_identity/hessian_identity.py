from dolfin import *
from dolfin_adjoint import *

mesh = UnitSquareMesh(3, 3)
V = FunctionSpace(mesh, "CG", 1)

test = TestFunction(V)
trial = TrialFunction(V)

def main(m):
  u = Function(V, name="Solution")

  F = inner(trial, test)*dx - inner(m, test)*dx
  solve(lhs(F) == rhs(F), u)

  return u

if __name__ == "__main__":
  m = interpolate(Constant(1), V, name="Parameter")
  u = main(m)

  parameters["adjoint"]["stop_annotating"] = True

  J = Functional((inner(u, u))**2*dx, name="NormSquared")
  dJdm = compute_gradient(J, TimeConstantParameter(m), forget=False)
  HJm  = hessian(J, TimeConstantParameter(m))

  adj_html("forward.html", "forward")

  def Jhat(m):
    u = main(m)
    return assemble(inner(u, u)**2*dx)

  Jm = Jhat(m)

  minconv = taylor_test(Jhat, TimeConstantParameter(m), Jm, dJdm, HJm=HJm)
  assert minconv > 2.9
