from dolfin import *
from dolfin_adjoint import *

if not hasattr(dolfin, "FunctionAssigner"):
  info_red("Need dolfin.FunctionAssigner for this test.")
  import sys
  sys.exit(0)

mesh = UnitSquareMesh(2, 2)
V = VectorFunctionSpace(mesh, "CG", 2)
P = FunctionSpace(mesh, "CG", 1)
Z = MixedFunctionSpace([V, P])

def main(u, p):
  assigner_u = FunctionAssigner(Z.sub(0), V)
  assigner_p = FunctionAssigner(Z.sub(1), P)

  z = Function(Z, name="Output")

  assigner_u.assign(z.sub(0), u)
  assigner_p.assign(z.sub(1), p)

  return z

if __name__ == "__main__":
  u = interpolate(Constant((1, 1)), V, name="Velocity")
  p = interpolate(Constant(0),      P, name="Pressure")
  z = main(u, p)

  assert adjglobals.adjointer.equation_count == 5

  success = replay_dolfin(tol=0.0, stop=True)
  assert success

  form = lambda z: inner(z, z)*dx

  J = Functional(form(z))
  m = InitialConditionParameter("Velocity")
  Jm = assemble(form(z))
  dJdm = compute_gradient(J, m, forget=False)

  def Jhat(u):
    z = main(u, p)
    return assemble(form(z))

  minconv = taylor_test(Jhat, m, Jm, dJdm, seed=1.0e-5)
  assert minconv > 1.8
