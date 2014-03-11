from dolfin import *
from dolfin_adjoint import *

if not hasattr(dolfin, "FunctionAssigner"):
  info_red("Need dolfin.FunctionAssigner for this test.")
  import sys
  sys.exit(0)

mesh = UnitIntervalMesh(2)
V = VectorFunctionSpace(mesh, "CG", 2)
P = FunctionSpace(mesh, "CG", 1)
Z = MixedFunctionSpace([V, P])

def main(p):
  assigner_p = FunctionAssigner(Z.sub(1), P)

  z = Function(Z, name="Output")

  assigner_p.assign(z.sub(1), p)

  return z

if __name__ == "__main__":
  #p = interpolate(Constant(1), P, name="Pressure")
  p = interpolate(Expression("x[0] + 1.0"), P, name="Pressure")
  z = main(p)

  print "Z.dim(): ", Z.dim()
  print "P.dim(): ", P.dim()
  print "p.vector(): ", list(p.vector())
  print "z.vector(): ", list(z.vector())

  A = tuple(p.vector())
  B = tuple(Function(z.sub(1)).vector())
  print "A: ", A
  print "B: ", B

  assert A == B

  assert adjglobals.adjointer.equation_count == 3

  success = replay_dolfin(tol=0.0, stop=True)
  assert success

  form = lambda z: inner(z, z)*dx

  J = Functional(form(z), name="a")
  m = InitialConditionParameter("Pressure")
  Jm = assemble(form(z))
  dJdm = compute_gradient(J, m, forget=False)
  #dJdm.vector()[0] = 1.7499999999999982
  #dJdm.vector()[1] = 0.8333333333333324
  print "dJdm:   ", list(dJdm.vector())

  eps = 0.0001
  dJdm_fd = Function(P)
  for i in range(P.dim()):
    p_ptb = Function(p)
    p_ptb.vector()[i] += eps
    z_ptb = main(p_ptb)
    J_ptb = assemble(form(z_ptb))
    dJdm_fd.vector()[i] = (J_ptb - Jm)/eps

  print "dJdm_fd: ", list(dJdm_fd.vector())

  def Jhat(p):
    z = main(p)
    return assemble(form(z))

  minconv = taylor_test(Jhat, m, Jm, dJdm, seed=1.0e-3)
  assert minconv > 1.8
