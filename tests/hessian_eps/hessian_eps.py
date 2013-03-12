from dolfin import *
from dolfin_adjoint import *

parameters["adjoint"]["cache_factorizations"] = True

mesh = UnitSquareMesh(3, 3)
V = FunctionSpace(mesh, "CG", 1)

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
  HJm  = hessian(J, TimeConstantParameter(m), warn=False)

  eps = HJm.eigendecomposition(n=3, solver='krylovschur')
  for i in range(len(eps)):
    (lamda, m) = eps[i]
    output = HJm(m)
    residual = assemble(inner(lamda*m - output, lamda*m - output)*dx)
    print "(%s) eigenvector: " % i, m.vector().array()
    print "(%s) lamda: " % i, lamda
    print "(%s) residual^2: " % i, residual

    assert residual < 1.0e-10
