from dolfin import *
from dolfin_adjoint import *
import ufl

mesh = UnitSquareMesh(3, 3)
V = FunctionSpace(mesh, "CG", 1)

test = TestFunction(V)
trial = TrialFunction(V)

def Fm(m):
  F = inner(trial, test)*dx - inner(m, test)*dx
  return F

def main(m):
  u = Function(V)

  F = Fm(m)
  solve(lhs(F) == rhs(F), u)

  return u

def HJ(u, m):

  J = inner(u, u)*dx
  dJdu = derivative(J, u)
  F = Fm(m)
  dFdu = lhs(F)
  dFdm = ufl.algorithms.expand_derivatives(derivative(F, m))

  def HJm(mdot):
    u_tlm = mdot

    u_soa = Function(V)
    d2Jdu2 = derivative(dJdu, u, u_tlm)
    solve(adjoint(dFdu) == d2Jdu2, u_soa)

    der = -action(adjoint(dFdm), u_soa)
    result = assemble(der)
    return Function(V, result)

  return HJm


if __name__ == "__main__":
  m = interpolate(Constant(1), V)
  u = main(m)

  J = inner(u, u)*dx
  dJdu = derivative(J, u)

  u_adj = Function(V)
  F = Fm(m)
  dFdu = lhs(F)
  solve(adjoint(dFdu) == dJdu, u_adj)
  dFdm = ufl.algorithms.expand_derivatives(derivative(F, m))
  dJdm_vec = assemble(-action(adjoint(dFdm), u_adj))
  dJdm = Function(V, dJdm_vec)

  def Jhat(m):
    u = main(m)
    return assemble(inner(u, u)*dx)

  Jm = Jhat(m)

  minconv = taylor_test(Jhat, TimeConstantParameter(m), Jm, dJdm, HJm=HJ(u, m))
