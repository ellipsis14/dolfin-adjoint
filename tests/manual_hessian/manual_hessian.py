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
import ufl.algorithms

parameters["adjoint"]["stop_annotating"] = True

mesh = UnitSquare(10, 10)
Vu = VectorFunctionSpace(mesh, "CG", 2)
Vm = VectorFunctionSpace(mesh, "CG", 1)
bcs = [DirichletBC(Vu, (1.0, 1.0), "on_boundary")]
hbcs = [homogenize(bc) for bc in bcs]
ufl_action = action

def F(u, m):
  u_test = TestFunction(Vu)

  F = (inner(dot(grad(u), u), u_test)*dx +
       inner(grad(u), grad(u_test))*dx +
      -inner(m, u_test)*dx)

  return F

def main(m):
  u = Function(Vu)
  Fm = F(u, m)
  solve(Fm == 0, u, J=derivative(Fm, u), bcs=bcs)
  return u

def J(u, m):
  return inner(u, u)*dx + 0.5*inner(m, m)*dx

def Jhat(m):
  u = main(m)
  Jm = J(u, m)
  return assemble(Jm)

def tlm(u, m, m_dot):
  Fm = F(u, m)
  dFmdu = derivative(Fm, u)
  dFmdm = derivative(Fm, m, m_dot)
  u_tlm = Function(Vu)

  solve(action(dFmdu, u_tlm) + dFmdm == 0, u_tlm, bcs=hbcs)
  return u_tlm

def adj(u, m):
  Fm = F(u, m)
  dFmdu = derivative(Fm, u)
  adFmdu = adjoint(dFmdu, reordered_arguments=ufl.algorithms.extract_arguments(dFmdu))

  Jm = J(u, m)
  dJdu = derivative(Jm, u, TestFunction(Vu))

  u_adj = Function(Vu)

  solve(action(adFmdu, u_adj) - dJdu == 0, u_adj, bcs=hbcs)
  return u_adj

def dJ(u, m, u_adj):
  Fm = F(u, m)
  Jm = J(u, m)
  dFmdm = derivative(Fm, m)
  adFmdm = adjoint(dFmdm) # the args argument to adjoint is the biggest time-waster ever. Everything else about the system is so beautiful :-/
  current_args = ufl.algorithms.extract_arguments(adFmdm)
  correct_args = [TestFunction(Vm), TrialFunction(Vu)]
  adFmdm = replace(adFmdm, dict(zip(current_args, correct_args)))

  dJdm = derivative(Jm, m, TestFunction(Vm))

  result = assemble(-action(adFmdm, u_adj) + dJdm)
  return Function(Vm, result)

def soa(u, m, u_tlm, u_adj, m_dot):
  Fm = F(u, m)
  dFmdu = derivative(Fm, u)
  adFmdu = adjoint(dFmdu, reordered_arguments=ufl.algorithms.extract_arguments(dFmdu))

  dFdudu = derivative(adFmdu, u, u_tlm)
  dFdudm = derivative(adFmdu, m, m_dot)

  Jm = J(u, m)
  dJdu = derivative(Jm, u, TestFunction(Vu))
  dJdudu = derivative(dJdu, u, u_adj)
  dJdudm = derivative(dJdu, m, m_dot)

  u_soa = Function(Vu)

  # Implement the second-order adjoint equation
  Fsoa = (action(dFdudu, u_adj) +
          action(dFdudu, u_adj) + 
          action(adFmdu, u_soa) + # <-- the lhs term
         -dJdudu
         -dJdudm)
  solve(Fsoa == 0, u_soa, bcs=hbcs)
  return u_soa

def HJ(u, m):
  def HJm(m_dot):
    u_tlm = tlm(u, m, m_dot)
    u_adj = adj(u, m)
    u_soa = soa(u, m, u_tlm, u_adj, m_dot)

    Fm = F(u, m)
    dFmdm = derivative(Fm, m)
    adFmdm = adjoint(dFmdm)
    current_args = ufl.algorithms.extract_arguments(adFmdm)
    correct_args = [TestFunction(Vm), TrialFunction(Vu)]
    adFmdm = replace(adFmdm, dict(zip(current_args, correct_args)))

    Jm = J(u, m)
    dJdm = derivative(Jm, m, TestFunction(Vm))

    # The following expression SHOULD work without the action hack
    # but UFL is pretty stupid here, and if the derivatives are null it
    # just crashes instead of gracefully dropping the terms

    #def action(A, x):
    #  A = ufl.algorithms.expand_derivatives(A)
    #  if A.integrals() != (): # form is not empty:
    #    return ufl_action(A, x)
    #  else:
    #    return A # form is empty, doesn't matter anyway

    #FH = (-action(derivative(adFmdm, u, u_tlm), u_adj) +
    #      -action(derivative(adFmdm, m, m_dot), u_adj) +
    #      -action(adFmdm, u_soa) +
    #       derivative(dJdm, u, u_tlm) +
    #       derivative(dJdm, m, m_dot))
    FH = (-action(adFmdm, u_soa) +
           derivative(dJdm, m, m_dot))

    result = assemble(FH)
    return Function(Vm, result)

  return HJm

def J_adj_m(m):
  '''J(lambda) = inner(lambda, lambda)*dx
  considered as a pure function of m
  for the purposes of Taylor verification'''
  u = main(m)
  u_adj = adj(u, m)
  return assemble(inner(u_adj, u_adj)*dx)

def grad_J_adj_m(m, m_dot):
  '''Gradient of the above function in the direction mdot.
  Correct if and only if the SOA solution is correct.'''
  u = main(m)
  u_adj = adj(u, m)
  u_tlm = tlm(u, m, m_dot)
  u_soa = soa(u, m, u_tlm, u_adj, m_dot)
  return 2 * u_adj.vector().inner(u_soa.vector())

def little_taylor_test(m):
  '''Implement my own Taylor test quickly for the above two functions.'''
  m_dot = interpolate(Constant((1.0, 1.0)), Vm)
  seed = 0.2
  without_gradient = []
  with_gradient = []
  Jm = J_adj_m(m)
  for h in [seed * 2**-i for i in range(5)]:
    m_ptb = Function(m_dot)
    m_ptb.vector()[:] *= h
    m_tilde = Function(m)
    m_tilde.vector()[:] += m_ptb.vector()
    without_gradient.append(J_adj_m(m_tilde) - Jm)
    with_gradient.append(without_gradient[-1] - grad_J_adj_m(m, m_ptb))

  print "Taylor remainders for J(adj(m)) without gradient information: ", without_gradient
  print "Convergence orders for above Taylor remainders: ", convergence_order(without_gradient)
  print "Taylor remainders for J(adj(m)) with gradient information: ", with_gradient
  print "Convergence orders for above Taylor remainders: ", convergence_order(with_gradient)

  assert min(convergence_order(with_gradient)) > 1.9

if __name__ == "__main__":
  m = interpolate(Expression(("sin(x[0])", "cos(x[1])")), Vm)
  u = main(m)
  Jm = assemble(J(u, m))

  m_dot = interpolate(Constant((1.0, 1.0)), Vm)

  u_tlm = tlm(u, m, m_dot)
  u_adj = adj(u, m)

  dJdm = dJ(u, m, u_adj)
  info_green("Applying Taylor test to gradient computed with adjoint ... ")
  minconv = taylor_test(Jhat, TimeConstantParameter(m), Jm, dJdm, value=m)
  assert minconv > 1.9

  info_green("Applying Taylor test to dlambda/dm ... ")
  little_taylor_test(m)

  HJm = HJ(u, m)
  info_green("Applying Taylor test to Hessian computed with second-order adjoint ... ")
  minconv = taylor_test(Jhat, TimeConstantParameter(m), Jm, dJdm, HJm=HJm, value=m, perturbation_direction=m_dot)
  assert minconv > 2.9