"""This script implements the Taylor remainder convergence test
for an individual form.

Imagine we have an expression F(T) that is a function of T. We
can check the correctness of the derivative dF/dT by noting that

||F(T + dT) - F(T)|| converges at first order

but that

||F(T + dT) - F(T) - dF/dT . dT|| converges at second order.

In this example, F(T) is the action of the Stokes operator
on a supplied velocity field:

F(T) = action(momentum(T), u).

"""

from numpy import random
from dolfin import *
from math import log

mesh = Mesh("mesh.xml.gz")
V = FunctionSpace(mesh, "CG", 1)
W = VectorFunctionSpace(mesh, "CG", 2) * FunctionSpace(mesh, "CG", 1)

T = Function(V, "temperature.xml.gz")
w = Function(W, "velocity.xml.gz")

t_pvd = File("temperature.pvd")
t_pvd << T
u_pvd = File("velocity.pvd")
u_pvd << w.split()[0]

def form(T):
  eta = exp(-log(1000)*T)
  Ra = 10000
  H = Ra*T
  g = Constant((0.0, -1.0))

  # Define basis functions
  (u, p) = TrialFunctions(W)
  (v, q) = TestFunctions(W)

  strain = lambda v: 0.5*(grad(v) + grad(v).T)

  # Define equation F((u, p), (v, q)) = 0
  F = (2.0*eta*inner(strain(u), strain(v))*dx
       + div(v)*p*dx
       + div(u)*q*dx
       + H*inner(g, v)*dx)

  return lhs(F)

def form_action(T):
  """This function computes F(T)."""
  F = form(T)
  return assemble(action(F, w))

def derivative_action(T, dT):
  """This function computes dF/dT . dT."""
  F = action(form(T), w)
  deriv = derivative(F, T, dT)
  return assemble(deriv)

def convergence_order(errors):
  import math

  orders = [0.0] * (len(errors)-1)
  for i in range(len(errors)-1):
    try:
      orders[i] = math.log(errors[i]/errors[i+1], 2)
    except ZeroDivisionError:
      orders[i] = numpy.nan

  return orders

if __name__ == "__main__":
  # We're going to choose a random perturbation direction, and then use that
  # direction 5 times, making the perturbation smaller each time.
  dT_dir = Function(V)
  dT_dir.vector()[:] = random.random((V.dim(),))

  # We need the unperturbed F(T) to compare against.
  unperturbed = form_action(T)

  # fd_errors will contain
  # ||F(T+dT) - F(T)||
  fd_errors = []

  # grad_errors will contain
  # ||F(T+dT) - F(T) - dF/dT . dT||
  grad_errors = []

  # h is the perturbation size
  for h in [1.0e-7/2**i for i in range(5)]:
    # Build the perturbation
    dT = Function(V)
    dT.vector()[:] = h * dT_dir.vector()

    # Compute the perturbed result
    TdT = Function(T) # T + dT
    TdT.vector()[:] += dT.vector()
    perturbed = form_action(TdT)

    fd_errors.append((perturbed - unperturbed).norm("l2"))
    grad_errors.append((perturbed - unperturbed - derivative_action(T, dT)).norm("l2"))

  # Now print the orders of convergence:
  print "Finite differencing errors: ", fd_errors
  print "Finite difference convergence order (should be 1): ", convergence_order(fd_errors)
  print "Gradient errors: ", grad_errors
  print "Gradient convergence order: ", convergence_order(grad_errors)
