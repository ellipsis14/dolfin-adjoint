from numpy import random
from dolfin import *

mesh = Mesh("mesh.xml.gz")
V = FunctionSpace(mesh, "DG", 1)
W = VectorFunctionSpace(mesh, "CG", 2) * FunctionSpace(mesh, "CG", 1)

w = Function(W, "velocity.xml.gz")
T = Function(V, "temperature.xml.gz")

u_pvd = File("velocity.pvd")
u_pvd << w.split()[0]
t_pvd = File("temperature.pvd")
t_pvd << T

def form(w):
  T = TrialFunction(V)
  v = TestFunction(V)

  h = CellSize(mesh)
  n = FacetNormal(mesh)

  u = split(w)[0]
  un = abs(dot(u('+'), n('+')))
  jump_v = v('+')*n('+') + v('-')*n('-')
  jump_T = T('+')*n('+') + T('-')*n('-')

  F = -dot(u*T, grad(v))*dx + (dot(u('+'), jump_v)*avg(T))*dS + dot(v, dot(u, n)*T)*ds + 0.5*un*dot(jump_T, jump_v)*dS
  return F

def form_action(w):
  F = form(w)
  return assemble(action(F, T))

def derivative_action(w, dw):
  F = action(form(w), T)
  deriv = derivative(F, w, dw)
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
  dw_dir = Function(W)
  dw_dir.vector()[:] = random.random((W.dim(),))

  unperturbed = form_action(w)
  fd_errors = []
  grad_errors = []

  for h in [0.1/2**i for i in range(5)]:
    dw = Function(W)
    dw.vector()[:] = h * dw_dir.vector()

    wdw = Function(w) # w + dw
    wdw.vector()[:] += dw.vector()
    perturbed = form_action(wdw)

    fd_errors.append((perturbed - unperturbed).norm("l2"))
    grad_errors.append((perturbed - unperturbed - derivative_action(w, dw)).norm("l2"))

  print "Finite differencing errors: ", fd_errors
  print "Finite difference convergence order: ", convergence_order(fd_errors)
  print "Gradient errors: ", grad_errors
  print "Gradient convergence order: ", convergence_order(grad_errors)
