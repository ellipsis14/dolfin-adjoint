import sys

from dolfin import *
from dolfin_adjoint import *
import libadjoint

f = Expression("x[0]*(x[0]-1)*x[1]*(x[1]-1)")
mesh = UnitSquare(4, 4)
V = FunctionSpace(mesh, "CG", 1)

def run_forward(initial_condition=None, annotate=True):
  u = TrialFunction(V)
  v = TestFunction(V)

  u_0 = Function(V, name="Velocity")
  if initial_condition is not None:
    u_0.assign(initial_condition)

  u_1 = Function(V)

  dt = 0.5
  T =  1.0

  F = ( (u - u_0)/dt*v + inner(grad(u), grad(v)) + f*v)*dx

  bc = DirichletBC(V, 1.0, "on_boundary")

  a, L = lhs(F), rhs(F)

  t = float(dt)
  n = 1


  adjointer.time.start(0)
  while t <= T:

      solve(a == L, u_0, bc, annotate=annotate)
      #solve(a == L, u_0, annotate=annotate)

      adj_inc_timestep(time=t)
      t += float(dt)

  adjointer.time.finish(1)

  return u_0

if __name__ == "__main__":

  u = run_forward()

  adj_html("forward.html", "forward")
  adj_html("adjoint.html", "adjoint")

  u00 = libadjoint.Variable("Velocity", 0, 0)
  u01 = libadjoint.Variable("Velocity", 0, 1)
  u10 = libadjoint.Variable("Velocity", 1, 0)

  # Integral over all time
  J = Functional(inner(u,u)*dx*dt[0:1])
  assert J.dependencies(adjointer, 0) == [u00]
  deps=J.dependencies(adjointer, 1)
  assert deps[0]==u01
  assert deps[1]==u10
  # This version of the test appears borked.
  #assert J.dependencies(adjointer, 1) == [u01, u10]

  # Integral over a certain time window
  J = Functional(inner(u,u)*dx*dt[0.5:1.0])
  assert J.dependencies(adjointer, 0) == []
  assert J.dependencies(adjointer, 1) == [u01, u10]

  # Pointwise evaluation (in the middle of a timestep)
  J = Functional(inner(u,u)*dx*dt[0.25])
  # assert J.dependencies(adjointer, 0) == [u00, u01]
  deps=J.dependencies(adjointer, 0)
  assert deps[0] in [u00, u01]
  assert deps[1] in [u00, u01]
  assert J.dependencies(adjointer, 1) == []

  # Pointwise evaluation (at a timelevel)
  J = Functional(inner(u,u)*dx*dt[0.5])
  assert J.dependencies(adjointer, 0) == []
  assert J.dependencies(adjointer, 1)[0] in [u01, u10]
  assert J.dependencies(adjointer, 1)[1] in [u01, u10]
  assert J.dependencies(adjointer, 1)[1] != J.dependencies(adjointer, 1)[0]

  # Integral over all time  
  # Functional.__hash__ is currently invalid
  #J = Functional(inner(u,u)*dx*dt[0:1])
  J = Functional(inner(u,u)*dx*dt[0:1],name="test")
  print adjointer.evaluate_functional(J,0)
