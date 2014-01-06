# 10.1007/s00158-005-0584-3, example 1


from dolfin import *
from dolfin_adjoint import *

import numpy
import pyipopt

V = Constant(0.4)
p = Constant(5)
eps = Constant(1.0e-3)

def k(a):
  return eps + (1 - eps) * a**p

# Define the discrete function spaces
n = 100
mesh = UnitSquareMesh(n, n)
A = FunctionSpace(mesh, "CG", 1) # function space for control
P = FunctionSpace(mesh, "CG", 1) # function space for solution

class WestNorth(SubDomain):
  def inside(self, x, on_boundary):
    return (x[0] == 0.0 or x[1] == 1.0) and on_boundary

bc = [DirichletBC(P, 0.0, WestNorth())]
f = interpolate(Constant(1.0e-2), P, name="SourceTerm")

def forward(a):
  # Define and solve the forward problem
  T = Function(P, name="Temperature")
  v = TestFunction(P)

  F = inner(grad(v), k(a)*grad(T))*dx - f*v*dx
  solve(F == 0, T, bc, solver_parameters={"newton_solver": {"absolute_tolerance": 1.0e-7, "maximum_iterations": 20}})

  return T

if __name__ == "__main__":
  a = interpolate(Constant(float(V)), A, name="Control")
  T = forward(a)
  controls = File("output/control_iterations.pvd")

  a_viz = Function(A, name="ControlVisualisation")
  def eval_cb(j, a):
    a_viz.assign(a)
    controls << a_viz

  # Define the reduced functionals
  J = Functional(f*T*dx + Constant(1.0e-8) * inner(grad(a), grad(a))*dx)
  m = SteadyParameter(a)
  Jhat = ReducedFunctional(J, m, eval_cb=eval_cb)
  rfn = ReducedFunctionalNumPy(Jhat)

  # Bound constraints
  lb = 0.0
  ub = 1.0

  # Volume constraints
  class VolumeConstraint(InequalityConstraint):
    def __init__(self, V):
      self.V  = float(V)
      self.smass  = assemble(TestFunction(A) * Constant(1) * dx)
      self.tmpvec = Function(A)

    def function(self, m):
      self.tmpvec.vector()[:] = m

      # Compute the integral of rho over the domain
      mass = self.smass.inner(self.tmpvec.vector())
      print "Current mass: ", mass 
      return [self.V - mass]

    def jacobian(self, m):
      return [-self.smass]

    def length(self):
      return 1

  # Solve the optimisation problem
  nlp = rfn.pyipopt_problem(bounds=(lb, ub), constraints=VolumeConstraint(V))
  a0 = rfn.get_parameters()
  results = nlp.solve(a0)

  File("ex1_reduced/a_soln.xml.gz") << a
