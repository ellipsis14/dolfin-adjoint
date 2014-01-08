#!/usr/bin/env python
# -*- coding: utf-8 -*- 

r"""
Solve example 4 of 

@article{borrvall2003,
author = {Borrvall, T. and Petersson, J.},
title = {Topology optimization of fluids in {S}tokes flow},
journal = {International Journal for Numerical Methods in Fluids},
volume = {41},
number = {1},
doi = {10.1002/fld.426},
pages = {77--107},
year = {2003},
}

This problem is to minimise the dissipated power in the fluid

0.5 * \int_\Omega \alpha(\rho) * u \cdot u + \mu + \int_\Omega \grad u : \grad u - \int_\Omega f * u

subject to the Stokes equations with Dirichlet conditions on the velocity:

\alpha(\rho) u - \mu \nabla^2 u + \nabla p = f  on \Omega
                                    div(u) = 0  on \Omega
                                         u = b  on \delta \Omega (see figure 10 for the BCs)

and to the control constraints on available fluid volume
               0 <= \rho <= 1
        \int_\Omega \rho <= V

where u is the velocity, p is the pressure, \rho is the control (\rho(x) == 1
means fluid present, \rho(x) == 0 means no fluid present), f is a prescribed
source term (here 0), \alpha(\rho) models the inverse permeability as a function
of the control

\alpha(\rho) = \bar{\alpha} + (\underline{\alpha} - \bar{\alpha} \rho \frac{1 +
q}{\rho + q}

with \bar{\alpha}, \underline{\alpha} and q prescribed constants, and V
is the volume bound on the control.

Physically, this corresponds to finding the fluid-solid distribution \rho(x) that
minimises the dissipated power in the fluid.

Like the poisson-topology example, this example will use general control
constraints and IPOPT to solve the optimisation problem.
"""

from dolfin import *
from dolfin_adjoint import *

try:
  import pyipopt
except ImportError:
  info_red("This example depends on IPOPT and pyipopt. When compiling IPOPT, make sure to link against HSL, as it is a necessity for practical problems.")
  raise

parameters["std_out_all_processes"] = False # turn off redundant output in parallel

mu = Constant(1.0)
V = Constant(1.0/3)
alphaunderbar = 2.5 * mu / (100**2)
alphabar = 2.5 * mu / (0.01**2)
q = Constant(0.1)

def alpha(rho):
  """Inverse permeability as a function of rho, equation (40)"""
  return alphabar + (alphaunderbar - alphabar) * rho * (1 + q) / (rho + q)

# Define the discrete function spaces
n = 100
mesh = UnitSquareMesh(n, n)
A = FunctionSpace(mesh, "DG", 0)       # control function space
U = VectorFunctionSpace(mesh, "CG", 2) # velocity function space
P = FunctionSpace(mesh, "CG", 1)       # pressure function space
W = MixedFunctionSpace([U, P])         # mixed Taylor-Hood function space

# Define the boundary condition on velocity
class InflowOutflow(Expression):
  def eval(self, values, x):
    values[1] = 0.0
    values[0] = 0.0
    l = 1.0/6.0
    delta = 1
    gbar = 1.0

    if x[0] == 0.0 or x[0] == delta:
      if (1.0/4 - l/2) < x[1] < (1.0/4 + l/2):
        t = x[1] - 1.0/4
        values[0] = gbar*(1 - (2*t/l)**2)
      if (3.0/4 - l/2) < x[1] < (3.0/4 + l/2):
        t = x[1] - 3.0/4
        values[0] = gbar*(1 - (2*t/l)**2)

  def value_shape(self):
    return (2,)

def forward(rho):
  """Solve the forward problem for a given fluid distribution rho(x)."""
  w = Function(W)
  (u, p) = split(w)
  (v, q) = TestFunctions(W)

  F = (alpha(rho) * inner(u, v) * dx + inner(grad(u), grad(v)) * dx +
       inner(grad(p), v) * dx  + inner(div(u), q) * dx)
  bc = DirichletBC(W.sub(0), InflowOutflow(), "on_boundary")
  solve(F == 0, w, bcs=bc)

  return w

if __name__ == "__main__":
  rho = interpolate(V, A, name="Control")
  w   = forward(rho)
  (u, p) = split(w)

  # Define the reduced functionals
  controls = File("output/control_iterations.pvd")
  rho_viz = Function(A, name="ControlVisualisation")
  def eval_cb(j, rho):
    rho_viz.assign(rho)
    controls << rho_viz

  J = Functional(0.5 * inner(alpha(rho) * u, u) * dx + mu * inner(grad(u), grad(u)) * dx)
  m = SteadyParameter(rho)
  Jhat = ReducedFunctional(J, m, eval_cb=eval_cb)
  rfn = ReducedFunctionalNumPy(Jhat)

  # Bound constraints
  lb = 0.0
  ub = 1.0

  # Volume constraints
  class VolumeConstraint(InequalityConstraint):
    """A class that enforces the volume constraint g(a) = V - a*dx >= 0."""
    def __init__(self, V):
      self.V  = float(V)

      self.smass  = assemble(TestFunction(A) * Constant(1) * dx)
      self.tmpvec = Function(A)

    def function(self, m):
      self.tmpvec.vector()[:] = m

      # Compute the integral of the control over the domain
      integral = self.smass.inner(self.tmpvec.vector())
      if MPI.process_number() == 0:
        print "Current control integral: ", integral
      return [self.V - integral]

    def jacobian(self, m):
      return [-self.smass]

    def length(self):
      """Return the number of components in the constraint vector (here, one)."""
      return 1

  # Solve the optimisation problem
  nlp = rfn.pyipopt_problem(bounds=(lb, ub), constraints=VolumeConstraint(V))
  nlp.int_option('max_iter', 30)

  rho0 = rfn.get_parameters()
  results = nlp.solve(rho0)

  File("output/control_solution.xml.gz") << rho
