#!/usr/bin/env python
# -*- coding: utf-8 -*- 

r"""
Solve example 5.2 of 

@article{hintermueller2011,
year={2011},
journal={Computational Optimization and Applications},
volume={50},
number={1},
doi={10.1007/s10589-009-9307-9},
title={A smooth penalty approach and a nonlinear multigrid algorithm for elliptic MPECs},
url={http://dx.doi.org/10.1007/s10589-009-9307-9},
publisher={Springer US},
author={Hinterm{\"u}ller, M. and Kopacka, I.},
pages={111-145},
}

The problem is to solve following Mathematical Problem with Equilibrium Constraints (MPEC):

  min 1/2 ||y-y_d||^2 + nu/2 ||u||^2

subject to the variational equality (VI)

  (\nabla y, \nabla(v - y)) >= (f+u, v-y) for all v >= 0,
                          y >= 0,

and bounds on the control functional

a <= u <= b,

where u is the control function, y is the solution of the variational inequality, y_d is data to be matched, f is a prescribed source term, \nu is a regularisation parameter and a, b are functions defining the upper and lower bound for the control.

The solution approach taken here is to approximate the variational inequality by a sequence of partial differential equation with the penalisation technique, 
and to solve the resulting sequence of PDE-constrained optimisation problems.

"""


from dolfin import *
from dolfin_adjoint import *
set_log_level(ERROR)

# A smooth approximation of the (pointwise) maximum operator
def smoothmax(r, eps=1e-4):
  return conditional(gt(r, eps), r - eps/2, conditional(lt(r, 0), 0, r**2 / (2*eps))) 

mesh = UnitSquareMesh(128, 128)
V = FunctionSpace(mesh, "CG", 1)  # The function space for the solution and control functions
y = Function(V, name="Solution")
u = Function(V, name="Control")
w = TestFunction(V)

# Define the PDE arising from the approximated variational inequality and solve it
alpha = Constant(1e-2)  # The penalisation parameter
f = interpolate(Expression("-std::abs(x[0]*x[1] - 0.5) + 0.25"), V)  # The source term
F = inner(grad(y), grad(w))*dx - 1 / alpha * inner(smoothmax(-y), w)*dx - inner(f + u, w)*dx  
bc = DirichletBC(V, 0.0, "on_boundary")
solve(F == 0, y, bcs=bc)

# Define the functional of interest
yd = Function(f, name="Data")
nu = 0.01
J = Functional(0.5*inner(y - yd, y - yd)*dx + nu/2*inner(u, u)*dx)

# Formulate the reduced problem
m = SteadyParameter(u)
Jhat = ReducedFunctional(J, m)

# Create output files
ypvd = File("output/y_opt.pvd") 
upvd = File("output/u_opt.pvd") 

# Solve the MPECs as a sequence of PDE-constrained optimisation problems 
ScalarParameter(alpha)  # Mark alpha to be a parameter. This allows us to change the 
                        # alpha value in the loop below and dolfin_adjoint will use the 
                        # new value automatically.
for i in range(4):
  # Update the penalisation value
  alpha.assign(float(alpha)/2)
  info_green("Set alpha to %f." % float(alpha))

  # Solve the optimisation problem
  u_opt = minimize(Jhat, bounds=(0.01, 0.03), options={"gtol": 1e-12, "ftol": 1e-100})
  
  # Use the optimised state solution as an initial guess 
  # for the Newton solver in the next optimisation round
  y_opt = DolfinAdjointVariable(y).tape_value()
  replace_parameter_value(InitialConditionParameter(y), y_opt)

  # Save the result and print some statistics
  ypvd << y_opt
  upvd << u_opt
  feasibility = sqrt(assemble(inner((Max(0, -y_opt)), (Max(0, -y_opt)))*dx))
  info_green("Feasibility: %s" % feasibility)
  info_green("Norm of y: %s" % sqrt(assemble(inner(y_opt, y_opt)*dx)))
  info_green("Norm of u_opt: %s" % sqrt(assemble(inner(u_opt, u_opt)*dx)))
