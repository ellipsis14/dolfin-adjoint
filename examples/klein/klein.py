#!/usr/bin/env python
# -*- coding: utf-8 -*-

# .. _klein:
#
# .. py:currentmodule:: dolfin_adjoint
#
# Sensitivity analysis of the heat equation on Gray's Klein bottle
# ================================================================
#
# .. sectionauthor:: Simon W. Funke <simon@simula.no>
#
#
# Background
# **********
#
# When working with a computational model, it is often desired to study 
# the impact of model input parameters on a particular model output variable 
# (which we call objective value from now on).
# The obvious approach is to perturb each input variable independently 
# and observe how the objective value changes. However, this quickly becomes
# infeasible if the number of input variables grows and/or 
# the model is computational expensive.
#
# One of the main advantages of the adjoint method is that the 
# cost for computing such sensitivities is nearly independent on the number of 
# input variables.
# This allows us to easily compute sensitivities with respect
# to hundreds of input variables, or even infinite dimensional functions!
#
# In the following example we apply dolfin-adjoint to compute the sensitivity of 
# a time-dependent model with respect to its initial condition. 
#
# Problem definition
# ******************
#
# The equation for this example is the two-dimensional, time-dependent heat-equation:
# computes the gradient of the solution norm at the final time
# with respect to the initial condition.
#
# .. math::
#            \frac{\partial u}{\partial t} - \nu \nabla^{2} = 0 
#             \quad & \textrm{in } \Omega \times (0, T), \\
#            u = g  \quad & \textrm{for } \Omega \times \{0\}.
#            
#
# where :math:`\Omega` is the spatial domain, :math:`T` is the final time, :math:`u` 
# is the unkown temperature variation, :math:`\nu` is the thermal diffusivity, and 
# :math:`g` is the initial temperature.
#
# The objective functional, that is the model output of interest, is the norm of the 
# temperature at the end of the time interval:
#
# .. math::
#            J(u) := \int_\Omega u(t=T)^2 \textrm{d} \Omega
# 
# We would like to compute the sensitivity of :math:`J` on the initial condition, that is:
#
# .. math::
#            \frac{\textrm{d}J(u)}{\textrm{d} g}
#
#
# Note that we do not enforce any boundary conditions for the heat equation. 
# The reason is that in this example the domain :math:`\Omega` is a closed 
# manifold, that is a manifold without boundary. More specifically the domain is
# a 2D manifold embedded in 3D: the `Gray's Klein bottle 
# <http://paulbourke.net/geometry/klein/>`_ with parameters :math:`a = 2.0`, :math:`n = 2` and :math:`m = 1`. The meshed domain looks like this:

# .. image:: klein-bottle.png
#     :scale: 50
#     :align: center


# Implementation
# **************

# We start the implementation by importing the :py:mod:`dolfin` and
# :py:mod:`dolfin_adjoint` modules

from dolfin import *
from dolfin_adjoint import *
from math import ceil

# We start with ...

#: Activate some FEniCS optimizations
parameters["form_compiler"]["cpp_optimize_flags"] = "-O3 -ffast-math -march=native"

#: Load the mesh  
mesh = Mesh("klein.xml.gz")

# Set the options for the time discretization
T = 1.
t = 0.0
step = 0.1

# 
#steps = int(ceil(T/float(step)))+1
#adj_checkpointing('multistage', steps, 2, 1, verbose=True)

#: Define the function space for the PDE solution
V = FunctionSpace(mesh, "CG", 1)

#: Define the initial condition
g = interpolate(Expression("sin(x[2])*cos(x[1])"), V)

#: Define the solution at the current time level
u = Function(V)

#: Define the solution at the previous time level
u_old = Function(V)

#: Define the test function
v = TestFunction(V)

#: Define the variational formulation of the problem
F = u*v*dx - u_old*v*dx + step*inner(grad(v), grad(u))*dx

#: Define the solver options
sps = {"nonlinear_solver": "newton"}
sps["newton_solver"] = {"maximum_iterations": 200, 
                        "relative_tolerance": 1.0e-200, 
                        "linear_solver": "lu"}

#: Solve the time-dependent forward problem
u_pvd = File("output/u.pvd")
u_pvd << g
fwd_timer = Timer("Forward run")
fwd_time = 0

u_old.assign(g, annotate=True)
plot(u_old, interactive=True)
while t <= T:
    t += step

    fwd_timer.start()
    solve(F == 0, u, solver_parameters=sps)
    u_old.assign(u)
    fwd_time += fwd_timer.stop()

    u_pvd << u
    adj_inc_timestep()


#: Define the functional and control
J = Functional(inner(u, u)*dx*dt[FINISH_TIME])
m = Control(g)
plot(u, interactive=True)

#: Compute the functional gradient with dolfin-adjoint
adj_timer = Timer("Adjoint run")
dJdm = compute_gradient(J, m)
adj_time = adj_timer.stop()

#: Print timing statistics 
print "Forward time: ", fwd_time
print "Adjoint time: ", adj_time
print "Adjoint to forward runtime ratio: ", adj_time / fwd_time

# Plot the computed functional gradient
plot(dJdm, title="Sensitivity of ||u(t=%f)||_L2 with respect to u(t=0)." % t)
interactive()

dJdm_pvd = File("output/dJdm.pvd")
dJdm_pvd << dJdm

# The following image on the left shows the initial temperature variation, that is :math:`u(T=0)` and the image on the right the final temperature variation, that is :math:`u(T=1)`.

# .. image:: u_combined.png
#     :scale: 30
#     :align: center

# The next image shows the senstivity computed that was computed with dolfin-adjoint, that is :math:`\textrm{d} (\|u(t=1)\|) / \textrm{d}(u(T=0))`:

# .. image:: klein-sensitivity.png
#     :scale: 30
#     :align: center
