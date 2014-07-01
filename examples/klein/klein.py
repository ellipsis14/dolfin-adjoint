#!/usr/bin/env python
# -*- coding: utf-8 -*-

# .. _klein:
#
# .. py:currentmodule:: dolfin_adjoint
#
# Sensitivity analysis of the heat equation on a Klein bottle
# ================================================================
#
# .. sectionauthor:: Simon W. Funke <simon@simula.no>
#
#
# Background
# **********
#
# When working with computational models, it is often desired to study 
# the impact of input parameters on a particular model output variable 
# (the objective value).
# The obvious approach to obtain sensitivity information is to perturb each 
# input variable independently and observe how the objective value changes. 
# However, this quickly becomes
# infeasible if the number of input variables grows and/or 
# the model is computational expensive.
#
# One of the key advantages of the adjoint method is that the 
# computational cost for obtaining sensitivities is nearly independent on the number of 
# input variables.
# This allows us to easily compute sensitivities with respect
# to hundreds of input variables, or even with respect to infinite dimensional functions!
#
# In the following example we consider a time-dependent model and apply dolfin-adjoint to 
# determine how sensitive the final solution is with respect to changes in its initial condition. 
#
# Problem definition
# ******************
#
# The partial differential equation for this example is the two-dimensional, time-dependent heat-equation:
#
# .. math::
#            \frac{\partial u}{\partial t} - \nu \nabla^{2} u= 0 
#             \quad & \textrm{in } \Omega \times (0, T), \\
#            u = g  \quad & \textrm{for } \Omega \times \{0\}.
#            
#
# where :math:`\Omega` is the spatial domain, :math:`T` is the final time, :math:`u` 
# is the unkown temperature variation, :math:`\nu` is the thermal diffusivity, and 
# :math:`g` is the initial temperature.
#
# The objective value, that is the model output of interest, is the norm of the 
# temperature variation at the final time:
#
# .. math::
#            J(u) := \int_\Omega u(t=T)^2 \textrm{d} \Omega
# 
# We are interested in the sensitivity of :math:`J` on the initial condition, that is:
#
# .. math::
#            \frac{\textrm{d}J}{\textrm{d} g}
#
#
# Note that we did not specify any boundary conditions for the heat equation above. 
# The reason is in this example the domain :math:`\Omega` is a closed 
# manifold, that is a manifold without boundary. More specifically the domain is
# a 2D manifold embedded in 3D: the `Gray's Klein bottle 
# <http://paulbourke.net/geometry/klein/>`_ with parameters :math:`a = 2.0`, :math:`n = 2` and :math:`m = 1`. The meshed Klein bottle looks like this:

# .. image:: klein-bottle.png
#     :scale: 50
#     :align: center


# Implementation
# **************

# We start the implementation by importing the :py:mod:`dolfin` and
# :py:mod:`dolfin_adjoint` modules.

from dolfin import *
from dolfin_adjoint import *

# Next we load a triangulation of the Klein bottle as a mesh file. 

mesh = Mesh("klein.xml.gz")

# FEniCS natively supports solving partial differential 
# equations on manifolds :cite:`rognes2013`, so nothing else needs to 
# be done here.
# If you are interested how this mesh was generated, you find the code 
# in ``examples/klein/make_mesh.py`` in the ``dolfin-adjoint``
# source tree.

# The next lines perform the setup of the heat equation model. 
# First we define a discrete function space based on a linear, continuous 
# finite element. Then we create the solution, test and trial 
# functions for the variational formulation.
# We also define the initial temperature and the thermal diffusivity coefficient.


# Function space for the PDE solution
V = FunctionSpace(mesh, "CG", 1)

# Solution at the current time level
u = Function(V)

# Solution at the previous time level
u_old = Function(V)

# Test function
v = TestFunction(V)

# Initial condition
g = interpolate(Expression("sin(x[2])*cos(x[1])"), V)

# Thermal diffusivity
nu = 1.0

# Now we can discretise the problem in time and specify the variational 
# formulation of the problem. 
# By multiplying the heat equation with a testfunction :math:`v \in V` and 
# applying a backward Euler time-discretisation, the discrete problem for one time level reads: Find :math:`u \in V` such that for all :math:`v \in V`:

# .. math::
#            \frac{u - u_{\textrm{old}}}{\textrm{step}} + \nu \left<\nabla u, \nabla v \right> = 0 
#            

# or in code:

# Set the options for the time discretization
T = 1.
t = 0.0
step = 0.1

# Define the variational formulation of the problem
F = (u*v*dx - u_old*v*dx)/step + nu*inner(grad(v), grad(u))*dx

# A small remark before we continue solving the forward problem.
# dolfin-adjoint supports optimal checkpointing based on the revolve library :cite:`griewank2000`.
# We leave it commented our here, but we will present some runtime results at the end of this example with checkpointing activated.

#adj_checkpointing('multistage', steps=11, snaps_on_disk=0, snaps_in_ram=5, verbose=True)

# Next, we solve the forward model.

# Solve the time-dependent forward problem
u_pvd = File("output/u.pvd")
u_pvd << g
fwd_timer = Timer("Forward run")
fwd_time = 0

u_old.assign(g, annotate=True)
while t <= T:
    t += step

    fwd_timer.start()
    solve(F == 0, u)
    u_old.assign(u)
    fwd_time += fwd_timer.stop()

    u_pvd << u
    adj_inc_timestep()

# Next we define the objetive functional.

J = Functional(inner(u, u)*dx*dt[FINISH_TIME])
m = Control(g)

# Now, we can compute the functional gradient with dolfin-adjoint

adj_timer = Timer("Adjoint run")
dJdm = compute_gradient(J, m)
adj_time = adj_timer.stop()

# Finally we plot the computed functional gradient and
# print some timing statistics. 

plot(dJdm, title="Sensitivity of ||u(t=%f)||_L2 with respect to u(t=0)." % t)
interactive()

print "Forward time: ", fwd_time
print "Adjoint time: ", adj_time
print "Adjoint to forward runtime ratio: ", adj_time / fwd_time

# The example code can be found in ``examples/klein`` in the ``dolfin-adjoint``
# source tree, and executed as follows:

# .. code-block:: bash

#   $ python klein.py
#   ...
#   Forward time:  8.62722325325
#   Adjoint time:  7.75998806953
#   Adjoint to forward runtime ratio:  0.899476904879

# Since the forward model is linear, we the adjoint model should optimally take 
# roughly the same time as the foward model. This is indeed the case here.

# The following image on the left shows the initial temperature variation, that is :math:`u(T=0)` and the image on the right the final temperature variation, that is :math:`u(T=1)`.

# .. image:: u_combined.png
#     :scale: 30
#     :align: center

# The next image shows the senstivity computed that was computed with dolfin-adjoint, that is :math:`\textrm{d} (\|u(t=1)\|) / \textrm{d}(u(T=0))`:

# .. image:: klein-sensitivity.png
#     :scale: 30
#     :align: center


# Checkpointing timings
# ---------------------

# The revolve library 

# 10 timesteps

# =====================================================    ====   ====  ====   ====
# Number of memory checkpoints                              2      3     4      5
# =====================================================    ====   ====  ====   ====
# Theoretical optimal adjoint to forward runtime ratio     5.00   2.18  1.63   1.45
# Observed adjoint to forward runtime ratio                5.07   2.26  1.73   1.53
# =====================================================    ====   ====  ====   ====


# .. rubric:: References

# .. bibliography:: /documentation/klein/klein.bib
#    :cited:
#    :labelprefix: 6E-

