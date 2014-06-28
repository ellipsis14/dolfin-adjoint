#!/usr/bin/env python
# -*- coding: utf-8 -*-

# This example solves the heat equation on Gray's Klein bottle and 
# computes the gradient of the solution norm at the final time
# with respect to the initial condition.


from dolfin import *
from dolfin_adjoint import *

# We start with ...

#: Activate some FEniCS optimizations
parameters["form_compiler"]["cpp_optimize_flags"] = "-O3 -ffast-math -march=native"

#: Load the mesh  
mesh = Mesh("klein.xml.gz")

# Set the options for the time discretization
T = 1.
t = 0.0
step = 0.1

#: Define the function space for the PDE solution
V = FunctionSpace(mesh, "CG", 1)

#: Define the initial condition
u_init = interpolate(Expression("sin(x[2])*cos(x[1])"), V)

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
fwd_timer = Timer("Forward run")
fwd_time = 0

u_old.assign(u_init, annotate=True)
while t <= T:
    t += step

    fwd_timer.start()
    solve(F == 0, u, solver_parameters=sps)
    u_old.assign(u)
    fwd_time += fwd_timer.stop()

    u_pvd << u


#: Define the functional and control
J = Functional(inner(u, u)*dx*dt[FINISH_TIME])
m = Control(u_init)

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
