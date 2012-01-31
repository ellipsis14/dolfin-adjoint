__author__ = "Lyudmyla Vynnytska and Marie E. Rognes"
__copyright__ = "Copyright (C) 2011 Simula Research Laboratory and %s" % __author__
__license__  = "GNU LGPL Version 3 or any later version"

# Last changed: 2011-10-17

import time
import numpy

from stokes import *
from composition import *
from temperature import *
from parameters import InitialTemperature, Ra, Rb, rho0, g
from parameters import eta0, b_val, c_val, deltaT

from dolfin import *; import dolfin
from dolfin_adjoint import *
debugging["record_all"] = True
debugging["fussy_replay"] = False
dolfin.parameters["form_compiler"]["representation"] = "quadrature"

def viscosity(T):
    eta = eta0 * exp(-b_val*T/deltaT + c_val*(1.0 - triangle.x[1])/height )
    return eta

def store(T, w, t):
    temperature_series.store(T.vector(), t)
    flow_series.store(w.vector(), t)
    if t == 0.0:
        temperature_series.store(mesh, t)

def message(t, dt):
    print "\n" + "-"*60
    print "t = %0.5g" % t
    print "dt = %0.5g" % dt
    print "-"*60

def compute_timestep(w):
    (u, p) = w.split(deepcopy=True)
    maxvel = numpy.max(numpy.abs(u.vector().array()))
    mesh = u.function_space().mesh()
    hmin = mesh.hmin()
    dt = CLFnum*hmin/maxvel
    return dt

def compute_initial_conditions(W, Q):
    begin("Computing initial conditions")

    # Define initial temperature (guess)
    T0 = InitialTemperature(Ra, length)

    # Temperature (T) at previous time step
    T_ = interpolate(T0, Q)

    # Solve Stokes problem with given initial temperature and
    # composition
    eta = viscosity(T_)
    (a, L, pre) = momentum(W, eta, (Ra*T_)*g)
    (A, b) = assemble_system(a, L, bcs)
    P = PETScMatrix()
    assemble(pre, tensor=P); [bc.apply(P) for bc in bcs]

    w = Function(W)
    solver = dolfin.KrylovSolver("tfqmr", "amg")
    solver.set_operators(A, P)
    solver.solve(w.vector(), b)

    end()
    return (T_, w, P)

parameters["form_compiler"]["cpp_optimize"] = True

# Define spatial domain
height = 1.0
length = 2.0
nx = 4
ny = 4
mesh = Rectangle(0, 0, length, height, nx, ny)

# Define initial and end time
t = 0.0
finish = 0.015

# Create function spaces
W = stokes_space(mesh)
V = W.sub(0).collapse()
Q = FunctionSpace(mesh, "DG", 1)

# Define boundary conditions for the velocity and pressure u
bottom = DirichletBC(W.sub(0), (0.0, 0.0), "x[1] == 0.0" )
top = DirichletBC(W.sub(0).sub(1), 0.0, "x[1] == %g" % height)
left = DirichletBC(W.sub(0).sub(0), 0.0, "x[0] == 0.0")
right = DirichletBC(W.sub(0).sub(0), 0.0, "x[0] == %g" % length)
bcs = [bottom, top, left, right]

# Define boundary conditions for the temperature
top_temperature = DirichletBC(Q, 0.0, "x[1] == %g" % height, "geometric")
bottom_temperature = DirichletBC(Q, 1.0, "x[1] == 0.0", "geometric")
T_bcs = [bottom_temperature, top_temperature]

rho = interpolate(rho0, Q)

# Functions at previous timestep (and initial conditions)
(T_, w_, P) = compute_initial_conditions(W, Q)

# Predictor functions
T_pr = Function(Q)      # Tentative temperature (T)

# Functions at this timestep
T = Function(Q)         # Temperature (T) at this time step
w = Function(W)

print "w: ", w
print "w_: ", w_
print "T: ", T
print "T_: ", T_
print "len(T_.vector()): ", len(T_.vector())
print "P: ", P
print "T_pr: ", T_pr

# Containers for storage
flow_series = TimeSeries("bin-final/flow")
temperature_series = TimeSeries("bin-final/temperature")

# Store initial data
store(T_, w_, 0.0)

# Define initial CLF and time step
CLFnum = 0.5
dt = compute_timestep(w_)
t += dt
n = 1

w_pr = Function(W)
print "w_pr: ", w_pr
(u_pr, p_pr) = split(w_pr)
(u_, p_) = split(w_)

# Solver for the Stokes systems
solver = AdjointPETScKrylovSolver("tfqmr", "amg")

while (t <= finish and n <= 2):

    message(t, dt)

    # Solve for predicted temperature in terms of previous velocity
    (a, L) = energy(Q, Constant(dt), u_, T_)
    solve(a == L, T_pr, T_bcs,
          solver_parameters={"linear_solver": "gmres"})

    # Solve for predicted flow
    eta = viscosity(T_pr)
    (a, L, precond) = momentum(W, eta, (Ra*T_pr)*g)

    b = assemble(L); [bc.apply(b) for bc in bcs]
    A = AdjointKrylovMatrix(a, bcs=bcs)

    solver.set_operators(A, P)
    solver.solve(w_pr.vector(), b)

    # Solve for corrected temperature T in terms of predicted and previous velocity
    (a, L) = energy_correction(Q, Constant(dt), u_pr, u_, T_)
    solve(a == L, T, T_bcs,
          solver_parameters={"linear_solver": "gmres"})

    # Solve for corrected flow
    eta = viscosity(T)
    (a, L, precond) = momentum(W, eta, (Ra*T)*g)

    b = assemble(L); [bc.apply(b) for bc in bcs]
    A = AdjointKrylovMatrix(a, bcs=bcs)

    solver.set_operators(A, P)
    solver.solve(w.vector(), b)

    # Store stuff
    store(T, w, t)

    # Compute time step
    dt = compute_timestep(w)

    # Move to new timestep and update functions
    T_.assign(T)
    w_.assign(w)
    t += dt
    n += 1
    adj_inc_timestep()

print "Replaying forward run ... "
adj_html("forward.html", "forward")
replay_dolfin()

#J = FinalFunctional(inner(u_, u_)*dx)
#adjoint = adjoint_dolfin(J)
