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

from dolfin import *
from dolfin_adjoint import *
debugging["record_all"] = True

def viscosity(T):
    eta = eta0 * exp(-b_val*T/deltaT + c_val*(1.0 - triangle.x[1])/height )
    return eta

def store(T, u, t):
    temperature_series.store(T.vector(), t)
    velocity_series.store(u.vector(), t)
    if t == 0.0:
        temperature_series.store(mesh, t)

def message(t, dt):
    print "\n" + "-"*60
    print "t = %0.5g" % t
    print "dt = %0.5g" % dt
    print "-"*60

def compute_timestep(u):
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

    # Velocity (u) at previous time step
    V = W.sub(0).collapse()
    u_ = Function(V)

    # Solve Stokes problem with given initial temperature and
    # composition
    eta = viscosity(T_)
    (a, L, pre) = momentum(W, eta, (Ra*T_)*g)
    (A, b) = assemble_system(a, L, bcs)
    P = PETScMatrix()
    assemble(pre, tensor=P); [bc.apply(P) for bc in bcs]

    velocity_pressure = Function(W)
    solver = KrylovSolver("tfqmr", "amg")
    solver.set_operators(A, P)
    solver.solve(velocity_pressure.vector(), b)

    u_.assign(velocity_pressure.split()[0])

    end()
    return (T_, u_, P)

parameters["form_compiler"]["cpp_optimize"] = True

# Define spatial domain
height = 1.0
length = 2.0
nx = 16
ny = 8
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
(T_, u_, P) = compute_initial_conditions(W, Q)

# Predictor functions
u_pr = Function(V)      # Tentative velocity (u)
T_pr = Function(Q)      # Tentative temperature (T)

# Functions at this timestep
u = Function(V)         # Velocity (u) at this time step
T = Function(Q)         # Temperature (T) at this time step
velocity_pressure = Function(W)

print "velocity_pressure: ", velocity_pressure
print "T: ", T
print "u: ", u
print "T_: ", T_
print "len(T_.vector()): ", len(T_.vector())
print "u_:", u_
print "P: ", P
print "u_pr: ", u_pr
print "T_pr: ", T_pr

# Containers for storage
velocity_series = TimeSeries("bin-final/velocity")
temperature_series = TimeSeries("bin-final/temperature")

# Store initial data
store(T_, u_, 0.0)

# Define initial CLF and time step
CLFnum = 0.5
dt = compute_timestep(u_)
t += dt
n = 1

# Solver for the Stokes systems
solver = PETScKrylovSolver("tfqmr", "amg")

while (t <= finish and n <= 1):

    message(t, dt)

    # Solve for predicted temperature
    (a, L) = energy(Q, Constant(dt), u_, T_)
    solve(a == L, T_pr, T_bcs,
          solver_parameters={"linear_solver": "gmres"})

    # Solve for predicted velocity
    eta = viscosity(T_pr)
    (a, L, precond) = momentum(W, eta, (Ra*T_pr)*g)

    b = assemble(L); [bc.apply(b) for bc in bcs]
    A = AdjointKrylovMatrix(a, bcs=bcs)

    solver.set_operators(A, P)
    solver.solve(velocity_pressure.vector(), b)
    u_pr.assign(velocity_pressure.split()[0])

    # Solve for corrected temperature T
    (a, L) = energy_correction(Q, Constant(dt), u_pr, u_, T_)
    solve(a == L, T, T_bcs,
          solver_parameters={"linear_solver": "gmres"})

    # Solve for corrected velocity
    eta = viscosity(T)
    (a, L, precond) = momentum(W, eta, (Ra*T)*g)

    b = assemble(L); [bc.apply(b) for bc in bcs]
    A = AdjointKrylovMatrix(a, bcs=bcs)

    solver.set_operators(A, P)
    solver.solve(velocity_pressure.vector(), b)
    u.assign(velocity_pressure.split()[0])

    # Store stuff
    store(T, u, t)

    # Compute time step
    dt = compute_timestep(u)

    # Move to new timestep and update functions
    T_.assign(T)
    u_.assign(u)
    t += dt
    n += 1

adj_html("forward.html", "forward")
replay_dolfin()
