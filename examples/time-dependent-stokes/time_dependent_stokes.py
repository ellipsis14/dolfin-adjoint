"""
This example demonstrates a tracking type optimal control problem for
time-dependent Stokes flow.

In particular, both the constraint and functional is time-dependent.

For details, see "Optimal error estimates and computations for
tracking-type control of the instationary Stokes system", Deckelnick
and Hinze, 2001.

Implicit Euler discretization in time and Taylor-Hood elements in
space

"""

__author__ = "Marie E. Rognes (meg@simula.no)"

from dolfin import *
from dolfin_adjoint import *

n = 10
mesh = UnitSquareMesh(n, n)
T = 0.1

def forward(mesh, T):

    V = VectorFunctionSpace(mesh, "CG", 2)
    Q = FunctionSpace(mesh, "CG", 1)
    W = V*Q

    # Function to store solution at current timestep
    w = Function(W)
    (y, p) = split(w)

    # Function to store solution at previous timestep
    w_old = Function(W)
    # FIXME: Set initial condition
    (y_old, p_old) = split(w_old)

    # Define trial and test functions
    test = TestFunction(W)
    (y, p) = split(test)
    trial = TrialFunction(W)
    (phi, q) = split(trial)

    # Define model parameters
    nu = Constant(1)
    dT = Constant(0.01)
    num_steps = int(T/float(dT))
    print "Number of timesteps: %i." % num_steps

    # Define initial control functions. 
    # The control is time-dependent, hence the control is a list 
    # if functions in V of length T/dT
    control = [Function(V)]*num_steps

    # Preassemble the left hand side of the weak Stokes equations
    a  = inner(y, phi)*dx + dT*inner(nu*grad(y), grad(phi))*dx - dT*inner(p, div(phi))*dx
    a += dT*inner(div(y), q)*dx

    A = assemble(a)
    # TODO: Apply bcs

    while i in range(num_steps):
        print "In timestep %i." % i

        L = dT*inner(u[i], phi)*dx + inner(y_old, phi)*dx
        assemble(L)

if __name__ == "__main__":
    forward(mesh, T)
