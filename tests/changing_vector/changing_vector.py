"""
Implementation of Burger's equation with nonlinear solve in each
timestep
"""

import sys

from dolfin import *
from dolfin_adjoint import *

from numpy import array

dolfin.parameters["adjoint"]["record_all"] = True
dolfin.parameters["adjoint"]["fussy_replay"] = False

n = 30
mesh = UnitInterval(n)
V = FunctionSpace(mesh, "CG", 2)

def Dt(u, u_, timestep):
    return (u - u_)/timestep

def main(ic, annotate=False):

    u_ = Function(ic, name="Velocity")
    u = Function(V, name="VelocityNext")
    v = TestFunction(V)

    nu = Function(V)
    nu.vector()[:] = 0.0001

    timestep = Constant(1.0/n)

    F = (Dt(u, u_, timestep)*v
         + u*u.dx(0)*v + nu*u.dx(0)*v.dx(0))*dx
    bc = DirichletBC(V, 0.0, "on_boundary")

    t = 0.0
    end = 0.2
    while (t <= end):
        nu.vector()[:] += array([0.0001] * V.dim()) # <--------- change nu here by hand
        solve(F == 0, u, bc, annotate=annotate)
        u_.assign(u, annotate=annotate)

        t += float(timestep)
        adj_inc_timestep()

    return u_

if __name__ == "__main__":
    ic = project(Expression("sin(2*pi*x[0])"),  V)
    main(ic, annotate=True)

    print "Running forward replay .... "
    success = replay_dolfin(forget=False) # <------ should catch it here

