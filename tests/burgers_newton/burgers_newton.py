"""
Implementation of Burger's equation with nonlinear solve in each
timestep
"""

import sys

from dolfin import *
from dolfin_adjoint import *

debugging["record_all"] = True

n = 30
mesh = UnitInterval(n)
V = FunctionSpace(mesh, "CG", 2)

def Dt(u, u_, timestep):
    return (u - u_)/timestep

def main(ic, annotate=False):

    u_ = Function(ic)
    u = Function(V)
    v = TestFunction(V)

    nu = Constant(0.0001)

    timestep = Constant(1.0/n)

    F = (Dt(u, u_, timestep)*v
         + u*grad(u)*v + nu*grad(u)*grad(v))*dx
    bc = DirichletBC(V, 0.0, "on_boundary")

    t = 0.0
    end = 0.2
    while (t <= end):
        solve(F == 0, u, bc, annotate=annotate)
        u_.assign(u, annotate=annotate)

        t += float(timestep)
        adj_inc_timestep()

    return u_

if __name__ == "__main__":

    ic = project(Expression("sin(2*pi*x[0])"),  V)
    forward = main(ic, annotate=True)

    adj_html("burgers_newton_forward.html", "forward")
    adj_html("burgers_newton_adjoint.html", "adjoint")

    print "Running forward replay .... "
    replay_dolfin()
    print "Running adjoint ... "

    J = Functional(forward*forward*dx)
    adjoint = adjoint_dolfin(J)

    def Jfunc(ic):
      forward = main(ic, annotate=False)
      return assemble(forward*forward*dx)

    minconv = test_initial_condition_adjoint(Jfunc, ic, adjoint, seed=1.0e-6)
