"""
Implementation of Burger's equation with nonlinear solve in each
timestep
"""

import sys

from dolfin import *
from dolfin_adjoint import *

debugging["record_all"] = True
debugging["fussy_replay"] = False

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
    ic_copy = Function(ic)
    forward = main(ic, annotate=True)
    forward_copy = Function(forward)

    adj_html("burgers_newton_forward.html", "forward")
    adj_html("burgers_newton_adjoint.html", "adjoint")

    print "Running forward replay .... "
    replay_dolfin(forget=False)
    print "Running adjoint ... "

    J = FinalFunctional(forward*forward*dx)
    for (adjoint, var) in compute_adjoint(J, forget=False):
      pass

    def Jfunc(ic):
      forward = main(ic, annotate=False)
      return assemble(forward*forward*dx)

    minconv = test_initial_condition_adjoint(Jfunc, ic, adjoint, seed=1.0e-5)
    if minconv < 1.9:
      sys.exit(1)

    dJ = assemble(derivative(forward_copy*forward_copy*dx, forward_copy))

    ic = forward
    ic.vector()[:] = ic_copy.vector()
    minconv = test_initial_condition_tlm(Jfunc, dJ, ic, seed=1.0e-5)
    if minconv < 1.9:
      sys.exit(1)
