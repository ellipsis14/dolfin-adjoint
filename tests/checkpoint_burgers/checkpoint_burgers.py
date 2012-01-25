"""
Naive implementation of Burgers' equation, goes oscillatory later
"""

import sys

from dolfin import *
from dolfin_adjoint import *
from math import ceil

n = 100
mesh = UnitInterval(n)
V = FunctionSpace(mesh, "CG", 2)

#debugging["record_all"] = True
#debugging["test_hermitian"] = (100, 1.0e-14)
#debugging["test_derivative"] = 6

def Dt(u, u_, timestep):
    return (u - u_)/timestep

def main(ic, annotate=False):

    u_ = Function(ic)
    u = TrialFunction(V)
    v = TestFunction(V)

    nu = Constant(0.0001)

    timestep = Constant(1.0/n)

    F = (Dt(u, u_, timestep)*v
         + u_*grad(u)*v + nu*grad(u)*grad(v))*dx

    (a, L) = system(F)

    bc = DirichletBC(V, 0.0, "on_boundary")

    t = 0.0
    end = 0.5
    if annotate: 
      adjoint_checkpointing('multistage', int(ceil(end/float(timestep))), 5, 10, verbose=True)

    u = Function(V)
    while (t <= end):
        solve(a == L, u, bc, annotate=annotate)

        u_.assign(u, annotate=annotate)

        t += float(timestep)
        adj_inc_timestep()
        #plot(u)

    #interactive()
    return u_

if __name__ == "__main__":

    
    ic = project(Expression("sin(2*pi*x[0])"),  V)
    forward = main(ic, annotate=True)
    adj_html("burgers_picard_checkpointing_forward.html", "forward")
    adj_html("burgers_picard_checkpointing_adjoint.html", "adjoint")
    #print "Running forward replay .... "
    #replay_dolfin()
    print "Running adjoint ... "

    J = Functional(forward*forward*dx)
    adjoint = adjoint_dolfin(J)

    def Jfunc(ic):
      forward = main(ic, annotate=False)
      return assemble(forward*forward*dx)

    minconv = test_initial_condition_adjoint(Jfunc, ic, adjoint, seed=1.0e-3)
    if minconv < 1.9:
      exit_code = 1
    else:
      exit_code = 0
    sys.exit(exit_code)
