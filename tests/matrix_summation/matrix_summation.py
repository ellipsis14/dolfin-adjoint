"""
Naive implementation of Burgers' equation, goes oscillatory later
"""

import sys

from dolfin import *
from dolfin_adjoint import *

n = 100
mesh = UnitInterval(n)
V = FunctionSpace(mesh, "CG", 2)

parameters["num_threads"] = 2

debugging["record_all"] = True
#debugging["test_hermitian"] = (100, 1.0e-14)
debugging["test_derivative"] = 6

def Dt(u, u_, timestep):
    return (u - u_)/timestep

def main(ic, annotate=False):

    u_ = ic
    u = TrialFunction(V)
    v = TestFunction(V)

    mass = assemble(inner(u, v) * dx)
    advec = assemble(u_*grad(u)*v * dx)
    rhs = inner(ic, v) * dx

    L = mass + advec
    solve(L == rhs, u_, annotate=annotate)

    return u_

if __name__ == "__main__":

    ic = project(Expression("sin(2*pi*x[0])"),  V)
    ic_copy = Function(ic)
    forward = main(ic, annotate=True)
    forward_copy = Function(forward)
    print "Running adjoint ... "

    J = FinalFunctional(forward*forward*dx)
    for (adjoint, var) in compute_adjoint(J, forget=False):
      pass

    def Jfunc(ic):
      forward = main(ic, annotate=False)
      return assemble(forward*forward*dx)

    ic.vector()[:] = ic_copy.vector()
    minconv = test_initial_condition_adjoint_cdiff(Jfunc, ic, adjoint, seed=0.0001)
    if minconv < 2.9:
      sys.exit(1)
