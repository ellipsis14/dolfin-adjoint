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
#debugging["test_derivative"] = 6

def Dt(u, u_, timestep):
    return (u - u_)/timestep

def main(ic, annotate=False):

    u_ = ic
    u = TrialFunction(V)
    v = TestFunction(V)

    nu = Constant(0.0001)

    timestep = Constant(1.0/n)

    F = (Dt(u, u_, timestep)*v
         + u_*grad(u)*v + nu*grad(u)*grad(v))*dx

    (a, L) = system(F)

    bc = DirichletBC(V, 0.0, "on_boundary")

    t = 0.0
    end = 0.025
    u = Function(V)

    solver_parameters = {"linear_solver": "default", "preconditioner": "none",
                         "krylov_solver": {"relative_tolerance": 1.0e-10}}
    while (t <= end):
        solve(a == L, u, bc, solver_parameters=solver_parameters, annotate=annotate)

        u_.assign(u, annotate=annotate)

        t += float(timestep)
        #plot(u)

    #interactive()
    return u_

if __name__ == "__main__":

    ic = project(Expression("sin(2*pi*x[0])"),  V)
    ic_copy = Function(ic)
    forward = main(ic, annotate=True)
    forward_copy = Function(forward)
    adj_html("burgers_picard_forward.html", "forward")
    adj_html("burgers_picard_adjoint.html", "adjoint")
    print "Running adjoint ... "

    J = FinalFunctional(forward*forward*dx)
    for (adjoint, var) in compute_adjoint(J, forget=False):
      pass

    def Jfunc(ic):
      forward = main(ic, annotate=False)
      return assemble(forward*forward*dx)

    ic.vector()[:] = ic_copy.vector()
    minconv = test_initial_condition_adjoint_cdiff(Jfunc, ic, adjoint, seed=5.0e-3)
    if minconv < 2.9:
      sys.exit(1)

    ic.vector()[:] = ic_copy.vector()
    dJ = assemble(derivative(forward_copy*forward_copy*dx, forward_copy))
    minconv = test_initial_condition_tlm(Jfunc, dJ, ic, seed=1.0e-5)
    if minconv < 1.9:
      sys.exit(1)
