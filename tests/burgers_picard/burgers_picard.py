"""
Naive implementation of Burgers' equation, goes oscillatory later
"""

import sys

from dolfin import *
from dolfin_adjoint import *

n = 100
mesh = UnitIntervalMesh(n)
V = FunctionSpace(mesh, "CG", 2)

dolfin.parameters["adjoint"]["test_derivative"] = True

def Dt(u, u_, timestep):
    return (u - u_)/timestep

def main(ic, annotate=False):

    u_ = Function(V, ic, name="Solution")
    u = TrialFunction(V)
    v = TestFunction(V)

    nu = Constant(0.0001)

    timestep = Constant(1.0/n)

    F = (Dt(u, u_, timestep)*v
         + u_*u.dx(0)*v + nu*u.dx(0)*v.dx(0))*dx

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

    return u_

if __name__ == "__main__":

    ic = project(Expression("sin(2*pi*x[0])"),  V)
    forward = main(ic, annotate=True)

    adj_html("burgers_picard_forward.html", "forward")
    adj_html("burgers_picard_adjoint.html", "adjoint")

    J = Functional(forward*forward*dx*dt[FINISH_TIME])
    m = InitialConditionParameter("Solution")

    Jm = assemble(forward*forward*dx)
    dJdm = compute_gradient(J, m, forget=False)

    def Jfunc(ic):
      forward = main(ic, annotate=False)
      return assemble(forward*forward*dx)

    minconv = taylor_test(Jfunc, m, Jm, dJdm, seed=5.0e-2)
    assert minconv > 1.9
