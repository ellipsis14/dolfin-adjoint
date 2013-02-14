"""
Naive implementation of Burgers' equation, goes oscillatory later
"""

import sys

from dolfin import *
from dolfin_adjoint import *

n = 2
mesh = UnitIntervalMesh(n)
V = FunctionSpace(mesh, "CG", 1)

def Dt(u, u_, timestep):
    return (u - u_)/timestep

def main(ic, annotate=False):

    u_ = Function(V, ic, name="Solution")
    u = TrialFunction(V)
    v = TestFunction(V)

    nu = Constant(0.0001)
    timestep = Constant(1.0)

    F = (Dt(u, u_, timestep)*v
         + u_*u.dx(0)*v + nu*u.dx(0)*v.dx(0))*dx

    (a, L) = system(F)

    bc = DirichletBC(V, 0.0, "on_boundary")

    t = 0.0
    end = float(timestep) - 1.0e-14 # only do one timestep please
    u = Function(V)

    solver_parameters = {"linear_solver": "default", "preconditioner": "none",
                         "krylov_solver": {"relative_tolerance": 1.0e-10}}
    while (t <= end):
        solve(a == L, u, bc, solver_parameters=solver_parameters, annotate=annotate)

        u_.assign(u, annotate=annotate)
        t += float(timestep)

    return u_

if __name__ == "__main__":

    ic = project(Constant(1.0),  V)
    forward = main(ic, annotate=True)

    adj_html("forward.html", "forward")
    adj_html("adjoint.html", "adjoint")

    J = Functional(inner(forward, forward)**2*dx*dt[FINISH_TIME])
    m = InitialConditionParameter("Solution")

    Jm   = assemble(inner(forward, forward)**2*dx)
    dJdm = compute_gradient(J, m, forget=False)
    HJm  = hessian(J, m)
    m_dot = interpolate(Constant(1.0), V)

    def Jfunc(ic):
      forward = main(ic, annotate=False)
      return assemble(inner(forward, forward)**2*dx)

    minconv = taylor_test(Jfunc, m, Jm, dJdm, perturbation_direction=m_dot, seed=0.2)
    assert minconv > 1.9
