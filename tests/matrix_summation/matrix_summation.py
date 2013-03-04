import sys

from dolfin import *
from dolfin_adjoint import *

n = 100
mesh = UnitIntervalMesh(n)
V = FunctionSpace(mesh, "CG", 2)

dolfin.parameters["adjoint"]["record_all"] = True

def Dt(u, u_, timestep):
    return (u - u_)/timestep

def main(ic, annotate=False):

    u_ = Function(ic, name="Velocity")
    u = TrialFunction(V)
    v = TestFunction(V)

    mass = assemble(inner(u, v) * dx)
    advec = assemble(u_*u.dx(0)*v * dx)
    rhs = assemble(inner(ic, v) * dx)

    L = mass + advec

    assert hasattr(L, 'form')
    solve(L, u_.vector(), rhs, annotate=annotate)

    return u_

if __name__ == "__main__":

    ic = project(Expression("sin(2*pi*x[0])"),  V)
    forward = main(ic, annotate=True)
    print "Running adjoint ... "

    J = Functional(forward*forward*dx*dt[FINISH_TIME])
    m = InitialConditionParameter("Velocity")
    Jm = assemble(forward*forward*dx)
    dJdm = compute_gradient(J, m, forget=False)

    def Jfunc(ic):
      forward = main(ic, annotate=False)
      return assemble(forward*forward*dx)

    minconv = taylor_test(Jfunc, m, Jm, dJdm)
    assert minconv > 1.8
