"""
Implementation of Burger's equation with nonlinear solve in each
timestep and a functional integrating over time
"""

import sys

from dolfin import *
from dolfin_adjoint import *

dolfin.parameters["adjoint"]["record_all"] = True
dolfin.parameters["adjoint"]["fussy_replay"] = False

n = 30
mesh = UnitIntervalMesh(n)
V = FunctionSpace(mesh, "CG", 2)

def Dt(u, u_, timestep):
    return (u - u_)/timestep

def main(ic, annotate=False):

    u_ = Function(ic, name="Velocity")
    u = Function(V, name="VelocityNext")
    v = TestFunction(V)

    nu = Constant(0.0001)

    timestep = Constant(1.0/n)

    F = (Dt(u, u_, timestep)*v
         + u*grad(u)*v + nu*grad(u)*grad(v))*dx
    bc = DirichletBC(V, 0.0, "on_boundary")

    t = 0.0
    end = 0.2
    j = 0
    j += 0.5*float(timestep)*assemble(u_*u_*dx)

    if annotate:
      adjointer.time.start(0)

    while (t <= end):
        solve(F == 0, u, bc, annotate=annotate)
        u_.assign(u, annotate=annotate)

        t += float(timestep)

        if t>end: 
          quad_weight = 0.5
        else:
          quad_weight = 1.0
        j += quad_weight*float(timestep)*assemble(u_*u_*dx)
        if annotate:
          adj_inc_timestep(time=t, finished=t>end)

    return j, u_

if __name__ == "__main__":

    ic = project(Expression("sin(2*pi*x[0])"),  V)
    ic_copy = Function(ic)
    j, forward = main(ic, annotate=True)
    forward_copy = Function(forward)
    ic = forward
    ic.vector()[:] = ic_copy.vector()

    adj_html("burgers_newton_forward.html", "forward")
    adj_html("burgers_newton_adjoint.html", "adjoint")

    print "Running forward replay .... "
    replay_dolfin(forget=False)

    print "Running adjoint ... "

    timestep = Constant(1.0/n)
    J = Functional(forward*forward*dx*dt)
    dJdic = compute_gradient(J, InitialConditionParameter(ic), forget=False)

    def Jfunc(ic):
      j, forward = main(ic, annotate=False)
      return j 

    minconv = test_initial_condition_adjoint(Jfunc, ic, dJdic, seed=1.0e-5)
    if minconv < 1.9:
      exit_code = 1
    else:
      exit_code = 0
    sys.exit(exit_code)
