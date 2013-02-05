"""
Implementation of Burger's equation with nonlinear solve in each
timestep
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

class BurgersProblem(NonlinearProblem):
  def __init__(self, F, u, bc):
    NonlinearProblem.__init__(self)
    self.f = F
    self.jacob = derivative(F, u)
    self.bc = bc

  def F(self, b, x):
    assemble(self.f, tensor=b)
    self.bc.apply(b)

  def J(self, A, x):
    assemble(self.jacob, tensor=A)
    self.bc.apply(A)

def main(ic, annotate=False):

    u_ = Function(ic)
    u = Function(V)
    v = TestFunction(V)

    nu = Constant(0.0001)

    timestep = Constant(1.0/n)

    solver = NewtonSolver("lu")
    solver.parameters["convergence_criterion"] = "incremental"
    solver.parameters["relative_tolerance"] = 1e-6

    bc = DirichletBC(V, 0.0, "on_boundary")
    burgers = BurgersProblem((Dt(u, u_, timestep)*v + u*grad(u)*v + nu*grad(u)*grad(v))*dx, u, bc)

    t = 0.0
    end = 0.2
    while (t <= end):
        solver.solve(burgers, u.vector(), annotate=annotate)
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

    J = Functional(forward*forward*dx*dt[FINISH_TIME])
    dJdm = compute_gradient(J, InitialConditionParameter(forward), forget = False)

    def Jfunc(ic):
      forward = main(ic, annotate=False)
      return assemble(forward*forward*dx)

    minconv = test_initial_condition_adjoint(Jfunc, ic, dJdm, seed=1.0e-5)
    if minconv < 1.9:
      info_red("Test failed. Convergence rate is %f < 1.9", minconv)
      sys.exit(1)

    dJ = assemble(derivative(forward_copy*forward_copy*dx, forward_copy))

    ic = forward
    ic.vector()[:] = ic_copy.vector()
    minconv = test_initial_condition_tlm(Jfunc, dJ, ic, seed=1.0e-5)
    if minconv < 1.9:
      info_red("Test failed. Convergence rate is %d < 1.9", minconv)
      sys.exit(1)
