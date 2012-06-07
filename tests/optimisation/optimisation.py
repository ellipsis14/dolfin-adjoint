""" Solves an optimisation problem with the Burgers equation as constraint """

import sys

from dolfin import *
from dolfin_adjoint import *

dolfin.set_log_level(ERROR)

n = 5
mesh = UnitInterval(n)
V = FunctionSpace(mesh, "CG", 2)

def Dt(u_next, u, timestep):
    return (u_next - u)/timestep

def main(u, annotate=False):

    u_next = Function(V, name="VelocityNext")
    v = TestFunction(V)

    nu = Constant(0.0001)

    timestep = Constant(1.0/n)

    F = (Dt(u_next, u, timestep)*v
         + u_next*grad(u_next)*v + nu*grad(u_next)*grad(v))*dx
    bc = DirichletBC(V, 0.0, "on_boundary")

    t = 0.0
    end = 0.2
    while (t <= end):
        solve(F == 0, u_next, bc, annotate=annotate)
        u.assign(u_next, annotate=annotate)

        t += float(timestep)
        adj_inc_timestep()

if __name__ == "__main__":

    ic = project(Expression("sin(2*pi*x[0])"),  V)
    u = Function(ic, name='Velocity')

    J = FinalFunctional(u*u*dx)
    def Jfunc(ic):
      u.assign(ic)
      main(u, annotate=True)
      return assemble(u*u*dx)

    # Start the optimisation 
    optimisation.minimise(Jfunc, J, InitialConditionParameter(u), ic)

    if Jfunc(ic) > 1e-9:
        print 'Test failed: Optimised functional value exceeds tolerance.' 
        sys.exit(1)
