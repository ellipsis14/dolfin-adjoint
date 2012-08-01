""" Solves an optimisation problem with the Burgers equation as constraint """

import sys

from dolfin import *
from dolfin_adjoint import *
import libadjoint

dolfin.set_log_level(ERROR)
dolfin.parameters["optimisation"]["test_gradient"] = True 

n = 10
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
    adjointer.time.start(t)
    while (t <= end):
        solve(F == 0, u_next, bc, annotate=annotate)
        u.assign(u_next, annotate=annotate)

        t += float(timestep)
        adj_inc_timestep(time=t, finished = t>end)

if __name__ == "__main__":

    ic = project(Expression("sin(2*pi*x[0])"),  V)
    u = Function(ic, name='Velocity')

    J = Functional(u*u*dx*dt[FINISH_TIME])

    # Run the model once to create the annotation
    u.assign(ic)
    main(u, annotate=True)

    # Run the optimisation 
    lb = project(Expression("-1"),  V)
    
    # Define the reduced funtional
    reduced_functional = ReducedFunctional(J, InitialConditionParameter(u))

    # Run the optimisation problem with gradient tests and L-BFGS-B
    minimize(reduced_functional, ic, algorithm = 'scipy.l_bfgs_b', pgtol=1e-6, factr=1e5, bounds = (lb, 1), iprint = 1)
    ic = project(Expression("sin(2*pi*x[0])"),  V)

    # Run the problem again with SQP, this time for performance reasons with the gradient test switched off
    dolfin.parameters["optimisation"]["test_gradient"] = False 
    minimize(reduced_functional, ic, algorithm = 'scipy.slsqp', bounds = (lb, 1), iprint = 2, acc = 1e-10)

    tol = 1e-9
    if reduced_functional(ic) > tol:
        print 'Test failed: Optimised functional value exceeds tolerance: ' , Jfunc(ic), ' > ', tol, '.'
        sys.exit(1)
