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

def main(u, annotate=False, J = None):

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
    #J = Functional(u*u*dx*dt)
    def Jfunc(ic):
      u.assign(ic)
      main(u, annotate=True, J=J)
      return assemble(u*u*dx)

    jfuncvalue = Jfunc(ic)
    print "Jfunc value", jfuncvalue 

    def reduced_functional(coeff):

        init_rhs = adjlinalg.Vector(coeff).duplicate()
        init_rhs.axpy(1.0,adjlinalg.Vector(coeff))
        rhs = adjrhs.RHS(init_rhs)
        class MyEquation(object):
            pass
        MyEquation()
        x = MyEquation()
        x.equation = adjointer.adjointer.equations[0]
        rhs.register(x)

        func_value = 0.
        for i in range(adjglobals.adjointer.equation_count):
            (fwd_var, output) = adjglobals.adjointer.get_forward_solution(i)

            storage = libadjoint.MemoryStorage(output)
            storage.set_overwrite(True)
            adjglobals.adjointer.record_variable(fwd_var, storage)
            if i == adjointer.timestep_end_equation(fwd_var.timestep):
                func_value += adjointer.evaluate_functional(J, fwd_var.timestep)

            #adjglobals.adjointer.forget_forward_equation(i)
        return func_value


    # Run the optimisation 
    lb = project(Expression("-1"),  V)

    optimisation.minimise(reduced_functional, J, InitialConditionParameter(u), ic, algorithm = 'scipy.l_bfgs_b', dontreset = True, pgtol=1e-6, factr=1e5, bounds = (lb, 1), iprint = 1)
    ic = project(Expression("sin(2*pi*x[0])"),  V)

    # For performance reasons, switch the gradient test off
    dolfin.parameters["optimisation"]["test_gradient"] = False 
    optimisation.minimise(reduced_functional, J, InitialConditionParameter(u), ic, algorithm = 'scipy.slsqp', dontreset = True, bounds = (lb, 1), iprint = 2, acc = 1e-10)

    tol = 1e-9
    if reduced_functional(ic) > tol:
        print 'Test failed: Optimised functional value exceeds tolerance: ' , Jfunc(ic), ' > ', tol, '.'
        sys.exit(1)
