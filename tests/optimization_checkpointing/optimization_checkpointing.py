""" Solves an optimisation problem with the Burgers equation as constraint """

import sys

from dolfin import *
from dolfin_adjoint import *
import scipy
import libadjoint

dolfin.set_log_level(ERROR)
dolfin.parameters["optimization"]["test_gradient"] = True 

n = 20
end = 0.2
timestep = Constant(1.0/n)
mesh = UnitIntervalMesh(n)
V = FunctionSpace(mesh, "CG", 2)

adj_checkpointing(strategy='multistage',
                  steps=int(end*n)+1,
                  snaps_on_disk=2,
                  snaps_in_ram=2,
                  verbose=True)

def Dt(u_next, u, timestep):
    return (u_next - u)/timestep

def main(u, annotate=False):

    u_next = Function(V, name="VelocityNext")
    v = TestFunction(V)

    nu = Constant(0.0001)

    F = (Dt(u_next, u, timestep)*v
         + u_next*u_next.dx(0)*v + nu*u_next.dx(0)*v.dx(0))*dx
    bc = DirichletBC(V, 0.0, "on_boundary")

    t = 0.0
    adjointer.time.start(t)
    while (t <= end):
        solve(F == 0, u_next, bc, annotate=annotate)
        u.assign(u_next, annotate=annotate)

        t += float(timestep)
        adj_inc_timestep(time=t, finished = t>end)

def derivative_cb(j, dj, m):
  print "j = %f, max(dj) = %f, max(m) = %f." % (j, dj.vector().max(), m.vector().max())

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
    reduced_functional = ReducedFunctional(J, InitialConditionParameter(u), derivative_cb = derivative_cb)

    print "\n === Solving problem with L-BFGS-B. === \n"
    # Run the optimisation problem with gradient tests and L-BFGS-B
    # scipt.optimize 0.11.0 introduced a new generic interface to the minimisation routines, 
    # which dolfin-adjoint.optimize automatically uses if available. Since the arguments changed, we need
    # to check for the version at this point.
    try:
        from scipy.optimize import minimize as scipy_minimize
        new_scipy = True
    except ImportError:
        new_scipy = False
        pass

    if new_scipy:
        u_opt = minimize(reduced_functional, method = 'L-BFGS-B', bounds = (lb, 1), tol = 1e-10, options = {'disp': True})
    else:
        u_opt = minimize(reduced_functional, method = 'L-BFGS-B', pgtol=1e-10, factr=1e5, bounds = (lb, 1), iprint = 1)

    tol = 1e-9
    final_functional = reduced_functional(u_opt)
    print "Final functional value: ", final_functional
    if final_functional > tol:
        print 'Test failed: Optimised functional value exceeds tolerance: ' , final_functional, ' > ', tol, '.'
        sys.exit(1)

    # Run the problem again with SQP, this time for performance reasons with the gradient test switched off
    dolfin.parameters["optimization"]["test_gradient"] = False 

    if new_scipy:
        # Method specific arguments:
        options = {"SLSQP": {"bounds": (lb, 1)},
                   "BFGS": {"bounds": None},
                   "COBYLA": {"bounds": None, "rhobeg": 0.1},
                   "TNC": {"bounds": None},
                   "L-BFGS-B": {"bounds": (lb, 1)},
                   "Newton-CG": {"bounds": None, "maxiter": 1},
                   "Nelder-Mead": {"bounds": None }, 
                   "Anneal": {"bounds": None, "lower": -0.1, "upper": 0.1},
                   "CG": {"bounds": None},
                   "Powell": {"bounds": None}
                  }

        for method in ["SLSQP", "BFGS", "COBYLA", "TNC", "L-BFGS-B", "Nelder-Mead", "Anneal", "CG"]: #, "Powell"]:
            print "\n === Solving problem with %s. ===\n" % method
            u_opt.assign(ic, annotate = False)
            reduced_functional(u_opt)
            u_opt = minimize(reduced_functional, 
                             bounds = options[method].pop("bounds"), 
                             method = method, tol = 1e-10, 
                             options = dict({'disp': True, "maxiter": 2}, **options[method]))
    else:
        print "You do not have a recent scipy.optimize version installed. Without it I can not run the remaining optimisation tests."
    info_green("Test passed")
