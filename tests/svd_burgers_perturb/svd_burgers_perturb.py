"""
Implementation of Burger's equation with nonlinear solve in each
timestep
"""

import sys
import numpy
import random

from dolfin import *
from dolfin_adjoint import *

dolfin.parameters["adjoint"]["record_all"] = True
dolfin.parameters["adjoint"]["fussy_replay"] = False

n = 50
mesh = UnitInterval(n)
V = FunctionSpace(mesh, "CG", 1)

def Dt(u, u_, timestep):
    return (u - u_)/timestep

def main(ic, annotate=False):

    u_ = Function(ic, name="State")
    u = Function(V, name="NextState")
    v = TestFunction(V)

    nu = Constant(0.1)

    timestep = Constant(1.0/n)

    F = (Dt(u, u_, timestep)*v
        + u*grad(u)*v + nu*grad(u)*grad(v))*dx
    bc = DirichletBC(V, 0.0, "on_boundary")

    t = 0.0
    end = 1.0
    while (t <= end):
        solve(F == 0, u, bc, annotate=annotate)
        u_.assign(u, annotate=annotate)

        t += float(timestep)
        adj_inc_timestep()

    return u_

if __name__ == "__main__":

    ic = project(Expression("sin(2*pi*x[0])"),  V)
    ic_copy = Function(ic)
    forward = main(ic, annotate=True)
    forward_copy = Function(forward)

    ic = forward
    ic.vector()[:] = ic_copy.vector()

    svd = adj_compute_propagator_svd("State", "State", nsv=1)
    (sigma, u, v) = svd.get_svd(0, return_vectors=True)

    ic_norm = sqrt(assemble(inner(v, v)*dx))

    perturbed_ic = Function(ic)
    perturbed_ic.vector().axpy(1.0, v.vector())
    perturbed_soln = main(perturbed_ic, annotate=False)

    soln_perturbation = perturbed_soln - forward_copy
    final_norm = sqrt(assemble(inner(soln_perturbation, soln_perturbation)*dx))
    print "Norm of initial perturbation: ", ic_norm
    print "Norm of final perturbation: ", final_norm
    ratio = final_norm / ic_norm
    print "Ratio: ", ratio
    print "Predicted growth of perturbation: ", sigma

    prediction_error = abs(sigma - ratio)/ratio * 100
    print "Prediction error: ", prediction_error,  "%"
    assert prediction_error < 2
