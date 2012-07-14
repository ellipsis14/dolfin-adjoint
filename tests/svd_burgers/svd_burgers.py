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

n = 10
mesh = UnitInterval(n)
V = FunctionSpace(mesh, "CG", 1)

def Dt(u, u_, timestep):
    return (u - u_)/timestep

def main(ic, annotate=False):

    u_ = Function(ic, name="State")
    u = Function(V, name="NextState")
    v = TestFunction(V)

    nu = Constant(0.0001)

    timestep = Constant(1.0/n)

    F = (Dt(u, u_, timestep)*v
         + u*grad(u)*v + nu*grad(u)*grad(v))*dx
    bc = DirichletBC(V, 0.0, "on_boundary")

    t = 0.0
    end = 0.2
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

    perturbation = Function(ic)
    vec = perturbation.vector()
    for i in range(len(vec)):
      vec[i] = random.random()

    info_blue("Computing the TLM the direct way ... ")
    param = InitialConditionParameter(ic, perturbation)
    for (tlm, var) in compute_tlm(param, forget=False):
      pass
    final_tlm = tlm

    ndof = V.dim()
    info_blue("Computing the TLM the SVD way ... ")
    svd = compute_gst("State", "State", nsv=ndof, ic_norm=None, final_norm=None)

    assert svd.ncv == ndof

    mat = compute_propagator_matrix(svd)
    tlm_output = numpy.dot(mat, perturbation.vector().array())
    norm = numpy.linalg.norm(final_tlm.vector().array() - tlm_output)

    print "Error norm: ", norm

    assert norm < 1.0e-8
