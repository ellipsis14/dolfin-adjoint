#!/usr/bin/env python
# -*- coding: utf-8 -*- 

r"""
Solve example 5.2 of 

@article{hintermueller2011,
year={2011},
journal={Computational Optimization and Applications},
volume={50},
number={1},
doi={10.1007/s10589-009-9307-9},
title={A smooth penalty approach and a nonlinear multigrid algorithm for elliptic MPECs},
url={http://dx.doi.org/10.1007/s10589-009-9307-9},
publisher={Springer US},
author={Hinterm{\"u}ller, M. and Kopacka, I.},
pages={111-145},
}

"""

from dolfin import *
from dolfin_adjoint import *

n = 128
mesh = UnitSquareMesh(n, n)

V = FunctionSpace(mesh, "CG", 1)
gamma = Constant(1e3) # 1.0 / alpha in the paper

y = Function(V, name="Solution")
u = Function(V, name="Control")
w = TestFunction(V)

gamma = Constant(1e3) # 1.0 / alpha in the paper
eps = 1e-4

def smoothmax(r):
  return conditional(gt(r, eps), r - eps/2, conditional(lt(r, 0), 0, r**2 / (2*eps))) 
def uflmax(a, b):
  return conditional(gt(a, b), a, b) 

f = interpolate(Expression("-std::abs(x[0]*x[1] - 0.5) + 0.25"), V)
F = inner(grad(y), grad(w))*dx - gamma * inner(smoothmax(-y), w)*dx - inner(f + u, w)*dx

bc = DirichletBC(V, 0.0, "on_boundary")
solve(F == 0, y, bcs = bc, solver_parameters={"newton_solver": {"maximum_iterations": 30}})

yd = Function(f)
alpha = 1e-2
J = Functional(0.5*inner(y - yd, y - yd)*dx + alpha/2*inner(u, u)*dx)

rf = ReducedFunctional(J, TimeConstantParameter(u))

y_hdr = adjglobals.adj_variables[y]
y_ic  = adjglobals.adj_variables[y]; y_ic.iteration = 0

ypvd = File("y_opt.pvd") 
upvd = File("u_opt.pvd") 
for g in [1000*2**i for i in range(12)]:
  print "Set gamma = ", g
  gamma.assign(g)
  u_opt = minimize(rf, bounds = (0.01, 0.03), options = {"disp": True, "gtol": 1e-12, "ftol": 1e-100})
  rf(u_opt)
  y_opt = adjglobals.adjointer.get_variable_value(y_hdr).data
  ypvd << y_opt
  upvd << u_opt

  feasibility = sqrt(assemble(inner((uflmax(0, -y_opt)), (uflmax(0, -y_opt)))*dx))
  info_red("Feasibility: %s"%  feasibility)
  info_red("Norm of y: %s"% sqrt(assemble(inner(y_opt, y_opt)*dx)))
  info_red("Norm of u_opt: %s"% sqrt(assemble(inner(u_opt, u_opt)*dx)))

  reduced_functional.replace_tape_ic_value(y_ic, y_opt)
