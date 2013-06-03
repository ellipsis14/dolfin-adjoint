"""
Naive implementation of Burgers' equation, goes oscillatory later
"""

import sys

from dolfin import *
from dolfin_adjoint import *
from math import ceil

n = 100
mesh = UnitIntervalMesh(n)
V = FunctionSpace(mesh, "CG", 2)

#dolfin.parameters["adjoint"]["record_all"] = True
#dolfin.parameters["adjoint"]["test_hermitian"] = (100, 1.0e-14)
#dolfin.parameters["adjoint"]["test_derivative"] = 6

def Dt(u, u_, timestep):
    return (u - u_)/timestep

def main(ic, timestep, end, annotate=False):

    u_ = Function(ic, name = "system state")
    u = TrialFunction(V)
    v = TestFunction(V)

    nu = Constant(0.0001, name = "nu")

    F = (Dt(u, u_, timestep)*v
         + u_*u.dx(0)*v + nu*u.dx(0)*v.dx(0))*dx

    (a, L) = system(F)

    bc = DirichletBC(V, 0.0, "on_boundary")

    t = 0.0

    u = Function(V)
    j = 0
    j += 0.5*float(timestep)*assemble(u_*u_*dx)
    if annotate:
      adjointer.time.start(t)

    while (t <= end):
        solve(a == L, u, bc, annotate=annotate)

        u_.assign(u, annotate=annotate)

        t += float(timestep)

        if t>end: 
          quad_weight = 0.5
        else:
          quad_weight = 1.0
        j += quad_weight*float(timestep)*assemble(u_*u_*dx)
        
        if annotate:
          adj_inc_timestep(time=t, finished=t>end)
        #plot(u)

    #interactive()
    return j, u_

if __name__ == "__main__":
    timestep = Constant(1.0/n, name = "timestep")
    end = 0.5
    
    adj_checkpointing('multistage', int(ceil(end/float(timestep))), 5, 10, verbose=True)
   
    ic = project(Expression("sin(2*pi*x[0])"),  V, name = "initial condition", annotate = True)
    j, forward = main(ic, timestep, end, annotate=True)
    adj_html("burgers_picard_checkpointing_forward.html", "forward")
    adj_html("burgers_picard_checkpointing_adjoint.html", "adjoint")

    J = Functional(forward*forward*dx*dt)

    def Jfunc(ic):
      j, forward = main(ic, timestep, end, annotate=False)
      return j 
    
    dJdf = compute_gradient(J, InitialConditionParameter(ic), forget = False)
    
    minconv = taylor_test(Jfunc, InitialConditionParameter(ic), j, dJdf, seed=1.0e-4)
    if minconv < 1.9:
      exit_code = 1
    else:
      exit_code = 0
    sys.exit(exit_code)
