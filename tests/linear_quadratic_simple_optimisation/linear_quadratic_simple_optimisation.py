""" 
Solves the linear-quadratic optimisation problem:

    j(m) = 0.5*m**2,
    
with steepest descent. Starting with x_0 = 5, we expect 
the optimisation to finish in exactly one iteration. 
"""
import sys
from dolfin import *
from dolfin_adjoint import *
dolfin.set_log_level(ERROR)
parameters['std_out_all_processes'] = False

def solve_pde(u, V, m):
    v = TestFunction(V)
    F = ((u-m)*v)*dx 
    solve(F == 0, u)

# Solves the linear optimisation problem 
n = 10
mesh = Mesh("mesh.xml")
V = FunctionSpace(mesh, "CG", 1)

u = project(Constant(5), V, name='State')
m = Function(V, name='Control')

J = Functional(0.5*u*u*dx)

# Run the forward model once to create the annotation
solve_pde(u, V, m)

# Run the optimisation 
p = InitialConditionParameter(m, value=m) 
rf = ReducedFunctional(J, p)
m, r = minimize_steepest_descent(rf, options={"gtol": 1e-16, "line_search": "fixed"})

assert r["Number of iterations"] == 1 
