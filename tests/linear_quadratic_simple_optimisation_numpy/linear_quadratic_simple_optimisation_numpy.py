""" 
Solves the linear-quadratic optimisation problem:

    j(m) = 0.5*m**2,
    
with steepest descent. Starting with x_0 = 5, we expect 
the optimisation to finish in exactly one iteration. 
"""
from dolfin import *
from dolfin_adjoint import *

def solve_pde(u, V, m):
    v = TestFunction(V)
    F = ((u-m)*v)*dx 
    solve(F == 0, u)

n = 10
mesh = Mesh("mesh.xml")
V = FunctionSpace(mesh, "CG", 1)

m = project(Constant(5), V, name='State')
u = Function(V, name='Control')

J = Functional(0.5*u*u*dx)

# Run the forward model once to create the annotation
solve_pde(u, V, m)

# Run the optimisation 
p = InitialConditionParameter(m, value=m) 
rf = ReducedFunctional(J, p)

rf_np = ReducedFunctionalNumPy(rf, in_euclidian_space=True)

m_np = rf_np.obj_to_array(rf.parameter[0].data())
j = rf_np(m_np)
assert abs(j-12.5) < 1e-10

# Perform a steepest descent step 
dj_np = rf_np.derivative()
m_new = m_np - dj_np
j = rf_np(m_new)
assert abs(j) < 1e-10

info_green("Test passed")
