""" 
Test for the mapping to Euclidian space in the ReducedFunctionalNumPy module.
"""
import sys
import numpy as np
from dolfin import *
from dolfin_adjoint import *

def solve_pde(u, V, m):
    v = TestFunction(V)
    F = ((u-m)*v)*dx 
    solve(F == 0, u)

# Solves the linear optimisation problem 
n = 10
mesh = Mesh("mesh.xml")
V = FunctionSpace(mesh, "CG", 1)

m = project(Constant(5), V, name='Control')
u = Function(V, name='State')

J = Functional(0.5*u*u*dx)

# Run the forward model once to create the annotation
solve_pde(u, V, m)

# Construct the reduced functionals
p = SteadyParameter(m, value=m) 

rf = ReducedFunctional(J, p)
rf_np = ReducedFunctionalNumPy(rf)
rf_np_euc = ReducedFunctionalNumPy(rf, in_euclidian_space=True)

m_eucl = rf_np_euc.obj_to_array(m) 

# Test equivalence of norms
assert abs(assemble(m*m*dx) - 25) < 1e-12
assert abs(np.dot(m_eucl, m_eucl) - 25) < 1e-12 

# Test equivalence of functionals
j = rf_np(m.vector().array())
j_eucl = rf_np_euc(m_eucl)

assert abs(j - 12.5) < 1e-12
assert abs(j_eucl - 12.5) < 1e-12

# Test equivalence of gradients
dj = rf.derivative(project=True, forget=False)[0]
dj_eucl = rf_np_euc.derivative(project=True, forget=False)

s = project(Expression("sin(x[0])"), V, annotate=False)
s_eucl = rf_np_euc.obj_to_array(s) 

djs = assemble(inner(dj, s)*dx)
djs_eucl = np.dot(dj_eucl, s_eucl)

assert abs(djs - djs_eucl) < 1e-12

# Test equivalence of gradients norms

djdj = assemble(inner(dj, dj)*dx)
djdj_eucl = np.dot(dj_eucl, dj_eucl)

assert abs(djdj - djdj_eucl) < 1e-12

# Test equivalence of gradients with project = False
dj_op = rf.derivative(project=False, forget=False)[0]
dj_eucl = rf_np_euc.derivative(project=False, forget=False)

djs = dj_op.vector().inner(s.vector())
djs_eucl = np.dot(dj_eucl, s_eucl)

assert abs(djs - djs_eucl) < 1e-12

# Test equivalence of gradients norms with project = False

djdj = dj_op.vector().inner(dj.vector())
djdj_eucl = np.dot(dj_eucl, dj_eucl)

assert abs(djdj - djdj_eucl) < 1e-12

# Test the equivalence of Hessians
m_dot = project(Expression("exp(x[0])"), V, annotate=False)
m_dot_eucl = rf_np_euc.obj_to_array(m_dot) 

H = rf.hessian(m_dot)[0]
H_eucl = rf_np_euc.hessian(None, m_dot_eucl)

Hs = H.vector().inner(s.vector())
Hs_eucl = np.dot(H_eucl, s_eucl)

assert abs(Hs - Hs_eucl) < 1e-12

info_green("Test passed")
