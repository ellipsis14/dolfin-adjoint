__author__ = "Lyudmyla Vynnytska and Marie E. Rognes"
__copyright__ = "Copyright (C) 2011 Simula Research Laboratory and %s" % __author__
__license__  = "GNU LGPL Version 3 or any later version"

# Last changed: 2011-10-17

from dolfin import *

def stokes_space(mesh):
    # Define spatial discretization (Taylor--Hood)
    V = VectorFunctionSpace(mesh, "CG", 2)
    Q = FunctionSpace(mesh, "CG", 1)
    W = V * Q
    return W

def strain(v):
    return sym(grad(v))

def momentum(W, eta, f):

    # Define basis functions
    (u, p) = TrialFunctions(W)
    (v, q) = TestFunctions(W)

    # Define equation F((u, p), (v, q)) = 0
    F = (2.0*eta*inner(strain(u), strain(v))*dx
         + div(v)*p*dx
         + div(u)*q*dx
         + inner(f, v)*dx)

    # Define form for preconditioner
    precond = inner(grad(v), grad(u))*dx + q*p*dx

    # Return left and right hand side and preconditioner
    return (lhs(F), rhs(F), precond)
