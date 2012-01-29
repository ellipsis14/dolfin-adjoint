__author__ = "Lyudmyla Vynnytska and Marie E. Rognes"
__copyright__ = "Copyright (C) 2011 Simula Research Laboratory and %s" % __author__
__license__  = "GNU LGPL Version 3 or any later version"

# Last changed: 2011-10-17

from dolfin import *
from numerics import advection, diffusion, backward_euler

def transport(Q, dt, u, phi_):

    # Define test and trial functions
    phi = TrialFunction(Q)
    v = TestFunction(Q)

    # Constants associated with DG scheme
    alpha = Constant(10.0)
    mesh = Q.mesh()
    h = CellSize(mesh)
    n = FacetNormal(mesh)

    # Diffusivity constant
    kappa = Constant(0.0001)

    # Define discrete time derivative operator
    Dt =  lambda phi:    backward_euler(phi, phi_, dt)
    a_A = lambda phi, v: advection(phi, v, u, n)
    a_D = lambda phi, v: diffusion(phi, v, kappa, alpha, n, h)

    # Define form
    F = Dt(phi)*v*dx + a_A(phi, v) + a_D(phi, v)

    return (lhs(F), rhs(F))
