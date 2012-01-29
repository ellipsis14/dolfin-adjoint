__author__ = "Lyudmyla Vynnytska and Marie E. Rognes"
__copyright__ = "Copyright (C) 2011 Simula Research Laboratory and %s" % __author__
__license__  = "GNU LGPL Version 3 or any later version"

# Last changed: 2011-10-17

from dolfin import *
from numerics import advection, diffusion, backward_euler

def energy(Q, dt, u, T_):

    # Define basis functions
    T = TrialFunction(Q)
    psi = TestFunction(Q)

    # Diffusivity constant
    kappa = Constant(1.0)

    # Variables associated with DG scheme
    alpha = Constant(50.0)
    mesh = Q.mesh()
    h = CellSize(mesh)
    n = FacetNormal(mesh)

    # Define discrete time derivative operator
    Dt =  lambda T:    backward_euler(T, T_, dt)
    a_A = lambda u, T, psi: advection(T, psi, u, n)
    a_D = lambda T, psi: diffusion(T, psi, kappa, alpha, n, h)

    # Define form
    F = Dt(T)*psi*dx + a_A(u, T, psi) + a_D(T, psi)

    return (lhs(F), rhs(F))

def energy_correction(Q, dt, u, u_, T_):

    # Define test and trial functions
    T = TrialFunction(Q)
    psi = TestFunction(Q)

    # Diffusivity constant
    k_c = Constant(1.0)

    # Constants associated with DG scheme
    alpha = Constant(50.0)
    mesh = Q.mesh()
    h = CellSize(mesh)
    n = FacetNormal(mesh)

    # Define discrete time derivative operator
    def Dt(T):
        return backward_euler(T, T_, dt)

    # Add syntactical sugar for a_A and a_D
    def a_A(u, T, psi):
        return advection(T, psi, u, n, theta=0.5)
    def a_D(T, psi):
        return diffusion(T, psi, k_c, alpha, n, h, theta=0.5)

    # Define form
    F = (Dt(T)*psi*dx
         + a_A(u, T, psi) + a_A(u_, T_, psi)
         + a_D(T, psi) + a_D(T_, psi))

    return (lhs(F), rhs(F))
