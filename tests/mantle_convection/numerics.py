__author__ = "Lyudmyla Vynnytska and Marie E. Rognes"
__copyright__ = "Copyright (C) 2011 Simula Research Laboratory and %s" % __author__
__license__  = "GNU LGPL Version 3 or any later version"

# Last changed: 2011-10-17

from dolfin import *

def advection(phi, psi, u, n, theta=1.0):

    # Define |u * n|
    un = abs(dot(u('+'), n('+')))

    # Contributions from cells
    a_cell = - dot(u*phi, grad(psi))*dx

    # Contributions from interior facets
    a_int = (dot(u('+'), jump(psi, n))*avg(phi)
             + 0.5*un*dot(jump(phi, n), jump(psi, n)))*dS

    return theta*(a_cell + a_int)

def diffusion(phi, psi, k_c, alpha, n, h, theta=1.0):

    # Contribution from the cells
    a_cell = k_c*dot(grad(phi), grad(psi))*dx

    # Contribution from the interior facets
    tmp = (alpha('+')/h('+')*dot(jump(psi, n), jump(phi, n))
           - dot(avg(grad(psi)), jump(phi, n))
           - dot(jump(psi, n), avg(grad(phi))))*dS
    a_int = k_c('+')*tmp

    return theta*(a_cell + a_int)


def backward_euler(u, u_, dt):
    return (u - u_)/dt
