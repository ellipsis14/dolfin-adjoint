"""
This example demonstrates a tracking type optimal control problem for
time-dependent Stokes flow.

In particular, both the constraint and functional is time-dependent.

For details, see "Optimal error estimates and computations for
tracking-type control of the instationary Stokes system", Deckelnick
and Hinze, 2001.

Implicit Euler discretization in time and Taylor-Hood elements in
space

"""

__author__ = "Marie E. Rognes (meg@simula.no)"

from dolfin import *
from dolfin_adjoint import *

n = 10
mesh = UnitSquareMesh(n, n)
T = 0.1

def forward(mesh, T):

    V = VectorFunctionSpace(mesh, "CG", 2)
    Q = FunctionSpace(mesh, "CG", 1)
    W = V*Q

    w = Function(W)
    (y, p) = split(w)


    while (
