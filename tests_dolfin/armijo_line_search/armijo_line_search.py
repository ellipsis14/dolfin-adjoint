""" A test for the Armijo steplength algorithm 
    that checks if the algorithm produces step sizes 
    that indeed satisfy the Armijo condition
"""

from dolfin_adjoint.optimization.line_search.armijo import ArmijoLineSearch
from math import sin, cos
import random
random.seed(11)

def phi(x, shift, scale=1):
    r = 0.5*(x-shift)**2 + sin(x)
    return scale*r

def dphi(x, shift, scale=1):
    r = x-shift + cos(x)
    return scale*r

def test_armijo(shift, ftol = 0.998):
    ls = ArmijoLineSearch(ftol=ftol)

    # Make sure that phi has always a negative gradient at 0
    if dphi(0, shift) > 0:
        scale = -1
    else:
        scale = 1

    myphi = lambda x: phi(x, shift, scale)
    mydphi = lambda x: dphi(x, shift, scale)
    myphi_dphi = lambda x: (myphi(x), mydphi(x))

    assert(mydphi(0) < 0)
    stp = ls.search(myphi, myphi_dphi)

    def decrease_condition(phi, dphi, stp, ftol):
        return phi(stp) <= (phi(0) + ftol*stp*dphi(0))

    print 'Decrease condition satisfied: ', decrease_condition(myphi, mydphi, stp, ftol)

    assert decrease_condition(myphi, mydphi, stp, ftol) 

for i in range(10):
    print "Performing linesearch at random point..."
    test_armijo(10*random.random())
