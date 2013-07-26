""" A test for the Strong Wolfe steplength algorithm 
    that checks if the algorithm produces step sizes 
    that indeed satisfy the Wolfe conditions 
"""

from dolfin_adjoint.optimization.line_search.strong_wolfe import StrongWolfeLineSearch
from math import sin, cos
import random
random.seed(11)

def phi(x, shift, scale=1):
    r = 0.5*(x-shift)**2 + sin(x)
    return scale*r

def dphi(x, shift, scale=1):
    r = x-shift + cos(x)
    return scale*r

def test_wolfe(shift, ftol = 0.998, gtol=0.999):
    ls = StrongWolfeLineSearch(ftol=ftol, gtol=gtol)
    assert(ftol < gtol) # Make sure that there exists a step

    # Make sure that phi has always a negative gradient at 0
    if dphi(0, shift) > 0:
        scale = -1
    else:
        scale = 1

    myphi = lambda x: phi(x, shift, scale)
    mydphi = lambda x: dphi(x, shift, scale)

    assert(mydphi(0) < 0)
    stp = ls.search(myphi, mydphi)

    def decrease_condition(phi, dphi, stp, ftol):
        return phi(stp) <= (phi(0) + ftol*stp*dphi(0))

    def curvature_condition(dphi, stp, gtol):
        return abs(dphi(stp)) <= gtol*abs(dphi(0))

    print 'Decrease condition satisfied: ', decrease_condition(myphi, mydphi, stp, ftol)
    print 'Curvature condition satisfied: ', curvature_condition(mydphi, stp, gtol)

    assert decrease_condition(myphi, mydphi, stp, ftol) 
    assert curvature_condition(mydphi, stp, gtol)

for i in range(10):
    print "Performing linesearch at random point..."
    test_wolfe(10*random.random())
