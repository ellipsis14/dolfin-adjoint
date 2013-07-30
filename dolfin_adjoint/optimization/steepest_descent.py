from dolfin import MPI, inner, assemble, dx, Function
from data_structures import CoefficientList
from line_search import FixedLineSearch, ArmijoLineSearch, StrongWolfeLineSearch 
import numpy

def minimize_steepest_descent(rf, tol=1e-16, options={}, **args):

    # Set the default options values
    gtol = options.get("gtol", 1e-4)
    maxiter = options.get("maxiter", 200)
    disp = options.get("disp", True)
    start_alpha = options.get("start_alpha", 1.0)
    line_search = options.get("line_search", "backtracking")
    c1 = options.get("c1", 1e-4)

    # Check the validness of the user supplied parameters
    assert 0 < c1 < 1

    # Define the norm and the inner product in the relevant function spaces
    def normL2(x):
        n = 0
        for c in x:
            # c is Function
            if hasattr(c, "vector"):
                n += assemble(inner(c, c)*dx)**0.5
            else:
            # c is Constant
                n += abs(float(c))
        return n

    def innerL2(x, y):
        return assemble(inner(x, y)*dx)

    if disp and MPI.process_number()==0:
        if line_search == "backtracking":
            print "Optimising using steepest descent with an Armijo line search." 
            print "Maximum optimisation iterations: %i" % maxiter 
            print "Armijo constant: c1 = %f" % c1
        elif line_search == "fixed":
            print "Optimising using steepest descent without line search." 
            print "Maximum optimisation iterations: %i" % maxiter 

    ls = get_line_search(line_search)

    m = CoefficientList([p.data() for p in rf.parameter]) 
    m_prev = m.deep_copy()
    J =  rf
    dJ = rf.derivative

    j = None 
    j_prev = None
    dj = None 
    s = None

    # Start the optimisation loop
    it = 0
    while True:

        # Evaluate the functional at the current iterate
        if j == None:
            j = J(m)
        dj = CoefficientList(dJ(forget=None))
        # TODO: Instead of reevaluating the gradient, we should just project dj 
        s = CoefficientList(dJ(forget=None, project=True)) # The search direction is the Riesz representation of the gradient
        s.scale(-1)

        # Check for convergence                                                              # Reason:
        if not ((gtol    == None or s == None or normL2(s) > gtol) and                       # ||\nabla j|| < gtol
                (tol     == None or j == None or j_prev == None or abs(j-j_prev)) > tol and  # \Delta j < tol
                (maxiter == None or it < maxiter)):                                          # maximum iteration reached
            break

        # Compute slope at current point
        djs = dj.inner(s) 

        if djs >= 0:
            raise RuntimeError, "Negative gradient is not a descent direction. Is your gradient correct?" 

        # Define the real-valued reduced function in the s-direction 
        def phi(alpha):
            m.assign(m_prev)
            m.axpy(alpha, s) # m = m_prev + alpha*s

            return J(m)

        def phi_dphi(alpha):
            p = phi(alpha) 
            dj = CoefficientList(dJ(forget=None))
            djs = dj.inner(s) 
            return p, djs

        alpha = ls.search(phi, phi_dphi)
        # For testing purposes: compare line search step produced by the python implementation with the original Fortran version 
        # In order for this to work, you need to compile the Fortran version with "cd dcsrch_fortran; python build", and make sure
        # that the build path is in the PYTHONPATH, i.e. import pswolfe should not through an error message.
        test_fortran = False
        if test_fortran:
            import pyswolfe

            dphi = lambda x: phi_dphi(x)[1]
            ls_fort = pyswolfe.StrongWolfeLineSearch(phi(0), dphi(0), 1.0, phi, dphi, gtol=0.01) 
            ls_fort.search()

            assert ls_fort.stp == alpha

        # update m and j_new
        j_new = phi(alpha)

        # Update the current iterate
        m_prev.assign(m)
        j_prev = j
        j = j_new
        it += 1

        if disp:
            n = normL2(s)
            if MPI.process_number()==0: 
                print "Iteration %i\tJ = %s\t|dJ| = %s" % (it, j, n)
        if "callback" in options:
            options["callback"](j, s, m)

    # Print the reason for convergence
    if disp:
        n = normL2(s)
        if MPI.process_number()==0:
            if maxiter != None and iter <= maxiter:
                print "\nMaximum number of iterations reached.\n"
            elif gtol != None and n <= gtol: 
                print "\nTolerance reached: |dJ| < gtol.\n"
            elif tol != None and j_prev != None and abs(j-j_prev) <= tol:
                print "\nTolerance reached: |delta j| < tol.\n"

    return m, {"Number of iterations": it}

def get_line_search(line_search):
    if line_search == "strong_wolfe":
        ls = StrongWolfeLineSearch(ftol=1e-4, gtol=0.1)
    elif line_search == "backtracking":
        ls = ArmijoLineSearch()
    elif line_search == "fixed":
        ls = FixedLineSearch()
    else:
        raise ValueError, "Unknown line search specified. Valid values are 'backtracking', 'strong_wolfe' and 'fixed'."

    return ls
