from dcsrch import dcsrch
from numpy import zeros

class StrongWolfeLineSearch:
    def __init__(self, ftol = 1e-4, gtol = 0.9, xtol = 1e-1, start_stp = 1.0, stpmin = None, stpmax = None):
        '''
        This class implements a line search algorithm whose steps 
        satisfy the strong Wolfe conditions (i.e. they satisfies a 
        sufficient decrease condition and a curvature condition).

        The algorithm is designed to find a step 'stp' that satisfies
        the sufficient decrease condition

               f(stp) <= f(0) + ftol*stp*f'(0),

        and the curvature condition

               abs(f'(stp)) <= gtol*abs(f'(0)).

        If ftol is less than gtol and if, for example, the function
        is bounded below, then there is always a step which satisfies
        both conditions.

        If no step can be found that satisfies both conditions, then
        the algorithm stops with a warning. In this case stp only
        satisfies the sufficient decrease condition.

        The function arguments are:

           ftol      | a nonnegative tolerance for the sufficient decrease condition.
           gtol      | a nonnegative tolerance for the curvature condition.
           xtol      | a nonnegative relative tolerance for an acceptable step.
           start_stp | a guess for an initial step size. 
           stpmin    | a nonnegative lower bound for the step.
           stpmax    | a nonnegative upper bound for the step.

        '''
        self.ftol       = ftol 
        self.gtol       = gtol
        self.xtol       = xtol
        self.start_stp  = start_stp
        self.stpmin     = stpmin
        self.stpmax     = stpmax

    def search(self, phi, dphi):
        ''' Performs the line search on the function phi. 

            dphi must implement the derivative of phi.
            Both phi and dphi must be functions [0, oo] -> R.

            The return value is a step that satisfies the strong Wolfe condition. 
        '''
            
        # Set up the variables for dscrch
        isave = zeros(3)
        dsave = zeros(14)
        task = "START"

        stp = self.start_stp
        f = phi(0)
        g = dphi(0)
        print "g", g

        while True:
            stp, task, isave, dsave = self.__csrch__(f, g, stp, task, isave, dsave)

            if task in ("START", "FG"):
                f = phi(stp)
                g = dphi(stp)
            else:
                break

        if "Error" in task:
            raise RuntimeError, task
        elif "Warning" in task:
            raise Warning, task
        else:
            assert task=="Convergence"
            return stp

    def __csrch__(self, f, g, stp, task, isave, dsave):
        print "g= ", g
        stp, task, isave, dsave = dcsrch(stp, f, g, self.ftol, self.gtol, self.xtol, task, self.stpmin, self.stpmax, isave, dsave)
        return stp, task, isave, dsave

