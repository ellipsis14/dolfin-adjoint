from ..reduced_functional import ReducedFunctional
from constraints import Constraint

__all__ = ['MinimizationProblem', 'MaximizationProblem']

class OptimizationProblem(object):
    """A class that encapsulates all the information required to formulate a
    reduced optimisation problem."""
    def __init__(self, reduced_functional, bounds=None, constraints=None):

        self.__check_arguments(reduced_functional, bounds, constraints)

        #: reduced_functional: a dolfin_adjoint.ReducedFunctional object that
        #: encapsulates a Function and Control
        self.reduced_functional = reduced_functional

        #: bounds: lower and upper bounds for the control (optional). None means
        #: unbounded. if not None, then it must be a list of the same length as
        #: the number controls for the reduced_functional. Each entry in the list
        #: must be a tuple (lb, ub), where ub and lb are floats, or objects
        #: of the same kind as the control.
        self.bounds = bounds

        #: constraints: general (possibly nonlinear) constraints on the controls.
        #: None means no constraints, otherwise a Constraint object or a list of 
        #: Constraints.
        self.constraints = constraints

    def __check_arguments(self, reduced_functional, bounds, constraints):
        if not isinstance(reduced_functional, ReducedFunctional):
            raise TypeError("reduced_functional should be a ReducedFunctional")

        if bounds is not None:
            if len(bounds) != len(reduced_functional.controls):
                raise TypeError("bounds should be of length number of controls of the ReducedFunctional")
            for bound in bounds:
                if len(bound) != 2:
                    raise TypeError("Each bound should be a tuple of length 2 (lb, ub)")

                for b in bound:
                    klass = reduced_functional.controls[i].klass()
                    if not (isinstance(b, (float, NoneType, klass))):
                        raise TypeError("This pair (lb, ub) should be None, a float, or a %s." % klass)

        if not ((constraints is None) or
                (isinstance(constraints, Constraint)) or
                (isinstance(constraints, list))):
            raise TypeError("constraints should be None or a Constraint or a list of Constraints")

class MinimizationProblem(OptimizationProblem):
    pass

class MaximizationProblem(OptimizationProblem):
    pass
