import optimization_problem
import dolfin

class OptimizationSolver(object):
    """An abstract base class that represents an optimization solver."""
    def __init__(self, problem, parameters=None):
        self.__check_arguments(problem, parameters)

        #: problem: an OptimizationProblem instance.
        self.problem = problem

        #: parameters: a dictionary of parameters.
        self.parameters = parameters

    def __check_arguments(self, problem, parameters):
        if not isinstance(problem, optimization_problem.OptimizationProblem):
            raise TypeError("problem should be an OptimizationProblem.")

        assert isinstance(parameters, (dict, dolfin.Parameters)) or parameters is None

        if parameters is not None:
            if not isinstance(parameters, (dict, dolfin.Parameters)):
                raise TypeError("parameters should be a dict or a dolfin.Parameters")

    def solve(self):
        raise NotImplementedError("This class is abstract.")
