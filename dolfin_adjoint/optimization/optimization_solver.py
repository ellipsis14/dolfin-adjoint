import optimization_problem
import dolfin

class OptimizationSolver(object):
    """An abstract base class that represents an optimization solver."""
    def __init__(self, problem, parameters=None):
        assert isinstance(problem, optimization_problem.OptimizationProblem)
        self.problem = problem

        assert isinstance(parameters, (dict, dolfin.Parameters)) or parameters is None
        self.parameters = parameters

    def solve(self):
        raise NotImplementedError("This class is abstract.")
