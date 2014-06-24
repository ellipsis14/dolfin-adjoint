import optimization_solver import OptimizationSolver
import dolfin
from ..reduced_functional_numpy import ReducedFunctionalNumPy

class IPOPTSolver(OptimizationSolver):
    """Use the pyipopt bindings to IPOPT to solve the given optimization problem.

    The pyipopt Problem instance is accessible as solver.pyipopt_problem."""

    def __init__(self, problem, parameters=None):
        try:
            import pyipopt
        except ImportError:
            print("You need to install pyipopt. Make sure to install IPOPT with HSL support!")
            raise

        OptimizationSolver.__init__(self, problem, parameters)

        self.__build_pyipopt_problem()
        self.__set_parameters()

    def __build_pyipopt_problem(self):
        """Build the pyipopt problem from the OptimizationProblem instance."""

        import numpy
        import pyipopt
        from functools import partial

        self.rfn = ReducedFunctionalNumPy(self.problem.reduced_functional)
        (lb, ub) = self.__get_bounds()
        (nconstraints, fun_g, jac_g, clb, cub) = self.__get_constraints()

        # A callback that evaluates the functional and derivative.
        J  = self.rfn.__call
        dJ = partial (self.rfn.derivative, forget=False)

        nlp = pyipopt.create(len(ub),           # length of parameter vector
                             lb,                # lower bounds on parameter vector
                             ub,                # upper bounds on parameter vector
                             nconstraints,      # number of constraints
                             clb,               # lower bounds on constraints,
                             cub,               # upper bounds on constraints,
                             nconstraints*n,    # number of nonzeros in the constraint Jacobian
                             0,                 # number of nonzeros in the Hessian
                             J,                 # to evaluate the functional
                             dJ,                # to evaluate the gradient
                             fun_g,             # to evaluate the constraints
                             jac_g)             # to evaluate the constraint Jacobian
        self.pyipopt_problem = nlp

    def __get_bounds(self):
        """Convert the bounds into the format accepted by pyipopt (two numpy arrays,
        one for the lower bound and one for the upper).

        FIXME: Do we really have to pass (-\infty, +\infty) when there are no bounds?"""

        bounds = self.problem.bounds

        if bounds is not None:
            lb_list = []
            ub_list = [] # a list of numpy arrays, one for each parameter

            for (bound, parameter) in zip(bounds, self.rfn.parameter):
                len_parameter = len(self.rfn.get_global(parameter))
                general_lb, general_ub = bound # could be float, Constant, or Function

                if isinstance(general_lb, float) or isinstance(general_lb, dolfin.Constant):
                    lb = np.array([float(general_lb)]*len_parameter)
                elif isinstance(general_lb, dolfin.Function):
                    assert general_lb.function_space().dim() == parameter.data().function_space().dim()
                    lb = self.rfn.get_global(general_lb)
                else:
                    raise TypeError("Unknown bound type %s" % general_lb.__class__)

                if isinstance(general_ub, float) or isinstance(general_ub, dolfin.Constant):
                    ub = np.array([float(general_ub)]*len_parameter)
                elif isinstance(general_ub, dolfin.Function):
                    assert general_ub.function_space().dim() == parameter.data().function_space().dim()
                    ub = self.rfn.get_global(general_ub)
                else:
                    raise TypeError("Unknown bound type %s" % general_ub.__class__)

            ub = numpy.concatenate(ub_list)
            lb = numpy.concatenate(lb_list)

        else:
            # Unfortunately you really need to specify bounds, I think?!
            max_float = numpy.finfo(numpy.double).max
            ub = numpy.array([max_float]*n)

            min_float = numpy.finfo(numpy.double).min
            lb = numpy.array([min_float]*n)

        return (lb, ub)

    def __get_constraints(self):
        constraints = self.problem.constraints

        if constraints is None:
            # The length of the constraint vector
            nconstraints = 0

            # The bounds for the constraint
            empty = np.array([], dtype=float)
            clb = empty
            cub = empty

            # The constraint function, should do nothing
            def fun_g(x, user_data=None):
                return empty

            # The constraint Jacobian
            def jac_g(x, flag, user_data=None):
                if flag:
                    rows = np.array([], dtype=int)
                    cols = np.array([], dtype=int)
                    return (rows, cols)
                else:
                    return empty

            return (nconstraints, fun_g, jac_g, clb, cub)

        else:
            # The length of the constraint vector
            nconstraints = len(constraints)

            # The constraint function
            def fun_g(x, user_data=None):
                return np.array(constraints.function(x))

            # The constraint Jacobian:
            # flag = True  means 'tell me the sparsity pattern';
            # flag = False means 'give me the damn Jacobian'.
            def jac_g(x, flag, user_data=None):
                if flag:
                    # FIXME: Don't have any sparsity information on constraints;
                    # pass in a dense matrix (it usually is anyway).
                    rows = []
                    for i in range(len(constraints)):
                        rows += [i] * n
                    cols = range(n) * len(constraints)
                    return (np.array(rows), np.array(cols))
                else:
                  return np.array(constraints.jacobian(x))

            # The bounds for the constraint: by the definition of our
            # constraint type, the lower bound is always zero,
            # whereas the upper bound is either zero or infinity,
            # depending on whether it's an equality constraint or inequalityconstraint.

            clb = np.array([0] * len(constraints))

            def constraint_ub(c):
                if isinstance(c, EqualityConstraint):
                    return [0] * len(c)
                elif isinstance(c, InequalityConstraint):
                    return [np.inf] * len(c)
            cub = np.array(sum([constraint_ub(c) for c in constraints], []))

            return (nconstraints, fun_g, jac_g, clb, cub)

    def __set_parameters(self):
        """Set some basic parameters from the parameters dictionary that the user
        passed in, if any."""

        if self.parameters is not None:
            if hasattr(self.parameters, 'tolerance'):
                tol = self.parameters['tolerance']
                self.pyipopt_problem.num_option('tol', tol)
            if hasattr(self.parameters, 'maximum_iterations'):
                maxiter = self.parameters['maximum_iterations']
                self.pyipopt_problem.int_option('maxiter', maxiter)

    def __copy_data(self, m):
        """Returns a deep copy of the given Function/Constant."""
        if hasattr(m, "vector"): 
            return Function(m.function_space())
        elif hasattr(m, "value_size"): 
            return Constant(m(()))
        else:
            raise TypeError, 'Unknown parameter type %s.' % str(type(m)) 

    def solve(self):
        """Solve the optimization problem and return the optimized parameters."""
        guess = self.rfn.get_parameters()
        results = self.pyipopt_problem.solve(guess)
        new_params = [self.__copy_data(p.data()) for p in self.rfn.parameter]
        self.rfn.set_local(new_params, results[0])

        # FIXME: if the parameters were passed as a list of length one, don't
        # un-list it.
        if len(new_params) == 1:
            new_params = new_params[0]

        return new_params
