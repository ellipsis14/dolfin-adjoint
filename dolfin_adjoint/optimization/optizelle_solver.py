from optimization_solver import OptimizationSolver
from optimization_problem import MaximizationProblem
import constraints
import numpy
import math
from ..enlisting import enlist, delist

from backend import *

class DolfinVectorSpace(object):
    def __init__(self, parameters):
        self.parameters = parameters

    @staticmethod
    def __deep_copy_obj(x):
        if isinstance(x, GenericFunction):
            return Function(x)
        elif isinstance(x, Constant):
            return Constant(float(x))
        else:
            raise NotImplementedError

    @staticmethod
    def __assign_obj(x, y):
        if isinstance(x, GenericFunction):
            y.assign(x)
        elif isinstance(x, Constant):
            y.assign(float(x))
        else:
            raise NotImplementedError

    @staticmethod
    def __scale_obj(alpha, x):
        if isinstance(x, GenericFunction):
            x.vector()[:] *= alpha
        elif isinstance(x, Constant):
            x.assign(alpha * float(x))
        else:
            raise NotImplementedError

    @staticmethod
    def __zero_obj(x):
        if isinstance(x, GenericFunction):
            x.vector().zero()
        elif isinstance(x, Constant):
            x.assign(0.0)
        else:
            raise NotImplementedError

    @staticmethod
    def __axpy_obj(alpha, x, y):
        if isinstance(x, GenericFunction):
            y.vector().axpy(alpha, x.vector())
        elif isinstance(x, Constant):
            y.assign(alpha * float(x) + float(y))
        else:
            raise NotImplementedError

    @staticmethod
    def __inner_obj(x, y):
        if isinstance(x, GenericFunction):
            return assemble(inner(x, y)*dx)
        elif isinstance(x, Constant):
            return float(x)*float(y)
        else:
            raise NotImplementedError

    @staticmethod
    def init(x):
        return [DolfinVectorSpace.__deep_copy_obj(xx) for xx in x]

    @staticmethod
    def copy(x, y):
        [DolfinVectorSpace.__assign_obj(xx, yy) for (xx, yy) in zip(x, y)]

    @staticmethod
    def scal(alpha, x):
        [DolfinVectorSpace.__scale_obj(alpha, xx) for xx in x]

    @staticmethod
    def zero(x):
        [DolfinVectorSpace.__zero_obj(xx) for xx in x]

    @staticmethod
    def axpy(alpha, x, y):
        [DolfinVectorSpace.__axpy_obj(alpha, xx, yy) for (xx, yy) in zip(x, y)]

    @staticmethod
    def innr(x, y):
        return sum(DolfinVectorSpace.__inner_obj(xx, yy) for (xx, yy) in zip(x, y))

    @staticmethod
    def rand(x):
        raise NotImplementedError

    @staticmethod
    def prod(x, y, z):
        raise NotImplementedError

    @staticmethod
    def id(x):
        raise NotImplementedError

    @staticmethod
    def linv(x, y, z):
        raise NotImplementedError

    @staticmethod
    def barr(x):
        raise NotImplementedError

    @staticmethod
    def srch(x,y):
        raise NotImplementedError

    @staticmethod
    def symm(x):
        raise NotImplementedError

    @staticmethod
    def normsqdiff(x, y):
        """
        Compute ||x -y||^2; the squared norm of the difference between two functions.

        Not part of the Optizelle vector space API, but useful for us."""

        xx = DolfinVectorSpace.init(x)
        DolfinVectorSpace.axpy(-1, y, xx)
        normsq = DolfinVectorSpace.innr(xx, xx)
        return normsq

try:
    import Optizelle
    import copy

    class RmVectorSpace(object):
        @staticmethod
        def init(x):
            """Memory allocation and size setting"""
            return copy.deepcopy(x) 

        @staticmethod
        def copy(x,y):
            """y <- x (Shallow.  No memory allocation.)"""
            numpy.copyto(y,x) 

        @staticmethod
        def scal(alpha,x):
            """x <- alpha * x"""
            x.__imul__(alpha)

        @staticmethod
        def zero(x):
            """x <- 0"""
            x.fill(0.)

        @staticmethod
        def axpy(alpha,x,y):
            """y <- alpha * x + y"""
            y.__iadd__(alpha*x)

        @staticmethod
        def innr(x,y):
            """<- <x,y>"""
            return numpy.inner(x,y) 

        @staticmethod
        def rand(x):
            """x <- random"""
            numpy.copyto(x,map(lambda x:random.normalvariate(0.,1.),x))

        @staticmethod
        def prod(x,y,z):
            """Jordan product, z <- x o y"""
            numpy.copyto(z,x*y)

        @staticmethod
        def id(x):
            """Identity element, x <- e such that x o e = x"""
            x.fill(1.)

        @staticmethod
        def linv(x,y,z):
            """Jordan product inverse, z <- inv(L(x)) y where L(x) y = x o y"""
            numpy.copyto(z,numpy.divide(y,x))

        @staticmethod
        def barr(x):
            """Barrier function, <- barr(x) where x o grad barr(x) = e"""
            try:
                return sum(math.log(xx) for xx in x)
            except ValueError:
                return -numpy.inf
            except:
                import traceback
                traceback.print_exc()
                raise
            
        @staticmethod
        def srch(x,y):
            """Line search, <- argmax {alpha \in Real >= 0 : alpha x + y >= 0} where y > 0"""
            alpha = float("inf")
            for i in xrange(0,len(x)):
                if x[i] < 0:
                    alpha0 = -y[i]/x[i]
                    if alpha0 < alpha:
                        alpha=alpha0
            return alpha

        @staticmethod
        def symm(x):
            """Symmetrization, x <- symm(x) such that L(symm(x)) is a symmetric operator"""
            pass


    # May not have optizelle installed.
    class OptizelleObjective(Optizelle.ScalarValuedFunction):
        def __init__(self, rf, scale=1):
            self.rf = rf
            self.last_x = None
            self.last_J = None
            self.scale = scale

        def eval(self, x):
            try:
                if self.last_x is not None:
                    normsq = DolfinVectorSpace.normsqdiff(x, self.last_x)
                    if normsq == 0.0:
                        return self.last_J

                self.last_x = DolfinVectorSpace.init(x)
                self.rf(x)
                self.last_J = self.scale*self.rf(x)
                return self.last_J
            except:
                import traceback
                traceback.print_exc()
                raise

        def grad(self, x, grad):
            try:
                self.eval(x)
                out = self.rf.derivative(forget=False, project=True)
                DolfinVectorSpace.scal(self.scale, out)
                DolfinVectorSpace.copy(out, grad)
            except:
                import traceback
                traceback.print_exc()
                raise

        def hessvec(self, x, dx, H_dx):
            try:
                self.eval(x)
                H = self.rf.hessian(dx, project=True)
                DolfinVectorSpace.scal(self.scale, H)
                DolfinVectorSpace.copy(H, H_dx)
            except:
                import traceback
                traceback.print_exc()
                raise


    class OptizelleConstraints(Optizelle.VectorValuedFunction):
        ''' This class generates a (equality and inequality) constraint object from 
            a dolfin_adjoint.Constraint which is compatible with the Optizelle 
            interface.
        '''

        def __init__(self, problem, constraints):
            self.constraints = constraints
            self.list_type = problem.reduced_functional.parameter

        def eval(self, x, y):
            ''' Evaluates the constraints and stores the result in y. '''


            if len(self.constraints) == 0:
                return


            try:
                x_list = delist(x, self.list_type)

                y[:] = self.constraints.function(x_list)
            except:
                import traceback
                traceback.print_exc()
                raise

        def p(self, x, dx, y):
            ''' Evaluates the Jacobian action and stores the result in y. '''

            if len(self.constraints) == 0:
                return

            try: 
                x_list = delist(x, self.list_type)
                dx_list = delist(dx, self.list_type)

                self.constraints.jacobian_action(x_list, dx_list, y)
            except:
                import traceback
                traceback.print_exc()
                raise

        def ps(self, x, dy, z):
            ''' 
                Evaluates the Jacobian adjoint action and stores the result in y. 
                    z=g'(x)*dy
            '''

            if len(self.constraints) == 0:
                return

            try: 
                x_list = delist(x, self.list_type)
                z_list = delist(z, self.list_type)

                self.constraints.jacobian_adjoint_action(x_list, dy, z_list)
            except:
                import traceback
                traceback.print_exc()
                raise

        def pps(self, x, dx, dy, z):
            ''' 
                Evaluates the Hessian adjoint action in directions dx and dy 
                and stores the result in z. 
                    z=(g''(x)dx)*dy
            '''

            if len(self.constraints) == 0:
                return

            try: 
                x_list = delist(x, self.list_type)
                dx_list = delist(dx, self.list_type)
                z_list = delist(z, self.list_type)

                self.constraints.hessian_action(x_list, dx_list, dy, z_list)
            except:
                import traceback
                traceback.print_exc()
                raise

except ImportError:
    pass


class OptizelleSolver(OptimizationSolver):
    """
    Use optizelle to solve the given optimisation problem.

    The optizelle State instance is accessible as solver.state.
    See dir(solver.state) for the parameters that can be set,
    and the optizelle manual for details.
    """
    def __init__(self, problem, parameters=None):
        """
        Create a new OptizelleSolver.

        To set optizelle-specific options, do e.g.

          solver = OptizelleSolver(problem, parameters={'maximum_iterations': 100,
                                            'optizelle_parameters': {'krylov_iter_max': 100}})
        """
        try:
            import Optizelle
            import Optizelle.Unconstrained.State
            import Optizelle.Unconstrained.Functions
            import Optizelle.Unconstrained.Algorithms
            import Optizelle.Constrained.State
            import Optizelle.Constrained.Functions
            import Optizelle.Constrained.Algorithms
            import Optizelle.EqualityConstrained.State
            import Optizelle.EqualityConstrained.Functions
            import Optizelle.EqualityConstrained.Algorithms
            import Optizelle.InequalityConstrained.State
            import Optizelle.InequalityConstrained.Functions
            import Optizelle.InequalityConstrained.Algorithms
        except ImportError:
            print("Could not import Optizelle.")
            raise

        OptimizationSolver.__init__(self, problem, parameters)

        # TODO: Add bound support to Optizelle
        assert self.problem.bounds is None

        self.__build_optizelle_state()

    def __build_optizelle_state(self):

        # Optizelle does not support maximization problem directly, 
        # hence we negate the functional instead
        if isinstance(self.problem, MaximizationProblem):
            scale = -1
        else:
            scale = +1

        # Create the appropriate Optizelle state, taking into account which
        # type of constraints we have (unconstrained, (in)-equality constraints).
        if self.problem.constraints is None:
            num_equality_constraints = 0
            num_inequality_constraints = 0
        else:
            num_equality_constraints = len(self.problem.constraints.equality_constraints())
            num_inequality_constraints = len(self.problem.constraints.inequality_constraints())

        x = [p.data() for p in self.problem.reduced_functional.parameter]

        # Unconstrained case
        if self.problem.constraints is None:
            self.state = Optizelle.Unconstrained.State.t(DolfinVectorSpace, Optizelle.Messaging(), x)
            self.fns = Optizelle.Unconstrained.Functions.t()
            self.fns.f = OptizelleObjective(self.problem.reduced_functional, scale=scale)

        # Equality constraints only
        elif num_equality_constraints > 0 and num_inequality_constraints == 0:

            # Allocate memory for the equality multiplier 
            y = numpy.zeros(num_equality_constraints)

            self.state = Optizelle.EqualityConstrained.State.t(DolfinVectorSpace, RmVectorSpace, Optizelle.Messaging(), x, y)
            self.fns = Optizelle.Constrained.Functions.t()

            equality_constraints = self.problem.constraints.equality_constraints()

            self.fns.f = OptizelleObjective(self.problem.reduced_functional, scale=scale)
            self.fns.g = OptizelleConstraints(self.problem, equality_constraints)

        # Inequality constraints only
        elif num_equality_constraints == 0 and num_inequality_constraints > 0:

            # Allocate memory for the inequality multiplier 
            z = numpy.zeros(num_inequality_constraints)

            self.state = Optizelle.InequalityConstrained.State.t(DolfinVectorSpace, RmVectorSpace, Optizelle.Messaging(), x, z)
            self.fns = Optizelle.InequalityConstrained.Functions.t()

            inequality_constraints = self.problem.constraints.inequality_constraints()

            self.fns.f = OptizelleObjective(self.problem.reduced_functional, scale=scale)
            self.fns.h = OptizelleConstraints(self.problem, inequality_constraints)

        # Inequality and equality constraints
        else:

            # Allocate memory for the equality multiplier 
            y = numpy.zeros(num_eq)

            # Allocate memory for the inequality multiplier 
            z = numpy.zeros(num_ineq)

            self.state = Optizelle.Constrained.State.t(DolfinVectorSpace, RmVectorSpace, RmVectorSpace, Optizelle.Messaging(), x, y, z)
            self.fns = Optizelle.Constrained.Functions.t()

            equality_constraints = self.problem.constraints.equality_constraints()
            inequality_constraints = self.problem.constraints.inequality_constraints()

            self.fns.f = OptizelleObjective(self.problem.reduced_functional, scale=scale)
            self.fns.g = OptizelleConstraints(self.problem, equality_constraints)
            self.fns.h = OptizelleConstraints(self.problem, inequality_constraints)

        # Set solver parameters
        self.__set_optizelle_parameters()

    def __set_optizelle_parameters(self):

        if self.parameters is None: return

        # First, set the default parameters.
        if 'maximum_iterations' in self.parameters:
            self.state.iter_max = self.parameters['maximum_iterations']

        # FIXME: is there a common 'tolerance' between all solvers supported by
        # dolfin-adjoint?

        # Then set any optizelle-specific ones.
        if 'optizelle_parameters' in self.parameters:
            optizelle_parameters = self.parameters['optizelle_parameters']
            for key in optizelle_parameters:
                try:
                    setattr(self.state, key, optizelle_parameters[key])
                except AttributeError:
                    print("Error: unknown optizelle option %s." % key)
                    raise

    def solve(self):
        """Solve the optimization problem and return the optimized parameters."""

        if self.problem.constraints is None:
            num_equality_constraints = 0
            num_inequality_constraints = 0
        else:
            num_equality_constraints = len(self.problem.constraints.equality_constraints())
            num_inequality_constraints = len(self.problem.constraints.inequality_constraints())

        # No constraints
        if self.problem.constraints is None:
            Optizelle.Unconstrained.Algorithms.getMin(DolfinVectorSpace, Optizelle.Messaging(), self.fns, self.state)

        # Equality constraints only
        elif num_equality_constraints > 0 and num_inequality_constraints == 0:
            Optizelle.EqualityConstrained.Algorithms.getMin(DolfinVectorSpace, RmVectorSpace, Optizelle.Messaging(), self.fns, self.state)

        # Inequality constraints only
        elif num_equality_constraints == 0 and num_inequality_constraints > 0:
            Optizelle.InequalityConstrained.Algorithms.getMin(DolfinVectorSpace, RmVectorSpace, Optizelle.Messaging(), self.fns, self.state)

        # Inequality and equality constraints
        else:
            Optizelle.Constrained.Algorithms.getMin(DolfinVectorSpace, RmVectorSpace, RmVectorSpace, Optizelle.Messaging(), self.fns, self.state)

        # Print out the reason for convergence
        print("The algorithm converged due to: %s" % (Optizelle.StoppingCondition.to_string(self.state.opt_stop)))

        # Return the optimal control
        list_type = self.problem.reduced_functional.parameter
        return delist(self.state.x, list_type)
