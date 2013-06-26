import dolfin
from dolfin import MPI 
from dolfin_adjoint import constant 
from reduced_functional import ReducedFunctional, get_global, set_local
import numpy
import sys

def serialise_bounds(bounds, m):
    ''' Converts bounds to an array of (min, max) tuples and serialises it in a parallel environment. ''' 

    if len(numpy.array(bounds).shape) == 1:
        bounds = numpy.array([[b] for b in bounds])

    if len(bounds) != 2:
        raise ValueError, "The 'bounds' parameter must be of the form [lower_bound, upper_bound] for one parameter or [ [lower_bound1, lower_bound2, ...], [upper_bound1, upper_bound2, ...] ] for multiple parameters."

    bounds_arr = [[], []]
    for i in range(2):
        for j in range(len(bounds[i])):
            bound = bounds[i][j]
            if type(bound) in [int,  float, numpy.int32, numpy.int64, numpy.float32, numpy.float64]:
                bound_len = len(get_global(m[j]))
                bounds_arr[i] += (bound*numpy.ones(bound_len)).tolist()
            else:
                bounds_arr[i] += get_global(bound).tolist()

    # Transpose and return the array to get the form [ [lower_bound1, upper_bound1], [lower_bound2, upper_bound2], ... ] 
    return numpy.array(bounds_arr).T

def minimize_scipy_generic(J, dJ, m, method, bounds = None, H = None, **kwargs):
    ''' Interface to the generic minimize method in scipy '''

    try:
        from scipy.optimize import minimize as scipy_minimize
        from scipy.optimize import basinhopping
    except ImportError:
        print "**************** Deprecated warning *****************"
        print "You have a old version of scipy (<0.11). This version is not supported by dolfin-adjoint."
        raise ImportError

    m_global = get_global(m)

    if not "options" in kwargs:
        kwargs["options"] = {}
    if MPI.process_number() != 0:
        # Shut up all processors except the first one.
        kwargs["options"]["disp"] = False
    else:
        # Print out progress information by default
        if not "disp" in kwargs["options"]:
            kwargs["options"]["disp"] = True

    # Make the default SLSLQP options more verbose
    if method == "SLSQP" and "iprint" not in kwargs["options"]:
        kwargs["options"]["iprint"] = 2

    # For gradient-based methods add the derivative function to the argument list 
    if method not in ["COBYLA", "Nelder-Mead", "Anneal", "Powell"]:
        kwargs["jac"] = dJ

    # For Hessian-based methods add the Hessian action function to the argument list
    if method in ["Newton-CG"]:
        kwargs["hessp"] = H

    if method=="basinhopping":
        del kwargs["options"]
        del kwargs["jac"]
        kwargs["minimizer_kwargs"]["jac"]=dJ

        res = basinhopping(J, m_global, **kwargs)

    elif bounds != None:
        bounds = serialise_bounds(bounds, m)
        res = scipy_minimize(J, m_global, method = method, bounds = bounds, **kwargs)
    else:
        res = scipy_minimize(J, m_global, method = method, **kwargs)

    set_local(m, numpy.array(res["x"]))
    return m

def minimize_custom(J, dJ, m, bounds = None, H = None, **kwargs):
    ''' Interface to the user-provided minimisation method '''

    try:
        algo = kwargs["algorithm"]
        del kwargs["algorithm"]
    except KeyError:
        raise KeyError, 'When using a "Custom" optimisation method, you must pass the optimisation function as the "algorithm" parameter. Make sure that this function accepts the same arguments as scipy.optimize.minimize.' 

    m_global = get_global(m)

    if bounds != None:
        bounds = serialise_bounds(bounds, m)

    res = algo(J, m_global, dJ, H, bounds, **kwargs)

    try:
        set_local(m, numpy.array(res))
    except Exception as e:
        raise e, "Failed to updated the optimised parameter value. Are you sure your custom optimisation algorithm returns an array containing the optimised values?" 
    return m

optimization_algorithms_dict = {'L-BFGS-B': ('The L-BFGS-B implementation in scipy.', minimize_scipy_generic),
                                'SLSQP': ('The SLSQP implementation in scipy.', minimize_scipy_generic),
                                'TNC': ('The truncated Newton algorithm implemented in scipy.', minimize_scipy_generic), 
                                'CG': ('The nonlinear conjugate gradient algorithm implemented in scipy.', minimize_scipy_generic), 
                                'BFGS': ('The BFGS implementation in scipy.', minimize_scipy_generic), 
                                'Nelder-Mead': ('Gradient-free Simplex algorithm.', minimize_scipy_generic),
                                'Powell': ('Gradient-free Powells method', minimize_scipy_generic),
                                'Newton-CG': ('Newton-CG method', minimize_scipy_generic),
                                'Anneal': ('Gradient-free simulated annealing', minimize_scipy_generic),
                                'basinhopping': ('Global basin hopping method', minimize_scipy_generic),
                                'COBYLA': ('Gradient-free constrained optimization by linear approxition method', minimize_scipy_generic),
                                'Custom': ('User-provided optimization algorithm', minimize_custom)
                                }

def print_optimization_methods():
    ''' Prints the available optimization methods '''

    print 'Available optimization methods:'
    for function_name, (description, func) in optimization_algorithms_dict.iteritems():
        print function_name, ': ', description

def minimize(reduced_func, method = 'L-BFGS-B', scale = 1.0, **kwargs):
    ''' Solves the minimisation problem with PDE constraint:

           min_m func(u, m) 
             s.t. 
           e(u, m) = 0
           lb <= m <= ub
           g(m) <= u
           
        where m is the control variable, u is the solution of the PDE system e(u, m) = 0, func is the functional of interest and lb, ub and g(m) constraints the control variables. 
        The optimization problem is solved using a gradient based optimization algorithm and the functional gradients are computed by solving the associated adjoint system.

        The function arguments are as follows:
        * 'reduced_func' must be a ReducedFunctional object. 
        * 'method' specifies the optimization method to be used to solve the problem. The available methods can be listed with the print_optimization_methods function.
        * 'scale' is a factor to scale to problem (default: 1.0). 
        * 'bounds' is an optional keyword parameter to support control constraints: bounds = (lb, ub). lb and ub must be of the same type than the parameters m. 
        
        Additional arguments specific for the optimization algorithms can be added to the minimize functions (e.g. iprint = 2). These arguments will be passed to the underlying optimization algorithm. For detailed information about which arguments are supported for each optimization algorithm, please refer to the documentaton of the optimization algorithm.
        '''

    reduced_func.scale = scale

    try:
        algorithm = optimization_algorithms_dict[method][1]
    except KeyError:
        raise KeyError, 'Unknown optimization method ' + method + '. Use print_optimization_methods() to get a list of the available methods.'

    if algorithm == minimize_scipy_generic:
        # For scipy's generic inteface we need to pass the optimisation method as a parameter. 
        kwargs["method"] = method 

    if method in ["Newton-CG", "Custom"]:
        dj = lambda m: reduced_func.derivative_array(m, taylor_test = dolfin.parameters["optimization"]["test_gradient"], 
                                                        seed = dolfin.parameters["optimization"]["test_gradient_seed"],
                                                        forget = None)
    else:
        dj = lambda m: reduced_func.derivative_array(m, taylor_test = dolfin.parameters["optimization"]["test_gradient"], 
                                                        seed = dolfin.parameters["optimization"]["test_gradient_seed"])

    opt = algorithm(reduced_func.eval_array, dj, [p.data() for p in reduced_func.parameter], H = reduced_func.hessian_array, **kwargs)
    if len(opt) == 1:
        return opt[0]
    else:
        return opt

def maximize(reduced_func, method = 'L-BFGS-B', scale = 1.0, **kwargs):
    ''' Solves the maximisation problem with PDE constraint:

           max_m func(u, m) 
             s.t. 
           e(u, m) = 0
           lb <= m <= ub
           g(m) <= u
           
        where m is the control variable, u is the solution of the PDE system e(u, m) = 0, func is the functional of interest and lb, ub and g(m) constraints the control variables. 
        The optimization problem is solved using a gradient based optimization algorithm and the functional gradients are computed by solving the associated adjoint system.

        The function arguments are as follows:
        * 'reduced_func' must be a ReducedFunctional object. 
        * 'method' specifies the optimization method to be used to solve the problem. The available methods can be listed with the print_optimization_methods function.
        * 'scale' is a factor to scale to problem (default: 1.0). 
        * 'bounds' is an optional keyword parameter to support control constraints: bounds = (lb, ub). lb and ub must be of the same type than the parameters m. 
        
        Additional arguments specific for the optimization methods can be added to the minimize functions (e.g. iprint = 2). These arguments will be passed to the underlying optimization method. For detailed information about which arguments are supported for each optimization method, please refer to the documentaton of the optimization algorithm.
        '''
    return minimize(reduced_func, method, scale = -scale, **kwargs)

minimise = minimize
maximise = maximize
