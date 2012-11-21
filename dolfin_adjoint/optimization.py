import dolfin
from dolfin import MPI 
from dolfin_adjoint import constant 
from reduced_functional import ReducedFunctional, get_global, set_local
import numpy
import sys

def serialise_bounds(bounds, m):
    ''' Converts bounds to an array of (min, max) tuples and serialises it in a parallel environment. '''

    # Convert the bounds into the canoncial array form [ [lower_bound1, lower_bound2, ... ], [upper_bound1, upper_bound2, ...] ]
    if len(numpy.array(bounds).shape) == 1:
        bounds = numpy.array([[b] for b in bounds])

    if len(bounds) != 2:
        raise ValueError, "The 'bounds' parameter must be of the form [lower_bound, upper_bound] for one parameter or [ [lower_bound1, lower_bound2, ...], [upper_bound1, upper_bound2, ...] ] for multiple parameters."

    bounds_arr = [[], []]
    for i in range(2):
        for j in range(len(bounds[i])):
            if type(bounds[i][j]) in [int,  float, numpy.int32, numpy.int64, numpy.float32, numpy.float64]:
                bounds_arr[i] += (bounds[i][j]*numpy.ones(m[j].vector().size())).tolist()
            else:
                bounds_arr[i] += get_global(bounds[i][j]).tolist()

    # Transpose and return the array to get the form [ [lower_bound1, upper_bound1], [lower_bound2, upper_bound2], ... ] 
    return numpy.array(bounds_arr).T

def minimize_scipy_generic(J, dJ, m, method, bounds = None, **kwargs):
    ''' Interface to the generic minimize method in scipy '''
    try:
        from scipy.optimize import minimize as scipy_minimize
    except ImportError:
        raise ImportError, "You need to install a scipy version > 0.11 in order to use this optimisation method."

    m_global = get_global(m)

    # Shut up all processors but the first one.
    if MPI.process_number() != 0:
        if not options in kwargs:
            kwargs["options"] = {}
        kwargs["options"]["disp"] = False

    if bounds:
        bounds = serialise_bounds(bounds, m)
        res = scipy_minimize(J, m_global, method = method, jac = dJ, bounds = bounds, **kwargs)
    else:
        res = scipy_minimize(J, m_global, method = method, jac = dJ, **kwargs)

    set_local(m, numpy.array(res["x"]))
    return m

def minimize_scipy_slsqp(J, dJ, m, bounds = None, **kwargs):
    ''' Interface to the SQP algorithm in scipy '''
    # If possible use scipy's generic interface 
    try:
        return minimize_scipy_generic(J, dJ, m, bounds = bounds, method = "SLSQP", **kwargs)
    except ImportError:
        pass
    from scipy.optimize import fmin_slsqp

    m_global = get_global(m)

    # Shut up all processors but the first one.
    if MPI.process_number() != 0:
        kwargs['iprint'] = 0

    if bounds:
        bounds = serialise_bounds(bounds, m)
        mopt = fmin_slsqp(J, m_global, fprime = dJ, bounds = bounds, **kwargs)
    else:
        mopt = fmin_slsqp(J, m_global, fprime = dJ, **kwargs)
    if type(mopt) == list:
        mopt = mopt[0]
    set_local(m, numpy.array(mopt))
    return m

def minimize_scipy_fmin_l_bfgs_b(J, dJ, m, bounds = None, **kwargs):
    ''' Interface to the L-BFGS-B algorithm in scipy '''
    # If possible use scipy's generic interface 
    try:
        return minimize_scipy_generic(J, dJ, m, bounds = bounds, method = "L-BFGS-B", **kwargs)
    except ImportError:
        pass
    from scipy.optimize import fmin_l_bfgs_b
    
    m_global = get_global(m)

    # Shut up all processors but the first one.
    if MPI.process_number() != 0:
        kwargs['iprint'] = -1

    if bounds:
        bounds = serialise_bounds(bounds, m)

    mopt, f, d = fmin_l_bfgs_b(J, m_global, fprime = dJ, bounds = bounds, **kwargs)
    set_local(m, mopt)
    return m

def minimize_scipy_tnc(J, dJ, m, bounds = None, **kwargs):
    # If possible use scipy's generic interface 
    try:
        return minimize_scipy_generic(J, dJ, m, bounds = bounds, method = "TNC", **kwargs)
    except ImportError:
        pass
    from scipy.optimize import fmin_tnc
    
    m_global = get_global(m)

    # Shut up all processors but the first one.
    if MPI.process_number() != 0:
        kwargs['iprint'] = -1

    if bounds:
        bounds = serialise_bounds(bounds, m)

    mopt, nfeval, rc = fmin_tnc(J, m_global, fprime = dJ, bounds = bounds, **kwargs)
    set_local(m, mopt)
    return m

def minimize_scipy_cg(J, dJ, m, **kwargs):
    # If possible use scipy's generic interface 
    try:
        return minimize_scipy_generic(J, dJ, m, bounds = bounds, method = "CG", **kwargs)
    except ImportError:
        pass
    from scipy.optimize import fmin_cg
    
    m_global = get_global(m)

    # Shut up all processors but the first one.
    if MPI.process_number() != 0:
        kwargs['iprint'] = -1
    kwargs['full_output'] = True

    result = fmin_cg(J, m_global, fprime = dJ, **kwargs)
    set_local(m, result[0])
    return m

def minimize_scipy_bfgs(J, dJ, m, **kwargs):
    # If possible use scipy's generic interface 
    try:
        return minimize_scipy_generic(J, dJ, m, bounds = bounds, method = "BFGS", **kwargs)
    except ImportError:
        pass
    from scipy.optimize import fmin_bfgs
    
    m_global = get_global(m)

    # Shut up all processors but the first one.
    if MPI.process_number() != 0:
        kwargs['iprint'] = -1

    mopt, fopt, gopt, Bopt, func_calls, grad_calls, warnflag, allvecs = fmin_bfgs(J, m_global, fprime = dJ, **kwargs)
    set_local(m, mopt)
    return m

optimization_algorithms_dict = {'L-BFGS-B': ('The L-BFGS-B implementation in scipy.', minimize_scipy_fmin_l_bfgs_b),
                                'SLSQP': ('The SLSQP implementation in scipy.', minimize_scipy_slsqp),
                                'TNC': ('The truncated Newton algorithm implemented in scipy.', minimize_scipy_tnc), 
                                'CG': ('The nonlinear conjugate gradient algorithm implemented in scipy.', minimize_scipy_cg), 
                                'BFGS': ('The BFGS implementation in scipy.', minimize_scipy_bfgs), 
                                'Nelder-Mead': ('Gradient-free Simplex algorithm.', minimize_scipy_generic),
                                'Powell': ('Gradient-free Powells method', minimize_scipy_generic),
                                'Newton-CG': ('Newton-CG method', minimize_scipy_generic),
                                'Anneal': ('Gradient-free simulated annealing', minimize_scipy_generic),
                                'COBYLA': ('Gradient-free constrained optimization by linear approxition method', minimize_scipy_generic)
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
        * 'scale' is a factor to scale to problem. Use a negative number to solve a maximisation problem.
        * 'bounds' is an optional keyword parameter to support control constraints: bounds = (lb, ub). lb and ub must be of the same type than the parameters m. 
        
        Additional arguments specific for the optimization algorithms can be added to the minimize functions (e.g. iprint = 2). These arguments will be passed to the underlying optimization algorithm. For detailed information about which arguments are supported for each optimization algorithm, please refer to the documentaton of the optimization algorithm.
        '''

    reduced_func.scale = scale

    try:
        algorithm = optimization_algorithms_dict[method][1]
    except KeyError:
        raise KeyError, 'Unknown optimization method ' + method + '. Use print_optimization_methods() to get a list of the available methods.'

    # For scipy's generic inteface we need to pass the optimisation method as a parameter. 
    if algorithm == "minimize_scipy_generic":
        kwargs["method"] = method 

    dj = lambda m: reduced_func.derivative_array(m, taylor_test = dolfin.parameters["optimization"]["test_gradient"], seed = dolfin.parameters["optimization"]["test_gradient_seed"])
    opt = algorithm(reduced_func.eval_array, dj, [p.data() for p in reduced_func.parameter], **kwargs)
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
        * 'scale' is a factor to scale to problem. Use a negative number to solve a maximisation problem.
        * 'bounds' is an optional keyword parameter to support control constraints: bounds = (lb, ub). lb and ub must be of the same type than the parameters m. 
        
        Additional arguments specific for the optimization methods can be added to the minimize functions (e.g. iprint = 2). These arguments will be passed to the underlying optimization method. For detailed information about which arguments are supported for each optimization method, please refer to the documentaton of the optimization algorithm.
        '''
    return minimize(reduced_func, method, scale = -scale, **kwargs)
