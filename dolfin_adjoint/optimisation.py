from dolfin import *
from dolfin_adjoint import * 
import numpy
import sys

def get_global(m):
    ''' Takes a distributed object and returns a numpy array that contains all global values '''
    if type(m) == float:
        return numpy.array(m)
    if type(m) == constant.Constant:
        a = numpy.zeros(m.value_size())
        p = numpy.zeros(m.value_size())
        m.eval(a, p)
        return a
    elif type(m) in (function.Function, functions.function.Function):
        m_v = m.vector()
        m_a = cpp.DoubleArray(m.vector().size())
        try:
            m.vector().gather(m_a, numpy.arange(m_v.size(), dtype='I'))
            return numpy.array(m_a.array())
        except TypeError:
            m_a = m.vector().gather(numpy.arange(m_v.size(), dtype='I'))
            return m_a 
    else:
        raise TypeError, 'Unknown parameter type %s.' % str(type(m)) 

def set_local(m, m_global_array):
    ''' Sets the local values of the distrbuted object m to the values contained in the global array m_global_array '''
    if type(m) == constant.Constant:
        if m.rank() == 0:
            m.assign(m_global_array[0])
        else:
            m.assign(Constant(tuple(m_global_array)))
    elif type(m) in (function.Function, functions.function.Function):
        range_begin, range_end = m.vector().local_range()
        m_a_local = m_global_array[range_begin:range_end]
        m.vector().set_local(m_a_local)
        m.vector().apply('insert')
    else:
        raise TypeError, 'Unknown parameter type'

def serialise_bounds(bounds, m):
    ''' Converts bounds to an array of tuples and serialises it in a parallel environment. '''
    
    bounds_arr = []
    for i in range(2):
        if type(bounds[i]) == int or type(bounds[i]) == float:
            bounds_arr.append(bounds[i]*numpy.ones(m.vector().size()))
        else:
            bounds_arr.append(get_global(bounds[i]))
            
    return numpy.array(bounds_arr).T

def minimise_scipy_slsqp(J, dJ, m, bounds = None, **kwargs):
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
    set_local(m, mopt)

def minimise_scipy_fmin_l_bfgs_b(J, dJ, m, bounds = None, **kwargs):
    from scipy.optimize import fmin_l_bfgs_b
    
    m_global = get_global(m)

    # Shut up all processors but the first one.
    if MPI.process_number() != 0:
        kwargs['iprint'] = -1

    if bounds:
        bounds = serialise_bounds(bounds, m)

    mopt, f, d = fmin_l_bfgs_b(J, m_global, fprime = dJ, bounds = bounds, **kwargs)
    set_local(m, mopt)

optimisation_algorithms_dict = {'scipy.l_bfgs_b': ('The L-BFGS-B implementation in scipy.', minimise_scipy_fmin_l_bfgs_b),
                                'scipy.slsqp': ('The SLSQP implementation in scipy.', minimise_scipy_slsqp) }

def print_optimisation_algorithms():
    ''' Prints the available optimisation algorithms '''

    print 'Available optimisation algorithms:'
    for function_name, (description, func) in optimisation_algorithms_dict.iteritems():
        print function_name, ': ', description

def minimise(reduced_functional, functional, parameter, m, algorithm, **kwargs):
    ''' Solves the minimisation problem with PDE constraint:

           min_m functional(u, m) 
             s.t. 
           e(u, m) = 0
           lb <= m <= ub
           g(m) <= u
           
        where m is the control variable, u is the solution of the PDE system e(u, m) = 0, functional is the functional of interest and lb, ub and g(m) constraints the control variables. 
        The optimisation problem is solved using a gradient based optimisation algorithm and the functional gradients are computed by solving the associated adjoint system.

        The functional arguments are as follows:
        * 'reduced_functional' must be a python function the implements the reduced functional (i.e. functional(u(m), m)). That is, it takes m as a parameter, solves the model and returns the functional value. 
        * 'functional' must be a dolfin_adjoint.functional object describing the functional of interest
        * 'parameter' must be a dolfin_adjoint.parameter that is to be minimised
        * 'm' must contain the control values. The optimisation algorithm uses these values as a initial guess and updates them after each optimisation iteration. The optimal control values can be accessed by reading 'm' after calling minimise.
        * 'bounds' is an optional keyword parameter to support control constraints: bounds = (lb, ub). lb and ub can either be floats to enforce a global bound or a dolfin.Function to define a varying bound.
        * 'algorithm' specifies the optimistation algorithm to be used to solve the problem. The available algorithms can be listed with the print_optimisation_algorithms function.
        
        Additional arguments specific for the optimisation algorithms can be added to the minimise functions (e.g. iprint = 2). These arguments will be passed to the underlying optimisation algorithm. For detailed information about which arguments are supported for each optimisation algorithm, please refer to the documentaton of the optimisation algorithm.
        '''
    def dJ_array(m_array):

        # In the case that the parameter values have changed since the last forward run, 
        # we need to rerun the forward model with the new parameters
        if (m_array != get_global(m)).any():
            reduced_functional_array(m_array) 

        dJdm = utils.compute_gradient(functional, parameter)
        dJdm_global = get_global(dJdm)

        if dolfin.parameters["optimisation"]["test_gradient"]:
            minconv = utils.test_gradient_array(reduced_functional_array, dJdm_global, m_array, 
                                                seed = dolfin.parameters["optimisation"]["test_gradient_seed"])
            if minconv < 1.9:
                raise RuntimeWarning, "A gradient test failed during execution."
            else:
                info("Gradient test succesfull.")
            reduced_functional_array(m_array) 

        return dJdm_global 

    def reduced_functional_array(m_array):

        # Reset any prior annotation of the adjointer as we are about to rerun the forward model.
        solving.adj_reset()
        # If functional is a FinalFunctinal, we need to set the activated flag to False
        if hasattr(functional, 'activated'):
            functional.activated = False

        set_local(m, m_array)
        return reduced_functional(m)

    if algorithm not in optimisation_algorithms_dict.keys():
        raise ValueError, 'Unknown optimisation algorithm ' + algorithm + '. Use the print_optimisation_algorithms to get a list of the available algorithms.'

    optimisation_algorithms_dict[algorithm][1](reduced_functional_array, dJ_array, m, **kwargs)
