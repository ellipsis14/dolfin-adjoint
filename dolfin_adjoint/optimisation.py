import dolfin
from dolfin import cpp, MPI, info_red, info
from dolfin_adjoint import constant, utils
from reduced_functional import ReducedFunctional
import numpy
import sys

def get_global(m_list):
    ''' Takes a (optional a list of) distributed object(s) and returns one numpy array containing their global values '''
    if not isinstance(m_list, (list, tuple)):
        m_list = [m_list]

    m_global = []
    for m in m_list:
        # Parameters of type float
        if m == None or type(m) == float:
            m_global.append(m)
        # Parameters of type Constant 
        elif type(m) == constant.Constant:
            a = numpy.zeros(m.value_size())
            p = numpy.zeros(m.value_size())
            m.eval(a, p)
            m_global += a.tolist()
        # Function parameters of type Function 
        elif hasattr(m, "vector"): 
            m_v = m.vector()
            m_a = cpp.DoubleArray(m.vector().size())
            try:
                m.vector().gather(m_a, numpy.arange(m_v.size(), dtype='I'))
                m_global += m_a.array().tolist()
            except TypeError:
                m_a = m.vector().gather(numpy.arange(m_v.size(), dtype='I'))
                m_global += m_a.tolist()
        else:
            raise TypeError, 'Unknown parameter type %s.' % str(type(m)) 

    return numpy.array(m_global, dtype='d')

def set_local(m_list, m_global_array):
    ''' Sets the local values of a (or optionally  a list of) distributed object(s) to the values contained in the global array m_global_array '''

    if not isinstance(m_list, (list, tuple)):
        m_list = [m_list]

    offset = 0
    for m in m_list:
        # Parameters of type dolfin.Constant 
        if type(m) == constant.Constant:
            m.assign(Constant(numpy.reshape(m_global_array[offset:offset+m.value_size()], m.shape())))
            offset += m.value_size()    
        # Function parameters of type dolfin.Function 
        elif hasattr(m, "vector"): 
            range_begin, range_end = m.vector().local_range()
            m_a_local = m_global_array[offset + range_begin:offset + range_end]
            m.vector().set_local(m_a_local)
            m.vector().apply('insert')
            offset += m.value_size() 
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

def minimize_scipy_slsqp(J, dJ, m, bounds = None, **kwargs):
    ''' Interface to the SQP algorithm in scipy '''
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

def minimize_scipy_fmin_l_bfgs_b(J, dJ, m, bounds = None, **kwargs):
    ''' Interface to the L-BFGS-B algorithm in scipy '''
    from scipy.optimize import fmin_l_bfgs_b
    
    m_global = get_global(m)

    # Shut up all processors but the first one.
    if MPI.process_number() != 0:
        kwargs['iprint'] = -1

    if bounds:
        bounds = serialise_bounds(bounds, m)

    mopt, f, d = fmin_l_bfgs_b(J, m_global, fprime = dJ, bounds = bounds, **kwargs)
    set_local(m, mopt)

optimisation_algorithms_dict = {'scipy.l_bfgs_b': ('The L-BFGS-B implementation in scipy.', minimize_scipy_fmin_l_bfgs_b),
                                'scipy.slsqp': ('The SLSQP implementation in scipy.', minimize_scipy_slsqp) }

def print_optimisation_algorithms():
    ''' Prints the available optimisation algorithms '''

    print 'Available optimisation algorithms:'
    for function_name, (description, func) in optimisation_algorithms_dict.iteritems():
        print function_name, ': ', description

def minimize(reduced_func, m, algorithm, **kwargs):
    ''' Solves the minimisation problem with PDE constraint:

           min_m func(u, m) 
             s.t. 
           e(u, m) = 0
           lb <= m <= ub
           g(m) <= u
           
        where m is the control variable, u is the solution of the PDE system e(u, m) = 0, func is the functional of interest and lb, ub and g(m) constraints the control variables. 
        The optimisation problem is solved using a gradient based optimisation algorithm and the functional gradients are computed by solving the associated adjoint system.

        The function arguments are as follows:
        * 'reduced_func' must be a python function the implements the reduced functional (i.e. func(u(m), m)). That is, it takes m as a parameter, solves the model and returns the functional value. A simple way of creating this function is the ReducedFunctional class.
        * 'm' must contain the control values. The optimisation algorithm uses these values as a initial guess and updates them after each optimisation iteration. The optimal control values can be accessed by reading 'm' after calling minimize.
        * 'bounds' is an optional keyword parameter to support control constraints: bounds = (lb, ub). lb and ub can either be floats to enforce a global bound or a dolfin.Function to define a varying bound.
        * 'algorithm' specifies the optimistation algorithm to be used to solve the problem. The available algorithms can be listed with the print_optimisation_algorithms function.
        
        Additional arguments specific for the optimisation algorithms can be added to the minimize functions (e.g. iprint = 2). These arguments will be passed to the underlying optimisation algorithm. For detailed information about which arguments are supported for each optimisation algorithm, please refer to the documentaton of the optimisation algorithm.
        '''

    def reduced_func_deriv_array(m_array):
        ''' An implementation of the reduced functional derivative that accepts the parameter as an array ''' 

        # In the case that the parameter values have changed since the last forward run, 
        # we first need to rerun the forward model with the new parameters to have the 
        # correct forward solutions
        if (m_array != get_global(m)).any():
            reduced_func_array(m_array) 

        dJdm = utils.compute_gradient(reduced_func.functional, reduced_func.parameter)
        dJdm_global = get_global(dJdm)

        # Perform the gradient test
        if dolfin.parameters["optimisation"]["test_gradient"]:
            minconv = utils.test_gradient_array(reduced_func_array, dJdm_global, m_array, 
                                                seed = dolfin.parameters["optimisation"]["test_gradient_seed"])
            if minconv < 1.9:
                raise RuntimeWarning, "A gradient test failed during execution."
            else:
                info("Gradient test succesfull.")
            reduced_func_array(m_array) 

        return dJdm_global 

    def reduced_func_array(m_array):
        ''' An implementation of the reduced functional that accepts the parameter as an array '''
        # In case the annotation is not reused, we need to reset any prior annotation of the adjointer before reruning the forward model.
        if not reduced_func.replays_annotation:
            solving.adj_reset()

        # Set the parameter values and execute the reduced functional
        set_local(m, m_array)
        return reduced_func(m)

    if algorithm not in optimisation_algorithms_dict.keys():
        raise ValueError, 'Unknown optimisation algorithm ' + algorithm + '. Use the print_optimisation_algorithms to get a list of the available algorithms.'

    optimisation_algorithms_dict[algorithm][1](reduced_func_array, reduced_func_deriv_array, m, **kwargs)
