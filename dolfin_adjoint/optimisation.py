from dolfin_adjoint import * 
import numpy

def generate_array_bounds(bounds, m):
    
    bounds_arr = []
    for i in range(2):
        if type(bounds[i]) == int or type(bounds[i]) == float:
            bounds_arr.append(bounds[i]*numpy.ones(len(m)))
        else:
            bounds_arr.append(bounds[i].vector().array())
            
    return numpy.array(bounds_arr).T

def minimise_scipy_slsqp(J, dJ, m0, bounds = None, **kwargs):
    from scipy.optimize import fmin_slsqp

    if bounds:
        bounds = generate_array_bounds(bounds, m0)
        mopt = fmin_slsqp(J, m0, fprime = dJ, bounds = bounds, **kwargs)
    else:
        mopt = fmin_slsqp(J, m0, fprime = dJ, **kwargs)
    return mopt

def minimise_scipy_fmin_l_bfgs_b(J, dJ, m0, bounds = None, **kwargs):
    from scipy.optimize import fmin_l_bfgs_b

    if bounds:
        bounds = generate_array_bounds(bounds, m0)

    mopt, f, d = fmin_l_bfgs_b(J, m0, fprime = dJ, bounds = bounds, **kwargs)
    return mopt

optimisation_algorithms_dict = {'scipy.l_bfgs_b': ('The L-BFGS-B implementation in scipy.', minimise_scipy_fmin_l_bfgs_b),
                                'scipy.slsqp': ('The SLSQP implementation in scipy.', minimise_scipy_slsqp) 
                               }

def print_optimisation_algorithms():
    ''' Prints the available optimisation algorithms '''

    print 'Available optimisation algorithms:'
    for function_name, (description, func) in optimisation_algorithms_dict.iteritems():
        print function_name, ': ', description

def minimise(Jfunc, J, m, m_init, algorithm, **kwargs):
    ''' Solves the minimisation problem with the specified optimisation algorithm:
           min_m J 
        where J is a dolfin_adjoint.functional and m is a dolfin_adjoint.parameter to be minimised. 
        Jfunc must be a python function with m as a parameter that runs and annotates the model and returns the functional value. '''

    def to_array(m_init):
        ''' Returns the values of the parameter object as an array '''
        return m_init.vector().array()

    def set_from_array(m_init, m_arr):
        ''' Sets the parameter object to the values given in the array '''
        m_init.vector().set_local(m_arr)

    def dJ_array(m_array):

        # In the case that the parameter values have changed since the last forward run, we need to rerun the forward model with the new parameters
        if (m_array != to_array(m_init)).any():
            Jfunc_array(m_array) 
        dJdm = utils.compute_gradient(J, m)
        return to_array(dJdm)

    def Jfunc_array(m_array):

        # Reset any prior annotation of the adjointer as we are about to rerun the forward model.
        solving.adj_reset()
        # If J is a FinalFunctinal, we need to reactived it
        if hasattr(J, 'activated'):
            J.activated = False

        m_init.vector().set_local(m_array)
        return Jfunc(m_init)

    if algorithm not in optimisation_algorithms_dict.keys():
        raise ValueError, 'Unknown optimisation algorithm ' + algorithm + '. Use the print_optimisation_algorithms to get a list of the available algorithms.'

    mopt_array = optimisation_algorithms_dict[algorithm][1](Jfunc_array, dJ_array, to_array(m_init), **kwargs)
    set_from_array(m_init, mopt_array)


