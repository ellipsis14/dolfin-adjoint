from dolfin import *
from dolfin_adjoint import * 
from mpi4py import MPI
import numpy

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

def serialise_array(comm, local_array):
    ''' Uses MPI to serialise an distributed array. The argument local_array must 
        be an array containg the local contribution of the array. The return value 
        will be the global array. '''

    # First, use a Allgather to collect the required information about the array size on each processor
    p = comm.Get_size()
    sendcount = numpy.zeros(p, dtype='i')
    lsendcount = numpy.array([len(local_array)], dtype='i')
    comm.Allgather([lsendcount,  MPI.INT],
                   [sendcount, MPI.INT])
    count = sum(sendcount) 
    displs = [sum(sendcount[:i]) for i in range(len(sendcount))]

    # Second use a Allgatherv to distribute the array to all processors
    array = numpy.zeros(count, dtype='d')
    comm.Allgatherv([local_array, MPI.DOUBLE],
                    [array, (sendcount, displs),  MPI.DOUBLE])
    return array

def serialise_bounds(bounds, m):
    
    bounds_arr = []
    for i in range(2):
        if type(bounds[i]) == int or type(bounds[i]) == float:
            bounds_arr.append(bounds[i]*numpy.ones(len(m)))
        else:
            bounds_arr.append(bounds[i].vector().array())
            
    sbounds_arr = [serialise_array(comm, b) for b in bounds_arr]
    return numpy.array(sbounds_arr).T

def minimise_scipy_slsqp(J, dJ, m0, bounds = None, **kwargs):
    from scipy.optimize import fmin_slsqp

    if bounds:
        bounds = serialise_bounds(bounds, m0)
        mopt = fmin_slsqp(J, m0, fprime = dJ, bounds = bounds, **kwargs)
    else:
        mopt = fmin_slsqp(J, m0, fprime = dJ, **kwargs)
    return mopt

def minimise_scipy_fmin_l_bfgs_b(J, dJ, m0, bounds = None, **kwargs):
    from scipy.optimize import fmin_l_bfgs_b

    if bounds:
        bounds = serialise_bounds(bounds, m0)

    mopt, f, d = fmin_l_bfgs_b(J, m0, fprime = dJ, bounds = bounds, **kwargs)
    return mopt

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

    def to_array(m):
        ''' Returns the values of the control variable as an array '''
        if type(m) == float:
            return [m]
        elif type(m) == numpy.ndarray:
            return m
        elif hasattr(m, 'vector'):
            # Use the vector attribute if available
            return m.vector().array()
        else:
            # Otherwise assume that the object is a Constant and cast/evaluate it depending on its shape 
            if m.rank() == 0:
                return [float(m)]
            else:
                a = numpy.zeros(m.shape())
                p = numpy.zeros(m.shape())
                m.eval(a, p)
                return a

    def set_from_array(m, m_array):
        ''' Sets the control variable m to the values given in the array '''
        if hasattr(m, 'vector'):
            m.vector().set_local(m_array)
        else:
            if m.rank() == 0:
                m.assign(m_array[0])
            else:
                m.assign(Constant(tuple(m_array)))

    def dJ_array(m_array):

        # In the case that the parameter values have changed since the last forward run, 
        # we need to rerun the forward model with the new parameters
        if (m_array != to_array(m)).any():
            reduced_functional_array(m_array) 

        dJdm = utils.compute_gradient(functional, parameter)

        if dolfin.parameters["optimisation"]["test_gradient"]:
            minconv = utils.test_gradient_array(reduced_functional_array, to_array(dJdm), m_array, 
                                                seed = dolfin.parameters["optimisation"]["test_gradient_seed"])
            if minconv < 1.9:
                raise RuntimeWarning, "A gradient test failed during execution."
            else:
                info("Gradient test succesfull.")
            reduced_functional_array(m_array) 

        return to_array(dJdm)

    def reduced_functional_array(m_array):

        # Reset any prior annotation of the adjointer as we are about to rerun the forward model.
        solving.adj_reset()
        # If functional is a FinalFunctinal, we need to set the actived flag to False
        if hasattr(functional, 'activated'):
            functional.activated = False

        set_from_array(m, m_array)
        return reduced_functional(m)

    if algorithm not in optimisation_algorithms_dict.keys():
        raise ValueError, 'Unknown optimisation algorithm ' + algorithm + '. Use the print_optimisation_algorithms to get a list of the available algorithms.'

    m_array = optimisation_algorithms_dict[algorithm][1](reduced_functional_array, dJ_array, to_array(m), **kwargs)
    set_from_array(m, m_array)
