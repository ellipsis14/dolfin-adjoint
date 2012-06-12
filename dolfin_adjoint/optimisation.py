from dolfin import *
from dolfin_adjoint import * 
from mpi4py import MPI
import numpy

comm = MPI.COMM_WORLD
rank = comm.Get_rank()


class darray():
    ''' A distributed array class '''

    def __init__(self, comm, local_size):
        ''' comm must be the MPI communicator and local_array an numpy.array containing the local values of the distributed array '''
        self.comm = comm
        self.local_array = numpy.zeros(local_size) 

        # First, use a Allgather to collect the required information about the local array sizes on each processor
        p = comm.Get_size()
        self.sendcount = numpy.zeros(p, dtype='i')
        lsendcount = numpy.array([local_size], dtype='i')
        comm.Allgather([lsendcount,  MPI.INT],
                       [self.sendcount, MPI.INT])
        # count is the global size of the array
        self.count = sum(self.sendcount) 
        # displs[i] is the offset index where the local_array data starts in the global array for processor i
        self.displs = [sum(self.sendcount[:i]) for i in range(len(self.sendcount))]

        # Allocate the space for the global array
        self.global_array = numpy.zeros(self.count, dtype='d')

    def update_global(self):
        ''' Update and return the global array by distributing the local arrays between all processors '''
        self.comm.Allgatherv([self.local_array, MPI.DOUBLE],
                        [self.global_array, (self.sendcount, self.displs),  MPI.DOUBLE])
        return self.global_array


    def update_local(self):
        ''' Update and return the local array by copying the relevant values from the global array '''

        rank = self.comm.Get_rank()
        local_start = self.displs[rank]
        if len(self.displs) == rank+1:
            self.local_array[:] = self.global_array[local_start:]
        else:
            local_end = self.displs[rank+1]
            self.local_array[:] = self.global_array[local_start:local_end]
        return self.local_array

def to_array(m):
    ''' Converts the value of m to an array. m can be a float, array, dolfin.Function or dolfin.Constant '''
    if type(m) == float:
        return numpy.array([m])
    elif type(m) == numpy.ndarray:
        return m
    elif hasattr(m, 'vector'):
        # Use the vector attribute if available
        return m.vector().array()
    else:
        # Otherwise assume that the object is a Constant and cast/evaluate it depending on its shape 
        if m.rank() == 0:
            return numpy.array([float(m)])
        else:
            a = numpy.zeros(m.shape())
            p = numpy.zeros(m.shape())
            m.eval(a, p)
            return a

def set_from_array(m, m_array):
    ''' Sets value of m to the values given in the array. m must be of type dolfin.Function or dolfin.Constant '''
    if hasattr(m, 'vector'):
        print 'new round'
        print 'len(m.vector().array())', len(m.vector().array())
        print 'len(m_array)', len(m_array)
        try:
            m.vector().set_local(m_array)
        except Exception:
            print 'ieeeeeeeeeeeeeeeeeeeeeeeeeeeek'
            print 'ieeeeeeeeeeeeeeeeeeeeeeeeeeeek'
            print 'ieeeeeeeeeeeeeeeeeeeeeeeeeeeek'
            print 'ieeeeeeeeeeeeeeeeeeeeeeeeeeeek'
            print 'ieeeeeeeeeeeeeeeeeeeeeeeeeeeek'
            print 'ieeeeeeeeeeeeeeeeeeeeeeeeeeeek'
            print 'ieeeeeeeeeeeeeeeeeeeeeeeeeeeek'
            print 'ieeeeeeeeeeeeeeeeeeeeeeeeeeeek'
            print 'ieeeeeeeeeeeeeeeeeeeeeeeeeeeek'
            import sys
            sys.exit(1)
    else:
        if m.rank() == 0:
            m.assign(m_array[0])
        else:
            m.assign(Constant(tuple(m_array)))

def serialise_bounds(bounds, m):
    ''' Converts bounds to an array of tuples and serialises it in a parallel environment. '''
    
    bounds_arr = []
    for i in range(2):
        if type(bounds[i]) == int or type(bounds[i]) == float:
            bounds_arr.append(bounds[i]*numpy.ones(len(m)))
        else:
            bounds_arr.append(bounds[i].vector().array())
            
    def serialise_array(comm, local_array):
        da = darray(comm, len(local_array))
        da.local_array[:] = local_array
        da.update_global()
        return da.global_array
                
    sbounds_arr = [serialise_array(comm, b) for b in bounds_arr]
    return numpy.array(sbounds_arr).T

def minimise_scipy_slsqp(J, dJ, m0, bounds = None, **kwargs):
    from scipy.optimize import fmin_slsqp

    # Create a distributed array containing the control variables
    dm0 = darray(comm, len(m0))
    dm0.local_array[:] = m0
    dm0.update_global()

    if bounds:
        bounds = serialise_bounds(bounds, m0)
        mopt = fmin_slsqp(J, dm0.global_array, fprime = dJ, bounds = bounds, **kwargs)
    else:
        mopt = fmin_slsqp(J, dm0.global_array, fprime = dJ, **kwargs)
    return mopt

def minimise_scipy_fmin_l_bfgs_b(J, dJ, m0, bounds = None, **kwargs):
    from scipy.optimize import fmin_l_bfgs_b

    if bounds:
        bounds = serialise_bounds(bounds, m0)

    # Create a distributed array containing the control variables
    dm0 = darray(comm, len(m0))
    dm0.local_array[:] = m0
    dm0.update_global()

    mopt, f, d = fmin_l_bfgs_b(J, dm0.global_array, fprime = dJ, bounds = bounds, **kwargs)
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
    def dJ_array(m_array):

        current_m = to_array(m)
        m_darray = darray(comm, len(current_m))
        m_darray.global_array[:] = m_array
        m_darray.update_local()

        # In the case that the parameter values have changed since the last forward run, 
        # we need to rerun the forward model with the new parameters
        if (m_darray.local_array != current_m).any():
            reduced_functional_array(m_array) 

        dJdm = utils.compute_gradient(functional, parameter)
        dJdm_array = to_array(dJdm)
        dJdm_darray = darray(comm, len(dJdm_array))
        dJdm_darray.local_array[:] = dJdm_array
        dJdm_darray.update_global()

        if dolfin.parameters["optimisation"]["test_gradient"]:
            minconv = utils.test_gradient_array(reduced_functional_array, dJdm_darray.global_array, m_darray.global_array, 
                                                seed = dolfin.parameters["optimisation"]["test_gradient_seed"])
            if minconv < 1.9:
                raise RuntimeWarning, "A gradient test failed during execution."
            else:
                info("Gradient test succesfull.")
            reduced_functional_array(m_darray.global_array) 


        return dJdm_darray.global_array 

    def reduced_functional_array(m_array):

        current_m = to_array(m)
        dm = darray(comm, len(current_m))
        print 'len(m_array)', len(m_array)
        print 'len(dm.global_array)', len(dm.global_array)
        dm.global_array[:] = m_array
        dm.update_local()
        print 'len(dm.local_array) after update', len(dm.local_array)
        print 'len(current_m)', len(current_m)

        # Reset any prior annotation of the adjointer as we are about to rerun the forward model.
        solving.adj_reset()
        # If functional is a FinalFunctinal, we need to set the actived flag to False
        if hasattr(functional, 'activated'):
            functional.activated = False

        set_from_array(m, dm.local_array)
        return reduced_functional(m)

    if algorithm not in optimisation_algorithms_dict.keys():
        raise ValueError, 'Unknown optimisation algorithm ' + algorithm + '. Use the print_optimisation_algorithms to get a list of the available algorithms.'

    m_array = optimisation_algorithms_dict[algorithm][1](reduced_functional_array, dJ_array, to_array(m), **kwargs)
    set_from_array(m, m_array)
