from dolfin_adjoint import * 

def minimise_scipy_fmin_l_bfgs_b(J, dJ, m0, bounds = None, **kwargs):
    from scipy.optimize import fmin_l_bfgs_b
    import numpy

    if bounds:
        bounds_arr = []
        for i in range(2):
            if type(bounds[i]) == int or type(bounds[i]) == float:
                bounds_arr.append(bounds[i]*numpy.ones(m0.size()))
            else:
                bounds_arr.append(bounds[i].vector().array())
                
        bounds = numpy.array(bounds_arr).T

    try:
      fmin_l_bfgs_b(J, m0, fprime = dJ, disp = 1, bounds = bounds, **kwargs)
    except TypeError:
      fmin_l_bfgs_b(J, m0, fprime = dJ, iprint = 1, bounds = bounds, **kwargs)

optimisation_algorithms_dict = {'scipy.l_bfgs_b': ('The L-BFGS-B implementation in scipy.', minimise_scipy_fmin_l_bfgs_b)}

def print_optimisation_algorithms():
    ''' Prints all available optimisation algorithms '''

    print 'Available optimisation algorithms:'
    for function_name, (description, func) in optimisation_algorithms_dict.iteritems():
        print function_name, ': ', description

def minimise(Jfunc, J, m, m_init, solver = 'scipy.l_bfgs_b', **kwargs):
    ''' Solves the minimisation problem:
           min_m J 
        with the specified optimisation algorithm,
        where J is a Funtional and m is the Parameter to be minimised. 
        Jfunc must be a python function with m as a parameter that runs and annotates the model and returns the functional value for the paramter m. '''

    def dJ_vec(m_vec):

        # In the case that the parameter values have changed since the last forward run, we need to rerun the forward model with the new parameters
        if (m_vec != m_init.vector().array()).any():
            Jfunc_vec(m_vec) 
        dJdm = utils.compute_gradient(J, m)
        return dJdm.vector().array()

    def Jfunc_vec(m_vec):

        # Reset any prior annotation of the adjointer as we are about to rerun the forward model.
        solving.adj_reset()
        # If J is a FinalFunctinal, we need to reactived it
        if hasattr(J, 'activated'):
            J.activated = False

        m_init.vector().set_local(m_vec)
        return Jfunc(m_init)

    if solver not in optimisation_algorithms_dict.keys():
        raise ValueError, 'Unknown optimisation algorithm ' + solver + '. Use the print_optimisation_algorithms to get a list of the available algorithms.'

    optimisation_algorithms_dict[solver][1](Jfunc_vec, dJ_vec, m_init.vector(), **kwargs)

