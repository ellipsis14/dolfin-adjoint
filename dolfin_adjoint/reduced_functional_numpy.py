import numpy
from dolfin import cpp, info, Constant, Function
from dolfin_adjoint import constant, utils 
from dolfin_adjoint.adjglobals import adjointer, adj_reset_cache
from reduced_functional import ReducedFunctional

class ReducedFunctionalNumPy(ReducedFunctional):
    ''' This class implements the reduced functional for a given functional/parameter combination. The core idea 
        of the reduced functional is to consider the problem as a pure function of the parameter value which 
        implicitly solves the recorded PDE. '''
    def __init__(self, rf):
        ''' Creates a reduced functional object, that evaluates the functional value for a given parameter value.
            This "numpy version" of a reduced functional is created from an existing ReducedFunctional object:
              rf_numpy = ReducedFunctionalNumPy(rf = rf)
            '''
        super(ReducedFunctionalNumPy, self).__init__(rf.functional, rf.parameter, scale = rf.scale, 
                                                     eval_cb = rf.eval_cb, derivative_cb = rf.derivative_cb, 
                                                     replay_cb = rf.replay_cb, hessian_cb = rf.hessian_cb, 
                                                     ignore = rf.ignore, cache = rf.cache)
        self.current_func_value = rf.current_func_value

    def __call__(self, m_array):
        ''' An implementation of the reduced functional evaluation
            that accepts the parameter as an array of scalars '''

        # In case the annotation is not reused, we need to reset any prior annotation of the adjointer before reruning the forward model.
        if not self.replays_annotation:
            solving.adj_reset()

        # We move in parameter space, so we also need to reset the factorisation cache
        adj_reset_cache()

        # Now its time to update the parameter values using the given array  
        m = [p.data() for p in self.parameter]
        set_local(m, m_array)

        return super(ReducedFunctionalNumPy, self).__call__(m)

    def derivative(self, m_array, taylor_test = False, seed = 0.001, forget = True):
        ''' An implementation of the reduced functional derivative evaluation 
            that accepts the parameter as an array of scalars  
            If taylor_test = True, the derivative is automatically verified 
            by the Taylor remainder convergence test. The perturbation direction 
            is random and the perturbation size can be controlled with the seed argument.
            '''

        # In the case that the parameter values have changed since the last forward run, 
        # we first need to rerun the forward model with the new parameters to have the 
        # correct forward solutions
        m = [p.data() for p in self.parameter]
        if (m_array != get_global(m)).any():
            self(m_array) 

        dJdm = super(ReducedFunctionalNumPy, self).derivative(forget=forget) 
        dJdm_global = get_global(dJdm)

        # Perform the gradient test
        if taylor_test:
            minconv = utils.test_gradient_array(self.__call__, self.scale * dJdm_global, m_array, 
                                                seed = seed) 
            if minconv < 1.9:
                raise RuntimeWarning, "A gradient test failed during execution."
            else:
                info("Gradient test succesfull.")
            self(m_array) 

        return dJdm_global 

    def hessian(self, m_array, m_dot_array):
        ''' An implementation of the reduced functional hessian action evaluation 
            that accepts the parameter as an array of scalars. ''' 

        # In the case that the parameter values have changed since the last forward run, 
        # we first need to rerun the forward model with the new parameters to have the 
        # correct forward solutions
        m = [p.data() for p in self.parameter]
        if (m_array != get_global(m)).any():
            self(m_array) 

            # Clear the adjoint solution as we need to recompute them 
            for i in range(adjointer.equation_count):
                adjointer.forget_adjoint_values(i)

        set_local(m, m_array)
        self.H.update(m)

        m_dot = [copy_data(p.data()) for p in self.parameter] 
        set_local(m_dot, m_dot_array)

        hess = super(ReducedFunctionalNumPy, self).hessian(m_dot) 
        return get_global(hess)

def copy_data(m):
    ''' Returns a deep copy of the given Function/Constant. '''
    if hasattr(m, "vector"): 
        return Function(m.function_space())
    elif hasattr(m, "value_size"): 
        return Constant(m(()))
    else:
        raise TypeError, 'Unknown parameter type %s.' % str(type(m)) 

def get_global(m_list):
    ''' Takes a list of distributed objects and returns one numpy array containing their (serialised) values '''
    if not isinstance(m_list, (list, tuple)):
        m_list = [m_list]

    m_global = []
    for m in m_list:

        # Parameters of type float
        if m == None or type(m) == float:
            m_global.append(m)

        elif hasattr(m, "tolist"): 
            m_global += m.tolist()

        # Function parameters of type Function 
        elif hasattr(m, "vector") or hasattr(m, "gather"): 
            if not hasattr(m, "gather"):
                m_v = m.vector()
            else:
                m_v = m
            m_a = cpp.DoubleArray(m_v.size())
            try:
                m_v.gather(m_a, numpy.arange(m_v.size(), dtype='I'))
                m_global += m_a.array().tolist()
            except TypeError:
                m_a = m_v.gather(numpy.arange(m_v.size(), dtype='intc'))
                m_global += m_a.tolist()

        # Parameters of type Constant 
        elif hasattr(m, "value_size"): 
            a = numpy.zeros(m.value_size())
            p = numpy.zeros(m.value_size())
            m.eval(a, p)
            m_global += a.tolist()

        else:
            raise TypeError, 'Unknown parameter type %s.' % str(type(m)) 

    return numpy.array(m_global, dtype='d')

def set_local(m_list, m_global_array):
    ''' Sets the local values of one or a list of distributed object(s) to the values contained in the global array m_global_array '''

    if not isinstance(m_list, (list, tuple)):
        m_list = [m_list]

    offset = 0
    for m in m_list:
        # Function parameters of type dolfin.Function 
        if hasattr(m, "vector"): 
            range_begin, range_end = m.vector().local_range()
            m_a_local = m_global_array[offset + range_begin:offset + range_end]
            m.vector().set_local(m_a_local)
            m.vector().apply('insert')
            offset += m.vector().size() 
        # Parameters of type dolfin.Constant 
        elif hasattr(m, "value_size"): 
            m.assign(constant.Constant(numpy.reshape(m_global_array[offset:offset+m.value_size()], m.shape())))
            offset += m.value_size()    
        elif isinstance(m, numpy.ndarray): 
            m[:] = m_global_array[offset:offset+len(m)]
            offset += len(m)
        else:
            raise TypeError, 'Unknown parameter type %s' % m.__class__

