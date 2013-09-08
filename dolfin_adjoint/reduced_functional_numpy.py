import numpy as np
from dolfin import cpp, info, info_red, Constant, Function, TestFunction, TrialFunction, assemble, inner, dx, as_backend_type, info_red
from dolfin_adjoint import constant, utils 
from dolfin_adjoint.adjglobals import adjointer, adj_reset_cache
from reduced_functional import ReducedFunctional

class ReducedFunctionalNumPy(ReducedFunctional):
    ''' This class implements the reduced functional for a given functional/parameter combination. The core idea 
        of the reduced functional is to consider the problem as a pure function of the parameter value which 
        implicitly solves the recorded PDE. '''

    def __init__(self, rf, in_euclidian_space=False):
        ''' Creates a reduced functional object, that evaluates the functional value for a given parameter value.
            This "NumPy version" of the reduced functional is created from an existing ReducedFunctional object:
              rf_np = ReducedFunctionalNumPy(rf = rf)

            If the optional parameter in_euclidian_space norm is True, the ReducedFunctionalNumPy will 
            perform a transformation from the L2-inner product given by the discrete Functionspace to Euclidian space.
            That is, the squared norm of the gradient can then for example be computed with:

               numpy.dot(dj, dj)

            instead of 

              assemble(inner(dj, dj)*dx).

            This is useful for example, if the reduced functional is to be used with a third party library (such as 
            optimisation libraries) that expect the Euclidian norm. 
            '''
        super(ReducedFunctionalNumPy, self).__init__(rf.functional, rf.parameter, scale = rf.scale, 
                                                     eval_cb = rf.eval_cb, derivative_cb = rf.derivative_cb, 
                                                     replay_cb = rf.replay_cb, hessian_cb = rf.hessian_cb, 
                                                     ignore = rf.ignore, cache = rf.cache)
        self.current_func_value = rf.current_func_value
        self.in_euclidian_space = in_euclidian_space

        self.__base_call__ = rf.__call__
        self.__base_derivative__ = rf.derivative
        self.__base_hessian__ = rf.hessian

        # Variables for storing the Cholesky factorisation
        self.L = None
        self.LT = None
        self.factor = None
        self.has_cholmod = False

        if self.in_euclidian_space:
            # Build up the Cholesky factorisation
            assert len(rf.parameter) == 1
            V = rf.parameter[0].data().function_space()
            r = TestFunction(V)
            q = TrialFunction(V)
            M = assemble(inner(r, q)*dx)

            # Try using PETSc4py and Cholmod for the Cholesky factorisation
            try:
                from scipy.sparse import csr_matrix
                from scikits.sparse.cholmod import cholesky 

                M_petsc = as_backend_type(M).mat()
                indptr, indices, data = M_petsc.getValuesCSR()
                s = len(indptr) - 1

                M_csr = csr_matrix( (data, indices, indptr), shape=(s, s) )
                factor = cholesky(M_csr)

                sqD = factor.L_D()[1]
                sqD.data[:] = np.sqrt(sqD.data)
                self.sqD = sqD

                sqDinv = factor.L_D()[1]
                sqDinv.data[:] = 1./np.sqrt(sqDinv.data)
                self.sqDinv = sqDinv

                self.factor = factor

                self.has_cholmod = True

            # Fallback to a dense Cholesky factorisation 
            except:
                info_red("Warning: Dolfin is not compiled with PETSc4Py support or scikits.sparse.cholmod is not installed.")
                info_red("dolfin-adjoint.optimize will use a dense Cholesky factorisation instead")
                info_red("which can be extremely slow.")

                from numpy.linalg import cholesky
                self.L = cholesky(M.array())
                self.LT = self.L.T


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
        self.set_local(m, m_array)

        return self.__base_call__(m)

    def set_local(self, m, m_array):
        if self.in_euclidian_space:
            if self.has_cholmod:
                set_local(m, m_array, True, True, None, self.factor, self.sqDinv)
            else:
                set_local(m, m_array, True, False, self.LT)
        else:
            set_local(m, m_array, False)

    def get_global(self, m):
        if self.in_euclidian_space:
            if self.has_cholmod:
                return get_global(m, True, True, None, self.factor, self.sqD)
            else:
                return get_global(m, True, False, self.LT)
        else:
            return get_global(m, False)

    def derivative(self, m_array=None, taylor_test=False, seed=0.001, forget=True, project=False):
        ''' An implementation of the reduced functional derivative evaluation 
            that accepts the parameter as an array of scalars. If no parameter values are given,
            the result is derivative at the last forward run.
            If taylor_test = True, the derivative is automatically verified 
            by the Taylor remainder convergence test. The perturbation direction 
            is random and the perturbation size can be controlled with the seed argument.
            '''

        # In the case that the parameter values have changed since the last forward run, 
        # we first need to rerun the forward model with the new parameters to have the 
        # correct forward solutions
        m = [p.data() for p in self.parameter]
        if m_array is not None and (m_array != self.get_global(m)).any():
            info_red("Rerunning forward model before computing derivative")
            self(m_array) 

        dJdm = self.__base_derivative__(forget=forget, project=project) 
        if project:
            dJdm_global = self.get_global(dJdm)
        else:
            dJdm_global = get_global(dJdm, False)
            if self.in_euclidian_space:

                if self.has_cholmod:
                    dJdm_global = self.sqDinv*self.factor.solve_L(self.factor.apply_P(dJdm_global))
                else:
                    dJdm_global = np.linalg.solve(self.L, dJdm_global)

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
            that accepts the parameter as an array of scalars. If m_array is None,
            the Hessian action at the latest forward run is returned. ''' 

        m = [p.data() for p in self.parameter]
        if m_array is not None:
            # In case the parameter values have changed since the last forward run, 
            # we first need to rerun the forward model with the new parameters to have the 
            # correct forward solutions
            if (m_array != self.get_global(m)).any():
                self(m_array) 

                # Clear the adjoint solution as we need to recompute them 
                for i in range(adjointer.equation_count):
                    adjointer.forget_adjoint_values(i)

            self.set_local(m, m_array)
        self.H.update(m)

        m_dot = [copy_data(p.data()) for p in self.parameter] 
        self.set_local(m_dot, m_dot_array)

        hess = self.__base_hessian__(m_dot) 
        hess_array = get_global(hess, False)

        if self.in_euclidian_space:

            if self.has_cholmod:
                hess_array = self.sqDinv*self.factor.solve_L(self.factor.apply_P(hess_array))
            else:
                hess_array = np.linalg.solve(self.L, hess_array)

        return hess_array
    
    def obj_to_array(self, obj):
        return self.get_global(obj)

    def get_parameters(self):
        m = [p.data() for p in self.parameter]
        return self.obj_to_array(m)

    def set_parameters(self, array):
        m = [p.data() for p in self.parameter]
        return self.set_local(m, array)

def copy_data(m):
    ''' Returns a deep copy of the given Function/Constant. '''
    if hasattr(m, "vector"): 
        return Function(m.function_space())
    elif hasattr(m, "value_size"): 
        return Constant(m(()))
    else:
        raise TypeError, 'Unknown parameter type %s.' % str(type(m)) 

def get_global(m_list, in_euclidian_space=False, has_cholmod=False, LT=None, factor=None, sqD=None):
    ''' Takes a list of distributed objects and returns one np array containing their (serialised) values '''
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
                m_v.gather(m_a, np.arange(m_v.size(), dtype='I'))
                m_a = m_a.array().tolist()
            except TypeError:
                m_a = m_v.gather(np.arange(m_v.size(), dtype='intc'))

            # Map the result to Euclidian space
            if in_euclidian_space:
                #info("Mapping to Euclidian space")
                if has_cholmod:
                    m_a = sqD*factor.L_D()[0].transpose()*factor.apply_P(m_a)
                else:
                    m_a = np.dot(LT, m_a)

            m_global += m_a.tolist()

        # Parameters of type Constant 
        elif hasattr(m, "value_size"): 
            a = np.zeros(m.value_size())
            p = np.zeros(m.value_size())
            m.eval(a, p)
            m_global += a.tolist()

        else:
            raise TypeError, 'Unknown parameter type %s.' % str(type(m)) 

    return np.array(m_global, dtype='d')

def set_local(m_list, m_global_array, in_euclidian_space=False, has_cholmod=False, LT=None, factor=None, sqDinv=None):
    ''' Sets the local values of one or a list of distributed object(s) to the values contained in the global array m_global_array '''

    if not isinstance(m_list, (list, tuple)):
        m_list = [m_list]

    offset = 0
    for m in m_list:
        # Function parameters of type dolfin.Function 
        if hasattr(m, "vector"): 

            if in_euclidian_space:
                #info("Mapping from Euclidian space")
                if has_cholmod:
                    m_global_array = factor.apply_Pt(factor.solve_Lt(sqDinv*m_global_array))
                else:
                    m_global_array = np.linalg.solve(LT, m_global_array)

            range_begin, range_end = m.vector().local_range()
            m_a_local = m_global_array[offset + range_begin:offset + range_end]
            m.vector().set_local(m_a_local)
            m.vector().apply('insert')
            offset += m.vector().size() 
        # Parameters of type dolfin.Constant 
        elif hasattr(m, "value_size"): 
            m.assign(constant.Constant(np.reshape(m_global_array[offset:offset+m.value_size()], m.shape())))
            offset += m.value_size()    
        elif isinstance(m, np.ndarray): 
            m[:] = m_global_array[offset:offset+len(m)]
            offset += len(m)
        else:
            raise TypeError, 'Unknown parameter type %s' % m.__class__

