import libadjoint
import numpy
from dolfin import cpp, info, project, Function, Constant, info_red, info_green
from dolfin_adjoint import adjlinalg, adjrhs, constant, utils, drivers
from dolfin_adjoint.adjglobals import adjointer, mem_checkpoints, disk_checkpoints, adj_reset_cache
import cPickle as pickle
import hashlib

def unlist(x):
    ''' If x is a list of length 1, return its element. Otherwise return x. '''
    if len(x) == 1:
        return x[0]
    else:
        return x

def copy_data(m):
    ''' Returns a deep copy of the given Function/Constant. '''
    if hasattr(m, "vector"): 
        return Function(m.function_space())
    elif hasattr(m, "value_size"): 
        return Constant(m(()))
    else:
        raise TypeError, 'Unknown parameter type %s.' % str(type(m)) 

def value_hash(value):
    if isinstance(value, Constant):
        return str(float(value))
    elif isinstance(value, Function):
        m = hashlib.md5()
        m.update(str(value.vector().norm("l2")) + str(value.vector().norm("l1")) + str(value.vector().norm("linf")))
        return m.hexdigest()
    elif isinstance (value, list):
        return "".join(map(value_hash, value))
    else:
        raise Exception, "Don't know how to take a hash of %s" % value

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

global_eqn_list = {}
def replace_tape_ic_value(parameter, new_value):
    ''' Replaces the initial condition value of the given parameter by registering a new equation of the rhs. '''

    # Case 1: The parameter value and new_vale are Functions
    if hasattr(new_value, 'vector'):
        # ... since these are duplicated and then occur as rhs in the annotation. 
        # Therefore, we need to update the right hand side callbacks for
        # the equation that targets the associated variable.

        # Create a RHS object with the new control values
        init_rhs = adjlinalg.Vector(new_value).duplicate()
        init_rhs.axpy(1.0, adjlinalg.Vector(new_value))
        rhs = adjrhs.RHS(init_rhs)
        # Register the new rhs in the annotation
        class DummyEquation(object):
            pass

        eqn = DummyEquation() 
        variable = parameter.var
        eqn_nb = variable.equation_nb(adjointer)
        eqn.equation = adjointer.adjointer.equations[eqn_nb]
        rhs.register(eqn)
        # Store the equation as a class variable in order to keep a python reference in the memory
        global_eqn_list[variable.equation_nb] = eqn

    # Case 2: The parameter value and new_value are Constants
    elif hasattr(new_value, "value_size"): 
        # Constants are not duplicated in the annotation. That is, changing a constant that occurs
        # in the forward model will also change the forward replay with libadjoint.
        constant = parameter.data()
        constant.assign(new_value(()))

    else:
        raise NotImplementedError, "Can only replace a dolfin.Functions or dolfin.Constants"

class ReducedFunctional(object):
    ''' This class implements the reduced functional for a given functional/parameter combination. The core idea 
        of the reduced functional is to consider the problem as a pure function of the parameter value which 
        implicitly solves the recorded PDE. '''
    def __init__(self, functional, parameter, scale = 1.0, eval_cb = None, derivative_cb = None, replay_cb = None, hessian_cb = None, ignore = [], cache = None):
        ''' Creates a reduced functional object, that evaluates the functional value for a given parameter value.
            The arguments are as follows:
            * 'functional' must be a dolfin_adjoint.Functional object. 
            * 'parameter' must be a single or a list of dolfin_adjoint.DolfinAdjointParameter objects.
            * 'scale' is an additional scaling factor. 
            * 'eval_cb' is an optional callback that is executed after each functional evaluation. 
              The interace must be eval_cb(j, m) where j is the functional value and 
              m is the parameter value at which the functional is evaluated.
            * 'derivative_cb' is an optional callback that is executed after each functional gradient evaluation. 
              The interface must be eval_cb(j, dj, m) where j and dj are the functional and functional gradient values, and 
              m is the parameter value at which the gradient is evaluated.
            * 'hessian_cb' is an optional callback that is executed after each hessian action evaluation. The interface must be
               hessian_cb(j, m, mdot, h) where mdot is the direction in which the hessian action is evaluated and h the value
               of the hessian action.
            '''
        self.functional = functional
        if not isinstance(parameter, (list, tuple)):
            parameter = [parameter]
        self.parameter = parameter
        # This flag indicates if the functional evaluation is based on replaying the forward annotation. 
        self.replays_annotation = True
        self.eqns = []
        self.scale = scale
        self.eval_cb = eval_cb
        self.derivative_cb = derivative_cb
        self.hessian_cb = hessian_cb
        self.replay_cb = replay_cb
        self.current_func_value = None
        self.ignore = ignore
        self.cache = cache

        # TODO: implement a drivers.hessian function that supports a list of parameters
        if len(parameter) == 1:
            self.H = drivers.hessian(functional, parameter[0], warn=False)

        if cache is not None:
            try:
                self._cache = pickle.load(open(cache, "r"))
            except IOError: # didn't exist
                self._cache = {"functional_cache": {},
                                "gradient_cache": {},
                                "hessian_cache": {}}

    def __del__(self):
        if self.cache is not None:
            pickle.dump(self._cache, open(self.cache, "w"))

    def __call__(self, value):
        ''' Evaluates the reduced functional for the given parameter value, by replaying the forward model.
            Note: before using this evaluation, make sure that the forward model has been annotated. '''

        if not isinstance(value, (list, tuple)):
            value = [value]
        if len(value) != len(self.parameter):
            raise ValueError, "The number of parameters must equal the number of parameter values."

        # Update the parameter values
        for i in range(len(value)):
            replace_tape_ic_value(self.parameter[i], value[i])

        if self.cache:
            hash = value_hash(value)
            if hash in self._cache["functional_cache"]:
                # Found a cache
                info_green("Got a functional cache hit")
                return self._cache["functional_cache"][hash]

        # Replay the annotation and evaluate the functional
        func_value = 0.
        for i in range(adjointer.equation_count):
            (fwd_var, output) = adjointer.get_forward_solution(i)
            if isinstance(output.data, Function):
              output.data.rename(str(fwd_var), "a Function from dolfin-adjoint")

            if self.replay_cb is not None:
              self.replay_cb(fwd_var, output.data, unlist(value))

            # Check if we checkpointing is active and if yes
            # record the exact same checkpoint variables as 
            # in the initial forward run 
            if adjointer.get_checkpoint_strategy() != None:
                if str(fwd_var) in mem_checkpoints:
                    storage = libadjoint.MemoryStorage(output, cs = True)
                    storage.set_overwrite(True)
                    adjointer.record_variable(fwd_var, storage)
                if str(fwd_var) in disk_checkpoints:
                    storage = libadjoint.MemoryStorage(output)
                    adjointer.record_variable(fwd_var, storage)
                    storage = libadjoint.DiskStorage(output, cs = True)
                    storage.set_overwrite(True)
                    adjointer.record_variable(fwd_var, storage)
                if not str(fwd_var) in mem_checkpoints and not str(fwd_var) in disk_checkpoints:
                    storage = libadjoint.MemoryStorage(output)
                    storage.set_overwrite(True)
                    adjointer.record_variable(fwd_var, storage)

            # No checkpointing, so we record everything
            else:
                storage = libadjoint.MemoryStorage(output)
                storage.set_overwrite(True)
                adjointer.record_variable(fwd_var, storage)

            if i == adjointer.timestep_end_equation(fwd_var.timestep):
                func_value += adjointer.evaluate_functional(self.functional, fwd_var.timestep)
                if adjointer.get_checkpoint_strategy() != None:
                    adjointer.forget_forward_equation(i)

        self.current_func_value = func_value 
        if self.eval_cb:
            self.eval_cb(self.scale * func_value, unlist(value))

        if self.cache:
            # Add result to cache
            info_red("Got a functional cache miss")
            self._cache["functional_cache"][hash] = self.scale*func_value

        return self.scale*func_value

    def derivative(self, forget=True, project=False):
        ''' Evaluates the derivative of the reduced functional for the lastly evaluated parameter value. ''' 
        dfunc_value = drivers.compute_gradient(self.functional, self.parameter, forget=forget, ignore=self.ignore, project=project)
        adjointer.reset_revolve()
        scaled_dfunc_value = []
        for df in list(dfunc_value):
            if hasattr(df, "function_space"):
                scaled_dfunc_value.append(Function(df.function_space(), self.scale * df.vector()))
            else:
                scaled_dfunc_value.append(self.scale * df)

        if self.derivative_cb:
            self.derivative_cb(self.scale * self.current_func_value, unlist(scaled_dfunc_value), unlist([p.data() for p in self.parameter]))

        return scaled_dfunc_value

    def hessian(self, m_dot):
        ''' Evaluates the Hessian action in direction m_dot. '''
        assert(len(self.parameter) == 1)

        if isinstance(m_dot, list):
          assert len(m_dot) == 1
          Hm = self.H(m_dot[0])
        else:
          Hm = self.H(m_dot)
        if self.hessian_cb:
            self.hessian_cb(self.scale * self.current_func_value,
                            unlist([p.data() for p in self.parameter]),
                            m_dot,
                            Hm.vector() * self.scale)
        
        if hasattr(Hm, 'function_space'):
            return [Function(Hm.function_space(), Hm.vector() * self.scale)]
        else:
            return [self.scale * Hm]

    def eval_array(self, m_array):
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

        return self(m)

    def derivative_array(self, m_array, taylor_test = False, seed = 0.001, forget = True):
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
            self.eval_array(m_array) 

        dJdm = self.derivative(forget=forget) 
        dJdm_global = get_global(dJdm)

        # Perform the gradient test
        if taylor_test:
            minconv = utils.test_gradient_array(self.eval_array, self.scale * dJdm_global, m_array, 
                                                seed = seed) 
            if minconv < 1.9:
                raise RuntimeWarning, "A gradient test failed during execution."
            else:
                info("Gradient test succesfull.")
            self.eval_array(m_array) 

        return dJdm_global 

    def hessian_array(self, m_array, m_dot_array):
        ''' An implementation of the reduced functional hessian action evaluation 
            that accepts the parameter as an array of scalars. ''' 

        # In the case that the parameter values have changed since the last forward run, 
        # we first need to rerun the forward model with the new parameters to have the 
        # correct forward solutions
        m = [p.data() for p in self.parameter]
        if (m_array != get_global(m)).any():
            self.eval_array(m_array) 

            # Clear the adjoint solution as we need to recompute them 
            for i in range(adjglobals.adjointer.equation_count):
                adjglobals.adjointer.forget_adjoint_values(i)

        set_local(m, m_array)
        self.H.update(m)

        m_dot = [copy_data(p.data()) for p in self.parameter] 
        set_local(m_dot, m_dot_array)

        return get_global(self.hessian(m_dot)) 
