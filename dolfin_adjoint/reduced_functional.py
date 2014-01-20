import libadjoint
from dolfin import Function, Constant, info_red, info_green, File
from dolfin_adjoint import adjlinalg, adjrhs, constant, drivers
from dolfin_adjoint.adjglobals import adjointer, mem_checkpoints, disk_checkpoints
import cPickle as pickle
import hashlib

global_eqn_list = {}
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
                                "derivative_cache": {},
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
            replace_parameter_value(self.parameter[i], value[i])

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

        if self.cache is not None:
            hash = value_hash([x.data() for x in self.parameter])
            fnspaces = [p.data().function_space() if isinstance(p.data(), Function) else None for p in self.parameter]

            if hash in self._cache["derivative_cache"]:
                info_green("Got a derivative cache hit.")
                return cache_load(self._cache["derivative_cache"][hash], fnspaces)

        dfunc_value = drivers.compute_gradient(self.functional, self.parameter, forget=forget, ignore=self.ignore, project=project)
        dfunc_value = enlist(dfunc_value)

        adjointer.reset_revolve()
        scaled_dfunc_value = []
        for df in list(dfunc_value):
            if hasattr(df, "function_space"):
                scaled_dfunc_value.append(Function(df.function_space(), self.scale * df.vector()))
            else:
                scaled_dfunc_value.append(self.scale * df)

        if self.derivative_cb:
            if self.current_func_value is not None:
              self.derivative_cb(self.scale * self.current_func_value, unlist(scaled_dfunc_value), unlist([p.data() for p in self.parameter]))
            else:
              info_red("Gradient evaluated without functional evaluation, not calling derivative callback function")

        if self.cache is not None:
            info_red("Got a derivative cache miss")
            self._cache["derivative_cache"][hash] = cache_store(scaled_dfunc_value, self.cache)

        return scaled_dfunc_value

    def hessian(self, m_dot):
        ''' Evaluates the Hessian action in direction m_dot. '''
        assert(len(self.parameter) == 1)

        if self.cache is not None:
            hash = value_hash([x.data() for x in self.parameter] + [m_dot])
            fnspaces = [p.data().function_space() if isinstance(p.data(), Function) else None for p in self.parameter]

            if hash in self._cache["hessian_cache"]:
                info_green("Got a Hessian cache hit.")
                return cache_load(self._cache["hessian_cache"][hash], fnspaces)

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
            val = [Function(Hm.function_space(), Hm.vector() * self.scale)]
        else:
            val = [self.scale * Hm]

        if self.cache is not None:
            info_red("Got a Hessian cache miss")
            self._cache["hessian_cache"][hash] = cache_store(val, self.cache)

        return val

def replace_parameter_value(parameter, new_value):
    ''' Replaces the parameter value with new_value. '''
    if hasattr(parameter, 'var'):
        replace_tape_value(parameter.var, new_value)

def replace_tape_value(variable, new_value):
    ''' Replaces the tape value of the given DolfinAdjointVariable with new_value. '''

    # Case 1: The parameter value and new_value are Functions
    if hasattr(new_value, 'vector'):
        # Functions are copied in da and occur as rhs in the annotation. 
        # Hence we need to update the right hand side callbacks for
        # the equation that targets the associated variable.

        # Create a RHS object with the new control values
        init_rhs = adjlinalg.Vector(new_value).duplicate()
        init_rhs.axpy(1.0, adjlinalg.Vector(new_value))
        rhs = adjrhs.RHS(init_rhs)
        # Register the new rhs in the annotation
        class DummyEquation(object):
            pass

        eqn = DummyEquation()
        eqn_nb = variable.equation_nb(adjointer)
        eqn.equation = adjointer.adjointer.equations[eqn_nb]
        rhs.register(eqn)

        # Keep a python reference of the equation in memory
        global_eqn_list[eqn_nb] = eqn

    # Case 2: The parameter value and new_value are Constants
    elif hasattr(new_value, "value_size"): 
        # Constants are not copied in the annotation. That is, changing a constant that occurs
        # in the forward model will also change the forward replay with libadjoint.
        constant = parameter.data()
        constant.assign(new_value(()))

    else:
        raise NotImplementedError, "Can only replace a dolfin.Functions or dolfin.Constants"

def unlist(x):
    ''' If x is a list of length 1, return its element. Otherwise return x. '''
    if len(x) == 1:
        return x[0]
    else:
        return x

def enlist(x):
    ''' Opposite of unlist '''
    if not isinstance(x, (list, tuple)):
        return [x]
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

def cache_load(value, V):
    if isinstance(value, (list, tuple)):
        return [cache_load(value[i], V[i]) for i in range(len(value))]
    elif isinstance(value, float):
        return Constant(value)
    elif isinstance(value, str):
        return Function(V, value)
    return

def cache_store(value, cache):
    if isinstance(value, (list, tuple)):
        return tuple(cache_store(x, cache) for x in value)
    elif isinstance(value, Constant):
        return float(value)
    elif isinstance(value, Function):
        hash = value_hash(value)
        filename = "%s_dir/%s.xml.gz" % (cache, hash)
        File(filename) << value
        return filename
    else:
        raise Exception, "Don't know how to store %s" % value
    return
