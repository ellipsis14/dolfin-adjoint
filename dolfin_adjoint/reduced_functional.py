import libadjoint
import numpy
from dolfin import cpp, info
from dolfin_adjoint import adjlinalg, adjrhs, constant, utils 
from dolfin_adjoint.adjglobals import adjointer

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
            m.assign(constant.Constant(numpy.reshape(m_global_array[offset:offset+m.value_size()], m.shape())))
            offset += m.value_size()    
        # Function parameters of type dolfin.Function 
        elif hasattr(m, "vector"): 
            range_begin, range_end = m.vector().local_range()
            m_a_local = m_global_array[offset + range_begin:offset + range_end]
            m.vector().set_local(m_a_local)
            m.vector().apply('insert')
            offset += m.vector().size() 
        else:
            raise TypeError, 'Unknown parameter type'

class DummyEquation(object):
    pass

class ReducedFunctional(object):
    ''' This class implements the reduced functional for a given functional/parameter combination. The core idea 
        of the reduced functional is to consider the problem as a pure function of the parameter value which 
        implicitly solves the recorded PDE. '''
    def __init__(self, functional, parameter, scale = 1.0):
        ''' Creates a reduced functional object, that evaluates the functional value for a given parameter value.
            The arguments are as follows:
            * 'functional' must be a dolfin_adjoint.Functional object. 
            * 'parameter' must be a single or a list of dolfin_adjoint.DolfinAdjointParameter objects.
            * 'scale' is an additional scaling factor. 
            '''
        self.functional = functional
        if not isinstance(parameter, (list, tuple)):
            parameter = [parameter]
        self.parameter = parameter
        # This flag indicates if the functional evaluation is based on replaying the forward annotation. 
        self.replays_annotation = True
        self.eqns = []
        self.scale = scale

    def eval_callback(self, value):
        ''' This function is called before the reduced functional is evaluated.
            It is intended to be overwritten by the user, for example to plot the control values 
            that are passed into the callback as "value". ''' 
        pass

    def __call__(self, value):
        ''' Evaluates the reduced functional for the given parameter value, by replaying the forward model.
            Note: before using this evaluation, make sure that the forward model has been annotated. '''

        self.eval_callback(value)
        if not isinstance(value, (list, tuple)):
            value = [value]
        if len(value) != len(self.parameter):
            raise ValueError, "The number of parameters must equal the number of parameter values."

        # Update the parameter values
        for i in range(len(value)):
            if type(value[i]) == constant.Constant:
                # Constants are not duplicated in the annotation. That is, changing a constant that occurs
                # in the forward model will also change the forward replay with libadjoint.
                # However, this is not the case for functions...
                pass
            elif hasattr(value[i], 'vector'):
                # ... since these are duplicated and then occur as rhs in the annotation. 
                # Therefore, we need to update the right hand side callbacks for
                # the equation that targets the associated variable.

                # Create a RHS object with the new control values
                init_rhs = adjlinalg.Vector(value[i]).duplicate()
                init_rhs.axpy(1.0, adjlinalg.Vector(value[i]))
                rhs = adjrhs.RHS(init_rhs)
                # Register the new rhs in the annotation
                eqn = DummyEquation() 
                eqn_nb = self.parameter[i].var.equation_nb(adjointer)
                eqn.equation = adjointer.adjointer.equations[eqn_nb]
                # Store the equation as a class variable in order to keep a python reference in the memory
                self.eqns.append(eqn)
                rhs.register(self.eqns[-1])
            else:
                raise NotImplementedError, "The ReducedFunctional class currently only works for parameters that are Functions"


        # Replay the annotation and evaluate the functional
        func_value = 0.
        for i in range(adjointer.equation_count):
            (fwd_var, output) = adjointer.get_forward_solution(i)

            storage = libadjoint.MemoryStorage(output)
            storage.set_overwrite(True)
            adjointer.record_variable(fwd_var, storage)
            if i == adjointer.timestep_end_equation(fwd_var.timestep):
                func_value += adjointer.evaluate_functional(self.functional, fwd_var.timestep)

            #adjglobals.adjointer.forget_forward_equation(i)
        return func_value

    def derivative(self):
        ''' Evaluates the derivative of the reduced functional for the lastly evaluated parameter value. ''' 
        return utils.compute_gradient(self.functional, self.parameter)

    def eval_array(self, m_array):
        ''' An implementation of the reduced functional evaluation
            that accepts the parameter as an array of scalars '''

        # In case the annotation is not reused, we need to reset any prior annotation of the adjointer before reruning the forward model.
        if not self.replays_annotation:
            solving.adj_reset()

        # Set the parameter values and execute the reduced functional
        m = [p.data() for p in self.parameter]
        set_local(m, m_array)
        return self.scale * self(m)

    def derivative_array(self, m_array, taylor_test = False, seed = 0.001):
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

        dJdm = self.derivative() 
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

        return self.scale * dJdm_global 
