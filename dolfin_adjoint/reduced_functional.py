import libadjoint
from dolfin_adjoint import adjlinalg, adjrhs, constant 
from dolfin_adjoint.adjglobals import adjointer

class DummyEquation(object):
    pass

class ReducedFunctional(object):
    ''' This class implements the reduced functional for a given functional/parameter combination. The core idea 
        of the reduced functional is to consider the problem as a pure function of the paramter value which 
        implicitly solves the recorded PDE. '''
    def __init__(self, functional, parameter):
        ''' Creates a reduced functional object, that evaluates the functional value for a given parameter value.
            The arguments are as follows:
            * 'functional' must be a dolfin_adjoint.Functional object. 
            * 'parameter' must be a single or a list of dolfin_adjoint.DolfinAdjointParameter objects.
            '''
        self.functional = functional
        if not isinstance(parameter, (list, tuple)):
            parameter = [parameter]
        self.parameter = parameter
        # This flag indicates if the functional evaluation is based on replaying the forward annotation. 
        self.replays_annotation = True
        self.eqns = []

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

