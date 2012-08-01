import libadjoint
from dolfin_adjoint import adjlinalg, adjrhs 
from dolfin_adjoint.adjglobals import adjointer

class ReducedFunctional(object):
    def __init__(self, functional, parameter):
        ''' Creates a reduced functional object, that evaluates the functional value for a given parameter value '''
        self.functional = functional
        self.parameter = parameter
        # This flag indicates if the functional evaluation is based on replaying the forward annotation. 
        self.replays_annotation = True

    def __call__(self, coeff):
        ''' Evaluates the reduced functional for the given parameter value '''
        # Create a RHS object with the new control values
        init_rhs = adjlinalg.Vector(coeff).duplicate()
        init_rhs.axpy(1.0,adjlinalg.Vector(coeff))
        rhs = adjrhs.RHS(init_rhs)
        # Register the new rhs in the annotation
        class DummyEquation(object):
            pass
        e = DummyEquation()
        eqn_nb = self.parameter.var.equation_nb(adjointer)
        e.equation = adjointer.adjointer.equations[eqn_nb]
        rhs.register(e)

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
