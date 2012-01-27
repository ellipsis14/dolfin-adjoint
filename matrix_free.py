import dolfin
import libadjoint
import solving
import hashlib
import expressions
import copy

def down_cast(*args, **kwargs):
  dc = dolfin.down_cast(*args, **kwargs)

  if hasattr(args[0], 'form'):
    dc.form = args[0].form

  if hasattr(args[0], 'function'):
    dc.function = args[0].function

  if hasattr(args[0], 'bcs'):
    dc.bcs = args[0].bcs

  return dc

class MatrixFree(solving.Matrix):
  def __init__(self, *args, **kwargs):
    self.fn_space = kwargs['fn_space']
    del kwargs['fn_space']
    solving.Matrix.__init__(self, *args, **kwargs)

  def solve(self, var, b):
    solver = dolfin.PETScKrylovSolver(*self.solver_parameters)
    x = dolfin.Function(self.fn_space)
    rhs = dolfin.assemble(b.data)
    for bc in self.bcs:
      bc.apply(rhs)

    solver.solve(self.data, dolfin.down_cast(x.vector()), dolfin.down_cast(rhs))
    return solving.Vector(x)

  def axpy(self, alpha, x):
    raise libadjoint.exceptions.LibadjointErrorNotImplemented("Can't add to a matrix-free matrix .. ")

class AdjointPETScKrylovSolver(dolfin.PETScKrylovSolver):
  def __init__(self, *args):
    dolfin.PETScKrylovSolver.__init__(self, *args)
    self.solver_parameters = args

  def solve(self, A, x, b, annotate=True):

    if annotate:
      if not hasattr(A, 'transpmult'):
        err = "Your PETScKrylovMatrix class has to implement a .transpmult method as well, as I need the transpose action. Note that if " + \
              "your forward problem has Dirichlet boundary conditions, the .transpmult MUST impose /homogenous/ Dirichlet boundary conditions " + \
              "on the resulting vector."
        raise libadjoint.exceptions.LibadjointErrorInvalidInputs(err)

      if not hasattr(x, 'function'):
        raise libadjoint.exceptions.LibadjointErrorInvalidInputs("Your x has to come from code like down_cast(my_function.vector()).")

      if not hasattr(b, 'form'):
        raise libadjoint.exceptions.LibadjointErrorInvalidInputs("Your b has to have the .form attribute: was it assembled with from dolfin_adjoint import *?")

      if not hasattr(A, 'dependencies'):
        print "A has no .dependencies method; assuming no nonlinear dependencies of the matrix-free operator."
        coeffs = []
        dependencies = []
      else:
        coeffs = [coeff for coeff in A.dependencies() if hasattr(coeff, 'function_space')]
        dependencies = [solving.adj_variables[coeff] for coeff in coeffs]

      if len(dependencies) > 0:
        assert hasattr(A, "set_dependencies"), "Need a set_dependencies method to replace your values, if you have nonlinear dependencies ... "

      rhs = solving.RHS(b.form)

      diag_name = hashlib.md5(str(hash(A))).hexdigest()
      diag_block = libadjoint.Block(diag_name, dependencies=dependencies, test_hermitian=solving.debugging["test_hermitian"], test_derivative=solving.debugging["test_derivative"])

      solving.register_initial_conditions(zip(rhs.coefficients(),rhs.dependencies()) + zip(coeffs, dependencies), linear=False, var=None)

      var = solving.adj_variables.next(x.function)

      frozen_expressions_dict = expressions.freeze_dict()

      def diag_assembly_cb(dependencies, values, hermitian, coefficient, context):
        '''This callback must conform to the libadjoint Python block assembly
        interface. It returns either the form or its transpose, depending on
        the value of the logical hermitian.'''

        assert coefficient == 1

        expressions.update_expressions(frozen_expressions_dict)

        if len(dependencies) > 0:
          A.set_dependencies(dependencies, [val.data for val in values])

        if hermitian:
          A_transpose = copy.copy(A)
          (A_transpose.transpmult, A_transpose.mult) = (A.mult, A.transpmult)
          return (MatrixFree(A_transpose, fn_space=x.function.function_space(), bcs=b.bcs, solver_parameters=self.solver_parameters), solving.Vector(None, fn_space=x.function.function_space()))
        else:
          return (MatrixFree(A, fn_space=x.function.function_space(), bcs=b.bcs, solver_parameters=self.solver_parameters), solving.Vector(None, fn_space=x.function.function_space()))
      diag_block.assemble = diag_assembly_cb

      eqn = libadjoint.Equation(var, blocks=[diag_block], targets=[var], rhs=rhs)
      cs = solving.adjointer.register_equation(eqn)
      solving.do_checkpoint(cs, var)

    out = dolfin.PETScKrylovSolver.solve(self, A, x, b)

    if annotate:
      if solving.debugging["record_all"]:
        solving.adjointer.record_variable(var, libadjoint.MemoryStorage(solving.Vector(x.function)))

    return out

