import dolfin
import libadjoint
import solving
import hashlib
import expressions

def down_cast(*args, **kwargs):
  dc = dolfin.down_cast(*args, **kwargs)

  if hasattr(args[0], 'form'):
    dc.form = args[0].form

  if hasattr(args[0], 'function'):
    dc.function = args[0].function

  return dc

class AdjointPETScKrylovSolver(dolfin.PETScKrylovSolver):
  def __init__(self, *args):
    if len(args) == 1:
      raise libadjoint.exceptions.LibadjointErrorNotImplemented("Sorry, solver/pc from KSP not implemented yet .. should be easy though!")

    dolfin.PETScKrylovSolver.__init__(self, *args)

    ksp = args[0]
    pc  = args[1]

    self.solver_parameters = {}
    self.solver_parameters["linear_solver"] = ksp
    self.solver_parameters["preconditioner"] = pc

  def solve(self, A, x, b):

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
      coeffs = A.dependencies()
      dependencies = [solving.adj_variables[x] for x in coeffs]

    if len(dependencies) > 0:
      raise libadjoint.exceptions.LibadjointErrorNotImplemented("Sorry, can't handle nonlinear dependencies yet ... ")

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

      if hermitian:
        A_transpose = copy.copy(A)
        (A_transpose.transpmult, A_transpose.mult) = (A.mult, A.transpmult)
        return (Matrix(A_transpose, solver_parameters=self.solver_parameters), Vector(None, fn_space=x.function.function_space()))
      else:
        return (Matrix(A, solver_parameters=self.solver_parameters), Vector(None, fn_space=x.function.function_space()))
    diag_block.assemble = diag_assembly_cb

    eqn = libadjoint.Equation(var, blocks=[diag_block], targets=[var], rhs=rhs)
    cs = solving.adjointer.register_equation(eqn)
    solving.do_checkpoint(cs, var)

    return dolfin.PETScKrylovSolver.solve(self, A, x, b)
