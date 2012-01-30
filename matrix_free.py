import ufl
import dolfin
import libadjoint
import solving
import hashlib
import expressions
import copy
import time

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

    self.operators = kwargs['operators']
    del kwargs['operators']

    solving.Matrix.__init__(self, *args, **kwargs)

  def solve(self, var, b):
    solver = dolfin.PETScKrylovSolver(*self.solver_parameters)

    x = dolfin.Function(self.fn_space)
    if b.data is None:
      dolfin.info_red("Warning: got zero RHS for the solve associated with variable %s" % var)
      return solving.Vector(x)

    rhs = dolfin.assemble(b.data)

    if var.type in ['ADJ_TLM', 'ADJ_ADJOINT']:
      self.bcs = [dolfin.homogenize(bc) for bc in self.bcs if isinstance(bc, dolfin.cpp.DirichletBC)]
    for bc in self.bcs:
      bc.apply(rhs)

    if self.operators[1] is not None: # we have a user-supplied preconditioner
      solver.set_operators(self.data, self.operators[1])
      solver.solve(dolfin.down_cast(x.vector()), dolfin.down_cast(rhs))
    else:
      solver.solve(self.data, dolfin.down_cast(x.vector()), dolfin.down_cast(rhs))

    return solving.Vector(x)

  def axpy(self, alpha, x):
    raise libadjoint.exceptions.LibadjointErrorNotImplemented("Can't add to a matrix-free matrix .. ")

class AdjointPETScKrylovSolver(dolfin.PETScKrylovSolver):
  def __init__(self, *args):
    dolfin.PETScKrylovSolver.__init__(self, *args)
    self.solver_parameters = args

    self.operators = (None, None)

  def set_operators(self, A, P):
    dolfin.PETScKrylovSolver.set_operators(self, A, P)
    self.operators = (A, P)

  def set_operator(self, A):
    dolfin.PETScKrylovSolver.set_operator(self, A)
    self.operators = (A, self.operators[1])

  def solve(self, *args, **kwargs):

    annotate = True
    if "annotate" in kwargs:
      annotate = kwargs["annotate"]
      del kwargs["annotate"]

    if len(args) == 3:
      A = args[0]
      x = args[1]
      b = args[2]
    elif len(args) == 2:
      A = self.operators[0]
      x = args[0]
      b = args[1]

    if annotate:
      if not isinstance(A, AdjointKrylovMatrix):
        raise libadjoint.exceptions.LibadjointErrorInvalidInputs("Sorry, we only support AdjointKrylovMatrix's.")

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

      diag_name = hashlib.md5(str(hash(A)) + str(time.time())).hexdigest()
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
          A_transpose = A.hermitian()
          return (MatrixFree(A_transpose, fn_space=x.function.function_space(), bcs=A_transpose.bcs, solver_parameters=self.solver_parameters, operators=transpose_operators(self.operators)), solving.Vector(None, fn_space=x.function.function_space()))
        else:
          return (MatrixFree(A, fn_space=x.function.function_space(), bcs=b.bcs, solver_parameters=self.solver_parameters, operators=self.operators), solving.Vector(None, fn_space=x.function.function_space()))
      diag_block.assemble = diag_assembly_cb

      def diag_action_cb(dependencies, values, hermitian, coefficient, input, context):
        expressions.update_expressions(frozen_expressions_dict)
        A.set_dependencies(dependencies, [val.data for val in values])

        if hermitian:
          acting_mat = A.transpose()
        else:
          acting_mat = A

        output_fn = dolfin.Function(input.data.function_space())
        acting_mat.mult(input.data.vector(), output_fn.vector())
        vec = output_fn.vector()
        for i in range(len(vec)):
          vec[i] = coefficient * vec[i]

        return solving.Vector(output_fn)
      diag_block.action = diag_action_cb

      if len(dependencies) > 0:
        def derivative_action(dependencies, values, variable, contraction_vector, hermitian, input, coefficient, context):
          expressions.update_expressions(frozen_expressions_dict)
          A.set_dependencies(dependencies, [val.data for val in values])

          action = A.derivative_action(values[dependencies.index(variable)].data, contraction_vector.data, hermitian, input.data, coefficient)
          return solving.Vector(action)
        diag_block.derivative_action = derivative_action

      eqn = libadjoint.Equation(var, blocks=[diag_block], targets=[var], rhs=rhs)
      cs = solving.adjointer.register_equation(eqn)
      solving.do_checkpoint(cs, var)

    out = dolfin.PETScKrylovSolver.solve(self, *args)

    if annotate:
      if solving.debugging["record_all"]:
        solving.adjointer.record_variable(var, libadjoint.MemoryStorage(solving.Vector(x.function)))

    return out

class AdjointKrylovMatrix(dolfin.PETScKrylovMatrix):
  def __init__(self, a, bcs=None):
    shapes = self.shape(a)
    dolfin.PETScKrylovMatrix.__init__(self, shapes[0], shapes[1])
    self.original_form = a
    self.current_form = a

    if bcs is None:
      self.bcs = []
    else:
      if isinstance(bcs, list):
        self.bcs = bcs
      else:
        self.bcs = [bcs]

  def shape(self, a):
    args = ufl.algorithms.extract_arguments(a)
    shapes = [arg.function_space().dim() for arg in args]
    return shapes

  def mult(self, *args):
    shapes = self.shape(self.current_form)
    y = dolfin.PETScVector(shapes[0])

    action_fn = dolfin.Function(ufl.algorithms.extract_arguments(self.current_form)[-1].function_space())
    action_vec = action_fn.vector()
    for i in range(len(args[0])):
      action_vec[i] = args[0][i]

    action_form = dolfin.action(self.current_form, action_fn)
    dolfin.assemble(action_form, tensor=y)

    for bc in self.bcs:
      bcvals = bc.get_boundary_values()
      for idx in bcvals:
        y[idx] = action_vec[idx]

    args[1].set_local(y.array())

  def dependencies(self):
    return ufl.algorithms.extract_coefficients(self.original_form)

  def set_dependencies(self, dependencies, values):
    replace_dict = dict(zip(self.dependencies(), values))
    self.current_form = dolfin.replace(self.original_form, replace_dict)

  def hermitian(self):
    adjoint_bcs = [dolfin.homogenize(bc) for bc in self.bcs if isinstance(bc, dolfin.cpp.DirichletBC)]
    return AdjointKrylovMatrix(dolfin.adjoint(self.original_form), bcs=adjoint_bcs)

  def derivative_action(self, variable, contraction_vector, hermitian, input, coefficient):
    deriv = dolfin.derivative(self.current_form, variable)
    args = ufl.algorithms.extract_arguments(deriv)
    deriv = dolfin.replace(deriv, {args[1]: contraction_vector})

    if hermitian:
      deriv = dolfin.adjoint(deriv)

    action = coefficient * dolfin.action(deriv, input)

    return action

def transpose_operators(operators):
  out = [None, None]

  for i in range(2):
    op = operators[i]
    if op is None: 
      out[i] = None
    elif isinstance(op, dolfin.cpp.GenericMatrix):
      out[i] = op.__class__()
      dolfin.assemble(dolfin.adjoint(op.form), tensor=out[i])
    elif isinstance(op, dolfin.Form):
      out[i] = dolfin.adjoint(op)

  return out
