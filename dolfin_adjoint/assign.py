import dolfin
import ufl
from solving import solve, annotate as solving_annotate, do_checkpoint, register_initial_conditions
import libadjoint
import adjlinalg
import adjglobals

def register_assign(new, old, op=None):

  if not isinstance(old, dolfin.Function):
    assert op is not None

  fn_space = new.function_space()
  block_name = "Identity: %s" % str(fn_space)
  if len(block_name) > int(libadjoint.constants.adj_constants["ADJ_NAME_LEN"]):
    block_name = block_name[0:int(libadjoint.constants.adj_constants["ADJ_NAME_LEN"])-1]
  identity_block = libadjoint.Block(block_name)

  def identity_assembly_cb(variables, dependencies, hermitian, coefficient, context):
    assert coefficient == 1
    return (adjlinalg.Matrix(adjlinalg.IdentityMatrix()), adjlinalg.Vector(None, fn_space=fn_space))

  identity_block.assemble = identity_assembly_cb
  dep = adjglobals.adj_variables.next(new)

  if dolfin.parameters["adjoint"]["record_all"] and isinstance(old, dolfin.Function):
    adjglobals.adjointer.record_variable(dep, libadjoint.MemoryStorage(adjlinalg.Vector(old)))

  rhs = IdentityRHS(old, fn_space, op)
  register_initial_conditions(zip(rhs.coefficients(),rhs.dependencies()), linear=True)
  initial_eq = libadjoint.Equation(dep, blocks=[identity_block], targets=[dep], rhs=rhs)
  cs = adjglobals.adjointer.register_equation(initial_eq)

  do_checkpoint(cs, dep, rhs)

class IdentityRHS(libadjoint.RHS):
  def __init__(self, var, fn_space, op):
    self.var = var
    self.fn_space = fn_space
    self.op = op
    if isinstance(var, dolfin.Function):
      self.dep = adjglobals.adj_variables[var]

  def __call__(self, dependencies, values):
    if len(values) > 0:
      return adjlinalg.Vector(values[0].data)
    else:
      return adjlinalg.Vector(self.op(self.var, self.fn_space))

  def derivative_action(self, dependencies, values, variable, contraction_vector, hermitian):

    # If you want to apply boundary conditions symmetrically in the adjoint
    # -- and you often do --
    # then we need to have a UFL representation of all the terms in the adjoint equation.
    # However!
    # Since UFL cannot represent the identity map,
    # we need to find an f such that when
    # assemble(inner(f, v)*dx)
    # we get the contraction_vector.data back.
    # This involves inverting a mass matrix.

    if dolfin.parameters["adjoint"]["symmetric_bcs"] and dolfin.__version__ <= '1.2.0':
      dolfin.info_red("Warning: symmetric BC application requested but unavailable in dolfin <= 1.2.0.")

    if dolfin.parameters["adjoint"]["symmetric_bcs"] and dolfin.__version__ > '1.2.0':

      V = contraction_vector.data.function_space()
      v = dolfin.TestFunction(V)

      if str(V) not in adjglobals.fsp_lu:
        u = dolfin.TrialFunction(V)
        A = dolfin.assemble(dolfin.inner(u, v)*dolfin.dx)
        lusolver = dolfin.LUSolver(A, "mumps")
        lusolver.parameters["symmetric"] = True
        lusolver.parameters["reuse_factorization"] = True
        adjglobals.fsp_lu[str(V)] = lusolver
      else:
        lusolver = adjglobals.fsp_lu[str(V)]

      riesz = dolfin.Function(V)
      lusolver.solve(riesz.vector(), contraction_vector.data.vector())
      return adjlinalg.Vector(dolfin.inner(riesz, v)*dolfin.dx)
    else:
      return adjlinalg.Vector(contraction_vector.data)

  def second_derivative_action(self, dependencies, values, inner_variable, inner_contraction_vector, outer_variable, hermitian, action):
    return None

  def dependencies(self):
    if isinstance(self.var, dolfin.Function):
      return [self.dep]
    else:
      return []

  def coefficients(self):
    if isinstance(self.var, dolfin.Function):
      return [self.var]
    else:
      return []

  def __str__(self):
    return "AssignIdentity(%s)" % str(self.dep)
