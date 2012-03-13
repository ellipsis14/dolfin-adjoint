import dolfin
import ufl
from solving import adjointer, adj_variables, debugging, solve, Vector, Matrix, annotate as solving_annotate
import libadjoint

dolfin_assign = dolfin.Function.assign
def dolfin_adjoint_assign(self, other, annotate=True):
  '''We also need to monkeypatch the Function.assign method, as it is often used inside 
  the main time loop, and not annotating it means you get the adjoint wrong for totally
  nonobvious reasons. If anyone objects to me monkeypatching your objects, my apologies
  in advance.'''

  # ignore anything not a dolfin.Function
  if not isinstance(other, dolfin.Function) or annotate is False:
    return dolfin_assign(self, other)

  # ignore anything that is an interpolation, rather than a straight assignment
  if str(self.function_space()) != str(other.function_space()):
    return dolfin_assign(self, other)

  other_var = adj_variables[other]
  # ignore any functions we haven't seen before -- we DON'T want to
  # annotate the assignment of initial conditions here. That happens
  # in the main solve wrapper.
  if not adjointer.variable_known(other_var):
    adj_variables.forget(other)
    return dolfin_assign(self, other)

  # OK, so we have a variable we've seen before. Beautiful.
  out = dolfin_assign(self, other)
  register_assign(self, other)
  return out

def register_assign(new, old):
  fn_space = new.function_space()
  block_name = "Identity: %s" % str(fn_space)
  if len(block_name) > int(libadjoint.constants.adj_constants["ADJ_NAME_LEN"]):
    block_name = block_name[0:int(libadjoint.constants.adj_constants["ADJ_NAME_LEN"])-1]
  identity_block = libadjoint.Block(block_name)

  def identity_assembly_cb(variables, dependencies, hermitian, coefficient, context):
    assert coefficient == 1
    return (Matrix(ufl.Identity(fn_space.dim())), Vector(dolfin.Function(fn_space)))

  identity_block.assemble = identity_assembly_cb
  dep = adj_variables.next(new)

  if debugging["record_all"]:
    adjointer.record_variable(dep, libadjoint.MemoryStorage(Vector(old)))

  initial_eq = libadjoint.Equation(dep, blocks=[identity_block], targets=[dep], rhs=IdentityRHS(old))
  adjointer.register_equation(initial_eq)

dolfin.Function.assign = dolfin_adjoint_assign

class IdentityRHS(libadjoint.RHS):
  def __init__(self, var):
    self.var = var
    self.dep = adj_variables[var]

  def __call__(self, dependencies, values):
    return Vector(values[0].data)

  def derivative_action(self, dependencies, values, variable, contraction_vector, hermitian):
    return Vector(contraction_vector.data)

  def dependencies(self):
    return [self.dep]

  def coefficients(self):
    return [self.var]

  def __str__(self):
    return "AssignIdentity(%s)" % str(self.dep)
