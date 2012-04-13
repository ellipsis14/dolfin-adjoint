import solving
import dolfin
import libadjoint
import ufl

def interpolate(v, V, annotate=True):

  out = dolfin.interpolate(v, V)

  if solving.debugging["stop_annotating"]:
    annotate = False

  if isinstance(v, dolfin.Function) and annotate:
    rhsdep = solving.adj_variables[v]
    if solving.adjointer.variable_known(rhsdep):
      block_name = "Identity: %s" % str(V)
      if len(block_name) > int(libadjoint.constants.adj_constants["ADJ_NAME_LEN"]):
        block_name = block_name[0:int(libadjoint.constants.adj_constants["ADJ_NAME_LEN"])-1]
      identity_block = libadjoint.Block(block_name)

      def identity_assembly_cb(variables, dependencies, hermitian, coefficient, context):
        assert coefficient == 1
        return (solving.Matrix(solving.IdentityMatrix()), solving.Vector(dolfin.Function(V)))

      identity_block.assemble = identity_assembly_cb

      rhs = InterpolateRHS(v, V)

      no_registered = solving.register_initial_conditions(zip(rhs.coefficients(),rhs.dependencies()), linear=True)

      if solving.adjointer.first_solve:
        solving.adjointer.first_solve = False
        if no_registered > 0:
          solving.adj_inc_timestep()

      dep = solving.adj_variables.next(out)

      if solving.debugging["record_all"]:
        solving.adjointer.record_variable(dep, libadjoint.MemoryStorage(solving.Vector(out)))

      initial_eq = libadjoint.Equation(dep, blocks=[identity_block], targets=[dep], rhs=rhs)
      cs = solving.adjointer.register_equation(initial_eq)

      solving.do_checkpoint(cs, dep)

  return out

class InterpolateRHS(libadjoint.RHS):
  def __init__(self, v, V):
    self.v = v
    self.V = V
    self.dep = solving.adj_variables[v]

  def __call__(self, dependencies, values):
    return solving.Vector(dolfin.interpolate(values[0].data, self.V))

  def derivative_action(self, dependencies, values, variable, contraction_vector, hermitian):
    if not hermitian:
      return solving.Vector(dolfin.interpolate(contraction_vector.data, self.V))
    else:
#      For future reference, the transpose action of the interpolation operator
#      is (in pseudocode!):
#
#      for target_dof in target:
#        figure out what element it lives in, to compute src_dofs
#        for src_dof in src_dofs:
#          basis = the value of the basis function of src_dof at the node of target_dof
#
#          # all of the above is exactly the same as the forward interpolation.
#          # forward interpolation would do:
#          # target_coefficients[target_dof] += basis * src_coefficients[src_dof]
#
#          # but the adjoint action is:
#          src_coefficients[src_dof] += basis * target_coefficients[target_dof]

      raise libadjoint.exceptions.LibadjointErrorNotImplemented("Can't transpose an interpolation operator yet, sorry!")

  def dependencies(self):
    return [self.dep]

  def coefficients(self):
    return [self.v]

  def __str__(self):
    return "InterpolateRHS(%s, %s)" % (str(self.v), str(self.V))

