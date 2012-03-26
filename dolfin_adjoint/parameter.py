import libadjoint
from solving import adj_variables, Vector
import ufl
import dolfin
from dolfin import info, info_blue, info_red

class InitialConditionParameter(libadjoint.Parameter):
  '''This Parameter is used as input to the tangent linear model (TLM)
  when one wishes to compute dJ/d(initial condition) in a particular direction (perturbation).'''
  def __init__(self, coeff, perturbation=None):
    '''coeff: the variable whose initial condition you wish to perturb.
       perturbation: the perturbation direction in which you wish to compute the gradient. Must be a Function.'''
    self.var = adj_variables[coeff]
    self.var.c_object.timestep = 0 # we want to put in the source term only at the initial condition.
    self.var.c_object.iteration = 0 # we want to put in the source term only at the initial condition.

    if perturbation:
      self.perturbation = Vector(perturbation).duplicate()
      self.perturbation.axpy(1.0, Vector(perturbation))
    else:
      self.perturbation = None

  def __call__(self, adjointer, i, dependencies, values, variable):
    # The TLM source term only kicks in at the start, for the initial condition:
    if self.var == variable:
      assert self.perturbation is not None, "Need to specify a perturbation if using in the TLM."
      return self.perturbation
    else:
      return None

  def __str__(self):
    return self.var.name + ':InitialCondition'

  def inner_adjoint(self, adjointer, adjoint, i, variable):
    if self.var == variable:
      return adjoint
    else:
      return None

class ScalarParameter(libadjoint.Parameter):
  '''This Parameter is used as input to the tangent linear model (TLM)
  when one wishes to compute dJ/da, where a is a single scalar parameter.'''
  def __init__(self, a):
    self.a = a

  def __call__(self, adjointer, i, dependencies, values, variable):
    (fwd_var, lhs, rhs) = adjointer.get_forward_equation(i)
    lhs = lhs.data; rhs = rhs.data

    if not isinstance(lhs, ufl.Identity):
      fn_space = ufl.algorithms.extract_arguments(rhs)[0].function_space()
      x = dolfin.Function(fn_space)
      form = rhs - dolfin.action(lhs, x)

      dparam = dolfin.Function(dolfin.FunctionSpace(fn_space.mesh(), "R", 0))
      dparam.vector()[:] = 1.0

      diff_form = dolfin.derivative(form, self.a, dparam)

      if x in ufl.algorithms.extract_coefficients(diff_form):
        #raise libadjoint.exceptions.LibadjointErrorNotImplemented("Sorry, the dF/dm term can't depend on the solution (yet!)")
        info_red("Warning: assuming dF/dm does not actually depend on the forward solution.")

      return solving.Vector(diff_form)
    else:
      return None

  def __str__(self):
    return str(self.a) + ':ScalarParameter'

  def inner_adjoint(self, adjointer, adjoint, i, variable):
    (fwd_var, lhs, rhs) = adjointer.get_forward_equation(i)
    lhs = lhs.data; rhs = rhs.data

    if not isinstance(lhs, ufl.Identity):
      fn_space = ufl.algorithms.extract_arguments(rhs)[0].function_space()
      x = dolfin.Function(fn_space)
      form = rhs - dolfin.action(lhs, x)

      dparam = dolfin.Function(dolfin.FunctionSpace(fn_space.mesh(), "R", 0))
      dparam.vector()[:] = 1.0

      diff_form = dolfin.derivative(form, self.a, dparam)

      if x in ufl.algorithms.extract_coefficients(diff_form):
        #raise libadjoint.exceptions.LibadjointErrorNotImplemented("Sorry, the dF/dm term can't depend on the solution (yet!)")
        info_red("Warning: assuming dF/dm does not actually depend on the forward solution.")

      dFdm = dolfin.assemble(diff_form) # actually - dF/dm
      assert isinstance(dFdm, dolfin.GenericVector)

      out = dFdm.inner(adjoint.vector())
      return dFdm.inner(adjoint.vector())
