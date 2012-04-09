import libadjoint
from solving import adj_variables, Vector, IdentityMatrix
import ufl
import dolfin
from dolfin import info, info_blue, info_red
import numpy

class InitialConditionParameter(libadjoint.Parameter):
  '''This Parameter is used as input to the tangent linear model (TLM)
  when one wishes to compute dJ/d(initial condition) in a particular direction (perturbation).'''
  def __init__(self, coeff, perturbation=None):
    '''coeff: the variable whose initial condition you wish to perturb.
       perturbation: the perturbation direction in which you wish to compute the gradient. Must be a Function.'''

    self.var = libadjoint.Variable(str(coeff), 0, 0)

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

    if not isinstance(lhs, IdentityMatrix):
      fn_space = ufl.algorithms.extract_arguments(rhs)[0].function_space()
      x = dolfin.Function(fn_space)
      form = rhs - dolfin.action(lhs, x)

      dparam = dolfin.Function(dolfin.FunctionSpace(fn_space.mesh(), "R", 0))
      dparam.vector()[:] = 1.0

      if isinstance(self.a, dolfin.Constant):
        diff_form = dolfin.derivative(form, self.a, dparam)
      elif isinstance(self.a, str):
        diff_form = None
        for coeff in ufl.algorithms.extract_coefficients(form):
          if hasattr(coeff, "adj_name"):
            if coeff.adj_name == self.a:
              diff_form = dolfin.derivative(form, coeff, dparam)
              break

        if diff_form is None:
          return None

      else:
        raise libadjoint.exceptions.LibadjointErrorNotImplemented("Don't know how to handle any other types!")

      if x in ufl.algorithms.extract_coefficients(diff_form):
        # We really need the forward solution to compute dF/dm.
        try:
          y = adjointer.get_variable_value(fwd_var).data
        except libadjoint.exceptions.LibadjointErrorNeedValue:
          info_red("Warning: recomputing forward solution to compute -dF/dm")
          y = adjointer.get_forward_solution(i)[1].data

        diff_form = dolfin.replace(diff_form, {x: y})

      return solving.Vector(diff_form)
    else:
      return None

  def __str__(self):
    return str(self.a) + ':ScalarParameter'

  def inner_adjoint(self, adjointer, adjoint, i, variable):
    (fwd_var, lhs, rhs) = adjointer.get_forward_equation(i)
    lhs = lhs.data; rhs = rhs.data

    if not isinstance(lhs, IdentityMatrix):
      fn_space = ufl.algorithms.extract_arguments(rhs)[0].function_space()
      x = dolfin.Function(fn_space)
      form = rhs - dolfin.action(lhs, x)

      dparam = dolfin.Function(dolfin.FunctionSpace(fn_space.mesh(), "R", 0))
      dparam.vector()[:] = 1.0

      if isinstance(self.a, dolfin.Constant):
        diff_form = dolfin.derivative(form, self.a, dparam)
      elif isinstance(self.a, str):
        diff_form = None
        for coeff in ufl.algorithms.extract_coefficients(form):
          if hasattr(coeff, "adj_name"):
            if coeff.adj_name == self.a:
              diff_form = dolfin.derivative(form, coeff, dparam)
              break

        if diff_form is None:
          return None

      else:
        raise libadjoint.exceptions.LibadjointErrorNotImplemented("Don't know how to handle any other types!")

      if x in ufl.algorithms.extract_coefficients(diff_form):
        # We really need the forward solution to compute dF/dm.
        try:
          y = adjointer.get_variable_value(fwd_var).data
        except libadjoint.exceptions.LibadjointErrorNeedValue:
          info_red("Warning: recomputing forward solution to compute -dF/dm")
          y = adjointer.get_forward_solution(i)[1].data

        diff_form = dolfin.replace(diff_form, {x: y})

      dFdm = dolfin.assemble(diff_form) # actually - dF/dm
      assert isinstance(dFdm, dolfin.GenericVector)

      out = dFdm.inner(adjoint.vector())
      return out

class ScalarParameters(libadjoint.Parameter):
  '''This Parameter is used as input to the tangent linear model (TLM)
  when one wishes to compute dJ/dv . delta v, where v is a vector of scalar parameters.'''
  def __init__(self, v, dv=None):
    self.v = v
    if dv is not None:
      self.dv = dv

  def __call__(self, adjointer, i, dependencies, values, variable):
    (fwd_var, lhs, rhs) = adjointer.get_forward_equation(i)
    lhs = lhs.data; rhs = rhs.data

    diff_form = None

    assert self.dv is not None, "Need a perturbation direction to use in the TLM."

    if not isinstance(lhs, IdentityMatrix):
      fn_space = ufl.algorithms.extract_arguments(rhs)[0].function_space()
      x = dolfin.Function(fn_space)
      form = rhs - dolfin.action(lhs, x)

      dparam = dolfin.Function(dolfin.FunctionSpace(fn_space.mesh(), "R", 0))
      dparam.vector()[:] = 1.0

      for (a, da) in zip(self.v, self.dv):
        out_form = da * dolfin.derivative(form, a, dparam)
        if diff_form is None:
          out_form = diff_form
        else:
          out_form += diff_form

        if x in ufl.algorithms.extract_coefficients(diff_form):
          # We really need the forward solution to compute dF/dm.
          try:
            y = adjointer.get_variable_value(fwd_var).data
          except libadjoint.exceptions.LibadjointErrorNeedValue:
            info_red("Warning: recomputing forward solution to compute -dF/dm")
            y = adjointer.get_forward_solution(i)[1].data

          diff_form = dolfin.replace(diff_form, {x: y})

      return solving.Vector(diff_form)
    else:
      return None

  def __str__(self):
    return str(self.v) + ':ScalarParameters'

  def inner_adjoint(self, adjointer, adjoint, i, variable):
    (fwd_var, lhs, rhs) = adjointer.get_forward_equation(i)
    lhs = lhs.data; rhs = rhs.data

    dJdv = numpy.zeros(len(self.v))

    if not isinstance(lhs, IdentityMatrix):
      fn_space = ufl.algorithms.extract_arguments(rhs)[0].function_space()
      x = dolfin.Function(fn_space)
      form = rhs - dolfin.action(lhs, x)

      dparam = dolfin.Function(dolfin.FunctionSpace(fn_space.mesh(), "R", 0))
      dparam.vector()[:] = 1.0

      for (i, a) in enumerate(self.v):
        diff_form = ufl.algorithms.expand_derivatives(dolfin.derivative(form, a, dparam))

        if x in ufl.algorithms.extract_coefficients(diff_form):
          # We really need the forward solution to compute dF/dm.
          try:
            y = adjointer.get_variable_value(fwd_var).data
          except libadjoint.exceptions.LibadjointErrorNeedValue:
            info_red("Warning: recomputing forward solution to compute -dF/dm")
            y = adjointer.get_forward_solution(i)[1].data

          diff_form = dolfin.replace(diff_form, {x: y})

        dFdm = dolfin.assemble(diff_form) # actually - dF/dm
        assert isinstance(dFdm, dolfin.GenericVector)

        out = dFdm.inner(adjoint.vector())
        dJdv[i] = out

      return dJdv
