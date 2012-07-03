import libadjoint
import ufl
import dolfin
from dolfin import info, info_blue, info_red
import numpy
import adjlinalg
import adjglobals
from adjrhs import adj_get_forward_equation
import adjresidual
from constant import get_constant

class DolfinAdjointParameter(libadjoint.Parameter):
  def inner_adjoint(self, adjointer, adjoint, i, variable):
    pass

  def partial_derivative(self, adjointer, J, timestep):
    pass

class InitialConditionParameter(DolfinAdjointParameter):
  '''This Parameter is used as input to the tangent linear model (TLM)
  when one wishes to compute dJ/d(initial condition) in a particular direction (perturbation).'''
  def __init__(self, coeff, perturbation=None):
    '''coeff: the variable whose initial condition you wish to perturb.
       perturbation: the perturbation direction in which you wish to compute the gradient. Must be a Function.'''

    self.var = libadjoint.Variable(str(coeff), 0, 0)

    if perturbation:
      self.perturbation = adjlinalg.Vector(perturbation).duplicate()
      self.perturbation.axpy(1.0, adjlinalg.Vector(perturbation))
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

class ScalarParameter(DolfinAdjointParameter):
  '''This Parameter is used as input to the tangent linear model (TLM)
  when one wishes to compute dJ/da, where a is a single scalar parameter.'''
  def __init__(self, a):
    self.a = a

  def __call__(self, adjointer, i, dependencies, values, variable):
    form = adjresidual.get_residual(i)
    if form is not None:
      form = -form

      fn_space = ufl.algorithms.extract_arguments(form)[0].function_space()
      dparam = dolfin.Function(dolfin.FunctionSpace(fn_space.mesh(), "R", 0))
      dparam.vector()[:] = 1.0

      diff_form = ufl.algorithms.expand_derivatives(dolfin.derivative(form, get_constant(self.a), dparam))

      if diff_form is None:
        return None

      return adjlinalg.adjlinalg.Vector(diff_form)
    else:
      return None

  def __str__(self):
    return str(self.a) + ':ScalarParameter'

  def inner_adjoint(self, adjointer, adjoint, i, variable):
    form = adjresidual.get_residual(i)
    if form is not None:
      form = -form

      mesh = ufl.algorithms.extract_arguments(form)[0].function_space().mesh()
      fn_space = dolfin.FunctionSpace(mesh, "R", 0)
      dparam = dolfin.Function(fn_space)
      dparam.vector()[:] = 1.0

      diff_form = ufl.algorithms.expand_derivatives(dolfin.derivative(form, get_constant(self.a), dparam))

      if diff_form is None:
        return None

      # Let's see if the form actually depends on the parameter m
      if diff_form.integrals() != ():
        dFdm = dolfin.assemble(diff_form) # actually - dF/dm
        assert isinstance(dFdm, dolfin.GenericVector)

        out = dFdm.inner(adjoint.vector())
        return out
      else:
        return None # dF/dm is zero, return None

  def partial_derivative(self, adjointer, J, timestep):
    form = J.get_form(adjointer, timestep)

    # OK. Now that we have the form for the functional at this timestep, let's differentiate it with respect to
    # my dear Constant, and be done.
    for coeff in ufl.algorithms.extract_coefficients(form):
      try:
        fn_space = coeff.function_space()
        break
      except:
        pass

    dparam = dolfin.Function(fn_space)
    dparam.vector()[:] = 1.0

    diff_form = ufl.algorithms.expand_derivatives(dolfin.derivative(form, get_constant(self.a), dparam))
    return dolfin.assemble(diff_form)

class ScalarParameters(DolfinAdjointParameter):
  '''This Parameter is used as input to the tangent linear model (TLM)
  when one wishes to compute dJ/dv . delta v, where v is a vector of scalar parameters.'''
  def __init__(self, v, dv=None):
    self.v = v
    if dv is not None:
      self.dv = dv

  def __call__(self, adjointer, i, dependencies, values, variable):
    diff_form = None
    assert self.dv is not None, "Need a perturbation direction to use in the TLM."

    form = adjresidual.get_residual(i)

    if form is None:
      return None
    else:
      form = -form

    fn_space = ufl.algorithms.extract_arguments(form)[0].function_space()
    dparam = dolfin.Function(dolfin.FunctionSpace(fn_space.mesh(), "R", 0))
    dparam.vector()[:] = 1.0

    for (a, da) in zip(self.v, self.dv):
      out_form = da * dolfin.derivative(form, a, dparam)
      if diff_form is None:
        diff_form = out_form
      else:
        diff_form += out_form

    return adjlinalg.adjlinalg.Vector(diff_form)

  def __str__(self):
    return str(self.v) + ':ScalarParameters'

  def inner_adjoint(self, adjointer, adjoint, i, variable):
    form = adjresidual.get_residual(i)

    if form is None:
      return None
    else:
      form = -form

    fn_space = ufl.algorithms.extract_arguments(form)[0].function_space()
    dparam = dolfin.Function(dolfin.FunctionSpace(fn_space.mesh(), "R", 0))
    dparam.vector()[:] = 1.0

    dJdv = numpy.zeros(len(self.v))
    for (i, a) in enumerate(self.v):
      diff_form = ufl.algorithms.expand_derivatives(dolfin.derivative(form, a, dparam))

      dFdm = dolfin.assemble(diff_form) # actually - dF/dm
      assert isinstance(dFdm, dolfin.GenericVector)

      out = dFdm.inner(adjoint.vector())
      dJdv[i] = out

    return dJdv
