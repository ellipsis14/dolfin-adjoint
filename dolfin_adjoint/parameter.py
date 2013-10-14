import libadjoint
import ufl
import backend
from backend import info, info_blue, info_red
import numpy
import adjlinalg
import adjglobals
from adjrhs import adj_get_forward_equation
import adjresidual
from constant import get_constant
import constant

class DolfinAdjointParameter(libadjoint.Parameter):
  def __call__(self, adjointer, i, dependencies, values, variable):
    '''This function gives the source term for the tangent linear model.
    variable gives the forward variable associated with this tangent
    linear solve; the other inputs (adjointer, i, dependencies, values)
    are there in case you need them.

    Return an adjlinalg.Vector to contribute a source term, or return
    None if there's nothing to do.'''
    raise NotImplementedError

  def equation_partial_derivative(self, adjointer, adjoint, i, variable):
    '''This function computes the contribution to the functional gradient
    associated with a particular equation.

    Given the adjoint solution adjoint, this function is to compute
    -inner(adjoint, derivative(F, m))

    where F is a particular equation (i is its number, variable is the forward
    variable associated with it)
    and m is the Parameter.'''
    raise NotImplementedError

  def equation_partial_second_derivative(self, adjointer, adjoint, i, variable, m_dot):
    '''This function computes the contribution to the functional gradient
    associated with a particular equation.

    Given the adjoint solution adjoint, this function is to compute
    -inner(adjoint, derivative(derivative(F, m), m_dot))

    where F is a particular equation (i is its number, variable is the forward
    variable associated with it)
    and m is the Parameter.'''
    pass

  def functional_partial_derivative(self, adjointer, J, timestep):
    '''Given a functional J, compute derivative(J, m) -- the partial derivative of
    J with respect to m. This is necessary to compute correct functional gradients.'''
    pass

  def functional_partial_second_derivative(self, adjointer, J, timestep, m_dot):
    '''Given a functional J, compute derivative(derivative(J, m), m, m_dot) -- the partial second derivative of
    J with respect to m in the direction m_dot. This is necessary to compute correct functional Hessians.'''
    pass

  def data(self):
    '''Return the data associated with the current values of the Parameter.'''
    raise NotImplementedError

  def set_perturbation(self, m_dot):
    '''Return another instance of the same class, representing the Parameter perturbed in a particular
    direction m_dot.'''
    raise NotImplementedError

class InitialConditionParameter(DolfinAdjointParameter):
  '''This Parameter is used as input to the tangent linear model (TLM)
  when one wishes to compute dJ/d(initial condition) in a particular direction (perturbation).'''
  def __init__(self, coeff, value=None, perturbation=None):
    '''coeff: the variable whose initial condition you wish to perturb.
       perturbation: the perturbation direction in which you wish to compute the gradient. Must be a Function.'''

    if not (isinstance(coeff, backend.Function) or isinstance(coeff, str)):
      raise TypeError, "The coefficient must be a Function or a String"
    self.coeff = coeff
    self.value = value
    self.var = None 
    # Find the first occurance of the coeffcient
    for t in range(adjglobals.adjointer.timestep_count):
      var = libadjoint.Variable(str(coeff), t, 0)
      if adjglobals.adjointer.variable_known(var):
        self.var = var 
        break
    # Fallback option for cases where the parameter is initialised before the annotation 
    if not self.var:
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

  def equation_partial_derivative(self, adjointer, adjoint, i, variable):
    if self.var == variable:
      return adjoint
    else:
      return None

  def data(self):
    if self.value is not None:
      return self.value
    else:
      return adjglobals.adjointer.get_variable_value(self.var).data

  def set_perturbation(self, m_dot):
    return InitialConditionParameter(self.coeff, perturbation=m_dot, value=self.value)

class ScalarParameter(DolfinAdjointParameter):
  '''This Parameter is used as input to the tangent linear model (TLM)
  when one wishes to compute dJ/da, where a is a single scalar parameter.'''
  def __init__(self, a, coeff=1):
    if not (isinstance(a, backend.Constant) or isinstance(a, str)):
      raise TypeError, "The coefficient must be a Constant or a String"
    self.a = a
    self.coeff = coeff

    # I can't believe I'm making this nauseous hack. I *hate* Constant.assign. It's
    # either constant or it isn't! Make up your minds!
    if isinstance(a, str):
      constant.scalar_parameters.append(a)
    else:
      constant.scalar_parameters.append(a.adj_name)

  def __call__(self, adjointer, i, dependencies, values, variable):
    form = adjresidual.get_residual(i)
    if form is not None:
      form = -form

      fn_space = ufl.algorithms.extract_arguments(form)[0].function_space()
      dparam = backend.Function(backend.FunctionSpace(fn_space.mesh(), "R", 0))
      dparam.vector()[:] = 1.0 * float(self.coeff)

      diff_form = ufl.algorithms.expand_derivatives(backend.derivative(form, get_constant(self.a), dparam))

      if diff_form is None:
        return None

      return adjlinalg.Vector(diff_form)
    else:
      return None

  def __str__(self):
    return str(self.coeff) + '*' + str(self.a) + ':ScalarParameter'

  def equation_partial_derivative(self, adjointer, adjoint, i, variable):
    form = adjresidual.get_residual(i)
    if form is not None:
      form = -form

      mesh = ufl.algorithms.extract_arguments(form)[0].function_space().mesh()
      fn_space = backend.FunctionSpace(mesh, "R", 0)
      dparam = backend.Function(fn_space)
      dparam.vector()[:] = 1.0 * float(self.coeff)

      diff_form = ufl.algorithms.expand_derivatives(backend.derivative(form, get_constant(self.a), dparam))

      if diff_form is None:
        return None

      # Let's see if the form actually depends on the parameter m
      if diff_form.integrals() != ():
        dFdm = backend.assemble(diff_form) # actually - dF/dm
        assert isinstance(dFdm, backend.GenericVector)

        out = dFdm.inner(adjoint.vector())
        return out
      else:
        return None # dF/dm is zero, return None

  def equation_partial_second_derivative(self, adjointer, adjoint, i, variable, m_dot):
    form = adjresidual.get_residual(i)
    if form is not None:
      form = -form

      mesh = ufl.algorithms.extract_arguments(form)[0].function_space().mesh()
      fn_space = backend.FunctionSpace(mesh, "R", 0)
      dparam = backend.Function(fn_space)
      dparam.vector()[:] = 1.0 * float(self.coeff)
      d2param = backend.Function(fn_space)
      d2param.vector()[:] = 1.0 * float(self.coeff) * m_dot

      diff_form = ufl.algorithms.expand_derivatives(backend.derivative(form, get_constant(self.a), dparam))
      if diff_form is None:
        return None

      diff_form  = ufl.algorithms.expand_derivatives(backend.derivative(diff_form, get_constant(self.a), d2param))
      if diff_form is None:
        return None

      # Let's see if the form actually depends on the parameter m
      if diff_form.integrals() != ():
        dFdm = backend.assemble(diff_form) # actually - dF/dm
        assert isinstance(dFdm, backend.GenericVector)

        out = dFdm.inner(adjoint.vector())
        return out
      else:
        return None # dF/dm is zero, return None

  def functional_partial_derivative(self, adjointer, J, timestep):
    form = J.get_form(adjointer, timestep)

    if form is None:
      return None

    # OK. Now that we have the form for the functional at this timestep, let's differentiate it with respect to
    # my dear Constant, and be done.
    for coeff in ufl.algorithms.extract_coefficients(form):
      try:
        mesh = coeff.function_space().mesh()
        fn_space = backend.FunctionSpace(mesh, "R", 0)
        break
      except:
        pass

    dparam = backend.Function(fn_space)
    dparam.vector()[:] = 1.0 * float(self.coeff)

    d = backend.derivative(form, get_constant(self.a), dparam)
    d = ufl.algorithms.expand_derivatives(d)
    if d.integrals() != ():
      return backend.assemble(d)
    else:
      return None

  def functional_partial_second_derivative(self, adjointer, J, timestep, m_dot):
    form = J.get_form(adjointer, timestep)

    if form is None:
      return None

    for coeff in ufl.algorithms.extract_coefficients(form):
      try:
        mesh = coeff.function_space().mesh()
        fn_space = backend.FunctionSpace(mesh, "R", 0)
        break
      except:
        pass

    dparam = backend.Function(fn_space)
    dparam.vector()[:] = 1.0 * float(self.coeff)

    d = backend.derivative(form, get_constant(self.a), dparam)
    d = ufl.algorithms.expand_derivatives(d)

    d2param = backend.Function(fn_space)
    d2param.vector()[:] = 1.0 * float(self.coeff) * m_dot

    d = backend.derivative(d, get_constant(self.a), d2param)
    d = ufl.algorithms.expand_derivatives(d)

    if d.integrals() != ():
      return backend.assemble(d)
    else:
      return None

  def data(self):
    return get_constant(self.a)

  def set_perturbation(self, m_dot):
    '''Return another instance of the same class, representing the Parameter perturbed in a particular
    direction m_dot.'''
    return ScalarParameter(self.a, coeff=m_dot)

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
    dparam = backend.Function(backend.FunctionSpace(fn_space.mesh(), "R", 0))
    dparam.vector()[:] = 1.0

    for (a, da) in zip(self.v, self.dv):
      out_form = da * backend.derivative(form, a, dparam)
      if diff_form is None:
        diff_form = out_form
      else:
        diff_form += out_form

    return adjlinalg.Vector(diff_form)

  def __str__(self):
    return str(self.v) + ':ScalarParameters'

  def equation_partial_derivative(self, adjointer, adjoint, i, variable):
    form = adjresidual.get_residual(i)

    if form is None:
      return None
    else:
      form = -form

    fn_space = ufl.algorithms.extract_arguments(form)[0].function_space()
    dparam = backend.Function(backend.FunctionSpace(fn_space.mesh(), "R", 0))
    dparam.vector()[:] = 1.0

    dJdv = numpy.zeros(len(self.v))
    for (i, a) in enumerate(self.v):
      diff_form = ufl.algorithms.expand_derivatives(backend.derivative(form, a, dparam))

      dFdm = backend.assemble(diff_form) # actually - dF/dm
      assert isinstance(dFdm, backend.GenericVector)

      out = dFdm.inner(adjoint.vector())
      dJdv[i] = out

    return dJdv

  def data(self):
    return self.v

class TimeConstantParameter(InitialConditionParameter):
  '''TimeConstantParameter is just another name for InitialConditionParameter,
  since from dolfin-adjoint's point of view they're exactly the same. But it
  confuses people to talk about initial conditions of data that doesn't change
  in time (like diffusivities, or bathymetries, or whatever), so hence this
  alias.'''
  pass

class SteadyParameter(InitialConditionParameter):
  '''SteadyParameter is just another name for InitialConditionParameter,
  since from dolfin-adjoint's point of view they're exactly the same. But it
  confuses people to talk about initial conditions of data in steady state problems, 
  so hence this alias.'''
  pass

class ListParameter(DolfinAdjointParameter):
  def __init__(self, parameters):
    for p in parameters:
      assert isinstance(p, DolfinAdjointParameter)

    self.parameters = parameters

  def __call__(self, adjointer, i, dependencies, values, variable):
    '''This function gives the source term for the tangent linear model.
    variable gives the forward variable associated with this tangent
    linear solve; the other inputs (adjointer, i, dependencies, values)
    are there in case you need them.

    Return an adjlinalg.Vector to contribute a source term, or return
    None if there's nothing to do.'''

    calls = [p(adjointer, i, dependencies, values, variable) for p in self.parameters]

    return reduce(_add, calls, None)

  def equation_partial_derivative(self, adjointer, adjoint, i, variable):
    '''This function computes the contribution to the functional gradient
    associated with a particular equation.

    Given the adjoint solution adjoint, this function is to compute
    -inner(adjoint, derivative(F, m))

    where F is a particular equation (i is its number, variable is the forward
    variable associated with it)
    and m is the Parameter.'''

    return [p.equation_partial_derivative(adjointer, adjoint, i, variable) for p in self.parameters]

  def equation_partial_second_derivative(self, adjointer, adjoint, i, variable, m_dot):
    '''This function computes the contribution to the functional gradient
    associated with a particular equation.

    Given the adjoint solution adjoint, this function is to compute
    -inner(adjoint, derivative(derivative(F, m), m_dot))

    where F is a particular equation (i is its number, variable is the forward
    variable associated with it)
    and m is the Parameter.'''
    return [p.equation_partial_second_derivative(adjointer, adjoint, i, variable, m) for (p, m) in zip(self.parameters, m_dot)]

  def functional_partial_derivative(self, adjointer, J, timestep):
    '''Given a functional J, compute derivative(J, m) -- the partial derivative of
    J with respect to m. This is necessary to compute correct functional gradients.'''
    return [p.functional_partial_derivative(adjointer, J, timestep) for p in self.parameters]

  def functional_partial_second_derivative(self, adjointer, J, timestep, m_dot):
    '''Given a functional J, compute derivative(derivative(J, m), m, m_dot) -- the partial second derivative of
    J with respect to m in the direction m_dot. This is necessary to compute correct functional Hessians.'''
    return [p.functional_partial_second_derivative(adjointer, J, timestep, m) for (p, m) in zip(self.parameters, m_dot)]

  def data(self):
    '''Return the data associated with the current values of the Parameter.'''
    return [p.data() for p in self.parameters]

  def set_perturbation(self, m_dot):
    '''Return another instance of the same class, representing the Parameter perturbed in a particular
    direction m_dot.'''
    return ListParameter([p.set_perturbation(m) for (p, m) in zip(self.parameters, m_dot)])

  def __getitem__(self, i):
    return self.parameters[i]

def _add(x, y):
  if x is None:
    return y

  if y is None:
    return x

  x.axpy(1.0, y)
  return x

