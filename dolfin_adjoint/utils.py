import libadjoint
from parameter import *
from dolfin import info_red, info_blue, info
import adjglobals
import dolfin
import numpy
import constant
import adjresidual
import ufl.algorithms
from solving import adj_html

def replay_dolfin(forget=False, tol=0.0, stop=False):
  if not dolfin.parameters["adjoint"]["record_all"]:
    info_red("Warning: your replay test will be much more effective with dolfin.parameters['adjoint']['record_all'] = True.")

  success = True
  for i in range(adjglobals.adjointer.equation_count):
      (fwd_var, output) = adjglobals.adjointer.get_forward_solution(i)

      storage = libadjoint.MemoryStorage(output)
      storage.set_compare(tol=tol)
      storage.set_overwrite(True)
      out = adjglobals.adjointer.record_variable(fwd_var, storage)
      success = success and out

      if forget:
        adjglobals.adjointer.forget_forward_equation(i)

      if not out and stop:
        return success

  return success

def convergence_order(errors):
  import math

  orders = [0.0] * (len(errors)-1)
  for i in range(len(errors)-1):
    try:
      orders[i] = math.log(errors[i]/errors[i+1], 2)
    except ZeroDivisionError:
      orders[i] = numpy.nan

  return orders

def compute_adjoint(functional, forget=True):

  for i in range(adjglobals.adjointer.timestep_count):
    adjglobals.adjointer.set_functional_dependencies(functional, i)

  for i in range(adjglobals.adjointer.equation_count)[::-1]:
      (adj_var, output) = adjglobals.adjointer.get_adjoint_solution(i, functional)

      storage = libadjoint.MemoryStorage(output)
      adjglobals.adjointer.record_variable(adj_var, storage)

      # forget is None: forget *nothing*.
      # forget is True: forget everything we can, forward and adjoint
      # forget is False: forget only unnecessary adjoint values
      if forget is None:
        pass
      elif forget:
        adjglobals.adjointer.forget_adjoint_equation(i)
      else:
        adjglobals.adjointer.forget_adjoint_values(i)

      yield (output.data, adj_var)

def compute_tlm(parameter, forget=False):

  for i in range(adjglobals.adjointer.equation_count):
      (tlm_var, output) = adjglobals.adjointer.get_tlm_solution(i, parameter)

      storage = libadjoint.MemoryStorage(output)
      storage.set_overwrite(True)
      adjglobals.adjointer.record_variable(tlm_var, storage)

      # forget is None: forget *nothing*.
      # forget is True: forget everything we can, forward and adjoint
      # forget is False: forget only unnecessary adjoint values
      if forget is None:
        pass
      elif forget:
        adjglobals.adjointer.forget_tlm_equation(i)
      else:
        adjglobals.adjointer.forget_tlm_values(i)

      yield (output.data, tlm_var)

def test_initial_condition_adjoint(J, ic, final_adjoint, seed=0.01, perturbation_direction=None):
  '''forward must be a function that takes in the initial condition (ic) as a dolfin.Function
     and returns the functional value by running the forward run:

       func = J(ic)

     final_adjoint is the adjoint associated with the initial condition
     (usually the last adjoint equation solved).

     This function returns the order of convergence of the Taylor
     series remainder, which should be 2 if the adjoint is working
     correctly.'''

  # We will compute the gradient of the functional with respect to the initial condition,
  # and check its correctness with the Taylor remainder convergence test.
  info_blue("Running Taylor remainder convergence analysis for the adjoint model ... ")
  import random

  # First run the problem unperturbed
  ic_copy = dolfin.Function(ic)
  f_direct = J(ic_copy)

  # Randomise the perturbation direction:
  if perturbation_direction is None:
    perturbation_direction = dolfin.Function(ic.function_space())
    vec = perturbation_direction.vector()
    for i in range(len(vec)):
      vec[i] = random.random()

  # Run the forward problem for various perturbed initial conditions
  functional_values = []
  perturbations = []
  perturbation_sizes = [seed/(2**i) for i in range(5)]
  for perturbation_size in perturbation_sizes:
    perturbation = dolfin.Function(perturbation_direction)
    vec = perturbation.vector()
    vec *= perturbation_size
    perturbations.append(perturbation)

    perturbed_ic = dolfin.Function(ic)
    vec = perturbed_ic.vector()
    vec += perturbation.vector()

    functional_values.append(J(perturbed_ic))

  # First-order Taylor remainders (not using adjoint)
  no_gradient = [abs(perturbed_f - f_direct) for perturbed_f in functional_values]

  info("Taylor remainder without adjoint information: " + str(no_gradient))
  info("Convergence orders for Taylor remainder without adjoint information (should all be 1): " + str(convergence_order(no_gradient)))

  adjoint_vector = final_adjoint.vector()

  with_gradient = []
  gradient_fd   = []
  for i in range(len(perturbations)):
    gradient_fd.append((functional_values[i] - f_direct)/perturbation_sizes[i])

    remainder = abs(functional_values[i] - f_direct - adjoint_vector.inner(perturbations[i].vector()))
    with_gradient.append(remainder)

  info("Taylor remainder with adjoint information: " + str(with_gradient))
  info("Convergence orders for Taylor remainder with adjoint information (should all be 2): " + str(convergence_order(with_gradient)))

  info("Gradients (finite differencing): " + str(gradient_fd))
  info("Gradient (adjoint): " + str(adjoint_vector.inner(perturbation_direction.vector())))

  return min(convergence_order(with_gradient))

def tlm_dolfin(parameter, forget=False):
  for i in range(adjglobals.adjointer.equation_count):
      (tlm_var, output) = adjglobals.adjointer.get_tlm_solution(i, parameter)

      storage = libadjoint.MemoryStorage(output)
      storage.set_overwrite(True)
      adjglobals.adjointer.record_variable(tlm_var, storage)

      if forget is None:
        pass
      elif forget:
        adjglobals.adjointer.forget_tlm_equation(i)
      else:
        adjglobals.adjointer.forget_tlm_values(i)

  return output

def test_initial_condition_tlm(J, dJ, ic, seed=0.01, perturbation_direction=None):
  '''forward must be a function that takes in the initial condition (ic) as a dolfin.Function
     and returns the functional value by running the forward run:

       func = J(ic)

     final_adjoint is the tangent linear variable for the solution on which the functional depends
     (usually the last TLM equation solved).

     dJ must be the derivative of the functional with respect to its argument, evaluated and assembled at
     the unperturbed solution (a dolfin Vector).

     This function returns the order of convergence of the Taylor
     series remainder, which should be 2 if the TLM is working
     correctly.'''

  # We will compute the gradient of the functional with respect to the initial condition,
  # and check its correctness with the Taylor remainder convergence test.
  info_blue("Running Taylor remainder convergence analysis for the tangent linear model... ")
  import random

  adj_var = adjglobals.adj_variables[ic]; adj_var.timestep = 0
  if not adjglobals.adjointer.variable_known(adj_var):
    info_red(str(adj_var) + " not known.")
    raise libadjoint.exceptions.LibadjointErrorInvalidInputs("Your initial condition must be the /exact same Function/ as the initial condition used in the forward model.")

  # First run the problem unperturbed
  ic_copy = dolfin.Function(ic)
  f_direct = J(ic_copy)

  # Randomise the perturbation direction:
  if perturbation_direction is None:
    perturbation_direction = dolfin.Function(ic.function_space())
    vec = perturbation_direction.vector()
    for i in range(len(vec)):
      vec[i] = random.random()

  # Run the forward problem for various perturbed initial conditions
  functional_values = []
  perturbations = []
  for perturbation_size in [seed/(2**i) for i in range(5)]:
    perturbation = dolfin.Function(perturbation_direction)
    vec = perturbation.vector()
    vec *= perturbation_size
    perturbations.append(perturbation)

    perturbed_ic = dolfin.Function(ic)
    vec = perturbed_ic.vector()
    vec += perturbation.vector()

    functional_values.append(J(perturbed_ic))

  # First-order Taylor remainders (not using adjoint)
  no_gradient = [abs(perturbed_f - f_direct) for perturbed_f in functional_values]

  info("Taylor remainder without tangent linear information: " + str(no_gradient))
  info("Convergence orders for Taylor remainder without tangent linear information (should all be 1): " + str(convergence_order(no_gradient)))

  with_gradient = []
  for i in range(len(perturbations)):
    param = InitialConditionParameter(ic, perturbations[i])
    final_tlm = tlm_dolfin(param, forget=False).data
    remainder = abs(functional_values[i] - f_direct - final_tlm.vector().inner(dJ))
    with_gradient.append(remainder)

  info("Taylor remainder with tangent linear information: " + str(with_gradient))
  info("Convergence orders for Taylor remainder with tangent linear information (should all be 2): " + str(convergence_order(with_gradient)))

  return min(convergence_order(with_gradient))

def test_initial_condition_adjoint_cdiff(J, ic, final_adjoint, seed=0.01, perturbation_direction=None):
  '''forward must be a function that takes in the initial condition (ic) as a dolfin.Function
     and returns the functional value by running the forward run:

       func = J(ic)

     final_adjoint is the adjoint associated with the initial condition
     (usually the last adjoint equation solved).

     This function returns the order of convergence of the Taylor
     series remainder of central finite differencing, which should be 3 
     if the adjoint is working correctly.'''

  # We will compute the gradient of the functional with respect to the initial condition,
  # and check its correctness with the Taylor remainder convergence test.
  info_blue("Running central differencing Taylor remainder convergence analysis for the adjoint model ... ")
  import random

  # First run the problem unperturbed
  ic_copy = dolfin.Function(ic)
  f_direct = J(ic_copy)

  # Randomise the perturbation direction:
  if perturbation_direction is None:
    perturbation_direction = dolfin.Function(ic.function_space())
    vec = perturbation_direction.vector()
    for i in range(len(vec)):
      vec[i] = random.random()

  # Run the forward problem for various perturbed initial conditions
  functional_values_plus = []
  functional_values_minus = []
  perturbations = []
  perturbation_sizes = [seed/(2**i) for i in range(4)]
  for perturbation_size in perturbation_sizes:
    perturbation = dolfin.Function(perturbation_direction)
    vec = perturbation.vector()
    vec *= perturbation_size
    perturbations.append(perturbation)

    perturbation = dolfin.Function(perturbation_direction)
    vec = perturbation.vector()
    vec *= perturbation_size/2.0

    perturbed_ic = dolfin.Function(ic)
    vec = perturbed_ic.vector()
    vec += perturbation.vector()
    functional_values_plus.append(J(perturbed_ic))

    perturbed_ic = dolfin.Function(ic)
    vec = perturbed_ic.vector()
    vec -= perturbation.vector()
    functional_values_minus.append(J(perturbed_ic))

  # First-order Taylor remainders (not using adjoint)
  no_gradient = [abs(functional_values_plus[i] - functional_values_minus[i]) for i in range(len(functional_values_plus))]

  info("Taylor remainder without adjoint information: " + str(no_gradient))
  info("Convergence orders for Taylor remainder without adjoint information (should all be 1): " + str(convergence_order(no_gradient)))

  adjoint_vector = final_adjoint.vector()

  with_gradient = []
  gradient_fd   = []
  for i in range(len(perturbations)):
    gradient_fd.append((functional_values_plus[i] - functional_values_minus[i])/perturbation_sizes[i])

    remainder = abs(functional_values_plus[i] - functional_values_minus[i] - adjoint_vector.inner(perturbations[i].vector()))
    with_gradient.append(remainder)

  info("Taylor remainder with adjoint information: " + str(with_gradient))
  info("Convergence orders for Taylor remainder with adjoint information (should all be 3): " + str(convergence_order(with_gradient)))

  info("Gradients (finite differencing): " + str(gradient_fd))
  info("Gradient (adjoint): " + str(adjoint_vector.inner(perturbation_direction.vector())))

  return min(convergence_order(with_gradient))

def compute_gradient(J, param, forget=True):
  dJdparam = None
  last_timestep = adjglobals.adjointer.timestep_count

  for i in range(adjglobals.adjointer.timestep_count):
    adjglobals.adjointer.set_functional_dependencies(J, i)

  for i in range(adjglobals.adjointer.equation_count)[::-1]:
    (adj_var, output) = adjglobals.adjointer.get_adjoint_solution(i, J)

    storage = libadjoint.MemoryStorage(output)
    adjglobals.adjointer.record_variable(adj_var, storage)
    fwd_var = libadjoint.Variable(adj_var.name, adj_var.timestep, adj_var.iteration)

    out = param.inner_adjoint(adjglobals.adjointer, output.data, i, fwd_var)
    dJdparam = _add(dJdparam, out)

    if last_timestep > adj_var.timestep:
      # We have hit a new timestep, and need to compute this timesteps' \partial J/\partial m contribution
      last_timestep = adj_var.timestep
      out = param.partial_derivative(adjglobals.adjointer, J, adj_var.timestep)
      dJdparam = _add(dJdparam, out)

    if forget is None:
      pass
    elif forget:
      adjglobals.adjointer.forget_adjoint_equation(i)
    else:
      adjglobals.adjointer.forget_adjoint_values(i)

  return dJdparam

def test_scalar_parameter_adjoint(J, a, dJda, seed=None):
  info_blue("Running Taylor remainder convergence analysis for the adjoint model ... ")

  functional_values = []
  f_direct = J(a)

  if seed is None:
    seed = float(a)/5.0
    if seed == 0.0:
      seed = 0.1

  perturbations = [seed / (2**i) for i in range(5)]

  for da in (dolfin.Constant(float(a) + x) for x in perturbations):
    functional_values.append(J(da))

  # First-order Taylor remainders (not using adjoint)
  no_gradient = [abs(perturbed_f - f_direct) for perturbed_f in functional_values]

  info("Taylor remainder without adjoint information: " + str(no_gradient))
  info("Convergence orders for Taylor remainder without adjoint information (should all be 1): " + str(convergence_order(no_gradient)))

  with_gradient = []
  gradient_fd   = []
  for i in range(len(perturbations)):
    gradient_fd.append((functional_values[i] - f_direct)/perturbations[i])

    remainder = abs(functional_values[i] - f_direct - dJda*perturbations[i])
    with_gradient.append(remainder)

  info("Taylor remainder with adjoint information: " + str(with_gradient))
  info("Convergence orders for Taylor remainder with adjoint information (should all be 2): " + str(convergence_order(with_gradient)))

  info("Gradients (finite differencing): " + str(gradient_fd))
  info("Gradient (adjoint): " + str(dJda))

  return min(convergence_order(with_gradient))

def test_scalar_parameters_adjoint(J, a, dJda, seed=0.1):
  info_blue("Running Taylor remainder convergence analysis for the adjoint model ... ")

  functional_values = []
  f_direct = J(a)

  a = numpy.array([float(x) for x in a])
  dJda = numpy.array(dJda)

  perturbation_direction = a/5.0
  perturbation_sizes = [seed / (2**i) for i in range(5)]
  perturbations = [a * i for i in perturbation_sizes]
  for x in perturbations:
    da = [dolfin.Constant(a[i] + x[i]) for i in range(len(a))]
    functional_values.append(J(da))

  # First-order Taylor remainders (not using adjoint)
  no_gradient = [abs(perturbed_f - f_direct) for perturbed_f in functional_values]

  info("Taylor remainder without adjoint information: " + str(no_gradient))
  info("Convergence orders for Taylor remainder without adjoint information (should all be 1): " + str(convergence_order(no_gradient)))

  with_gradient = []
  for i in range(len(perturbations)):
    remainder = abs(functional_values[i] - f_direct - numpy.dot(dJda, perturbations[i]))
    with_gradient.append(remainder)

  info("Taylor remainder with adjoint information: " + str(with_gradient))
  info("Convergence orders for Taylor remainder with adjoint information (should all be 2): " + str(convergence_order(with_gradient)))

  return min(convergence_order(with_gradient))

def test_gradient_array(J, dJdx, x, seed = 0.01, perturbation_direction = None):
  '''Checks the correctness of the derivative dJ.
     x must be an array that specifies at which point in the parameter space
     the gradient is to be checked, and dJdx must be an array containing the gradient. 
     The function J(x) must return the functional value. 

     This function returns the order of convergence of the Taylor
     series remainder, which should be 2 if the gradient is correct.'''

  import random
  # We will compute the gradient of the functional with respect to the initial condition,
  # and check its correctness with the Taylor remainder convergence test.
  info("Running Taylor remainder convergence analysis to check the gradient ... ")

  # First run the problem unperturbed
  j_direct = J(x)

  # Randomise the perturbation direction:
  if perturbation_direction is None:
    perturbation_direction = x.copy()
    for i in range(len(x)):
      perturbation_direction[i] = random.random()

  # Run the forward problem for various perturbed initial conditions
  functional_values = []
  perturbations = []
  perturbation_sizes = [seed/(2**i) for i in range(5)]
  for perturbation_size in perturbation_sizes:
    perturbation = perturbation_direction.copy() * perturbation_size
    perturbations.append(perturbation)

    perturbed_x = x.copy() + perturbation 
    functional_values.append(J(perturbed_x))

  # First-order Taylor remainders (not using adjoint)
  no_gradient = [abs(perturbed_j - j_direct) for perturbed_j in functional_values]

  info("Absolute functional evaluation differences: %s" % str(no_gradient))
  info("Convergence orders for Taylor remainder without adjoint information (should all be 1): %s" % str(convergence_order(no_gradient)))

  with_gradient = []
  for i in range(len(perturbations)):
      remainder = abs(functional_values[i] - j_direct - numpy.dot(perturbations[i], dJdx))
      with_gradient.append(remainder)

  info("Absolute functional evaluation differences with adjoint: %s" % str(with_gradient))
  info("Convergence orders for Taylor remainder with adjoint information (should all be 2): %s" % str(convergence_order(with_gradient)))

  return min(convergence_order(with_gradient))

def taylor_test(J, m, Jm, dJdm, seed=None, perturbation_direction=None, value=None):
  '''J must be a function that takes in a parameter value m and returns the value
     of the functional:

       func = J(m)

     Jm is the value of the function J at the parameter m. 
     dJdm is the gradient of J evaluated at m, to be tested for correctness.

     This function returns the order of convergence of the Taylor
     series remainder, which should be 2 if the adjoint is working
     correctly.'''

  info_blue("Running Taylor remainder convergence test ... ")
  import random

  def get_const(val):
    if isinstance(val, str):
      return float(constant.constant_values[val])
    else:
      return float(val)

  def get_value(param, value):
    if value is not None:
      return value
    else:
      try:
        return adjglobals.adjointer.get_variable_value(param.var).data
      except libadjoint.exceptions.LibadjointErrorNeedValue:
        info_red("Do you need to pass forget=False to compute_gradient?")
        raise

  # First, compute perturbation sizes.
  if seed is None:
    if isinstance(m, ScalarParameter):
      seed = get_const(m.a)/5.0

      if seed == 0.0: seed = 0.1
    else:
      seed = 0.01

  perturbation_sizes = [seed/(2**i) for i in range(5)]

  # Next, compute the perturbation direction.
  if perturbation_direction is None:
    if isinstance(m, ScalarParameter):
      perturbation_direction = 1
    elif isinstance(m, ScalarParameters):
      perturbation_direction = numpy.array([get_const(x)/5.0 for x in m.v])
    elif isinstance(m, InitialConditionParameter):
      ic = get_value(m, value)
      perturbation_direction = dolfin.Function(ic)
      vec = perturbation_direction.vector()
      for i in range(len(vec)):
        vec[i] = random.random()
    else:
      raise libadjoint.exceptions.LibadjointErrorNotImplemented("Don't know how to compute a perturbation direction")

  # So now compute the perturbations:
  if not isinstance(perturbation_direction, dolfin.Function):
    perturbations = [x*perturbation_direction for x in perturbation_sizes]
  else:
    perturbations = []
    for x in perturbation_sizes:
      perturbation = dolfin.Function(perturbation_direction)
      vec = perturbation.vector()
      vec *= x
      perturbations.append(perturbation)

  # And now the perturbed inputs:
  if isinstance(m, ScalarParameter):
    pinputs = [dolfin.Constant(get_const(m.a) + x) for x in perturbations]
  elif isinstance(m, ScalarParameters):
    a = numpy.array([get_const(x) for x in m.v])

    def make_const(arr):
      return [dolfin.Constant(x) for x in arr]

    pinputs = [make_const(a + x) for x in perturbations]
  elif isinstance(m, InitialConditionParameter):
    pinputs = []
    for x in perturbations:
      pinput = dolfin.Function(x)
      pinput.vector()[:] += ic.vector()
      pinputs.append(pinput)

  # At last: the common bit!
  functional_values = []
  for pinput in pinputs:
    Jp = J(pinput)
    functional_values.append(Jp)

  # First-order Taylor remainders (not using adjoint)
  no_gradient = [abs(perturbed_J - Jm) for perturbed_J in functional_values]

  info("Taylor remainder without adjoint information: " + str(no_gradient))
  info("Convergence orders for Taylor remainder without adjoint information (should all be 1): " + str(convergence_order(no_gradient)))

  with_gradient = []
  if isinstance(m, ScalarParameter):
    for i in range(len(perturbations)):
      remainder = abs(functional_values[i] - Jm - dJdm*perturbations[i])
      with_gradient.append(remainder)
  elif isinstance(m, ScalarParameters):
    for i in range(len(perturbations)):
      remainder = abs(functional_values[i] - Jm - numpy.dot(dJdm, perturbations[i]))
      with_gradient.append(remainder)
  elif isinstance(m, InitialConditionParameter):
    for i in range(len(perturbations)):
      remainder = abs(functional_values[i] - Jm - dJdm.vector().inner(perturbations[i].vector()))
      with_gradient.append(remainder)

  info("Taylor remainder with adjoint information: " + str(with_gradient))
  info("Convergence orders for Taylor remainder with adjoint information (should all be 2): " + str(convergence_order(with_gradient)))

  return min(convergence_order(with_gradient))

def estimate_error(J, forget=True):
  err = 0.0
  i = adjglobals.adjointer.equation_count - 1

  for (adj, var) in compute_adjoint(J, forget=forget):
    form = adjresidual.get_residual(i)
    if form is not None:
      Vplus = dolfin.increase_order(adj.function_space())
      adj_h = dolfin.Function(Vplus)
      adj_h.extrapolate(adj)

      args = ufl.algorithms.extract_arguments(form)
      assert len(args) == 1
      estimator = dolfin.replace(form, {args[0]: adj_h})
      err += dolfin.assemble(estimator)

    i = i - 1

  return err

def _add(value, increment):
  # Add increment to value correctly taking into account None.
  if increment is None:
    return value
  elif value is None:
    return increment
  else:
    return value+increment

