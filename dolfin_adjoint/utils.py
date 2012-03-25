import libadjoint
from solving import *
from parameter import *
from dolfin import info_red, info_blue, info
import ufl
import ufl.algorithms
import dolfin

def replay_dolfin(forget=False, tol=0.0):
  if "record_all" not in debugging or debugging["record_all"] is not True:
    info_red("Warning: your replay test will be much more effective with debugging['record_all'] = True.")

  success = True
  for i in range(adjointer.equation_count):
      (fwd_var, output) = adjointer.get_forward_solution(i)

      storage = libadjoint.MemoryStorage(output)
      storage.set_compare(tol=tol)
      storage.set_overwrite(True)
      out = adjointer.record_variable(fwd_var, storage)
      success = success and out

      if forget:
        adjointer.forget_forward_equation(i)

  return success

def convergence_order(errors):
  import math

  orders = [0.0] * (len(errors)-1)
  for i in range(len(errors)-1):
    orders[i] = math.log(errors[i]/errors[i+1], 2)

  return orders

def adjoint_dolfin(functional, forget=True):

  for i in range(adjointer.equation_count)[::-1]:
      (adj_var, output) = adjointer.get_adjoint_solution(i, functional)
      
      storage = libadjoint.MemoryStorage(output)
      adjointer.record_variable(adj_var, storage)

      if forget:
        adjointer.forget_adjoint_equation(i)
      else:
        adjointer.forget_adjoint_values(i)

  return output.data # return the last adjoint state

def compute_adjoint(functional, forget=True):

  for i in range(adjointer.equation_count)[::-1]:
      (adj_var, output) = adjointer.get_adjoint_solution(i, functional)

      storage = libadjoint.MemoryStorage(output)
      adjointer.record_variable(adj_var, storage)

      if forget:
        adjointer.forget_adjoint_equation(i)
      else:
        adjointer.forget_adjoint_values(i)

      yield (output.data, adj_var)

def compute_tlm(parameter, forget=False):

  for i in range(adjointer.equation_count):
      (tlm_var, output) = adjointer.get_tlm_solution(i, parameter)

      storage = libadjoint.MemoryStorage(output)
      storage.set_overwrite(True)
      adjointer.record_variable(tlm_var, storage)

      if forget:
        adjointer.forget_tlm_equation(i)
      else:
        adjointer.forget_tlm_values(i)

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
  for i in range(adjointer.equation_count):
      (tlm_var, output) = adjointer.get_tlm_solution(i, parameter)

      storage = libadjoint.MemoryStorage(output)
      storage.set_overwrite(True)
      adjointer.record_variable(tlm_var, storage)

      if forget:
        adjointer.forget_tlm_equation(i)
      else:
        adjointer.forget_tlm_values(i)

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

  adj_var = adj_variables[ic]; adj_var.timestep = 0
  if not adjointer.variable_known(adj_var):
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

  for i in range(adjointer.equation_count)[::-1]:
    (adj_var, output) = adjointer.get_adjoint_solution(i, J)

    storage = libadjoint.MemoryStorage(output)
    adjointer.record_variable(adj_var, storage)

    (fwd_var, lhs, rhs) = adjointer.get_forward_equation(i)
    lhs = lhs.data; rhs = rhs.data; adjoint = output.data

    if not isinstance(lhs, ufl.Identity):
      fn_space = ufl.algorithms.extract_arguments(rhs)[0].function_space()
      x = dolfin.Function(fn_space)
      form = rhs - dolfin.action(lhs, x)

      if isinstance(param, dolfin.Constant):
        dparam = dolfin.Function(fn_space)
      else:
        dparam = None

      diff_form = dolfin.derivative(form, param, dparam)

      if x in ufl.algorithms.extract_coefficients(diff_form):
        #raise libadjoint.exceptions.LibadjointErrorNotImplemented("Sorry, the dF/dm term can't depend on the solution (yet!)")
        info_red("Warning: assuming dF/dm does not actually depend on the forward solution.")

      dFdm = dolfin.assemble(diff_form) # actually - dF/dm
      print "dFdm: ", list(dFdm)
      if isinstance(dFdm, dolfin.GenericVector):
        if dJdparam is None:
          dJdparam = dFdm.inner(adjoint.vector())
        else:
          dJdparam += dFdm.inner(adjoint.vector())
      elif isinstance(dFdm, dolfin.GenericMatrix):
        if dJdparam is None:
          dJdparam = dFdm.mult(adjoint.vector())
        else:
          dJdparam += dFdm.mult(adjoint.vector())
      else:
        raise libadjoint.exceptions.LibadjointErrorNotImplemented("Sorry, don't know how to handle anything else!")

    if forget:
      adjointer.forget_adjoint_equation(i)
    else:
      adjointer.forget_adjoint_values(i)

  return dJdparam

