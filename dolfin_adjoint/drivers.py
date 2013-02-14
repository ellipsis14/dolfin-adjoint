import libadjoint
from parameter import *
from dolfin import info_red, info_blue, info
import adjglobals
import dolfin
import numpy
import constant
import adjresidual
import ufl.algorithms

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

def compute_adjoint(functional, forget=True, ignore=[]):

  ignorelist = []
  for fn in ignore:
    if isinstance(fn, dolfin.Function):
      ignorelist.append(adjglobals.adj_variables[fn])
    elif isinstance(fn, str):
      ignorelist.append(libadjoint.Variable(fn, 0, 0))
    else:
      ignorelist.append(fn)

  for i in range(adjglobals.adjointer.timestep_count):
    adjglobals.adjointer.set_functional_dependencies(functional, i)

  for i in range(adjglobals.adjointer.equation_count)[::-1]:
      fwd_var = adjglobals.adjointer.get_forward_variable(i)
      if fwd_var in ignorelist:
        info("Ignoring the adjoint equation for %s" % fwd_var)
        continue

      (adj_var, output) = adjglobals.adjointer.get_adjoint_solution(i, functional)

      storage = libadjoint.MemoryStorage(output)
      storage.set_overwrite(True)
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
      # forget is False: forget only unnecessary tlm values
      if forget is None:
        pass
      elif forget:
        adjglobals.adjointer.forget_tlm_equation(i)
      else:
        adjglobals.adjointer.forget_tlm_values(i)

      yield (output.data, tlm_var)

def compute_gradient(J, param, forget=True, ignore=[], callback=lambda var, output: None):
  try:
    scalar = False
    dJdparam = [None for i in range(len(param))]
    lparam = param
  except TypeError:
    scalar = True
    dJdparam = [None]
    lparam = [param]
  last_timestep = adjglobals.adjointer.timestep_count

  ignorelist = []
  for fn in ignore:
    if isinstance(fn, dolfin.Function):
      ignorelist.append(adjglobals.adj_variables[fn])
    elif isinstance(fn, str):
      ignorelist.append(libadjoint.Variable(fn, 0, 0))
    else:
      ignorelist.append(fn)

  for i in range(adjglobals.adjointer.timestep_count):
    adjglobals.adjointer.set_functional_dependencies(J, i)

  for i in range(adjglobals.adjointer.equation_count)[::-1]:
    fwd_var = adjglobals.adjointer.get_forward_variable(i)
    if fwd_var in ignorelist:
      info("Ignoring the adjoint equation for %s" % fwd_var)
      continue

    (adj_var, output) = adjglobals.adjointer.get_adjoint_solution(i, J)

    callback(adj_var, output.data)

    storage = libadjoint.MemoryStorage(output)
    adjglobals.adjointer.record_variable(adj_var, storage)
    fwd_var = libadjoint.Variable(adj_var.name, adj_var.timestep, adj_var.iteration)

    for j in range(len(lparam)):
      out = lparam[j].inner_adjoint(adjglobals.adjointer, output.data, i, fwd_var)
      dJdparam[j] = _add(dJdparam[j], out)

      if last_timestep > adj_var.timestep:
        # We have hit a new timestep, and need to compute this timesteps' \partial J/\partial m contribution
        last_timestep = adj_var.timestep
        out = lparam[j].partial_derivative(adjglobals.adjointer, J, adj_var.timestep)
        dJdparam[j] = _add(dJdparam[j], out)

    if forget is None:
      pass
    elif forget:
      adjglobals.adjointer.forget_adjoint_equation(i)
    else:
      adjglobals.adjointer.forget_adjoint_values(i)

  if scalar:
    return dJdparam[0]
  else:
    return dJdparam

class hessian(object):
  def __init__(self, J, m):
    self.J = J
    self.m = m

    dolfin.info_red("Warning: Hessian computation is still experimental and is known to not work for some problems. Please Taylor test thoroughly.")

    if not isinstance(m, (InitialConditionParameter, ScalarParameter)):
      raise libadjoint.exceptions.LibadjointErrorNotImplemented("Sorry, Hessian computation only works for InitialConditionParameter|SteadyParameter|TimeConstantParameter|ScalarParameter so far.")

    if isinstance(m, ScalarParameter):
      dolfin.info_red("Warning: Hessian computation with ScalarParameter will only work if your equations depend *linearly* on your parameter. This is not checked.")

  def __call__(self, m_dot):

    self.m_p = self.m.set_perturbation(m_dot)

    if hasattr(m_dot, 'function_space'):
      Hm = dolfin.Function(m_dot.function_space())
    elif isinstance(m_dot, float):
      Hm = 0.0
    else:
      raise NotImplementedError("Sorry, don't know how to handle this")

    # run the tangent linear model
    for output in compute_tlm(self.m_p, forget=None):
      pass

    # run the adjoint and second-order adjoint equations.
    i = adjglobals.adjointer.equation_count
    for (adj, adj_var) in compute_adjoint(self.J, forget=None):
      i = i - 1
      (soa_var, soa_vec) = adjglobals.adjointer.get_soa_solution(i, self.J, self.m_p)
      soa = soa_vec.data

      # now implement the Hessian action formula.
      out = self.m.inner_adjoint(adjglobals.adjointer, soa, i, soa_var.to_forward())
      if out is not None:
        if isinstance(Hm, dolfin.Function):
          Hm.vector().axpy(1.0, out.vector())
        elif isinstance(Hm, float):
          Hm += out

      storage = libadjoint.MemoryStorage(soa_vec)
      storage.set_overwrite(True)
      adjglobals.adjointer.record_variable(soa_var, storage)

    return Hm

def _add(value, increment):
  # Add increment to value correctly taking into account None.
  if increment is None:
    return value
  elif value is None:
    return increment
  else:
    return value+increment

