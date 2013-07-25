import libadjoint
from parameter import *
from dolfin import info_red, info_blue, info
import adjglobals
import dolfin
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
      if output.data:
        output.data.rename(str(adj_var), "a Function from dolfin-adjoint")

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
      if output.data:
        output.data.rename(str(tlm_var), "a Function from dolfin-adjoint")

      storage = libadjoint.MemoryStorage(output)
      storage.set_overwrite(True)
      adjglobals.adjointer.record_variable(tlm_var, storage)

      yield (output.data, tlm_var)

      # forget is None: forget *nothing*.
      # forget is True: forget everything we can, forward and adjoint
      # forget is False: forget only unnecessary tlm values
      if forget is None:
        pass
      elif forget:
        adjglobals.adjointer.forget_tlm_equation(i)
      else:
        adjglobals.adjointer.forget_tlm_values(i)


def compute_gradient(J, param, forget=True, ignore=[], callback=lambda var, output: None, project=False):
  dolfin.parameters["adjoint"]["stop_annotating"] = True

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
    storage.set_overwrite(True)
    adjglobals.adjointer.record_variable(adj_var, storage)
    fwd_var = libadjoint.Variable(adj_var.name, adj_var.timestep, adj_var.iteration)

    for j in range(len(lparam)):
      out = lparam[j].equation_partial_derivative(adjglobals.adjointer, output.data, i, fwd_var)
      dJdparam[j] = _add(dJdparam[j], out)

      if last_timestep > adj_var.timestep:
        # We have hit a new timestep, and need to compute this timesteps' \partial J/\partial m contribution
        out = lparam[j].functional_partial_derivative(adjglobals.adjointer, J, adj_var.timestep)
        dJdparam[j] = _add(dJdparam[j], out)

    last_timestep = adj_var.timestep

    if forget is None:
      pass
    elif forget:
      adjglobals.adjointer.forget_adjoint_equation(i)
    else:
      adjglobals.adjointer.forget_adjoint_values(i)

  def project_test(func):
    if isinstance(func, dolfin.Function):
      V = func.function_space()
      u = dolfin.TrialFunction(V)
      v = dolfin.TestFunction(V)
      M = dolfin.assemble(dolfin.inner(u, v)*dolfin.dx)
      proj = dolfin.Function(V)
      dolfin.solve(M, proj.vector(), func.vector())
      return proj
    else:
      return func


  for i, parameter in enumerate(lparam):
    if isinstance(dJdparam[i], dolfin.Function):
      dJdparam[i].rename("d(%s)/d(%s)" % (str(J), str(parameter)), "a Function from dolfin-adjoint")

  return postprocess(dJdparam, project)

def postprocess(dJdparam, project):
  if project:
    dJdparam = map(project_test, dJdparam)

  dJdparam = [dolfin.Constant(x) if isinstance(x, float) else x for x in dJdparam]

  if len(dJdparam) == 1:
    return dJdparam[0]
  else:
    return dJdparam

def hessian(J, m, warn=True):
  '''Choose which Hessian the user wants.'''
  dolfin.parameters["adjoint"]["stop_annotating"] = True
  return BasicHessian(J, m, warn=warn)

class BasicHessian(libadjoint.Matrix):
  '''A basic implementation of the Hessian class that recomputes the tangent linear, adjoint and second-order adjoint
  equations on each action. Should be the slowest, but safest, with the lowest memory requirements.'''
  def __init__(self, J, m, warn=True):
    self.J = J
    self.m = m

    if warn:
      dolfin.info_red("Warning: Hessian computation is still experimental and is known to not work for some problems. Please Taylor test thoroughly.")

    if not isinstance(m, (InitialConditionParameter, ScalarParameter)):
      raise libadjoint.exceptions.LibadjointErrorNotImplemented("Sorry, Hessian computation only works for InitialConditionParameter|SteadyParameter|TimeConstantParameter|ScalarParameter so far.")

    self.update(m)

  def update(self, m):
    pass

  def __call__(self, m_dot):

    hess_action_timer = dolfin.Timer("Hessian action")

    m_p = self.m.set_perturbation(m_dot)
    last_timestep = adjglobals.adjointer.timestep_count

    if hasattr(m_dot, 'function_space'):
      Hm = dolfin.Function(m_dot.function_space())
    elif isinstance(m_dot, float):
      Hm = 0.0
    else:
      raise NotImplementedError("Sorry, don't know how to handle this")

    tlm_timer = dolfin.Timer("Hessian action (TLM)")
    # run the tangent linear model
    for (tlm, tlm_var) in compute_tlm(m_p, forget=None):
      pass

    tlm_timer.stop()

    # run the adjoint and second-order adjoint equations.
    for i in range(adjglobals.adjointer.equation_count)[::-1]:
      adj_var = adjglobals.adjointer.get_forward_variable(i).to_adjoint(self.J)
      # Only recompute the adjoint variable if we do not have it yet
      try:
        adj = adjglobals.adjointer.get_variable_value(adj_var)
      except (libadjoint.exceptions.LibadjointErrorHashFailed, libadjoint.exceptions.LibadjointErrorNeedValue):
        adj_timer = dolfin.Timer("Hessian action (ADM)")
        adj = adjglobals.adjointer.get_adjoint_solution(i, self.J)[1]
        adj_timer.stop()

        storage = libadjoint.MemoryStorage(adj)
        adjglobals.adjointer.record_variable(adj_var, storage)

      adj = adj.data
      
      soa_timer = dolfin.Timer("Hessian action (SOA)")
      (soa_var, soa_vec) = adjglobals.adjointer.get_soa_solution(i, self.J, m_p)
      soa_timer.stop()
      soa = soa_vec.data

      func_timer = dolfin.Timer("Hessian action (derivative formula)")
      # now implement the Hessian action formula.
      out = self.m.equation_partial_derivative(adjglobals.adjointer, soa, i, soa_var.to_forward())
      if out is not None:
        if isinstance(Hm, dolfin.Function):
          Hm.vector().axpy(1.0, out.vector())
        elif isinstance(Hm, float):
          Hm += out

      out = self.m.equation_partial_second_derivative(adjglobals.adjointer, adj, i, soa_var.to_forward(), m_dot)
      if out is not None:
        if isinstance(Hm, dolfin.Function):
          Hm.vector().axpy(1.0, out.vector())
        elif isinstance(Hm, float):
          Hm += out

      if last_timestep > adj_var.timestep:
        # We have hit a new timestep, and need to compute this timesteps' \partial^2 J/\partial m^2 contribution
        last_timestep = adj_var.timestep
        out = self.m.functional_partial_second_derivative(adjglobals.adjointer, self.J, adj_var.timestep, m_dot)
        if out is not None:
          if isinstance(Hm, dolfin.Function):
            Hm.vector().axpy(1.0, out.vector())
          elif isinstance(Hm, float):
            Hm += out

      func_timer.stop()

      storage = libadjoint.MemoryStorage(soa_vec)
      storage.set_overwrite(True)
      adjglobals.adjointer.record_variable(soa_var, storage)

    if isinstance(Hm, dolfin.Function):
      Hm.rename("d^2(%s)/d(%s)^2" % (str(self.J), str(self.m)), "a Function from dolfin-adjoint")

    return Hm

  def action(self, x, y):
    assert isinstance(x.data, dolfin.Function)
    assert isinstance(y.data, dolfin.Function)

    Hm = self.__call__(x.data)
    y.data.assign(Hm)

  def eigendecomposition(self, **kwargs):
    '''Compute the eigendecomposition of the Hessian.'''

    params = {'solver': 'krylovschur',
              'spectrum': 'largest magnitude',
              'type': 'hermitian',
              'monitor': True,
              'n': 1}

    params.update(kwargs)

    # We take in the options in "DOLFIN" syntax (the same
    # as SLEPcEigenSolver. libadjoint uses a different
    # syntax for the same things. Here we translate.
    # Sorry for the confusion.

    pairs = {'method': 'solver',
             'type': 'type',
             'which': 'spectrum',
             'monitor': 'monitor',
             'neigenpairs': 'n'}

    options = {}
    for key in pairs:
      options[key] = params[pairs[key]]

    # OK! Now add the model input and output vectors.
    data = adjlinalg.Vector(self.m.data())
    options['input'] = data
    options['output'] = data

    eps = adjglobals.adjointer.compute_eps(self, options)

    retval = []
    for i in range(eps.ncv):
      (lamda, u) = eps.get_eps(i)
      u.data.rename("eigenvector %s of d^2(%s)/d(%s)^2" % (i, str(self.J), str(self.m)), "a Function from dolfin-adjoint")
      retval += [(lamda, u.data)]

    return retval

def _add(value, increment):
  # Add increment to value correctly taking into account None.
  if increment is None:
    return value
  elif value is None:
    return increment
  else:
    return value+increment

class compute_gradient_tlm(object):
  '''Rather than compute the gradient of a functional all at once with
  the adjoint, compute its action with the tangent linear model. Useful
  for testing tangent linear models, and might be useful in future where
  you have many functionals and few parameters.'''
  def __init__(self, J, m, forget=True, callback=lambda var, output: None, project=False):
    self.J = J
    self.m = m
    self.forget = forget
    self.cb = callback
    self.project = project

    if not isinstance(m, InitialConditionParameter):
      raise Exception("Sorry, only works for InitialConditionParameter at the minute")

  def vector(self):
    return self

  def inner(self, vec):
    '''Compute the action of the gradient on the vector vec.'''
    mdot = self.m.set_perturbation(dolfin.Function(self.m.data().function_space(), vec))
    
    grad = 0.0

    for (tlm, tlm_var) in compute_tlm(mdot, forget=self.forget):
      self.cb(tlm_var, tlm)
      fwd_var = tlm_var.to_forward()
      dJdu = adjglobals.adjointer.evaluate_functional_derivative(self.J, fwd_var)
      if dJdu is not None:
        dJdu_vec = dolfin.assemble(dJdu.data)
        grad = _add(grad, dJdu_vec.inner(tlm.vector()))

      # skip the dJdm term for now, don't need it for InitialConditionParameter you see

    return grad
