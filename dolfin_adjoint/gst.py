import adjglobals
import adjlinalg
import libadjoint
import dolfin

def compute_gst(ic, final, nsv, ic_norm="mass", final_norm="mass"):
  '''This function computes the generalised stability analysis of a simulation.
  Generalised stability theory computes the perturbations to a field (such as an
  initial condition, forcing term, etc.) that /grow the most/ over the finite
  time window of the simulation. For more details, see the mathematical documentation
  on `the website <http://dolfin-adjoint.org>`_.

  :py:data:`ic` -- the input of the propagator
  :py:data:`final` -- the output of the propagator
  :py:data:`nsv` -- the number of optimal perturbations to compute
  :py:data:`ic_norm` -- a symmetric positive-definite bilinear form that defines the norm on the input space
  :py:data:`final_norm` -- a symmetric positive-definite bilinear form that defines the norm on the output space

  You can supply :py:data:`"mass"` for :py:data:`ic_norm` and :py:data:`final_norm` to use the (default) mass matrices associated
  with these spaces.

  For example:

  .. code-block:: python

    gst = compute_gst("State", "State", nsv=10)
    for i in range(gst.ncv): # number of converged vectors
      (sigma, u, v) = gst.get_gst(i, return_vectors=True)
  '''

  ic_var = adjglobals.adj_variables[ic]; ic_var.c_object.timestep = 0; ic_var.c_object.iteration = 0
  final_var = adjglobals.adj_variables[final]

  if final_norm == "mass":
    final_value = adjglobals.adjointer.get_variable_value(final_var).data
    final_fnsp  = final_value.function_space()
    u = dolfin.TrialFunction(final_fnsp)
    v = dolfin.TestFunction(final_fnsp)
    final_mass = dolfin.inner(u, v)*dolfin.dx
    final_norm = adjlinalg.Matrix(final_mass)
  elif final_norm is not None:
    final_norm = adjlinalg.Matrix(final_norm)

  if ic_norm == "mass":
    ic_value = adjglobals.adjointer.get_variable_value(ic_var).data
    ic_fnsp  = ic_value.function_space()
    u = dolfin.TrialFunction(ic_fnsp)
    v = dolfin.TestFunction(ic_fnsp)
    ic_mass = dolfin.inner(u, v)*dolfin.dx
    ic_norm = adjlinalg.Matrix(ic_mass)
  elif ic_norm is not None:
    ic_norm = adjlinalg.Matrix(ic_norm)

  return adjglobals.adjointer.compute_gst(ic_var, ic_norm, final_var, final_norm, nsv)

orig_get_gst = libadjoint.GSTHandle.get_gst
def new_get_gst(self, *args, **kwargs):
  '''Process the output of get_gst to return dolfin.Function's instead of adjlinalg.Vector's.'''
  retvals = orig_get_gst(self, *args, **kwargs)
  new_retvals = []
  try:
    for retval in retvals:
      if isinstance(retval, adjlinalg.Vector):
        new_retvals.append(retval.data)
      else:
        new_retvals.append(retval)

    return new_retvals

  except TypeError:
    return retvals

libadjoint.GSTHandle.get_gst = new_get_gst

def compute_propagator_matrix(gst):
  # Warning: for testing purposes only -- it's far too expensive to do on big models.
  # This also only works in serial.

  import numpy

  (sigma, u, v) = gst.get_gst(0, return_vectors=True)
  (u, v) = (u.vector().array(), v.vector().array())

  mat = sigma * numpy.outer(u, v)

  for i in range(1, gst.ncv):
    (sigma, u, v) = gst.get_gst(i, return_vectors=True)
    (u, v) = (u.vector().array(), v.vector().array())

    sum_mat = sigma * numpy.outer(u, v)
    mat += sum_mat

  return mat

def perturbed_replay(parameter, perturbation, perturbation_norm, perturbation_scale, observation, observation_norm):
  r"""Perturb the forward run and compute

  .. math::

    \frac{
    \left|\left| \delta \mathrm{observation} \right|\right| 
    }{
    \left|\left| \delta \mathrm{input} \right| \right|
    }

  as a function of time.

  :py:data:`parameter` -- an InitialConditionParameter to say what variable should be perturbed (e.g. InitialConditionParameter('InitialConcentration'))
  :py:data:`perturbation` -- a Function to give the perturbation direction (from a GST analysis, for example)
  :py:data:`perturbation_norm` -- a bilinear Form which induces a norm on the space of perturbation inputs
  :py:data:`perturbation_scale` -- how big the norm of the initial perturbation should be
  :py:data:`observation` -- the variable to observe (e.g. 'Concentration')
  :py:data:`observation_norm` -- a bilinear Form which induces a norm on the space of perturbation outputs
  """

  if not dolfin.parameters["adjoint"]["record_all"]:
    info_red("Warning: your replay test will be much more effective with dolfin.parameters['adjoint']['record_all'] = True.")

  assert isinstance(parameter, InitialConditionParameter)

  perturbation_norm = assemble(perturbation_norm)

  def compute_norm(perturbation, norm):
    # Need to compute <x, Ax> and then take its sqrt
    # where x is perturbation, A is norm
    try:
      vec = perturbation.vector()
    except:
      vec = perturbation

    Ax = norm.mult(vec)
    xAx = vec.inner(Ax)
    return sqrt(xAx)

  growths = []

  for i in range(adjglobals.adjointer.equation_count):
      (fwd_var, output) = adjglobals.adjointer.get_forward_solution(i)

      if fwd_var == parameter.var: # we've hit the initial condition we want to perturb
        current_norm = compute_norm(perturbation, perturbation_norm)
        output.data.vector()[:] += (perturbation_scale/current_norm) * perturbation.vector()

      if fwd_var.name == observation: # we've hit something we want to observe
        # Fetch the unperturbed result from the record
        unperturbed = adjglobals.adjointer.get_variable_value(fwd_var).data
        diff = output.data.vector() - unperturbed.vector()
        growths.append(compute_norm(diff, observation_norm)/perturbation_scale) # <--- the action line

      storage = libadjoint.MemoryStorage(output)
      storage.set_compare(tol=None)
      storage.set_overwrite(True)
      out = adjglobals.adjointer.record_variable(fwd_var, storage)

      if forget:
        adjglobals.adjointer.forget_forward_equation(i)

  return growths

