import adjglobals
import adjlinalg
import libadjoint
import dolfin

def compute_gst(ic, final, nsv, ic_norm=None, final_norm="mass"):
  ic_var = adjglobals.adj_variables[ic]; ic_var.c_object.timestep = 0; ic_var.c_object.iteration = 0
  final_var = adjglobals.adj_variables[final]

  if final_norm == "mass":
    final_value = adjglobals.adjointer.get_variable_value(final_var).data
    final_fnsp  = final_value.function_space()
    u = dolfin.TrialFunction(final_fnsp)
    v = dolfin.TestFunction(final_fnsp)
    final_mass = dolfin.inner(u, v)*dolfin.dx
    final_norm = adjlinalg.Matrix(final_mass)
    print "compute_gst: final_norm.__class__ == ", final_norm.__class__

  return adjglobals.adjointer.compute_gst(ic_var, ic_norm, final_var, final_norm, nsv)

orig_get_gst = libadjoint.GSTHandle.get_gst
def new_get_gst(self, *args, **kwargs):
  '''Process the output of get_gst to return dolfin.Function's instead of adjlinalg.Vector's.'''
  retvals = orig_get_gst(self, *args, **kwargs)
  new_retvals = []
  for retval in retvals:
    if isinstance(retval, adjlinalg.Vector):
      new_retvals.append(retval.data)
    else:
      new_retvals.append(retval)

  return new_retvals
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
