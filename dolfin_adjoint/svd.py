import adjglobals
import adjlinalg
import libadjoint

def adj_compute_propagator_svd(ic, final, nsv):
  ic_var = adjglobals.adj_variables[ic]; ic_var.c_object.timestep = 0; ic_var.c_object.iteration = 0
  final_var = adjglobals.adj_variables[final]

  return adjglobals.adjointer.compute_propagator_svd(ic_var, final_var, nsv)

orig_get_svd = libadjoint.SVDHandle.get_svd
def new_get_svd(self, *args, **kwargs):
  '''Process the output of get_svd to return dolfin.Function's instead of adjlinalg.Vector's.'''
  retvals = orig_get_svd(self, *args, **kwargs)
  new_retvals = []
  for retval in retvals:
    if isinstance(retval, adjlinalg.Vector):
      new_retvals.append(retval.data)
    else:
      new_retvals.append(retval)

  return new_retvals
libadjoint.SVDHandle.get_svd = new_get_svd

def adj_compute_propagator_matrix(svd):
  # Warning: for testing purposes only -- it's far too expensive to do on big models.
  # This also only works in serial.

  import numpy

  (sigma, u, v) = svd.get_svd(0, return_vectors=True)
  (u, v) = (u.vector().array(), v.vector().array())

  mat = sigma * numpy.outer(u, v)

  for i in range(1, svd.ncv):
    (sigma, u, v) = svd.get_svd(i, return_vectors=True)
    (u, v) = (u.vector().array(), v.vector().array())
    sum_mat = sigma * numpy.outer(u, v)
    mat += sum_mat

  return mat
