from solving import adjointer, adj_variables

def adj_compute_propagator_svd(ic, final, nsv):
  ic_var = adj_variables[ic]; ic_var.c_object.timestep = 0; ic_var.c_object.iteration = 0
  final_var = adj_variables[final]
  return adjointer.compute_propagator_svd(ic_var, final_var, nsv)

def adj_compute_propagator_matrix(svd):
  # Warning: for testing purposes only -- it's far too expensive to do on big models.
  # This also only works in serial.

  import numpy

  (sigma, u, v) = svd.get_svd(0, return_vectors=True)
  (u, v) = (u.data.vector().array(), v.data.vector().array())

  mat = sigma * numpy.outer(u, v)

  for i in range(1, svd.ncv):
    (sigma, u, v) = svd.get_svd(i, return_vectors=True)
    (u, v) = (u.data.vector().array(), v.data.vector().array())
    sum_mat = sigma * numpy.outer(u, v)
    mat += sum_mat

  return mat
