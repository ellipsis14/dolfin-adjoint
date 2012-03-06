from solving import adjointer, adj_variables

def adj_compute_tlm_svd(ic, final, nsv):
  ic_var = adj_variables[ic]; ic_var.timestep = 0; ic_var.iteration = 0
  final_var = adj_variables[final]
  return adjointer.compute_tlm_svd(ic_var, final_var, nsv)
