import re
import ufl.algorithms

### A general dictionary that applies a key function before lookup
class KeyedDict(dict):
  def __init__(self, keyfunc, *args, **kwargs):
    self.keyfunc = keyfunc
    dict.__init__(self, *args, **kwargs)

  def __getitem__(self, x):
    return dict.__getitem__(self, self.keyfunc(x))

  def __setitem__(self, x, y):
    return dict.__setitem__(self, self.keyfunc(x), y)

  def __delitem__(self, x):
    return dict.__delitem__(self, self.keyfunc(x))

  def __contains__(self, x):
    return dict.__contains__(self, self.keyfunc(x))

### Stuff for LU caching

soa_to_adj = re.compile(r'\[(?P<func>Functional:.*?):.*\]')

# For caching strategies: a dictionary that maps adj_variable to LUSolver
# Not used by default

def lu_canonicalisation(var):
  # Return a string representation of var for indexing into the LU cache.

  s = str(var)

  # Since the SOA operator is always the same as the ADM, we can replace all
  # requests for SOA operators with ADM ones
  if var.type == 'ADJ_SOA':
    s = soa_to_adj.sub(r'[\1]', s).replace("SecondOrderAdjoint", "Adjoint")

  return s

lu_solvers = KeyedDict(keyfunc=lu_canonicalisation)

### Stuff for preassembly caching

def form_key(form):
  try:
    return ufl.algorithms.expand_indices(form)
  except:
    return form

assembled_fwd_forms = set()
assembled_adj_forms = KeyedDict(keyfunc=form_key)

### Stuff for PointIntegralSolver caching
pis_fwd_to_tlm = {}
pis_fwd_to_adj = {}
