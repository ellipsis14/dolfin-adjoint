import re

soa_to_adj = re.compile(r'\[(?P<func>Functional:.*?):Solution:.*\]')

# For caching strategies: a dictionary that maps adj_variable to LUSolver
# Not used by default

def canonicalise(var):
  # Return a string representation of var for indexing into the LU cache.

  s = str(var)

  # Since the SOA operator is always the same as the ADM, we can replace all
  # requests for SOA operators with ADM ones
  if var.type == 'ADJ_SOA':
    s = soa_to_adj.sub(r'[\1]', s).replace("SecondOrderAdjoint", "Adjoint")

  return s

class LUCache(dict):
  def __getitem__(self, x):
    return dict.__getitem__(self, canonicalise(x))

  def __setitem__(self, x, y):
    return dict.__setitem__(self, canonicalise(x), y)

  def __delitem__(self, x):
    return dict.__delitem__(self, canonicalise(x))

  def __contains__(self, x):
    return dict.__contains__(self, canonicalise(x))

