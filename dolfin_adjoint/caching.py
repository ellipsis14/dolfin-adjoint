
# For caching strategies: a dictionary that maps adj_variable to LUSolver
# Not used by default
class LUCache(dict):
  def __getitem__(self, x):
    return dict.__getitem__(self, str(x))

  def __setitem__(self, x, y):
    return dict.__setitem__(self, str(x), y)

  def __delitem__(self, x):
    return dict.__delitem__(self, str(x))

  def __contains__(self, x):
    return dict.__contains__(self, str(x))

