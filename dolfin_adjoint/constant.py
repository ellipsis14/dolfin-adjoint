import dolfin

class Constant(dolfin.Constant):
  def __init__(self, value, cell=None, name=None):
    dolfin.Constant.__init__(self, value, cell)
    if name is not None:
      self.adj_name = name
