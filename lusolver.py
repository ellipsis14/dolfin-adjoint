import dolfin

class LUSolver(dolfin.LUSolver):
  def solve(self, *args, **kwargs):
    dolfin.info_red("Warning: LUSolver.solve is not currently annotated.")
    return dolfin.LUSolver.solve(self, *args, **kwargs)
