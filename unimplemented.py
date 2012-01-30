import dolfin

class KrylovSolver(dolfin.KrylovSolver):
  def solve(self, *args, **kwargs):
    dolfin.info_red("Warning: KrylovSolver.solve is not currently annotated.")
    return dolfin.KrylovSolver.solve(self, *args, **kwargs)

class NonlinearVariationalProblem(dolfin.NonlinearVariationalProblem):
  def __init__(self, *args, **kwargs):
    dolfin.info_red("Warning: NonlinearVariationalProblem is not currently annotated.")
    dolfin.NonlinearVariationalProblem.__init__(self, *args, **kwargs)

class NonlinearProblem(dolfin.NonlinearVariationalProblem):
  def __init__(self, *args, **kwargs):
    dolfin.info_red("Warning: NonlinearProblem is not currently annotated.")
    dolfin.NonlinearProblem.__init__(self, *args, **kwargs)

class LinearVariationalProblem(dolfin.LinearVariationalProblem):
  def __init__(self, *args, **kwargs):
    dolfin.info_red("Warning: LinearVariationalProblem is not currently annotated.")
    dolfin.LinearVariationalProblem.__init__(self, *args, **kwargs)
