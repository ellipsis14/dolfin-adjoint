import dolfin

class PETScKrylovSolver(dolfin.PETScKrylovSolver):
  def solve(self, *args, **kwargs):
    dolfin.info_red("Warning: PETScKrylovSolver.solve is not currently annotated.")
    return dolfin.PETScKrylovSolver.solve(self, *args, **kwargs)

