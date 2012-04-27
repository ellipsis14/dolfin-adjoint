import dolfin
import solving
import libadjoint
import adjglobals
import adjlinalg

class NewtonSolver(dolfin.NewtonSolver):
  def solve(self, *args, **kwargs):
    to_annotate = True
    if "annotate" in kwargs:
      to_annotate = kwargs["annotate"]
      del kwargs["annotate"] # so we don't pass it on to the real solver

    if to_annotate:
      factory = args[0]
      vec = args[1]
      b = dolfin.PETScVector()

      factory.F(b=b, x=vec)

      F = b.form
      bcs = b.bcs

      u = vec.function
      var = adjglobals.adj_variables[u]

      solving.annotate(F == 0, u, bcs, solver_parameters={"newton_solver": self.parameters.to_dict()})

    newargs = [self] + list(args)
    out = dolfin.NewtonSolver.solve(*newargs, **kwargs)

    if to_annotate and dolfin.parameters["adjoint"]["record_all"]:
      adjglobals.adjointer.record_variable(adjglobals.adj_variables[u], libadjoint.MemoryStorage(adjlinalg.Vector(u)))

    return out
