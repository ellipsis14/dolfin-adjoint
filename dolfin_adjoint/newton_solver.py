import dolfin
import solving
import libadjoint

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
      var = solving.adj_variables[u]

      solving.annotate(F == 0, u, bcs, solver_parameters={"newton_solver": dict(self.parameters)})

    newargs = [self] + list(args)
    out = dolfin.NewtonSolver.solve(*newargs, **kwargs)

    if to_annotate and solving.debugging["record_all"]:
      solving.adjointer.record_variable(solving.adj_variables[u], libadjoint.MemoryStorage(solving.Vector(u)))

    return out
