import solving
import backend
if backend.__name__ == "dolfin":
  import backend.fem.projection
import libadjoint
import adjglobals
import adjlinalg
import utils

def project(v, V=None, bcs=None, mesh=None, solver_type="cg", preconditioner_type="default", form_compiler_parameters=None, annotate=None, name=None):
  '''The project call performs an equation solve, and so it too must be annotated so that the
  adjoint and tangent linear models may be constructed automatically by libadjoint.

  To disable the annotation of this function, just pass :py:data:`annotate=False`. This is useful in
  cases where the solve is known to be irrelevant or diagnostic for the purposes of the adjoint
  computation (such as projecting fields to other function spaces for the purposes of
  visualisation).'''

  to_annotate = utils.to_annotate(annotate)

  if isinstance(v, backend.Expression) and (annotate is not True):
    to_annotate = False

  if isinstance(v, backend.Constant) and (annotate is not True):
    to_annotate = False

  if backend.__name__ == "dolfin":
    out = backend.project(v, V, bcs, mesh, solver_type, preconditioner_type, form_compiler_parameters)
  else:
    out = backend.project(v, V)

  if name is not None:
    out.adj_name = name
    out.rename(name, "a Function from dolfin-adjoint")

  if to_annotate:
    # reproduce the logic from project. This probably isn't future-safe, but anyway

    if V is None:
      V = backend.fem.projection._extract_function_space(v, mesh)

    # Define variational problem for projection
    w = backend.TestFunction(V)
    Pv = backend.TrialFunction(V)
    a = backend.inner(w, Pv)*backend.dx
    L = backend.inner(w, v)*backend.dx

    solving.annotate(a == L, out, bcs, solver_parameters={"linear_solver": solver_type, "preconditioner": preconditioner_type, "symmetric": True})

    if backend.parameters["adjoint"]["record_all"]:
      adjglobals.adjointer.record_variable(adjglobals.adj_variables[out], libadjoint.MemoryStorage(adjlinalg.Vector(out)))

  return out

