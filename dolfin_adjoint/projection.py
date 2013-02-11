import solving
import dolfin
import libadjoint
import adjglobals
import adjlinalg

def project(v, V=None, bcs=None, mesh=None, solver_type="cg", preconditioner_type="default", form_compiler_parameters=None, annotate=True, name=None):
  '''The project call performs an equation solve, and so it too must be annotated so that the
  adjoint and tangent linear models may be constructed automatically by libadjoint.

  To disable the annotation of this function, just pass :py:data:`annotate=False`. This is useful in
  cases where the solve is known to be irrelevant or diagnostic for the purposes of the adjoint
  computation (such as projecting fields to other function spaces for the purposes of
  visualisation).'''

  if dolfin.parameters["adjoint"]["stop_annotating"]:
    annotate = False

  if isinstance(v, dolfin.Expression):
    annotate = False

  out = dolfin.project(v, V, bcs, mesh, solver_type, preconditioner_type, form_compiler_parameters)
  if name is not None:
    out.adj_name = name

  if annotate:
    # reproduce the logic from project. This probably isn't future-safe, but anyway

    if V is None:
      V = dolfin._extract_function_space(v, mesh)

    # Define variational problem for projection
    w = dolfin.TestFunction(V)
    Pv = dolfin.TrialFunction(V)
    a = dolfin.inner(w, Pv)*dolfin.dx
    L = dolfin.inner(w, v)*dolfin.dx

    solving.annotate(a == L, out, bcs, solver_parameters={"linear_solver": solver_type, "preconditioner": preconditioner_type, "symmetric": True})

    if dolfin.parameters["adjoint"]["record_all"]:
      adjglobals.adjointer.record_variable(adjglobals.adj_variables[out], libadjoint.MemoryStorage(adjlinalg.Vector(out)))

  return out

