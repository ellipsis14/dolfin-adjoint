import solving
import dolfin
import libadjoint

def project(v, V=None, bcs=None, mesh=None, solver_type="cg", preconditioner_type="default", form_compiler_parameters=None, annotate=True):

  if isinstance(v, dolfin.Expression):
    annotate = False

  out = dolfin.project(v, V, bcs, mesh, solver_type, preconditioner_type, form_compiler_parameters)

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

    if solving.debugging["record_all"]:
      solving.adjointer.record_variable(solving.adj_variables[out], libadjoint.MemoryStorage(solving.Vector(out)))

  return out

