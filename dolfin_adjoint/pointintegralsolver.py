import dolfin
import ufl.algorithms
import libadjoint
import solving
import adjlinalg
import adjglobals
import hashlib

if dolfin.__version__ > '1.2.0':
  class PointIntegralSolver(dolfin.PointIntegralSolver):
    def step(self, dt, annotate=True):
      super(PointIntegralSolver, self).step(dt)

      to_annotate = annotate
      if dolfin.parameters["adjoint"]["stop_annotating"]:
        to_annotate = False

      if to_annotate:
        scheme = self.scheme()
        var = scheme.solution()
        fn_space = var.function_space()

        identity_block = solving.get_identity(fn_space)
        rhs = PointIntegralRHS(self)
        next_var = adjglobals.adj_variables.next(var)

        eqn = libadjoint.Equation(next_var, blocks=[identity_block], targets=[next_var], rhs=rhs)
        cs  = adjglobals.adjointer.register_equation(eqn)
        solving.do_checkpoint(cs, next_var, rhs)

        if dolfin.parameters["adjoint"]["record_all"]:
          adjglobals.adjointer.record_variable(next_var, libadjoint.MemoryStorage(adjlinalg.Vector(var)))

  class PointIntegralRHS(libadjoint.RHS):
    def __init__(self, solver):
      self.solver = solver

      scheme = solver.scheme()
      self.form = scheme.rhs_form()

      self.coeffs = [coeff for coeff in ufl.algorithms.extract_coefficients(self.form) if hasattr(coeff, "function_space")]
      self.deps   = [adjglobals.adj_variables[coeff] for coeff in self.coeffs]

      solving.register_initial_conditions(zip(self.coeffs, self.deps), linear=True)

    def dependencies(self):
      return self.deps

    def coefficients(self):
      return self.coeffs

    def __str__(self):
      return hashlib.md5(str(self.form)).hexdigest()

  __all__ = ['PointIntegralSolver']
else:
  __all__ = []
