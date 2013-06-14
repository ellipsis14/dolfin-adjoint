import dolfin
import ufl.algorithms
import libadjoint
import solving
import adjlinalg
import adjglobals
import hashlib
import utils

if dolfin.__version__ > '1.2.0':
  class PointIntegralSolver(dolfin.PointIntegralSolver):
    def step(self, dt, annotate=None):

      to_annotate = utils.to_annotate(annotate)

      if to_annotate:
        scheme = self.scheme()
        var = scheme.solution()
        fn_space = var.function_space()

        current_var = adjglobals.adj_variables[var]
        if not adjglobals.adjointer.variable_known(current_var):
          solving.register_initial_conditions([(var, current_var)], linear=True)

        identity_block = solving.get_identity(fn_space)
        rhs = PointIntegralRHS(self, dt, current_var)
        next_var = adjglobals.adj_variables.next(var)

        eqn = libadjoint.Equation(next_var, blocks=[identity_block], targets=[next_var], rhs=rhs)
        cs  = adjglobals.adjointer.register_equation(eqn)

      super(PointIntegralSolver, self).step(dt)

      if to_annotate:
        solving.do_checkpoint(cs, next_var, rhs)

        if dolfin.parameters["adjoint"]["record_all"]:
          adjglobals.adjointer.record_variable(next_var, libadjoint.MemoryStorage(adjlinalg.Vector(var)))

  class PointIntegralRHS(libadjoint.RHS):
    def __init__(self, solver, dt, ic_var):
      self.solver = solver
      self.solver.reset()
      self.dt = dt

      self.scheme = scheme = solver.scheme()
      self.time = float(scheme.t())
      self.form = scheme.rhs_form()
      self.fn_space = scheme.solution().function_space()

      self.coeffs = [coeff for coeff in ufl.algorithms.extract_coefficients(self.form) if hasattr(coeff, "function_space")]
      self.deps   = [adjglobals.adj_variables[coeff] for coeff in self.coeffs]

      solving.register_initial_conditions(zip(self.coeffs, self.deps), linear=True)

      self.ic_var = ic_var
      if scheme.solution() not in self.coeffs:
        self.coeffs.append(scheme.solution())
        self.deps.append(ic_var)

    def dependencies(self):
      return self.deps

    def coefficients(self):
      return self.coeffs

    def __str__(self):
      return hashlib.md5(str(self.form)).hexdigest()

    def new_form(self, new_time, new_solution, dependencies, values):
      new_form = dolfin.replace(self.form, dict(zip(self.coeffs, [val.data for val in values])))

      if self.ic_var is not None:
        ic_value = values[dependencies.index(self.ic_var)].data
        new_solution.assign(ic_value, annotate=False)
        new_form = dolfin.replace(new_form, {ic_value: new_solution})

      new_form = dolfin.replace(new_form, {self.scheme.t(): new_time})
      return new_form

    def new_scheme(self, dependencies, values):
      new_time = dolfin.Constant(self.time)
      new_solution = dolfin.Function(self.fn_space)
      new_form = self.new_form(new_time, new_solution, dependencies, values)

      new_scheme = self.scheme.__class__(new_form, new_solution, new_time)
      return new_scheme

    def __call__(self, dependencies, values):
      new_scheme = self.new_scheme(dependencies, values)
      new_solver = dolfin.PointIntegralSolver(new_scheme)
      new_solver.parameters.update(self.solver.parameters)
      new_solver.step(self.dt)

      return adjlinalg.Vector(new_scheme.solution())

    def derivative_action(self, dependencies, values, variable, contraction_vector, hermitian):
      assert not hermitian

      new_scheme = self.new_scheme(dependencies, values)
      tlm_scheme = new_scheme.to_tlm(contraction_vector.data)
      tlm_solver = dolfin.PointIntegralSolver(tlm_scheme)
      tlm_solver.parameters.update(self.solver.parameters)
      tlm_solver.step(self.dt)

      return adjlinalg.Vector(tlm_scheme.solution())

  __all__ = ['PointIntegralSolver']
else:
  __all__ = []
