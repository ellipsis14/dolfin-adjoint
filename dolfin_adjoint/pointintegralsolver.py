import dolfin
import ufl.algorithms
import libadjoint
import solving
import adjlinalg
import adjglobals
import hashlib
import utils
import caching

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
      if hasattr(self.solver, 'reset'): self.solver.reset()
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
      coeffs = [x for x in ufl.algorithms.extract_coefficients(self.scheme.rhs_form()) if hasattr(x, 'function_space')]
      for (coeff, value) in zip(coeffs, values):
        coeff.assign(value.data)

      self.scheme.t().assign(self.time)
      self.solver.step(self.dt)

      return adjlinalg.Vector(self.scheme.solution())

    def derivative_action(self, dependencies, values, variable, contraction_vector, hermitian):
      new_scheme = self.new_scheme(dependencies, values)
      if not hermitian:
        if self.solver not in caching.pis_fwd_to_tlm:
          dolfin.info_blue("No TLM solver, creating ... ")
          tlm_scheme = self.scheme.to_tlm(contraction_vector.data)

          tlm_solver = dolfin.PointIntegralSolver(tlm_scheme)
          tlm_solver.parameters.update(self.solver.parameters)
          caching.pis_fwd_to_tlm[self.solver] = tlm_solver
        else:
          dolfin.info_green("Got an TLM solver, using ... ")
          tlm_solver = caching.pis_fwd_to_tlm[self.solver]
          tlm_scheme = tlm_solver.scheme()
          tlm_scheme.contraction.assign(contraction_vector.data)

        coeffs = [x for x in ufl.algorithms.extract_coefficients(tlm_scheme.rhs_form()) if hasattr(x, 'function_space')]
        for (coeff, value) in zip(coeffs, values):
          coeff.assign(value.data)
        tlm_scheme.t().assign(self.time)

        tlm_solver.step(self.dt)

        return adjlinalg.Vector(tlm_scheme.solution())

      else:
        if self.solver not in caching.pis_fwd_to_adj:
          dolfin.info_blue("No ADM solver, creating ... ")
          adm_scheme = self.scheme.to_adm(contraction_vector.data)

          adm_solver = dolfin.PointIntegralSolver(adm_scheme)
          adm_solver.parameters.update(self.solver.parameters)
          caching.pis_fwd_to_adj[self.solver] = adm_solver
        else:
          dolfin.info_green("Got an ADM solver, using ... ")
          adm_solver = caching.pis_fwd_to_adj[self.solver]
          adm_scheme = adm_solver.scheme()
          adm_scheme.contraction.assign(contraction_vector.data)

        coeffs = [x for x in ufl.algorithms.extract_coefficients(adm_scheme.rhs_form()) if hasattr(x, 'function_space')]
        for (coeff, value) in zip(coeffs, values):
          coeff.assign(value.data)
        adm_scheme.t().assign(self.time)

        adm_solver.step(self.dt)

        return adjlinalg.Vector(adm_scheme.solution())

  __all__ = ['PointIntegralSolver']
else:
  __all__ = []
