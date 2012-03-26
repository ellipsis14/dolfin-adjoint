import dolfin
import solving
import libadjoint

class NonlinearVariationalProblem(dolfin.NonlinearVariationalProblem):
  def __init__(self, F, u, bcs=None, J=None, *args, **kwargs):
    dolfin.NonlinearVariationalProblem.__init__(self, F, u, bcs, J, *args, **kwargs)
    self.F = F
    self.u = u
    self.bcs = bcs
    self.J = J

class NonlinearVariationalSolver(dolfin.NonlinearVariationalSolver):
  def __init__(self, problem):
    dolfin.NonlinearVariationalSolver.__init__(self, problem)
    self.problem = problem

  def solve(self, annotate=True):
    if annotate:
      problem = self.problem
      solving.annotate(problem.F == 0, problem.u, problem.bcs, J=problem.J, solver_parameters=self.parameters.to_dict())

    out = dolfin.NonlinearVariationalSolver.solve(self)

    if annotate and solving.debugging["record_all"]:
      solving.adjointer.record_variable(solving.adj_variables[self.problem.u], libadjoint.MemoryStorage(solving.Vector(self.problem.u)))

    return out

class LinearVariationalProblem(dolfin.LinearVariationalProblem):
  def __init__(self, a, L, u, bcs, *args, **kwargs):
    dolfin.LinearVariationalProblem.__init__(self, a, L, u, bcs, *args, **kwargs)
    self.a = a
    self.L = L
    self.u = u
    self.bcs = bcs

class LinearVariationalSolver(dolfin.LinearVariationalSolver):
  def __init__(self, problem):
    dolfin.LinearVariationalSolver.__init__(self, problem)
    self.problem = problem

  def solve(self, annotate=True):
    if annotate:
      problem = self.problem
      solving.annotate(problem.a == problem.L, problem.u, problem.bcs, solver_parameters=self.parameters.to_dict())

    out = dolfin.LinearVariationalSolver.solve(self)

    if annotate and solving.debugging["record_all"]:
      solving.adjointer.record_variable(solving.adj_variables[self.problem.u], libadjoint.MemoryStorage(solving.Vector(self.problem.u)))

    return out

