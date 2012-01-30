import dolfin
from solving import adjointer, adj_variables, debugging, solve, Vector, annotate as solving_annotate
import libadjoint

dolfin_assign = dolfin.Function.assign
def dolfin_adjoint_assign(self, other, annotate=True):
  '''We also need to monkeypatch the Function.assign method, as it is often used inside 
  the main time loop, and not annotating it means you get the adjoint wrong for totally
  nonobvious reasons. If anyone objects to me monkeypatching your objects, my apologies
  in advance.'''

  # ignore anything not a dolfin.Function
  if not isinstance(other, dolfin.Function) or annotate is False:
    return dolfin_assign(self, other)

  # ignore anything that is an interpolation, rather than a straight assignment
  if self.function_space() != other.function_space():
    return dolfin_assign(self, other)

  other_var = adj_variables[other]
  # ignore any functions we haven't seen before -- we DON'T want to
  # annotate the assignment of initial conditions here. That happens
  # in the main solve wrapper.
  if not adjointer.variable_known(other_var):
    adj_variables.forget(other)
    return dolfin_assign(self, other)

  # OK, so we have a variable we've seen before. Beautiful.
  fn_space = other.function_space()
  u, v = dolfin.TestFunction(fn_space), dolfin.TrialFunction(fn_space)
  M = dolfin.inner(u, v)*dolfin.dx
  if debugging["fussy_replay"]:
    return solve(M == dolfin.action(M, other), self) # this takes care of all the annotation etc
  else:
    solving_annotate(M == dolfin.action(M, other), self)
    if debugging["record_all"]:
      adjointer.record_variable(adj_variables[self], libadjoint.MemoryStorage(Vector(self)))
    return dolfin_assign(self, other)

dolfin.Function.assign = dolfin_adjoint_assign

