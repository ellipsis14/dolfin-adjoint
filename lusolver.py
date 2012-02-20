import dolfin
import solving
import libadjoint

class LUSolver(dolfin.LUSolver):
  def __init__(self, *args):
    try:
      self.operator = args[0].form
    except AttributeError:
      raise libadjoint.exceptions.LibadjointErrorInvalidInputs("Your matrix A has to have the .form attribute: was it assembled after from dolfin_adjoint import *?")

    dolfin.LUSolver.__init__(self, *args)

  def solve(self, *args, **kwargs):

    if len(args) != 2:
      raise libadjoint.exceptions.LibadjointErrorInvalidInputs("The annotated LUSolver.solve must be called like solve(x, b).")

    A = self.operator

    try:
      x = args[0].function
    except AttributeError:
      raise libadjoint.exceptions.LibadjointErrorInvalidInputs("Your solution x has to have a .function attribute; is it the .vector() of a Function?")

    try:
      b = args[1].form
    except AttributeError:
      raise libadjoint.exceptions.LibadjointErrorInvalidInputs("Your RHS b has to have the .form attribute: was it assembled after from dolfin_adjoint import *?")

    try:
      eq_bcs = list(set(A.bcs + b.bcs))
    except AttributeError:
      assert not hasattr(A, 'bcs') and not hasattr(b, 'bcs')
      eq_bcs = []

    solving.annotate(A == b, x, eq_bcs, solver_parameters={"linear_solver": "lu"})

    if self.parameters["reuse_factorization"]:
      dolfin.info_red('Warning: the annotation of LUSolver.solve currently ignores parameters["reuse_factorizarion"] = True.')

    out = dolfin.LUSolver.solve(self, *args, **kwargs)

    if solving.debugging["record_all"]:
      solving.adjointer.record_variable(solving.adj_variables[x], libadjoint.MemoryStorage(solving.Vector(x)))

    return out
