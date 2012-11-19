import dolfin
import solving
import assembly
import libadjoint
import adjglobals
import adjlinalg

lu_solvers = {}
adj_lu_solvers = {}

def make_LUSolverMatrix(form, reuse_factorization):
  class LUSolverMatrix(adjlinalg.Matrix):
    def solve(self, var, b):

      if reuse_factorization is False:
        return adjlinalg.Matrix.solve(self, var, b)

      if var.type in ['ADJ_TLM', 'ADJ_ADJOINT']:
        bcs = [dolfin.homogenize(bc) for bc in self.bcs if isinstance(bc, dolfin.DirichletBC)] + [bc for bc in self.bcs if not isinstance(bc, dolfin.DirichletBC)]
      else:
        bcs = self.bcs

      if var.type in ['ADJ_FORWARD', 'ADJ_TLM']:
        solver = lu_solvers[form]
      else:
        if adj_lu_solvers[form] is None:
          A = assembly.assemble(self.data); [bc.apply(A) for bc in bcs]
          adj_lu_solvers[form] = LUSolver(A)
          adj_lu_solvers[form].parameters["reuse_factorization"] = True

        solver = adj_lu_solvers[form]

      x = adjlinalg.Vector(dolfin.Function(self.test_function().function_space()))

      if b.data is None:
        # This means we didn't get any contribution on the RHS of the adjoint system. This could be that the
        # simulation ran further ahead than when the functional was evaluated, or it could be that the
        # functional is set up incorrectly.
        dolfin.info_red("Warning: got zero RHS for the solve associated with variable %s" % var)
      else:
        if isinstance(b.data, dolfin.Function):
          b_vec = b.data.vector().copy()
        else:
          b_vec = dolfin.assemble(b.data)

        [bc.apply(b_vec) for bc in bcs]
        solver.solve(x.data.vector(), b_vec, annotate=False)

      return x
  return LUSolverMatrix

class LUSolver(dolfin.LUSolver):
  '''This object is overloaded so that solves using this class are automatically annotated,
  so that libadjoint can automatically derive the adjoint and tangent linear models.'''
  def __init__(self, *args):

    if len(args) > 0:
      try:
        self.operator = args[0].form
      except AttributeError:
        raise libadjoint.exceptions.LibadjointErrorInvalidInputs("Your matrix A has to have the .form attribute: was it assembled after from dolfin_adjoint import *?")

      try:
        self.op_bcs = args[0].bcs
      except AttributeError:
        self.op_bcs = []

    dolfin.LUSolver.__init__(self, *args)

  def set_operator(self, operator):
    try:
      self.operator = operator.form
    except AttributeError:
      raise libadjoint.exceptions.LibadjointErrorInvalidInputs("Your matrix A has to have the .form attribute: was it assembled after from dolfin_adjoint import *?")

    try:
      self.op_bcs = operator.bcs
    except AttributeError:
      self.op_bcs = []

    dolfin.LUSolver.set_operator(self, operator)

  def solve(self, *args, **kwargs):
    '''To disable the annotation, just pass :py:data:`annotate=False` to this routine, and it acts exactly like the
    Dolfin solve call. This is useful in cases where the solve is known to be irrelevant or diagnostic
    for the purposes of the adjoint computation (such as projecting fields to other function spaces
    for the purposes of visualisation).'''

    annotate = True
    if "annotate" in kwargs:
      annotate = kwargs["annotate"]
      del kwargs["annotate"]

    if dolfin.parameters["adjoint"]["stop_annotating"]:
      annotate = False

    if annotate:
      if len(args) == 2:
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
          eq_bcs = list(set(self.op_bcs + args[1].bcs))
        except AttributeError:
          eq_bcs = self.op_bcs

      elif len(args) == 3:
        A = args[0].form
        try:
          x = args[1].function
        except AttributeError:
          raise libadjoint.exceptions.LibadjointErrorInvalidInputs("Your solution x has to have a .function attribute; is it the .vector() of a Function?")

        try:
          b = args[2].form
        except AttributeError:
          raise libadjoint.exceptions.LibadjointErrorInvalidInputs("Your RHS b has to have the .form attribute: was it assembled after from dolfin_adjoint import *?")

        try:
          eq_bcs = list(set(self.op_bcs + args[2].bcs))
        except AttributeError:
          eq_bcs = self.op_bcs

      else:
        raise libadjoint.exceptions.LibadjointErrorInvalidInputs("LUSolver.solve() must be called with either (A, x, b) or (x, b).")

      if self.parameters["reuse_factorization"]:
        lu_solvers[A] = self
        adj_lu_solvers[A] = None

      solving.annotate(A == b, x, eq_bcs, solver_parameters={"linear_solver": "lu"}, matrix_class=make_LUSolverMatrix(A, self.parameters["reuse_factorization"]))

    out = dolfin.LUSolver.solve(self, *args, **kwargs)

    if annotate:
      if dolfin.parameters["adjoint"]["record_all"]:
        adjglobals.adjointer.record_variable(adjglobals.adj_variables[x], libadjoint.MemoryStorage(adjlinalg.Vector(x)))

    return out
