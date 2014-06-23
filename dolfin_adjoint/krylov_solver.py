import dolfin
import ufl
import solving
import libadjoint
import adjlinalg
import adjglobals
import misc
import utils

class KrylovSolver(dolfin.KrylovSolver):
  '''This object is overloaded so that solves using this class are automatically annotated,
  so that libadjoint can automatically derive the adjoint and tangent linear models.'''
  def __init__(self, *args):
    dolfin.KrylovSolver.__init__(self, *args)
    self.solver_parameters = args
    self.nsp = None
    self.tnsp = None

    self.operators = (None, None)
    if len(args) > 0 and isinstance(args[0], dolfin.GenericMatrix):
      self.operators = (args[0], None)

  def set_operators(self, A, P):
    dolfin.KrylovSolver.set_operators(self, A, P)
    self.operators = (A, P)

  def set_nullspace(self, nsp):
    dolfin.KrylovSolver.set_nullspace(self, nsp)
    self.nsp = nsp

  def set_transpose_nullspace(self, tnsp):
    if hasattr(dolfin.KrylovSolver, 'set_transpose_nullspace'): dolfin.KrylovSolver.set_transpose_nullspace(self, tnsp)
    self.tnsp = tnsp

  def set_operator(self, A):
    dolfin.KrylovSolver.set_operator(self, A)
    self.operators = (A, self.operators[1])

  def solve(self, *args, **kwargs):
    '''To disable the annotation, just pass :py:data:`annotate=False` to this routine, and it acts exactly like the
    Dolfin solve call. This is useful in cases where the solve is known to be irrelevant or diagnostic
    for the purposes of the adjoint computation (such as projecting fields to other function spaces
    for the purposes of visualisation).'''

    to_annotate = utils.to_annotate(kwargs.pop("annotate", None))

    if to_annotate:
      if len(args) == 3:
        A = args[0]
        x = args[1]
        b = args[2]
      elif len(args) == 2:
        A = self.operators[0]
        x = args[0]
        b = args[1]

      bcs = []
      if hasattr(A, 'bcs'):
        bcs += A.bcs
      if hasattr(b, 'bcs'):
        bcs += b.bcs
      bcs = misc.uniq(bcs)

      assemble_system = A.assemble_system

      A = A.form
      u = x.function
      b = b.form

      if self.operators[1] is not None:
        P = self.operators[1].form
      else:
        P = None

      solver_parameters = self.solver_parameters
      parameters = self.parameters.to_dict()
      fn_space = u.function_space()
      has_preconditioner = P is not None
      nsp = self.nsp
      tnsp = self.tnsp

      if nsp is not None:
        msg = """
        The transpose nullspace is not set.

        The nullspace of the KrylovSolver is set. In this case,
        the transpose nullspace must also be set, use:

          solver.set_transpose_nullspace(nullspace)
        """
        assert tnsp is not None, msg

      class KrylovSolverMatrix(adjlinalg.Matrix):
        def __init__(self, *args, **kwargs):
          if 'initial_guess' in kwargs:
            self.initial_guess = kwargs['initial_guess']
            del kwargs['initial_guess']
          else:
            self.initial_guess = None

          replace_map = kwargs['replace_map']
          del kwargs['replace_map']

          adjlinalg.Matrix.__init__(self, *args, **kwargs)

          self.adjoint = kwargs['adjoint']
          if P is None:
            self.operators = (dolfin.replace(A, replace_map), None)
          else:
            self.operators = (dolfin.replace(A, replace_map), dolfin.replace(P, replace_map))

        def axpy(self, alpha, x):
          raise libadjoint.exceptions.LibadjointErrorNotImplemented("Shouldn't ever get here")

        def solve(self, var, b):
          if self.adjoint:
            operators = transpose_operators(self.operators)
          else:
            operators = self.operators

          solver = dolfin.KrylovSolver(*solver_parameters)
          solver.parameters.update(parameters)

          if self.adjoint:
            # swap nullspaces
            (nsp_, tnsp_) = (tnsp, nsp)
          else:
            (nsp_, tnsp_) = (nsp, tnsp)

          if nsp_ is not None:
            solver.set_nullspace(nsp_)
          if tnsp_ is not None and hasattr(solver, 'set_transpose_nullspace'):
            solver.set_transpose_nullspace(tnsp_)

          x = dolfin.Function(fn_space)
          if self.initial_guess is not None and var.type == 'ADJ_FORWARD':
            x.vector()[:] = self.initial_guess.vector()

          if b.data is None:
            dolfin.info_red("Warning: got zero RHS for the solve associated with variable %s" % var)
            return adjlinalg.Vector(x)

          if var.type in ['ADJ_TLM', 'ADJ_ADJOINT']:
            self.bcs = [dolfin.homogenize(bc) for bc in self.bcs if isinstance(bc, dolfin.cpp.DirichletBC)] + [bc for bc in self.bcs if not isinstance(bc, dolfin.cpp.DirichletBC)]

          # This is really hideous. Sorry.
          if isinstance(b.data, dolfin.Function):
            rhs = b.data.vector().copy()
            [bc.apply(rhs) for bc in self.bcs]

            if assemble_system: # if we called assemble_system, rather than assemble
              v = dolfin.TestFunction(fn_space)
              (A, rhstmp) = dolfin.assemble_system(operators[0], dolfin.inner(b.data, v)*dolfin.dx, self.bcs)
              if has_preconditioner:
                (P, rhstmp) = dolfin.assemble_system(operators[1], dolfin.inner(b.data, v)*dolfin.dx, self.bcs)
                solver.set_operators(A, P)
              else:
                solver.set_operator(A)
            else: # we called assemble
              A = dolfin.assemble(operators[0])
              [bc.apply(A) for bc in self.bcs]
              if has_preconditioner:
                P = dolfin.assemble(operators[1])
                [bc.apply(P) for bc in self.bcs]
                solver.set_operators(A, P)
              else:
                solver.set_operator(A)
          else:

            if assemble_system: # if we called assemble_system, rather than assemble
              (A, rhs) = dolfin.assemble_system(operators[0], b.data, self.bcs)
              if has_preconditioner:
                (P, rhstmp) = dolfin.assemble_system(operators[1], b.data, self.bcs)
                solver.set_operators(A, P)
              else:
                solver.set_operator(A)
            else: # we called assemble
              A = dolfin.assemble(operators[0])
              rhs = dolfin.assemble(b.data)
              [bc.apply(A) for bc in self.bcs]
              [bc.apply(rhs) for bc in self.bcs]
              if has_preconditioner:
                P = dolfin.assemble(operators[1])
                [bc.apply(P) for bc in self.bcs]
                solver.set_operators(A, P)
              else:
                solver.set_operator(A)

          if tnsp_ is not None:
            tnsp_.orthogonalize(rhs)

          solver.solve(x.vector(), rhs)
          return adjlinalg.Vector(x)

      solving.annotate(A == b, u, bcs, matrix_class=KrylovSolverMatrix, initial_guess=parameters['nonzero_initial_guess'], replace_map=True)

    out = dolfin.KrylovSolver.solve(self, *args, **kwargs)

    if to_annotate and dolfin.parameters["adjoint"]["record_all"]:
      adjglobals.adjointer.record_variable(adjglobals.adj_variables[u], libadjoint.MemoryStorage(adjlinalg.Vector(u)))

    return out

def transpose_operators(operators):
  out = [None, None]

  for i in range(2):
    op = operators[i]

    if op is None:
      out[i] = None
    elif isinstance(op, dolfin.cpp.GenericMatrix):
      out[i] = op.__class__()
      dolfin.assemble(dolfin.adjoint(op.form), tensor=out[i])

      if hasattr(op, 'bcs'):
        adjoint_bcs = [dolfin.homogenize(bc) for bc in op.bcs if isinstance(bc, dolfin.cpp.DirichletBC)] + [bc for bc in op.bcs if not isinstance(bc, dolfin.DirichletBC)]
        [bc.apply(out[i]) for bc in adjoint_bcs]

    elif isinstance(op, dolfin.Form) or isinstance(op, ufl.form.Form):
      out[i] = dolfin.adjoint(op)

      if hasattr(op, 'bcs'):
        out[i].bcs = [dolfin.homogenize(bc) for bc in op.bcs if isinstance(bc, dolfin.cpp.DirichletBC)] + [bc for bc in op.bcs if not isinstance(bc, dolfin.DirichletBC)]

    elif isinstance(op, AdjointKrylovMatrix):
      pass

    else:
      print "op.__class__: ", op.__class__
      raise libadjoint.exceptions.LibadjointErrorNotImplemented("Don't know how to transpose anything else!")

  return out
