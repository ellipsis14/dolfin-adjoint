import dolfin
import solving
import libadjoint
from matrix_free import transpose_operators

class KrylovSolver(dolfin.KrylovSolver):
  def __init__(self, *args):
    dolfin.KrylovSolver.__init__(self, *args)
    self.solver_parameters = args

    self.operators = (None, None)

  def set_operators(self, A, P):
    dolfin.KrylovSolver.set_operators(self, A, P)
    self.operators = (A, P)

  def set_operator(self, A):
    dolfin.KrylovSolver.set_operator(self, A)
    self.operators = (A, self.operators[1])

  def solve(self, *args, **kwargs):
    to_annotate = True
    if "annotate" in kwargs:
      to_annotate = kwargs["annotate"]
      del kwargs["annotate"] # so we don't pass it on to the real solver

    if to_annotate:
      if len(args) == 3:
        A = args[0]
        x = args[1]
        b = args[2]
      elif len(args) == 2:
        A = self.operators[0]
        x = args[0]
        b = args[1]

      bcs = list(set(A.bcs + b.bcs))
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

      class KrylovSolverMatrix(solving.Matrix):
        def __init__(self, *args, **kwargs):
          solving.Matrix.__init__(self, *args, **kwargs)
          self.adjoint = kwargs['adjoint']
          self.operators = (A, P)

        def axpy(self, alpha, x):
          raise libadjoint.exceptions.LibadjointErrorNotImplemented("Shouldn't ever get here")

        def solve(self, var, b):
          if self.adjoint:
            operators = transpose_operators(self.operators)
          else:
            operators = self.operators

          solver = dolfin.KrylovSolver(*solver_parameters)
          solver.parameters.update(parameters)

          x = dolfin.Function(fn_space)
          if b.data is None:
            dolfin.info_red("Warning: got zero RHS for the solve associated with variable %s" % var)
            return solving.Vector(x)

          if var.type in ['ADJ_TLM', 'ADJ_ADJOINT']:
            self.bcs = [dolfin.homogenize(bc) for bc in self.bcs if isinstance(bc, dolfin.cpp.DirichletBC)]

          if isinstance(b.data, dolfin.Function):
            rhs = b.data.vector().copy()
          else:
            # This is really hideous. Sorry.

            if assemble_system:
              (A, rhs) = dolfin.assemble_system(operators[0], b.data, self.bcs)
              if has_preconditioner:
                (P, rhstmp) = dolfin.assemble_system(operators[1], b.data, self.bcs)
                solver.set_operators(A, P)
              else:
                solver.set_operator(A)
            else:
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

          solver.solve(x.vector(), rhs)
          return solving.Vector(x)

      solving.annotate(A == b, u, bcs, matrix_class=KrylovSolverMatrix)

    out = dolfin.KrylovSolver.solve(self, *args, **kwargs)

    if to_annotate and solving.debugging["record_all"]:
      solving.adjointer.record_variable(solving.adj_variables[u], libadjoint.MemoryStorage(solving.Vector(u)))

    return out
