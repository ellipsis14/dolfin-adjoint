import dolfin
import solving
import assembly
import libadjoint
import adjglobals
import adjlinalg
import misc
import utils

class LocalSolverMatrix(adjlinalg.Matrix):
    def solve(self, var, b):
        newsolver = dolfin.LocalSolver(self.data, b.data)
        x = dolfin.Function(self.test_function().function_space())
        newsolver.solve(x.vector())

        x_vec = adjlinalg.Vector(x)
        return x_vec

class LocalSolver(dolfin.LocalSolver):
    def __init__(self, a, L):
        dolfin.LocalSolver.__init__(self, a, L)
        self.a = a
        self.L = L

    def solve(self, x_vec, **kwargs):
        to_annotate = utils.to_annotate(kwargs.pop("annotate", None))
        x = x_vec.function

        if to_annotate:
            solving.annotate(self.a == self.L, x, matrix_class=LocalSolverMatrix)

        out = dolfin.LocalSolver.solve(self, x_vec)

        if to_annotate:
            if dolfin.parameters["adjoint"]["record_all"]:
                adjglobals.adjointer.record_variable(adjglobals.adj_variables[x],
                           libadjoint.MemoryStorage(adjlinalg.Vector(x)))

        return out
