import dolfin
import solving
import assembly
import libadjoint
import adjglobals
import adjlinalg
import misc
import utils
import caching

class LocalSolverMatrix(adjlinalg.Matrix):
    def solve(self, var, b):
        x = dolfin.Function(self.test_function().function_space())

        # b is a libadjoint object (form or function)
        (a, L) = (self.data, b.data)

        # First: check if L is None (meaning zero)
        if L is None:
            x_vec = adjlinalg.Vector(x)
            return x_vec	

        if isinstance(L, dolfin.Function):
            L = L.form

        # Next: if necessary, create a new solver and add to dictionary
        if (a, L) not in caching.localsolvers:
            if dolfin.parameters["adjoint"]["debug_cache"]:
                dolfin.info_red("Creating new LocalSolver")
            newsolver = dolfin.LocalSolver(a, L)
            caching.localsolvers[(a, L)] = newsolver
        else:
            if dolfin.parameters["adjoint"]["debug_cache"]:
                dolfin.info_green("Reusing LocalSolver")            

        # Get the right solver from the solver dictionary
        solver = caching.localsolvers[(a, L)]
        solver.solve(x.vector())

        x_vec = adjlinalg.Vector(x)
        return x_vec

class LocalSolver(dolfin.LocalSolver):
    def __init__(self, a, L):
        dolfin.LocalSolver.__init__(self, a, L)
        self.a = a
        self.L = L

    def solve(self, x_vec, **kwargs):
        # Figure out whether to annotate or not
        to_annotate = utils.to_annotate(kwargs.pop("annotate", None))
        x = x_vec.function

        if to_annotate:
            # Set Matrix class for solving the adjoint systems
            solving.annotate(self.a == self.L, x, matrix_class=LocalSolverMatrix)

        # Use standard local solver
        out = dolfin.LocalSolver.solve(self, x_vec)

        if to_annotate:
            # checkpointing
            if dolfin.parameters["adjoint"]["record_all"]:
                adjglobals.adjointer.record_variable(adjglobals.adj_variables[x],
                           libadjoint.MemoryStorage(adjlinalg.Vector(x)))

        return out
