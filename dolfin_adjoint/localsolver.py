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
            b_vec = L.vector()
            L = None
        else:
            b_vec = dolfin.assemble(L)

        # Next: if necessary, create a new solver and add to dictionary
        if a not in caching.localsolvers:
            if dolfin.parameters["adjoint"]["debug_cache"]:
                dolfin.info_red("Creating new LocalSolver")
            newsolver = dolfin.LocalSolver(a, None, solver_type=self.solver_parameters["solver_type"])
            newsolver.factorize()
            caching.localsolvers[a] = newsolver
        else:
            if dolfin.parameters["adjoint"]["debug_cache"]:
                dolfin.info_green("Reusing LocalSolver")

        # Get the right solver from the solver dictionary
        solver = caching.localsolvers[a]
        solver.solve_local(x.vector(), b_vec, b.fn_space.dofmap())

        x_vec = adjlinalg.Vector(x)
        return x_vec

class LocalSolver(dolfin.LocalSolver):
    def __init__(self, a, L = None, solver_type = dolfin.LocalSolver.LU):
        dolfin.LocalSolver.__init__(self, a, L, solver_type)
        self.a = a
        self.L = L
        self.solver_type = solver_type

    def solve(self, x_vec, b_vec, b_dofmap, **kwargs):
        # Figure out whether to annotate or not
        to_annotate = utils.to_annotate(kwargs.pop("annotate", None))
        x = x_vec.function

        if to_annotate:
            L = b_vec.form

            # Set Matrix class for solving the adjoint systems
            solving.annotate(self.a == L, x, solver_parameters={"solver_type": self.solver_type}, matrix_class=LocalSolverMatrix)

        # Use standard local solver
        out = dolfin.LocalSolver.solve_local(self, x_vec, b_vec, b_dofmap)

        if to_annotate:
            # checkpointing
            if dolfin.parameters["adjoint"]["record_all"]:
                adjglobals.adjointer.record_variable(adjglobals.adj_variables[x],
                           libadjoint.MemoryStorage(adjlinalg.Vector(x)))

        return out

    def solve_global_rhs(*args, **kwargs):
        error("ERROR: Only use LocalSolver.solve(), solve_global_rhs() doesn't get annotated.")
