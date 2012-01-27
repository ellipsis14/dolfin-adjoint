from dolfin import *
from dolfin_adjoint import *

mesh = UnitSquare(32, 32)
V = FunctionSpace(mesh, 'CG', 1)
bc = DirichletBC(V, Constant(0.0), lambda x, on_boundary: on_boundary)
u = TrialFunction(V); v = TestFunction(V);
A, b = assemble_system( inner(grad(u), grad(v))*dx, Constant(1.0)*v*dx, bc)

class KrylovMatrix(PETScKrylovMatrix) :
    def __init__(self) :
        PETScKrylovMatrix.__init__(self, V.dim(), V.dim())

    def mult(self, *args):
        y = PETScVector(V.dim())
        A.mult(args[0], y)
        args[1].set_local(y.array())

    def transpmult(self, *args):
        y = PETScVector(V.dim())
        A.transpmult(args[0], y)
        args[1].set_local(y.array())

    def dependencies(self):
      return []

y = Function(V)
solve(A, y.vector(), b, "cg", "none")

x = Function(V)
KrylovSolver = AdjointPETScKrylovSolver("cg","none")
KrylovSolver.solve(KrylovMatrix(), down_cast(x.vector()), down_cast(b))

print (y.vector()-x.vector()).norm("l2")
