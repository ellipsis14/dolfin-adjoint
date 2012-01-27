"""
Naive implementation of Burgers' equation, goes oscillatory later
"""

import sys

from dolfin import *
from dolfin_adjoint import *

n = 100
mesh = UnitInterval(n)
V = FunctionSpace(mesh, "CG", 2)

parameters["num_threads"] = 2

debugging["record_all"] = True
#debugging["test_hermitian"] = (100, 1.0e-14)
#debugging["test_derivative"] = 6

def Dt(u, u_, timestep):
    return (u - u_)/timestep

def main(ic, annotate=False):

    u_ = ic
    u = TrialFunction(V)
    v = TestFunction(V)

    nu = Constant(0.0001)

    timestep = Constant(1.0/n)

    F = (Dt(u, u_, timestep)*v
         + u_*grad(u)*v + nu*grad(u)*grad(v))*dx

    (a, L) = system(F)
    bc = DirichletBC(V, 0.0, "on_boundary")

    class KrylovMatrix(PETScKrylovMatrix):
      def __init__(self, a):
        shapes = self.shape(a)
        PETScKrylovMatrix.__init__(self, shapes[0], shapes[1])
        self.original_form = a
        self.current_form = a

      def shape(self, a):
        args = ufl.algorithms.extract_arguments(a)
        shapes = [arg.function_space().dim() for arg in args]
        return shapes

      def mult(self, *args):
        shapes = self.shape(self.current_form)
        y = PETScVector(shapes[0])

        action_fn = Function(ufl.algorithms.extract_arguments(self.current_form)[-1].function_space())
        action_vec = action_fn.vector()
        for i in range(len(args[0])):
          action_vec[i] = args[0][i]

        action_form = action(self.current_form, action_fn)
        assemble(action_form, tensor=y)
        bc.apply(y)
        args[1].set_local(y.array())

      def transpmult(self, *args):
        shapes = self.shape(self.current_form)
        y = PETScVector(shapes[1])
        action_form = action(adjoint(self.current_form), args[0])
        assemble(action_form, tensor=y)
        bc.apply(y)
        args[1].set_local(y.array())

      def dependencies(self):
        return ufl.algorithms.extract_coefficients(self.original_form)

      def set_dependencies(self, dependencies, values):
        replace_dict = dict(zip(self.dependencies(), values))
        self.current_form = replace(self.original_form, replace_dict)

    t = 0.0
    end = 0.025
    u = Function(V)

    KrylovSolver = AdjointPETScKrylovSolver("gmres", "none")
    MatFreeBurgers = KrylovMatrix(a)

    while (t <= end):
        b_rhs = assemble(L)
        bc.apply(b_rhs)
        KrylovSolver.solve(MatFreeBurgers, down_cast(u.vector()), down_cast(b_rhs), annotate=annotate)

        u_.assign(u, annotate=annotate)

        t += float(timestep)
        #plot(u)

    #interactive()
    return u_

if __name__ == "__main__":

    ic = project(Expression("sin(2*pi*x[0])"),  V)
    ic_copy = Function(ic)
    forward = main(ic, annotate=True)
    forward_copy = Function(forward)
    adj_html("burgers_matfree_forward.html", "forward")
    adj_html("burgers_matfree_adjoint.html", "adjoint")

    replay_dolfin()

#    print "Running adjoint ... "
#    J = FinalFunctional(forward*forward*dx)
#    adjoint = adjoint_dolfin(J, forget=False)
#
#    def Jfunc(ic):
#      forward = main(ic, annotate=False)
#      return assemble(forward*forward*dx)
#
#    ic.vector()[:] = ic_copy.vector()
#    minconv = test_initial_condition_adjoint(Jfunc, ic, adjoint, seed=1.0e-3)
#    if minconv < 1.9:
#      sys.exit(1)
#
#    ic.vector()[:] = ic_copy.vector()
#    dJ = assemble(derivative(forward_copy*forward_copy*dx, forward_copy))
#    minconv = test_initial_condition_tlm(Jfunc, dJ, ic, seed=1.0e-5)
#    if minconv < 1.9:
#      sys.exit(1)
