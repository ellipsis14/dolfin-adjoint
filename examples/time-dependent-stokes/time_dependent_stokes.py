"""
This example demonstrates a tracking type optimal control problem for
time-dependent Stokes flow.

In particular, both the constraint and functional is time-dependent.

For details, see "Optimal error estimates and computations for
tracking-type control of the instationary Stokes system", Deckelnick
and Hinze, 2001.

Implicit Euler discretization in time and Taylor-Hood elements in
space

"""

__author__ = "Marie E. Rognes (meg@simula.no)"

from dolfin import *
from dolfin_adjoint import *

parameters["form_compiler"]["optimize"] = True
parameters["form_compiler"]["cpp_optimize"] = True

set_log_level(PROGRESS)

def forward(mesh, T):

    # Define Taylor-Hood function spaces
    V = VectorFunctionSpace(mesh, "CG", 2)
    Q = FunctionSpace(mesh, "CG", 1)
    W = V*Q

    # Definition of the initial condition for y
    y0 = Expression(("exp(1)*(cos(2*pi*x[0]) - 1)*sin(2*pi*x[1])",
                     "- exp(1)*(cos(2*pi*x[1]) - 1)*sin(2*pi*x[0])"), degree=2)
    y0 = interpolate(y0, V)

    # Assign y0 to the first component of w_:
    w_ = Function(W)
    dolfin.assign(w_.sub(0), y0)
    (y_, p_) = split(w_)

    # Define trial and test functions
    w = Function(W)
    (y, p) = TrialFunctions(W)
    (phi, q) = TestFunctions(W)

    # Define model parameters
    Re = 10.0
    nu = Constant(1.0/Re)
    dT = Constant(0.01)
    num_steps = int(T/float(dT))
    print "Number of timesteps: %i." % num_steps

    # Define initial control functions.
    # The control is time-dependent, hence the control is a list
    # if functions in V of length T/dT
    u = [Function(V)]*num_steps

    # Preassemble the left hand side of the weak Stokes equations
    a  = (inner(y, phi)*dx + dT*inner(nu*grad(y), grad(phi))*dx
          - dT*inner(div(phi), p)*dx)
    a += - dT*inner(div(y), q)*dx
    A = assemble(a)

    # Define right hand vector side for later reuse
    b = Vector(mesh.mpi_comm(), W.dim())

    # Define Dirichlet boundary condition
    bc = DirichletBC(W.sub(0), (0.0, 0.0), "on_boundary")
    bc.apply(A)

    # Create (suboptimal) Krylov solver
    solver = KrylovSolver(A, "gmres")

    # Create null space basis and attach it to the Krylov solver
    null_vec = Vector(w.vector())
    Q.dofmap().set(null_vec, 1.0)
    null_vec *= 1.0/null_vec.norm("l2")
    nullspace = VectorSpaceBasis([null_vec])
    solver.set_nullspace(nullspace)

    # For the adjoint, the transpose nullspace must also be set. This
    # system is symmetric, so the transpose nullspace is the same as
    # the nullspace
    solver.set_transpose_nullspace(nullspace);

    # Time-stepping
    progress = Progress("Time-stepping", num_steps)
    for n in range(num_steps):

        # Assemble left hand side and apply boundary condition
        L = inner(dT*u[n] + y_, phi)*dx
        assemble(L, tensor=b)
        bc.apply(b)

        # Solve linear system
        solver.solve(w.vector(), b)

        # Plot solution
        #(y, p) = w.split(deepcopy=True)
        #plot(p, title="p")

        # Update previous solution
        w_.assign(w)

        progress += 1

    return (y, p, u)


if __name__ == "__main__":

    # Define mesh and end time
    n = 20
    mesh = UnitSquareMesh(n, n)
    T = 0.1

    # Set-up forward problem
    (y, p, u) = forward(mesh, T)

    # Define tracking type functional:
    alpha = 1.0
    #alpha = 0.01
    z = Constant((0.0, 0.0))
    j = 1./2*inner(y-z, y-z)*dx*dt #+ alpha/2.*u**2*dx*dt
    J = Functional(j)
