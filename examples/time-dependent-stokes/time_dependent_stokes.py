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

set_log_level(PROGRESS)
parameters["form_compiler"]["optimize"] = True
parameters["form_compiler"]["cpp_optimize"] = True

#def J(y, z, u, alpha):
#    1./2*inner(y - z, y - z)*dx + alpha/2.*inner(u, u)*dx

def forward(u, m, mesh, T, J=None):

    # Define Taylor-Hood function spaces
    V = VectorFunctionSpace(mesh, "CG", 2)
    Q = FunctionSpace(mesh, "CG", 1)
    W = V*Q

    # Definition of the initial condition for y
    y0 = Expression(("exp(1)*(cos(2*pi*x[0]) - 1)*sin(2*pi*x[1])",
                     "- exp(1)*(cos(2*pi*x[1]) - 1)*sin(2*pi*x[0])"), degree=2)
    y0 = interpolate(y0, V)

    # Assign y0 to the first component of w_:
    w_ = Function(W, name="Initial velocity-pressure")
    dolfin.assign(w_.sub(0), y0) # FIXME
    (y_, p_) = split(w_)

    # Define trial and test functions
    w = Function(W, name="Velocity-pressure")
    (y, p) = TrialFunctions(W)
    (phi, q) = TestFunctions(W)

    # Define model parameters
    Re = 10.0
    nu = Constant(1.0/Re)

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
    #solver = LUSolver(A)

    if isinstance(solver, KrylovSolver):
        # Create null space basis and attach it to the Krylov solver
        null_vec = Vector(w.vector())
        Q.dofmap().set(null_vec, 1.0)
        null_vec *= 1.0/null_vec.norm("l2")
        nullspace = VectorSpaceBasis([null_vec])
        solver.set_nullspace(nullspace)

        # For the adjoint, the transpose nullspace must also be
        # set. This system is symmetric, so the transpose nullspace is
        # the same as the nullspace
        solver.set_transpose_nullspace(nullspace);

    # Time-stepping
    progress = Progress("Time-stepping", num_steps)
    t = 0.0
    adj_start_timestep(t)
    for n in range(num_steps):

        t += dT

        u.assign(m[n], annotate=True)
        #u.assign(m[n])

        # Assemble left hand side and apply boundary condition
        L = inner(dT*u + y_, phi)*dx
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

        # Update the adjoint time-step
        # FUTURE: this can go away
        if n == (num_steps - 1):
            adj_inc_timestep(t, finished=True)
        else:
            adj_inc_timestep(t)

    return w


if __name__ == "__main__":

    # Define mesh and time parameters
    n = 16
    mesh = UnitSquareMesh(n, n)
    T = 0.1
    dT = Constant(0.001)
    num_steps = int(T/float(dT))
    print "Number of timesteps: %i." % num_steps

    # Define the control function u
    V = VectorFunctionSpace(mesh, "CG", 2)
    u = Function(V, name="Control")
    m = [Function(V, name="Control_%d" % i) for i in range(num_steps)]

    # Set-up forward problem
    w = forward(u, m, mesh, T)
    (y, p) = split(w)

    # Define tracking type functional via the observations z:
    alpha = 1.0
    #alpha = 0.01
    z = Function(w.function_space().sub(0).collapse(), name="Observations")

    def trapezoidal(u, dT, alpha):
        N = len(u)
        value = (0.5*alpha/2.0*dT*inner(u[0], u[0])*dx
                 + sum(alpha/2.0*dT*inner(m, m)*dx for m in u[1:N-2])
                 + 0.5*alpha/2.0*dT*inner(u[N-1], u[N-1])*dx)
        return value

    # Note that y and z are the same dolfin type objects here, but
    # dolfin-adjoint treats them differently because of what has
    # happened to them though the course of the tape. y has varied, z
    # has not.
    #j = 1./2*inner(y-z, y-z)*dx*dt + trapezoidal(u, dT, alpha)
    j = 1./2*inner(y-z, y-z)*dx*dt + alpha/2.*inner(u, u)*dx*dt
    J = Functional(j)

    # Define the reduced functional (see tutorial on PDE-constrained
    # optimisation)
    #
    # Jtilde = ReducedFunctional(J, u) # FUTURE
    #replay_dolfin(forget=False, stop=True)

    controls = map(SteadyParameter, m)
    #compute_gradient(J, controls)

    # Optimize
    set_log_level(ERROR)
    Jtilde = ReducedFunctional(J, controls)
    u_opt = minimize(Jtilde, tol=1.e-10)
    [plot(m) for m in u_opt]
    interactive()

# Problems I ran into
# 1. What Parameter to use
#
# 3.
#   FErari not installed, skipping tensor optimizations
# Traceback (most recent call last):
#   File "_ctypes/callbacks.c", line 314, in 'calling callback function'
#   File "/usr/lib/python2.7/dist-packages/libadjoint/libadjoint.py", line 494, in cfunc
#     output = self.derivative(adjointer, variable, dependencies, values)
#   File "/home/meg/local/fenics-dev/lib/python2.7/site-packages/dolfin_adjoint/functional.py", line 128, in derivative
#     self._substitute_form(adjointer, timestep, dependencies, values))
#   File "/home/meg/local/fenics-dev/lib/python2.7/site-packages/dolfin_adjoint/functional.py", line 213, in _substitute_form
#     functional_value = _add(functional_value, trapezoidal(integral_interval, 0))
#   File "/home/meg/local/fenics-dev/lib/python2.7/site-packages/dolfin_adjoint/functional.py", line 208, in trapezoidal
#     quad_weight = 0.5*(this_interval.stop-this_interval.start)
# TypeError: unsupported operand type(s) for -: 'FinishTimeConstant' and 'StartTimeConstant'
# Conclusion: need to label timesteps

# 4. Tutorial/Manual should be available as pdf so that one can search
# through the entire thing.

# 5. %
# iteration_count
#     clib.adj_iteration_count(adjointer.adjointer, self.var, iteration_count)
#   File "/usr/lib/python2.7/dist-packages/libadjoint/libadjoint.py", line 18, in handle_error
#     raise exception, errstr
# libadjoint.exceptions.LibadjointErrorInvalidInputs: Error in adj_iteration_count: No iteration found for supplied variable Control_2:8:0:Forward.
