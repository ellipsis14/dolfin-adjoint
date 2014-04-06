""" Solves a optimal control problem constrained by the Poisson equation:

    min_(u, m) \int_\Omega 1/2 || u - u_d ||^2 + alpha/2 ||m||^2
    
    subjecct to 

    grad \cdot \grad u = m    in \Omega
    u = 0                     on \partial \Omega


"""
from dolfin import *
from dolfin_adjoint import *
try:
    import moola
except ImportError:
    import sys
    info_blue("Moola bindings unavailable, skipping test")
    sys.exit(0)

dolfin.set_log_level(ERROR)
parameters['std_out_all_processes'] = False
x = triangle.x

def solve_pde(u, V, m):
    v = TestFunction(V)
    F = (inner(grad(u), grad(v)) - m*v)*dx 
    bc = DirichletBC(V, 0.0, "on_boundary")
    solve(F == 0, u, bc)

def compute_errors(u, m):
    solve_pde(u, V, m)

    

    # Define the analytical expressions
    m_analytic = Expression("sin(pi*x[0])*sin(pi*x[1])")
    u_analytic = Expression("1/(2*pi*pi)*sin(pi*x[0])*sin(pi*x[1])")

    # Compute the error
    control_error = errornorm(m_analytic, m)
    state_error = errornorm(u_analytic, u)

    print "Error in state: {}.".format(state_error)
    print "Error in control: {}.".format(control_error)

    return control_error, state_error


if __name__ == "__main__":

    n = 128
    mesh = UnitSquareMesh(n, n)

    V = FunctionSpace(mesh, "CG", 1)
    u = Function(V, name='State')
    W = FunctionSpace(mesh, "DG", 0)
    m = Function(W, name='Control')

    u_d = 1/(2*pi**2)*sin(pi*x[0])*sin(pi*x[1]) 

    J = Functional((inner(u-u_d, u-u_d))*dx)

    # Run the forward model once to create the annotation
    solve_pde(u, V, m)

    # Run the optimisation 
    rf = ReducedFunctional(J, InitialConditionParameter(m, value=m))
    problem = rf.moola_problem()
    
    solver = moola.NewtonCG(options={'gtol': 1e-5, 'maxiter': 20})
    m_moola = moola.DolfinPrimalVector(m)

    sol = solver.solve(problem, m_moola)

    m_opt = sol['Optimizer'].data

    plot(m_opt, interactive=True)
    compute_errors(u, m_opt)
