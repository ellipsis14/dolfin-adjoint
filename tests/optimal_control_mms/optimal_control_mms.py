""" Solves a MMS problem with smooth control """

from dolfin import *
from dolfin_adjoint import *

dolfin.set_log_level(ERROR)
parameters['std_out_all_processes'] = False

x = triangle.x

def solve_pde(u, V, m):
    v = TestFunction(V)
    F = (inner(grad(u), grad(v)) - m*v)*dx 
    bc = DirichletBC(V, 0.0, "on_boundary")
    solve(F == 0, u, bc)

def solve_optimal_control(n):
    mesh = UnitSquare(n, n)
    V = FunctionSpace(mesh, "CG", 1)
    u = Function(V, name='State')
    W = FunctionSpace(mesh, "DG", 0)
    m = Function(W, name='Control')

    u_d = 1/(2*pi**2)*sin(pi*x[0])*sin(pi*x[1]) 

    J = Functional((inner(u-u_d, u-u_d))*dx*dt[FINISH_TIME])
    def Jfunc(m):
      solve_pde(u, V, m)
      return assemble(inner(u-u_d, u-u_d)*dx)

    # Run the optimisation 
    optimisation.minimise(Jfunc, J, InitialConditionParameter(m), m, algorithm = 'scipy.l_bfgs_b', pgtol=1e-16, factr=1, bounds = (-1, 1), iprint = 1, maxfun = 20)
    #optimisation.minimise(Jfunc, J, InitialConditionParameter(m), m, algorithm = 'scipy.slsqp', bounds = (-1, 1), iprint = 3, iter = 60)
    Jfunc(m)

    m_analytic = sin(pi*x[0])*sin(pi*x[1]) 
    u_analytic = 1/(2*pi**2)*sin(pi*x[0])*sin(pi*x[1])

    control_error = assemble(inner(m_analytic-m, m_analytic-m)*dx)
    state_error = assemble(inner(u_analytic-u, u_analytic-u)*dx)
    return control_error, state_error

control_errors = []
state_errors = []
element_sizes = []
for i in range(3,7):
    n = 2**i
    control_error, state_error = solve_optimal_control(n)
    control_errors.append(control_error)
    state_errors.append(state_error)
    element_sizes.append(1./n)

info_green("Control errors: " + str(control_errors))
info_green("Control convergence: " + str(convergence_order(control_errors, base = 2)))
info_green("State errors: " + str(state_errors))
info_green("State convergence: " + str(convergence_order(state_errors, base = 2)))

if min(convergence_order(control_errors)) < 2.0:
    sys.exit(1)
if min(convergence_order(state_errors)) < 4.0:
    sys.exit(1)
info_green("Test passed")    
