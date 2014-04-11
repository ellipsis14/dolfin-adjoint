""" Solves a optimal control problem constrained by the Poisson equation:

    min_(u, m) \int_\Omega 1/2 || u - d ||^2 + 1/2 || f ||^2
    
    subjecct to 

    grad \cdot \grad u = f    in \Omega
    u = 0                     on \partial \Omega


"""
from dolfin import *
from dolfin_adjoint import *
import moola

# Create mesh, refined in the center
n = 64
mesh = UnitSquareMesh(n, n)

cf = CellFunction("bool", mesh)
subdomain = CompiledSubDomain('std::abs(x[0]-0.5)<.25 && std::abs(x[1]-0.5)<0.25')
subdomain.mark(cf, True)
mesh = refine(mesh, cf)

# Define discrete function spaces and funcions
V = FunctionSpace(mesh, "CG", 1)
W = FunctionSpace(mesh, "DG", 0)

f = Function(W, name='Control')
u = Function(V, name='State')
v = TestFunction(V)

# Define and solve the Poisson equation to generate the dolfin-adjoint annotation
F = (inner(grad(u), grad(v)) - f*v)*dx 
bc = DirichletBC(V, 0.0, "on_boundary")
solve(F == 0, u, bc)

# Define functional of interest and the reduced functional
x = triangle.x
d = 1/(2*pi**2)*sin(pi*x[0])*sin(pi*x[1]) # the desired temperature profile

alpha = Constant(1e-2)
J = Functional((0.5*inner(u-d, u-d))*dx + alpha/2*f**2*dx)
control = SteadyParameter(f)
rf = ReducedFunctional(J, control)

# Solve the optimsiation problem with the Moola optimisation package
problem = rf.moola_problem()
solver = moola.NewtonCG(options={'gtol': 1e-5, 'maxiter': 20})
f_moola = moola.DolfinPrimalVector(f)
sol = solver.solve(problem, f_moola)
f_opt = sol['Optimizer'].data

plot(f_opt, interactive=True)

# Define the expressions of the analytical solution
f_analytic = Expression("sin(pi*x[0])*sin(pi*x[1])")
u_analytic = Expression("1/(2*pi*pi)*sin(pi*x[0])*sin(pi*x[1])")

# Compute errors between numerical and analytical solutions
f.assign(f_opt)  # Solve the Poisson problem again for the optimal m
solve(F == 0, u, bc)  
control_error = errornorm(f_analytic, f_opt)
state_error = errornorm(u_analytic, u)
print "Error in state: {}.".format(state_error)
print "Error in control: {}.".format(control_error)
