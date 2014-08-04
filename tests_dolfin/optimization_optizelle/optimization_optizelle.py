""" Solves a optimal control problem constrained by the Poisson equation:

    min_(u, m) \int_\Omega 1/2 || u - d ||^2 + 1/2 || f ||^2

    subjecct to

    grad \cdot \grad u = f    in \Omega
    u = 0                     on \partial \Omega


"""
from dolfin import *
from dolfin_adjoint import *
import Optizelle
set_log_level(ERROR)

parameters["adjoint"]["cache_factorizations"] = True

# Create mesh
n = 6
mesh = UnitSquareMesh(n, n)

# Define discrete function spaces and funcions
V = FunctionSpace(mesh, "CG", 1)
W = FunctionSpace(mesh, "DG", 0)

f = interpolate(Expression("0.11"), W, name='Control')
u = Function(V, name='State')
v = TestFunction(V)

# Define and solve the Poisson equation to generate the dolfin-adjoint annotation
F = (inner(grad(u), grad(v)) - f*v)*dx 
bc = DirichletBC(V, 0.0, "on_boundary")
solve(F == 0, u, bc)

# Define functional of interest and the reduced functional
x = SpatialCoordinate(mesh)
d = 1/(2*pi**2)*sin(pi*x[0])*sin(pi*x[1]) # the desired temperature profile

alpha = Constant(1e-6)
J = Functional((0.5*inner(u-d, u-d))*dx + alpha/2*f**2*dx)
control = SteadyParameter(f)
rf = ReducedFunctional(J, control)

ub = Function(W)
ub.vector()[:] = 0.5
lb = -0.5
problem = MinimizationProblem(rf, bounds=[(lb, ub)])
parameters = {
             "maximum_iterations": 10,
             "optizelle_parameters":
                 {
                 "msg_level" : 10,
                 "algorithm_class" : Optizelle.AlgorithmClass.TrustRegion,
                 "H_type" : Optizelle.Operators.UserDefined,
                 "dir" : Optizelle.LineSearchDirection.NewtonCG,
                 "ipm": Optizelle.InteriorPointMethod.PrimalDualLinked,
                 "sigma": 0.001,
                 "gamma": 0.995,
                 "linesearch_iter_max" : 50,
                 "krylov_iter_max" : 100,
                 "eps_krylov" : 1e-4
                 }
             }

solver = OptizelleSolver(problem, parameters=parameters)
f_opt = solver.solve()
cmax = f_opt.vector().max()
cmin = f_opt.vector().min()

# Check that the bounds are satisfied
assert cmin >= lb
assert cmax <= ub.vector().max()

# Check that the functional value is below the threshold
assert rf(f_opt) < 0.6e-4
