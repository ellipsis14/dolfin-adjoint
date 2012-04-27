from dolfin import *
from dolfin_adjoint import *

n = 30
mesh = UnitInterval(n)
V = FunctionSpace(mesh, "CG", 2)

ic = project(Expression("sin(2*pi*x[0])"),  V)
u = Function(ic)
u_next = Function(V)
v = TestFunction(V)

nu = Constant(0.0001)

timestep = Constant(1.0/n)

F = ((u_next - u)/timestep*v
     + u_next*grad(u_next)*v + nu*grad(u_next)*grad(v))*dx
bc = DirichletBC(V, 0.0, "on_boundary")

t = 0.0
end = 0.2
while (t <= end):
    solve(F == 0, u_next, bc)
    u.assign(u_next)
    t += float(timestep)

J = FinalFunctional(inner(u, u)*dx)
dJdnu = compute_gradient(J, ScalarParameter(nu))
