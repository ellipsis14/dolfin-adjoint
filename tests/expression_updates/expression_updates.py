from dolfin import *
from dolfin_adjoint import *

mesh = UnitSquareMesh(4,4)
V = FunctionSpace(mesh, "CG", 1)

u = TrialFunction(V)
v = TestFunction(V)

a = inner(grad(u),  grad(v))*dx
u =  Function(V, name = "solution")

bc = DirichletBC(V, 0, "on_boundary")

source = Expression("t*sin(x[0])*sin(x[1])", t = 0.0)
f = source*v*dx

t = 0.0
dt = 0.1

A = assemble(a)
for i in range(2):
    t += dt
    source.t = t
    F = assemble(f)
    bc.apply(A)
    bc.apply(F)
    solve(A, u.vector(), F, "cg", "ilu")

adj_html("forward.html", "forward")
assert replay_dolfin()

