""" This program optimises an control problem constrained by the Stokes equations """
from dolfin import *
from dolfin_adjoint import *

# Create a rectangluar domain with a circular hole.
rect = Rectangle(0, 0, 30, 10) 
circ = Circle(10, 5, 2.5)
domain = rect - circ
N = 50
mesh = Mesh(domain, N)

# Define function spaces (P2-P1)
V = VectorFunctionSpace(mesh, "CG", 2)  # Velocity
Q = FunctionSpace(mesh, "CG", 1)        # Pressure
W = MixedFunctionSpace([V, Q])

# Define a measure on the circle boundary
class Circle(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and (x[0]-10)**2 + (x[1]-5)**2 < 3**2

facet_marker = FacetFunction("size_t",mesh)
facet_marker.set_all(10)
Circle().mark(facet_marker,2)

ds = Measure("ds")[facet_marker]

# Define test and solution functions
v, q = TestFunctions(W)
x = TrialFunction(W)
u, p = split(x)
s = Function(W, name="State")
g = Function(V, name="Control")

# Set parameter values
nu = Constant(1)     # Viscosity coefficient
gamma = Constant(1000) # Nitsche penalty parameter
n = FacetNormal(mesh)
h = CellSize(mesh)

# Define boundary conditions
u_inflow = Expression(("x[1]*(10-x[1])/25","0"))
noslip  = DirichletBC(W.sub(0), (0, 0), "on_boundary && (x[1] >= 9.9 || x[1] < 0.1)")
inflow  = DirichletBC(W.sub(0), u_inflow, "on_boundary && x[0] <= 0.0")
bcs = [inflow, noslip]

# Define the variational formulation of the Navier-Stokes equations
a = (nu*inner(grad(u), grad(v))*dx 
        - nu*inner(grad(u)*n, v)*ds(2)
        - nu*inner(grad(v)*n, u)*ds(2)
        + gamma/h*nu*inner(u,v)*ds(2)
        - inner(p, div(v))*dx
        + inner(p*n, v)*ds(2)
        - inner(q, div(u))*dx
        + inner(q*n, u)*ds(2)
        )
L = ( - nu*inner(grad(v)*n, g)*ds(2)
        + gamma/h* nu*inner(g,v)*ds(2)
        + inner(q*n, g)*ds(2)
    )

# Solve the Stokes equations
A, b = assemble_system(a, L, bcs)
solve(A, s.vector(), b)

# Define Stress functional
u, p = split(s)
alpha = Constant(10)

J = Functional(inner(grad(u), grad(u))*dx + alpha*inner(g, g)*ds(2))
m = SteadyParameter(g)
Jhat = ReducedFunctional(J, m)

g_opt = minimize(Jhat)
plot(g_opt, title="Optimised boundary")

g.assign(g_opt)
A, b = assemble_system(a, L, bcs)
solve(A, s.vector(), b)
plot(s.sub(0), title="Velocity")
plot(s.sub(1), title="Pressure")
interactive()
