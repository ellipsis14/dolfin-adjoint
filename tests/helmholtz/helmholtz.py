"""This demo program solves Helmholtz's equation

  - div grad u(x, y) + u(x,y) = f(x, y)

on the unit square with source f given by

  f(x, y) = (1.0 + 8.0*pi**2)*cos(x[0]*2*pi)*cos(x[1]*2*pi)

and the analytical solution

  u(x, y) = cos(x[0]*2*pi)*cos(x[1]*2*pi)
"""

# Begin demo
from firedrake import *
from dolfin_adjoint import *

def model(n, degree=2):
    # Create mesh and define function space
    mesh = UnitSquareMesh(2 ** n, 2 ** n)
    V = FunctionSpace(mesh, "CG", degree)

    # Define variational problem
    lmbda = 1
    u = TrialFunction(V)
    v = TestFunction(V)
    f = Function(V)
    f.interpolate(Expression("(1+8*pi*pi)*cos(x[0]*pi*2)*cos(x[1]*pi*2)"))
    a = (dot(grad(v), grad(u)) + lmbda * v * u) * dx
    L = f * v * dx

    # Compute solution
    assemble(a)
    assemble(L)
    x = Function(V)
    solve(a == L, x)

    # Analytical solution
    f.interpolate(Expression("cos(x[0]*pi*2)*cos(x[1]*pi*2)"))
    return sqrt(assemble(dot(x - f, x - f) * dx)), x, f

if __name__ == '__main__':
    print "Running forward model"
    model(5, degree=1)

  
    adj_html("forward.html", "forward")
    print "Replaying forward model"
    print replay_dolfin(tol=0.0, stop=True)
