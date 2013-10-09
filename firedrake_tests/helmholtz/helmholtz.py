"""This demo program solves Helmholtz's equation

  - div grad u(x, y) + u(x,y) = f(x, y)

on the unit square with source f given by

  f(x, y) = (1.0 + 8.0*pi**2)*cos(x[0]*2*pi)*cos(x[1]*2*pi)

and the analytical solution

  u(x, y) = cos(x[0]*2*pi)*cos(x[1]*2*pi)
"""

# Begin demo
from firedrake import *
from firedrake_adjoint import *
# Create mesh and define function space
n = 5
mesh = UnitSquareMesh(2 ** n, 2 ** n)
V = FunctionSpace(mesh, "CG", 1)

def model(s):
    # Define variational problem
    lmbda = 1
    u = TrialFunction(V)
    v = TestFunction(V)
    a = (dot(grad(v), grad(u)) + lmbda * v * u) * dx
    L = s * v * dx

    # Compute solution
    assemble(a)
    assemble(L)
    x = Function(V, name="State")
    solve(a == L, x)

    # Analytical solution
    f = Function(V)
    f.interpolate(Expression("cos(x[0]*pi*2)*cos(x[1]*pi*2)"))
    
    j = assemble(dot(x - f, x - f) * dx)
    return j, x, f

if __name__ == '__main__':

    s = Function(V)
    s.interpolate(Expression("(1+8*pi*pi)*cos(x[0]*pi*2)*cos(x[1]*pi*2)"))

    print "Running forward model"
    j, x, f = model(s)
  
    adj_html("forward.html", "forward")
    print "Replaying forward model"
    print replay_dolfin(tol=0.0, stop=True)

    J = Functional(inner(x-f, x-f)*dx*dt[FINISH_TIME]) 
    m = InitialConditionParameter(s)

    print "Running adjoint model"
    dJdm = compute_gradient(J, m, forget=None)

    parameters["adjoint"]["stop_annotating"] = True

    Jhat = lambda s: model(s)[0]
    conv_rate = taylor_test(Jhat, m, j, dJdm)
    assert conv_rate > 1.9
