from dolfin import *
from dolfin_adjoint import *
import ufl

parameters["adjoint"]["debug_cache"] = True
ufl.set_level(INFO)

def main(c, annotate=False):
    # Prepare a mesh
    mesh = UnitIntervalMesh(50)

    # Define function space
    U = FunctionSpace(mesh, "DG", 2)

    # Set some expressions
    uinit = Expression("cos(pi*x[0])")
    ubdr = Constant(1.0)
    #c = Constant(1.0)

    # Set initial values
    u0 = interpolate(uinit, U)

    # Define test and trial functions
    v = TestFunction(U)
    u = TrialFunction(U)

    # Set time step size
    DT = Constant(2.e-5)
    
    # Define fluxes on interior and exterior facets
    uhat    = avg(u0) + 0.25*jump(u0)
    uhatbnd = -u0 + 0.25*(u0-ubdr)

    # Define variational formulation
    a = u*v*dx
    L = (u0*v + DT*c*u0*v.dx(0))*dx \
        -DT* uhat * jump(v)*dS \
        -DT* uhatbnd * v*ds

    # Prepare solution
    u_ls = Function(U)

    # Prepare LocalSolver
    local_solver = LocalSolver(a, L)

    # The acutal timestepping
    for i in range(2):
        local_solver.solve(u_ls.vector())
#        u0.assign(u_ls)
    
    return u_ls

if __name__ == "__main__":
    c = Constant(1.0, name="BoundaryValue")
    u = main(c, annotate=True)

#    plot(u, interactive=True)
#    v = main(Constant(1.1), annotate=False)
#    plot(v, interactive=True)
#    print "assemble(inner(u, u)*dx): ", assemble(inner(u, u)*dx)
#    print "assemble(inner(v, v)*dx): ", assemble(inner(v, v)*dx)
#    import sys; sys.exit(1)


    info_blue("Replaying")
    replay_dolfin(forget=False, tol=0.0, stop=True)

    info_blue("Computing adjoint")
    J = Functional(inner(u, u)*dx)
    m = Control(c)
    Jm = assemble(inner(u, u)*dx)
    dJdm = compute_gradient(J, m, forget=False)

    def Jhat(m):
        print "Evaluating with c: ", float(m)
        u = main(m, annotate=False)
        J = assemble(inner(u, u)*dx)
        print "Functional: ", J
        return J

    minconv = taylor_test(Jhat, m, Jm, dJdm, seed=1.0e-3)
    assert minconv > 1.8
