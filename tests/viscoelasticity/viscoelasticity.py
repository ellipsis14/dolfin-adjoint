"""
Standard linear solid (SLS) viscoelastic model:

  A_E^0 \dot \sigma_0 + A_V^0 \sigma_0 = strain(u)
  A_E^1 \dot \sigma_1 = strain(v)

  \sigma = \sigma_0 + \sigma_1

  \div \sigma = g

  \skew \sigma = 0
"""

from dolfin import *
from dolfin import div as d

def div(v):
    return as_vector((d(v[0]), d(v[1]), d(v[2])))

# Vectorized skew
def skw(tau):
    s = 2*skew(tau)
    return as_vector((s[0][1], s[0][2], s[1][2]))


# Compliance tensors
def A00(tau):
    mu = 5.0; lamda = 100.0
    foo = 1.0/(2*mu)*(tau - lamda/(2*mu + 3*lamda)*tr(tau)*Identity(3))
    return foo

def A10(tau):
    mu = 30.0; lamda = 100.0
    foo = 1.0/(2*mu)*(tau - lamda/(2*mu + 3*lamda)*tr(tau)*Identity(3))
    return foo

def A11(tau):
    mu = 20.0; lamda = 100.0
    foo = 1.0/(2*mu)*(tau - lamda/(2*mu + 3*lamda)*tr(tau)*Identity(3))
    return foo

def main():
    parameters["form_compiler"]["optimize"] = True
    parameters["form_compiler"]["cpp_optimize"] = True
    set_log_level(DEBUG)

    dt = 0.01
    T = 1.0
    coarseness = 4
    #mesh = Mesh("mesh_edgelength%d.xml.gz" % coarseness)
    #mesh = UnitCube(3, 3, 3)
    mesh = Box(0., 0., 0., 0.5, 0.5, 1.0, 4, 4, 8)
    #plot(mesh, interactive=True)

    # Mark all facets by 0, exterior facets by 1, and then top and
    # bottom by 2
    boundaries = FacetFunction("uint", mesh)
    boundaries.set_all(0)
    on_bdry = AutoSubDomain(lambda x, on_boundary: on_boundary)
    top = AutoSubDomain(lambda x, on_boundary: near(x[2], 1.0))
    bottom = AutoSubDomain(lambda x, on_boundary: near(x[2], 0.0))
    on_bdry.mark(boundaries, 1)
    top.mark(boundaries, 2)
    bottom.mark(boundaries, 2)
    ds = Measure("ds")[boundaries]

    # Define function spaces
    S = VectorFunctionSpace(mesh, "BDM", 1)
    V = VectorFunctionSpace(mesh, "DG", 0)
    Q = VectorFunctionSpace(mesh, "DG", 0)
    Z = MixedFunctionSpace([S, S, V, Q])

    # Define trial and test functions
    (sigma0, sigma1, v, gamma) = TrialFunctions(Z)
    (tau0, tau1, w, eta) = TestFunctions(Z)

    f_ = Function(V)
    f = Function(V)

    z_ = Function(Z)
    (sigma0_, sigma1_, v_, gamma_) = split(z_)

    #z_star = Function(Z)
    z = Function(Z)

    k_n = Constant(dt)
    t = dt

    def avg(q, q_):
        return 0.5*(q + q_)

    sigma0_mid = avg(sigma0, sigma0_)
    sigma1_mid = avg(sigma1, sigma1_)
    v_mid = avg(v, v_)
    gamma_mid = avg(gamma, gamma_)
    f_mid = avg(f, f_)

    n = FacetNormal(mesh)

    v_D_mid = Function(V) # 0.5*(v^* +  v^n) Dirichlet condition
    #v_D_mid = Expression(("0.0", "0.0", "x[2]*(1-x[2])*sin(t)"), t=0)

    # Boundary traction
    #p = Expression("sin(2*pi*t)*x[2]", t=0)
    #p = Expression("x[2]")
    p = Expression("1.0")
    g = p*n
    beta = Constant(10000.0)
    h = tetrahedron.volume

    F = (inner(inv(k_n)*A10(sigma0 - sigma0_), tau0)*dx
         + inner(A00(sigma0_mid), tau0)*dx
         + inner(inv(k_n)*A11(sigma1 - sigma1_), tau1)*dx
         + inner(div(tau0 + tau1), v_mid)*dx
         + inner(skw(tau0 + tau1), gamma_mid)*dx
         + inner(div(sigma0_mid + sigma1_mid), w)*dx
         + inner(skw(sigma0_mid + sigma1_mid), eta)*dx
         #- inner(f_mid, w)*dx # Zero body source
         #- inner(v_mid, (tau0 + tau1)*n)*ds(1) # Traction bdry cf penalty
         - inner(v_D_mid, (tau0 + tau1)*n)*ds(2) # Velocity on dO_D
         )

    # Tricky to enforce Dirichlet boundary conditions on varying sums
    # of components (same deal as for slip for Stokes for
    # instance). Use penalty instead
    #F_penalty = (beta*inv(h)*inner((tau0 + tau1)*n,
    #                               (sigma0 + sigma1)*n - g)*ds(1))
    F_penalty = (beta*inv(h)*inner((tau0 + tau1)*n,
                                   (sigma0_mid + sigma1_mid)*n - g)*ds(1))

    F = F + F_penalty

    (a, L) = system(F)
    A = assemble(a)
    solver = LUSolver(A)
    solver.parameters["reuse_factorization"] = True

    current_v = Function(V)
    pvds = File("results/velocities.pvd")
    while (t <= T):
        print "t = ", t
        #p.t = t
        #v_D_mid.t = t

        # Assemble right-hand side
        b = assemble(L)

        # Solve system
        solver.solve(z.vector(), b)

        # Update time
        z_.assign(z)
        current_v.assign(z.split()[2])
        plot(current_v)
        pvds << current_v
        t += dt

    interactive()


if __name__ == "__main__":

    main()
