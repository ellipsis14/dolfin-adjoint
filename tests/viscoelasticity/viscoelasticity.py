"""
Standard linear solid (SLS) viscoelastic model:

  A_E^0 \dot \sigma_0 + A_V^0 \sigma_0 = strain(u)
  A_E^1 \dot \sigma_1 = strain(v)

  \sigma = \sigma_0 + \sigma_1

  \div \sigma = g

  \skew \sigma = 0
"""

import sys

from dolfin import *
from dolfin import div as d

from dolfin_adjoint import *
debugging["record_all"] = True

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

def get_box():
    mesh = Box(0., 0., 0., 0.2, 0.2, 1.0, 3, 3, 3)

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

    return (mesh, boundaries)

def get_spinal_cord():
    mesh = Mesh("mesh_edgelength4.xml.gz")
    boundaries = mesh.domains().facet_domains(mesh)
    for (i, a) in enumerate(boundaries.array()):
        if a > 10:
            boundaries.array()[i] = 0
        if a == 3:
            boundaries.array()[i] = 2

    print "Boundary markers: ", set(boundaries.array())
    return (mesh, boundaries)

#(mesh, boundaries) = get_spinal_cord()
(mesh, boundaries) = get_box()
dt = 0.01
T = 0.02
ds = Measure("ds")[boundaries]

# Define function spaces
S = VectorFunctionSpace(mesh, "BDM", 1)
V = VectorFunctionSpace(mesh, "DG", 0)
Q = VectorFunctionSpace(mesh, "DG", 0)
CG1 = VectorFunctionSpace(mesh, "CG", 1)
Z = MixedFunctionSpace([S, S, V, Q])

def main(ic=None, annotate=False):
    parameters["form_compiler"]["optimize"] = True
    parameters["form_compiler"]["cpp_optimize"] = True
    set_log_level(DEBUG)

    # Define trial and test functions
    (sigma0, sigma1, v, gamma) = TrialFunctions(Z)
    (tau0, tau1, w, eta) = TestFunctions(Z)

    f_ = Function(V)
    f = Function(V)

    z_ = Function(Z)
    if ic is not None:
      z_ = Function(ic)

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

    v_D = Function(V)     # Velocity boundary value
    v_D_mid = Function(V) # 0.5*(v^* +  v^n) Dirichlet condition

    # Boundary traction (pressure originating from CSF flow)
    p = Expression("sin(2*pi*t)*(1.0/(171 - 78)*(x[2] - 78))", t=0)
    # p = Expression("sin(2*pi*t)*x[2]", t=0) # For box
    g = - p*n
    beta = Constant(10000.0)
    h = tetrahedron.volume

    disc = "BE"
    if disc == "TR":
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
        # F_penalty = (beta*inv(h)*inner((tau0 + tau1)*n,
        #                               (sigma0 + sigma1)*n - g)*ds(1))
        F_penalty = (beta*inv(h)*inner((tau0 + tau1)*n,
                                       (sigma0_mid + sigma1_mid)*n - g)*ds(1))

        F = F + F_penalty
    elif disc == "BE":
        F = (inner(inv(k_n)*A10(sigma0 - sigma0_), tau0)*dx
             + inner(A00(sigma0), tau0)*dx
             + inner(inv(k_n)*A11(sigma1 - sigma1_), tau1)*dx
             + inner(div(tau0 + tau1), v)*dx
             + inner(skw(tau0 + tau1), gamma)*dx
             + inner(div(sigma0 + sigma1), w)*dx
             + inner(skw(sigma0 + sigma1), eta)*dx
             )

        # Use penalty to enforce essential bc on sigma
        F_penalty = (beta*inv(h)*inner((tau0 + tau1)*n,
                                       (sigma0 + sigma1)*n - g)*ds(1))
        F = F + F_penalty


    (a, L) = system(F)
    A = assemble(a)
    solver = LUSolver(A)
    solver.parameters["reuse_factorization"] = True

    velocities = File("results/velocities.pvd")
    displacements = File("results/displacement.pvd")
    displacement = Function(CG1)
    while (t <= T):

        # Update source(s)
        print "t = ", t
        p.t = t

        # Assemble right-hand side
        b = assemble(L)

        # Solve system
        solver.solve(z.vector(), b, annotate=annotate)

        return z

        # Store velocities and displacements
        cg_v = project(z.split()[2], CG1)
        cg_d = project(displacement + k_n*z.split()[2], CG1)
        displacements << cg_d
        velocities << cg_v

        # Update time
        z_.assign(z)
        displacement.assign(cg_d)
        t += dt

    return z_


if __name__ == "__main__":

    ic = Function(Z)
    ic_copy = Function(ic)
    z = main(ic, annotate=True)

    info_blue("Replaying forward run ... ")
    adj_html("forward.html", "forward")
    replay_dolfin(forget=False)

    J = FinalFunctional(inner(z, z)*dx)
    info_blue("Running adjoint ... ")
    adjoint = adjoint_dolfin(J, forget=False)

    def Jfunc(ic):
      z = main(ic, annotate=False)
      J = assemble(inner(z, z)*dx)
      print "J(.): ", J
      return J

    ic.vector()[:] = ic_copy.vector()
    info_blue("Checking adjoint correctness ... ")
    minconv = test_initial_condition_adjoint(Jfunc, ic, adjoint, seed=1.0e-5)

    if minconv < 1.9:
      sys.exit(1)
