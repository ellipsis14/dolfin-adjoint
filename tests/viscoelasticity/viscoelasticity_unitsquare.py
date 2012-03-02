# TODO: Add copyrights

"""
Schematic drawing (starts with 1 springs, starts with 0 dashpots)

      | A10 --- A00 |
----- |             | --------
      |     A11     |

Standard linear solid (SLS) viscoelastic model:

  A_E^0 \dot \sigma_0 + A_V^0 \sigma_0 = strain(u)
  A_E^1 \dot \sigma_1 = strain(v)

  \sigma = \sigma_0 + \sigma_1

  \div \sigma = gx
  \skew \sigma = 0

NB: Mesh in mm, remember that Pa = N/m^2 = kg/(m s^2) = g/(mm s^2)
Give bc and Lame parameters in kPa -> displacements in mm, velocities
in mm/s, stresses in kPa
"""

import sys
import pylab

from dolfin import *
from dolfin import div as d

# Adjoint stuff
from dolfin_adjoint import *

factor = 1.
penalty_beta = 10**6 # NB: Sensitive to this for values less than 10^6
#penalty_beta = 0.0 # NB: Sensitive to this for values less than 10^6
dim = 2

# Vectorized div
def div(v):
    return as_vector((d(v[0]), d(v[1])))

# Scalar skew
def skw(tau):
    s = 2*skew(tau) # FIXME: Why did I put a 2 here?
    return s[0][1]

# Compliance tensors (Semi-arbitrarily chosen values and units)
def A00(tau):
    "Maxwell dashpot (eta)"
    mu = Constant(1.0) # kPa
    lamda = Constant(10) # kPa
    foo = 1.0/(2*mu)*(tau - lamda/(2*mu + dim*lamda)*tr(tau)*Identity(dim))
    return foo

def A10(tau):
    "Maxwell spring (A2)"
    mu = Constant(10)
    lamda = Constant(10) # kPa
    foo = 1.0/(2*mu)*(tau - lamda/(2*mu + dim*lamda)*tr(tau)*Identity(dim))
    return foo

def A11(tau):
    "Elastic spring (A1)"
    mu = Constant(1) # kPa
    lamda = Constant(10) # kPa
    foo = 1.0/(2*mu)*(tau - lamda/(2*mu + dim*lamda)*tr(tau)*Identity(dim))
    return foo

def get_square():
    "Use this for simple testing."
    n = 10
    mesh = UnitSquare(n, n)

    # Mark all facets by 0, exterior facets by 1, and then top and
    # bottom by 2
    boundaries = FacetFunction("uint", mesh)
    boundaries.set_all(0)
    on_bdry = AutoSubDomain(lambda x, on_boundary: on_boundary)
    top = AutoSubDomain(lambda x, on_boundary: near(x[1], 1.0))
    bottom = AutoSubDomain(lambda x, on_boundary: near(x[1], 0.0))
    on_bdry.mark(boundaries, 1)
    top.mark(boundaries, 2)
    bottom.mark(boundaries, 2)
    return (mesh, boundaries)

def crank_nicolson_step(Z, z_, k_n, g, v_D_mid, ds):

    # Define trial and test functions
    (sigma0, sigma1, v, gamma) = TrialFunctions(Z)
    (tau0, tau1, w, eta) = TestFunctions(Z)

    # Extract previous components
    (sigma0_, sigma1_, v_, gamma_) = split(z_)

    # Define midpoint values for brevity
    def avg(q, q_):
        return 0.5*(q + q_)
    sigma0_mid = avg(sigma0, sigma0_)
    sigma1_mid = avg(sigma1, sigma1_)
    v_mid = avg(v, v_)
    gamma_mid = avg(gamma, gamma_)

    # Define form
    n = FacetNormal(Z.mesh())
    F = (inner(inv(k_n)*A10(sigma0 - sigma0_), tau0)*dx
         + inner(A00(sigma0_mid), tau0)*dx
         + inner(inv(k_n)*A11(sigma1 - sigma1_), tau1)*dx
         + inner(div(tau0 + tau1), v_mid)*dx
         + inner(skw(tau0 + tau1), gamma_mid)*dx
         + inner(div(sigma0_mid + sigma1_mid), w)*dx
         + inner(skw(sigma0_mid + sigma1_mid), eta)*dx
         - inner(0.5*v_, (tau0 + tau1)*n)*ds(1)
         - inner(v_D_mid, (tau0 + tau1)*n)*ds(2) # Velocity on dO_D
         )

    # Tricky to enforce Dirichlet boundary conditions on varying sums
    # of components (same deal as for slip for Stokes for
    # instance). Use penalty instead
    beta = Constant(penalty_beta)
    h = tetrahedron.volume
    F_penalty = 0.5*(beta*inv(h)*inner((tau0 + tau1)*n,
                                       (sigma0 + sigma1)*n - g)*ds(1))
    F = F + F_penalty

    return F

def bdf2_step(Z, z_, z__, k_n, g, v_D, ds):

    # Define trial and test functions
    (sigma0, sigma1, v, gamma) = TrialFunctions(Z)
    (tau0, tau1, w, eta) = TestFunctions(Z)

    # Extract previous components
    (sigma0_, sigma1_, v_, gamma_) = split(z_)
    (sigma0__, sigma1__, v__, gamma__) = split(z__)

    # Define complete form
    n = FacetNormal(Z.mesh())
    F = (inner(inv(k_n)*A10(1.5*sigma0 - 2.*sigma0_ + 0.5*sigma0__), tau0)*dx
         + inner(A00(sigma0), tau0)*dx
         + inner(inv(k_n)*A11(1.5*sigma1 - 2.*sigma1_ + 0.5*sigma1__), tau1)*dx
         + inner(div(tau0 + tau1), v)*dx
         + inner(skw(tau0 + tau1), gamma)*dx
         + inner(div(sigma0 + sigma1), w)*dx
         + inner(skw(sigma0 + sigma1), eta)*dx
         - inner(v_D, (tau0 + tau1)*n)*ds(2)
         )

    # Enforce essential bc on stress by penalty
    beta = Constant(penalty_beta)
    h = tetrahedron.volume
    F_penalty = beta*inv(h)*inner((tau0 + tau1)*n,
                                  (sigma0 + sigma1)*n - g)*ds(1)
    F = F + F_penalty
    return F

# Quick testing for box:
(mesh, boundaries) = get_square()
p = Expression("sin(2*pi*t)*1.0*x[1]", t=0)

# Define function spaces
S = VectorFunctionSpace(mesh, "BDM", 1)
V = VectorFunctionSpace(mesh, "DG", 0)
Q = FunctionSpace(mesh, "DG", 0)
CG1 = VectorFunctionSpace(mesh, "CG", 1)
Z = MixedFunctionSpace([S, S, V, Q])

def main(ic, T=1.0, dt=0.01, annotate=False, verbose=False):
    # dk = half the timestep
    dk = dt/2.0

    parameters["form_compiler"]["optimize"] = True
    parameters["form_compiler"]["cpp_optimize"] = True

    ds = Measure("ds")[boundaries]

    # Define functions for previous timestep (z_), half-time (z_star)
    # and current (z)
    z_ = Function(ic)
    z_star = Function(Z)
    z = Function(Z)

    # Boundary conditions
    #v_D = Expression(("x[1]*(1. - x[1])*sin(2*pi*t)", "0.0"), t = 0.0)
    #v_D_mid = v_D
    v_D_mid = Function(V) # Velocity condition at half time
    v_D = Function(V)     # Velocity condition at time

    # Boundary traction (pressure originating from CSF flow)
    n = FacetNormal(mesh)
    g = - p*n

    F_cn = crank_nicolson_step(Z, z_, Constant(dk), g, v_D_mid, ds)
    (a_cn, L_cn) = system(F_cn)
    A_cn = assemble(a_cn)
    cn_solver = LUSolver(A_cn)
    cn_solver.parameters["reuse_factorization"] = True

    F_bdf = bdf2_step(Z, z_star, z_, Constant(dk), g, v_D, ds)
    (a_bdf, L_bdf) = system(F_bdf)
    A_bdf = assemble(a_bdf)
    bdf_solver = LUSolver(A_bdf)
    bdf_solver.parameters["reuse_factorization"] = True

    velocities = File("results/velocities.pvd")
    displacements = File("results/displacement.pvd")
    stresses = File("results/stresses.pvd")
    displacement = Function(CG1)
    displacements << displacement
    velocities << displacement # Just zero
    stresses << displacement   # Just zero

    progress = Progress("Time-iteration", int(T/dt))
    t = dk

    if verbose:
        set_log_level(DEBUG)

    sqrt_volume = sqrt(assemble(Constant(1.0)*dx, mesh=mesh))
    while (t <= T):

        # Half-time step:
        # Update source(s)
        p.t = t
        v_D.t = t

        # Assemble right-hand side for CN system
        b = assemble(L_cn)

        # Solve Crank-Nicolson system
        cn_solver.solve(z_star.vector(), b, annotate=annotate)

        # Increase time
        t += dk

        # Next-time step:
        # Update sources
        p.t = t
        v_D_mid.t = t

        # Assemble right-hand side for BDF system
        b = assemble(L_bdf)

        # Solve BDF system
        bdf_solver.solve(z.vector(), b, annotate=annotate)

        # Store solutions
        (sigma0, sigma1, v, gamma) = z.split()
        cg_v = project(v, CG1)
        cg_d = project(displacement + Constant(dt)*v, CG1)
        #stress_02 = project(sigma0[2] + sigma1[2], CG1)
        displacements << cg_d
        velocities << cg_v

        #stresses << stress_02

        # Print some output
        if verbose:
            d_norm = norm(cg_d)
            v_norm = norm(cg_v)
            s_norm = norm(stress_02)
            print ("t = %g, ||u|| = %g, ||v|| = %g, ||s|| = %g"
                   % (t, d_norm/sqrt_volume, v_norm/sqrt_volume,
                      s_norm/sqrt_volume))

        # Update time and variables
        t += dk
        z_.assign(z)
        displacement.assign(cg_d)
        #plot(displacement)

        progress += 1

    #interactive()
    return z_

if __name__ == "__main__":

    debugging["record_all"] = True

    ic = Function(Z)

    ic_copy = Function(ic)

    T = 0.5
    dt = 0.01
    #z = main(ic, T=T, dt=dt, annotate=False, verbose=True)
    z = main(ic, T=T, dt=dt, annotate=True, verbose=False)

    #info_blue("Replaying forward run ... ")
    #adj_html("forward.html", "forward")
    #replay_dolfin(forget=False)
    adj_html("adjoint.html", "adjoint")

    # Use traction on vertical plane as measure
    (sigma0, sigma1, v, gamma) = split(z)
    #sigma = sigma0 + sigma1
    #J = FinalFunctional(inner(sigma[2], sigma[2])*dx)
    J = FinalFunctional(inner(sigma0[1], sigma0[1])*dx)
    #adjoint = adjoint_dolfin(J, forget=False)

    info_blue("Running adjoint ... ")
    adjoints = File("results/adjoints.pvd")
    norms = []
    adjoint = Function(CG1)
    for i in range(adjointer.equation_count)[::-1]:
        print "i = ", i
        (adj_var, output) = adjointer.get_adjoint_solution(i, J)

        storage = libadjoint.MemoryStorage(output)
        adjointer.record_variable(adj_var, storage)

        print adj_var.name
        #print output.data.__class__
        #print output.data.function_space()
        if adj_var.name == "w_3":
            (tau0, tau1, w, eta) = output.data.split()
            no = norm(output.data)
            adjoint.assign(project(tau1[1], CG1))
            plot(adjoint)
            norms += [(i, no)]
    print "norm = ", norms
    pylab.figure()
    data = zip(*norms)
    print "data = ", data
    x = data[0]
    y = data[1]
    pylab.plot(x, y, '*-')
    pylab.show()
    interactive()
        #plot(w, title="Adjoint velocity")
        #plot(eta, title="Adjoint vorticity", interactive=True)

    #print adjointer
    #print adjoint.__class__
    exit()

    def Jfunc(ic):
      z = main(ic, T=T, dt=dt, annotate=False)
      (sigma0, sigma1, v, gamma) = split(z)
      sigma = sigma0 + sigma1
      J = assemble(inner(sigma[2], sigma[2])*dx)
      print "J(.): ", J
      return J

    ic.vector()[:] = ic_copy.vector()
    info_blue("Checking adjoint correctness ... ")
    minconv = test_initial_condition_adjoint(Jfunc, ic, adjoint, seed=1.0e-4)

    if minconv < 1.9:
      sys.exit(1)


