from dolfin import *
import random
import sys

H = 0.1
L = 0.8

mesh = Rectangle(0, 0, L, H, 200, 50)

V = VectorFunctionSpace(mesh, "CG", 2)
Q = FunctionSpace(mesh, "CG", 1)
P = FunctionSpace(mesh, "DG", 1)
Z = MixedFunctionSpace([V, Q, P])

g = Constant((0.0, -10.0))
rho_0 = 1
alpha = Constant(10**-3)
kappa = 0
nu = Constant(10**-6)

dt = 0.2
start = 0
end = 40.0
theta = 0.5

temp_pvd = File("results/temperature.pvd")
u_pvd = File("results/velocity.pvd")
p_pvd = File("results/pressure.pvd")

no_slip = DirichletBC(Z.sub(0), (0.0, 0.0), "on_boundary && x[1] < DOLFIN_EPS")
free_left = DirichletBC(Z.sub(0).sub(0), 0.0, "on_boundary && x[0] < DOLFIN_EPS")
free_right = DirichletBC(Z.sub(0).sub(0), 0.0, "on_boundary && x[0] > 0.8 - DOLFIN_EPS")
free_top = DirichletBC(Z.sub(0).sub(1), 0.0, "on_boundary && x[1] > 0.1 - DOLFIN_EPS")
bcs = [no_slip, free_left, free_right, free_top]

parameters["num_threads"] = 8

def store(z, t):
  t = float(t)
  print "Storing variables at t=%s" % t
  (u, p, temp) = z.split()

  temp_pvd << (temp, t)
  u_pvd << (u, t)
  p_pvd << (p, t)

def print_cfl(z, mesh, dt):
  import numpy
  u = interpolate(z.split()[0], V)

  umax = numpy.max(numpy.abs(u.vector().array()))
  hmin = mesh.hmin()
  CFL = umax * dt/hmin
  info_blue("CFL number: %s" % CFL)

  new_dt = min(1.0 * hmin/umax, 0.5)
  info_blue("New dt: %s" % new_dt)
  return new_dt

def rho(T):
  return rho_0*(1 - alpha * T)

def Dt(u_old, u_new, dt):
  return (u_new - u_old)/Constant(dt)

def cn(u_old, u_new): # Crank-Nicolson
  return (1 - theta)*u_old + theta*u_new

def main(ic, start, end, dt, bcs):

  store(ic, t=start)
  z_old = ic
  (u_old, p_old, temp_old) = split(z_old)

  z_new = TrialFunction(Z)
  (u_new, p_new, temp_new) = split(z_new)

  z = Function(ic)
  (u, p, temp) = split(z)

  (u_test, p_test, temp_test) = split(TestFunction(Z))

  u_cn = cn(u_old, u_new)
  p_cn = cn(p_old, p_new)
  temp_cn = cn(temp_old, temp_new)

  n = FacetNormal(mesh)
  un = abs(dot(u('+'), n('+')))

  dts = []

  t = start
  while t < end:
    t += dt
    L = inner(Dt(u_old, u_new, dt), u_test)*dx + inner(grad(u_cn)*u, u_test)*dx + \
        nu*inner(grad(u_cn), grad(u_test))*dx - inner((rho(temp_cn)/rho_0)*g, u_test)*dx + \
        -div(u_test)*p_cn*dx + p_test*div(u_cn)*dx + \
        inner(Dt(temp_old, temp_new, dt), temp_test)*dx - dot(u*temp_new, grad(temp_test))*dx + (dot(u('+'), jump(temp_test, n))*avg(temp_new) + 0.5*un*dot(jump(temp_new, n), jump(temp_test, n)))*dS

    F = replace(L, {z_new: z})
    J = derivative(F, z)
    solve(F == 0, z, bcs=bcs, J=J, solver_parameters={"linear_solver": "mumps"})
    z_old.assign(z)
    store(z_old, t=t)
    dts.append(dt)
    dt = print_cfl(z_old, mesh, dt)

  return z_old 

if __name__ == "__main__":

  class ICExpression(Expression):
    def __init__(self, delta=0.03):
      self.delta = delta
      pass

    def eval(self, values, x):
      delta = self.delta

      if x[0] >= (0.4 + delta):
        values[-1] = +0.5
      elif x[0] <= (0.4 - delta):
        values[-1] = -0.5
      else:
        xdash = x[0] - (0.4 - delta)
        m = -H / (2*delta)
        if x[1] <= m*xdash+H:
          values[-1] = -0.5
        else:
          values[-1] = +0.5

    def value_shape(self):
      return (4,)

  ic = interpolate(ICExpression(), Z)

  soln = main(ic, start, end, dt, bcs)
