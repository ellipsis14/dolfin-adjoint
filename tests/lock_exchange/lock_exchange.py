from dolfin import *

H = 0.1
L = 0.8
n = 10

mesh = Rectangle(0, 0, L, H, n, n)

V = VectorFunctionSpace(mesh, "CG", 2)
Q = FunctionSpace(mesh, "CG", 1)
P = FunctionSpace(mesh, "DG", 1)
Z = MixedFunctionSpace([V, Q, P])

g = Constant((0.0, -10.0))
rho_0 = 1
alpha = Constant(10**-3)
kappa = 0
nu = Constant(10**-6)

dt = 0.025
start = 0
end = 0.5
theta = 0.5

temp_pvd = File("results/temperature.pvd")
u_pvd = File("results/velocity.pvd")
p_pvd = File("results/pressure.pvd")

def store(z, t):
  (u, p, temp) = z.split()

  temp_pvd << (temp, t)
  u_pvd << (u, t)
  p_pvd << (p, t)

def rho(T):
  return rho_0*(1 - alpha * T)

def Dt(u_old, u_new):
  return (u_new - u_old)/Constant(dt)

def cn(u_old, u_new): # Crank-Nicolson
  return (1 - theta)*u_old + theta*u_new

def main(ic):

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

  no_slip = DirichletBC(Z.sub(0), (0.0, 0.0), "on_boundary && x[1] < DOLFIN_EPS")
  free_left = DirichletBC(Z.sub(0).sub(0), 0.0, "on_boundary && x[0] < DOLFIN_EPS")
  free_right = DirichletBC(Z.sub(0).sub(0), 0.0, "on_boundary && x[0] > 0.8 - DOLFIN_EPS")
  free_top = DirichletBC(Z.sub(0).sub(1), 0.0, "on_boundary && x[1] > 0.1 - DOLFIN_EPS")
  u_bcs = [no_slip, free_left, free_right, free_top]

  L = inner(Dt(u_old, u_new), u_test)*dx + inner(grad(u_cn)*u, u_test)*dx + \
      nu*inner(grad(u_cn), grad(u_test))*dx + inner(rho(temp_cn)*g, u_test)*dx + \
      -div(u_test)*p_cn*dx + p_test*div(u_cn)*dx + \
      inner(temp_test, temp_new)*dx - inner(temp_test, temp_old)*dx

  t = start
  while t < end:
    t += dt
    F = replace(L, {z_new: z})
    solve(F == 0, z, bcs=u_bcs)
    z_old.assign(z)

    return z_old # indented to only do one timestep

if __name__ == "__main__":
  class ICExpression(Expression):
    def __init__(self):
      pass

    def eval(self, values, x):
      if x[0] < 0.4:
        values[-1] = -0.5
      else:
        values[-1] = +0.5

    def value_shape(self):
      return (4,)

  ic = interpolate(ICExpression(), Z)

  soln = main(ic)
  store(soln, t=dt)
