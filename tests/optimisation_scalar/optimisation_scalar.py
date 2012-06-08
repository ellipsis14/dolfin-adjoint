from dolfin import *
from dolfin_adjoint import *
import sys

dolfin.set_log_level(ERROR)

n = 10
mesh = UnitInterval(n)
V = FunctionSpace(mesh, "CG", 2)

ic = project(Expression("sin(2*pi*x[0])"),  V)
u = Function(ic)

def main(nu):
  u_next = Function(V)
  v = TestFunction(V)

  timestep = Constant(1.0/n)

  F = ((u_next - u)/timestep*v
      + u_next*grad(u_next)*v 
      + nu*grad(u_next)*grad(v))*dx
  bc = DirichletBC(V, 0.0, "on_boundary")

  t = 0.0
  end = 0.1
  while (t <= end):
    solve(F == 0, u_next, bc)
    u.assign(u_next)
    t += float(timestep)
    adj_inc_timestep()

if __name__ == "__main__":
  nu = Constant(0.0001, name = 'Diffusivity')
  main(nu)

  J = FinalFunctional(inner(u, u)*dx)

  def Jhat(nu): 
    u.assign(ic)
    main(nu)
    return assemble(inner(u, u)*dx)

  # Run the optimisation 
  nu = optimisation.minimise(Jhat, J, ScalarParameter(nu), nu, 'scipy.slsqp', iprint = 2)
  print 'Final control value: ', float(nu)

  tol = 1e-4
  if Jhat(nu) > tol:
    print 'Test failed: Optimised functional value exceeds tolerance: ' , Jhat(nu), ' > ', tol, '.'
    sys.exit(1)



