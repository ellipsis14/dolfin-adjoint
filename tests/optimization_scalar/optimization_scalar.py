from dolfin import *
from dolfin_adjoint import *
import sys

dolfin.set_log_level(ERROR)
dolfin.parameters["optimization"]["test_gradient"] = True

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

def eval_cb(j, m):
  print "j = %f, m = %f." % (j, float(m))

def derivative_cb(j, dj, m):
  print "j = %f, dj = %f, m = %f." % (j, dj, float(m))

if __name__ == "__main__":
  nu = Constant(0.0001, name="Nu")
  # Run the forward model once to have the annotation
  main(nu)

  J = Functional(inner(u, u)*dx*dt[FINISH_TIME])

  # Run the optimisation 
  reduced_functional = ReducedFunctional(J, ScalarParameter("Nu"), eval_cb = eval_cb, derivative_cb = derivative_cb)
  nu_opt = minimize(reduced_functional, 'SLSQP')

  tol = 1e-4
  if reduced_functional(nu_opt) > tol:
    print 'Test failed: Optimised functional value exceeds tolerance: ', reduced_functional(nu_opt), ' > ', tol, '.'
    sys.exit(1)
