try:
  from dolfin import BackwardEuler
except ImportError:
  from dolfin import info_red
  info_red("Need dolfin > 1.2.0 for ode_solver test.")
  import sys; sys.exit(0)

from dolfin import *
from dolfin_adjoint import *
import ufl.algorithms

mesh = UnitIntervalMesh(2)
#R = FunctionSpace(mesh, "R", 0) # in my opinion, should work, but doesn't
R = FunctionSpace(mesh, "CG", 1)

def main(u0, c, Solver, dt):

  u = Function(u0, name="Solution")
  v = TestFunction(R)
  form = inner(c*u, v)*dP

  time = Constant(0.0)
  scheme = Solver(form, u, time)
  scheme.t().assign(float(time))

  xs = [float(time)]
  ys = [u.vector().array()[0]]

  solver = PointIntegralSolver(scheme)
  solver.parameters["newton_solver"]["iterations_to_retabulate_jacobian"] = 1

  for i in range(int(2.0/dt)):
    solver.step(dt)
    xs.append(float(time))
    ys.append(u.vector().array()[0])

  return (u, xs, ys)

if __name__ == "__main__":
  u0_f = 1.0
  u0 = interpolate(Constant(u0_f), R, name="InitialValue")
  c_f = 1.0
  c  = interpolate(Constant(c_f), R, name="GrowthRate")
  Solver = BackwardEuler

  exact_u = lambda t: u0_f*exp(c_f*t)

  plot = True
  if plot:
    import matplotlib.pyplot as plt

  dts = [0.1, 0.05, 0.025]

  errors = []
  for dt in dts:
    adj_reset()
    (u, xs, ys) = main(u0, c, Solver, dt=dt)

    exact_ys = [exact_u(t) for t in xs]
    errors.append(abs(ys[-1] - exact_ys[-1]))

    if plot:
      plt.plot(xs, ys, label="Approximate solution (dt %s)" % dt)
      if dt == dts[-1]:
        plt.plot(xs, exact_ys, label="Exact solution")

  print "Errors: ", errors
  print "Convergence order: ", convergence_order(errors)

  if plot:
    plt.legend(loc="best")
    plt.show()
  
  replay = True

  if replay:
    assert adjglobals.adjointer.equation_count > 0
    adj_html("forward.html", "forward")
    success = replay_dolfin(tol=0.0, stop=True)
    assert success
