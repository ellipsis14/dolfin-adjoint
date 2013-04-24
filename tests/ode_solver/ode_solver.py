try:
  from dolfin import BackwardEuler
except ImportError:
  from dolfin import info_red
  info_red("Need dolfin > 1.2.0 for ode_solver test.")
  import sys; sys.exit(0)

from dolfin import *
from dolfin_adjoint import *

mesh = UnitIntervalMesh(2)
#R = FunctionSpace(mesh, "R", 0) # in my opinion, should work, but doesn't
R = FunctionSpace(mesh, "CG", 1)

def main(u0, c, Solver):

  u = Function(u0, name="Solution")
  v = TestFunction(R)
  form = inner(c*u, v)*dP

  time = Constant(0.0)
  scheme = Solver(form, u, time)
  scheme.t().assign(float(time))

  xs = [float(time)]
  ys = [u.vector().array()[0]]

  solver = PointIntegralSolver(scheme)
  dt = 0.2

  for i in range(1):
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

  (u, xs, ys) = main(u0, c, Solver)

  plot = False

  if plot:
    import matplotlib.pyplot as plt
    plt.plot(xs, ys, label="Approximate solution")

    exact_u = lambda t: u0_f*exp(c_f*t)
    exact_ys = [exact_u(t) for t in xs]

    print "xs:" , xs
    print "ys:" , ys
    print "exact_ys: ", exact_ys
    plt.plot(xs, exact_ys, label="Exact solution")
    plt.legend(loc="best")
    plt.show()

  replay = True

  if replay:
    assert adjglobals.adjointer.equation_count > 0
    adj_html("forward.html", "forward")
    #success = replay_dolfin(tol=0.0, stop=True)
    #assert success
