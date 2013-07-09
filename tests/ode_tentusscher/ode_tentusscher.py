try:
  from dolfin import BackwardEuler
except ImportError:
  from dolfin import info_red
  info_red("Need dolfin > 1.2.0 for ode_solver test.")
  import sys; sys.exit(0)

from dolfin import *
from dolfin_adjoint import *
import ufl.algorithms

if not hasattr(MultiStageScheme, "to_tlm"):
  info_red("Need dolfin > 1.2.0 for ode_solver test.")
  import sys; sys.exit(0)

# Import cell model (rhs, init_values, default_parameters)
import tentusscher_2004_mcell as model

params = model.default_parameters()
state_init = model.init_values()

mesh = UnitIntervalMesh(1)
#R = FunctionSpace(mesh, "R", 0) # in my opinion, should work, but doesn't
num_states = state_init.value_size()
V = VectorFunctionSpace(mesh, "CG", 1, dim=num_states)

def main(u, form, time, Scheme, dt):

  scheme = Scheme(form, u, time)
  scheme.t().assign(float(time))

  xs = [float(time)]
  ys = [u.vector().array()[0]]

  solver = PointIntegralSolver(scheme)
  solver.parameters.reset_stage_solutions = True
  solver.parameters.newton_solver.reset_each_step = True

  for i in range(4):
    solver.step(dt)
    xs.append(float(time))
    ys.append(u.vector().array()[15])

  return (u, xs, ys)

if __name__ == "__main__":
  u0 = interpolate(state_init, V, name="InitialValue")
  Scheme = BackwardEuler

  u = Function(V, name="Solution")
  v = TestFunction(V)
  time = Constant(0.0)
  form = model.rhs(u, time, params)*dP

  ## Step 0. Check forward order-of-convergence (nothing to do with adjoints)
  check = False
  plot = False

  dt = 0.01
  u.assign(u0)
  (u, xs, ys) = main(u, form, time, Scheme, dt=dt)
  print "Solution: ", ys[-1]

  ## Step 1. Check replay correctness
  
  replay = True
  if replay:
    info_blue("Checking replay correctness .. ")
    assert adjglobals.adjointer.equation_count > 0
    adj_html("forward.html", "forward")
    success = replay_dolfin(tol=1.0e-15, stop=True)
    assert success

  ## Step 2. Check TLM correctness

  seed = 1e-8
  dtm = TimeMeasure()
  J = Functional(inner(u[15], u[15])*dx*dtm[FINISH_TIME])
  m = InitialConditionParameter(u)
  assert m.data().vector()[15] == u0.vector()[15]
  Jm = assemble(inner(u[15], u[15])*dx)

  def Jhat(ic):
    time = Constant(0.0)
    form = model.rhs(ic, time, params)*dP
    
    (u, xs, ys) = main(ic, form, time, Scheme, dt=dt)
    print "Perturbed functional value: ", assemble(inner(u[15], u[15])*dx)
    return assemble(inner(u[15], u[15])*dx)

  tlm = False
  # FIXME: Takes for ever...
  if tlm:
    dJdm = compute_gradient_tlm(J, m, forget=False)
    minconv_tlm = taylor_test(Jhat, m, Jm, dJdm, \
                              perturbation_direction=interpolate(Constant((0.1,)*num_states), V), seed=seed)
    assert minconv_tlm > 1.8

  ## Step 3. Check ADM correctness

  dJdm = compute_gradient(J, m, forget=False)
  minconv_adm = taylor_test(Jhat, m, Jm, dJdm, \
                            perturbation_direction=interpolate(Constant((0.1,)*num_states), V), seed=seed)
  assert minconv_adm > 1.8
