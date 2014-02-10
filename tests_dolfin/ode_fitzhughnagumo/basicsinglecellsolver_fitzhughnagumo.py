from dolfin import *

try:
  from beatadjoint import BasicSingleCellSolver
except ImportError:
  info_red("Need beatadjoint to run")
  import sys; sys.exit(0)

try:
  from dolfin import BackwardEuler
except ImportError:
  from dolfin import info_red
  info_red("Need dolfin > 1.2.0 for ode_solver test.")
  import sys; sys.exit(0)

from dolfin_adjoint import *
import ufl.algorithms

if not hasattr(MultiStageScheme, "to_tlm"):
  info_red("Need dolfin > 1.2.0 for ode_solver test.")
  import sys; sys.exit(0)

# Import cell model (rhs, init_values, default_parameters)
from beatadjoint.cellmodels.fitzhughnagumo import Fitzhughnagumo as model

domain = UnitIntervalMesh(1)
num_states = len(model().initial_conditions()((0.0,)))
V = None

def main(model, ics=None, annotate=False):

  params = BasicSingleCellSolver.default_parameters()
  params["theta"] = 1.0
  solver = BasicSingleCellSolver(model, None, params=params, domain=domain)
  solver.parameters["enable_adjoint"] = annotate

  if ics is None:
    global V
    ics = project(model.initial_conditions(), solver.VS)
    V = solver.VS

  (vs_, vs) = solver.solution_fields()
  vs_.assign(ics)

  dt = 0.1
  T = 0.2
  solutions = solver.solve((0.0, T), dt)
  for x in solutions:
    pass

  (vs_, vs) = solver.solution_fields()
  return vs

if __name__ == "__main__":

  u = main(model(), annotate=True)

  ## Step 1. Check replay correctness

  replay = True
  if replay:
    info_blue("Checking replay correctness .. ")
    assert adjglobals.adjointer.equation_count > 0
    adj_html("forward.html", "forward")
    success = replay_dolfin(tol=1.0e-15, stop=True)
    assert success

  ## Step 2. Check TLM correctness

  dtm = TimeMeasure()
  J = Functional(inner(u, u)*dx*dtm[FINISH_TIME])
  m = InitialConditionParameter(u)
  Jm = assemble(inner(u, u)*dx)

  def Jhat(ic):
    u = main(model(), ics=ic)
    print "Perturbed functional value: ", assemble(inner(u, u)*dx)
    return assemble(inner(u, u)*dx)

  dJdm = compute_gradient_tlm(J, m, forget=False)
  minconv_tlm = taylor_test(Jhat, m, Jm, dJdm, \
                            perturbation_direction=interpolate(Constant((0.1,)*num_states), V), seed=1.0e-1)
  assert minconv_tlm > 1.8

  ## Step 3. Check ADM correctness

  dJdm = compute_gradient(J, m, forget=False)
  minconv_adm = taylor_test(Jhat, m, Jm, dJdm, \
                            perturbation_direction=interpolate(Constant((0.1,)*num_states), V), seed=1.0e-1)
  assert minconv_adm > 1.8
