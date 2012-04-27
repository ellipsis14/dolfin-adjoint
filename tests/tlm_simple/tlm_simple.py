import random

from dolfin import *
from dolfin_adjoint import *
import sys

mesh = UnitSquare(4, 4)
V = FunctionSpace(mesh, "CG", 3)
dolfin.parameters["adjoint"]["record_all"] = True

def main(ic, annotate=False):
  u = TrialFunction(V)
  v = TestFunction(V)

  bc = DirichletBC(V, "-1.0", "on_boundary")

  mass = inner(u, v)*dx
  soln = Function(V)

  solve(mass == action(mass, ic), soln, bc, annotate=annotate)
  return soln

if __name__ == "__main__":

  ic = project(Expression("x[0]*(x[0]-1)*x[1]*(x[1]-1)"), V)
  soln = main(ic, annotate=True)

  perturbation_direction = Function(V)
  vec = perturbation_direction.vector()
  for i in range(len(vec)):
    vec[i] = random.random()

  ICParam = InitialConditionParameter(ic, perturbation_direction)

  for (dudm, tlm_var) in compute_tlm(ICParam):
    # just keep iterating until we get the last dudm
    pass

  dudm_np = dudm.vector()

  dJdu = derivative(soln*dx, soln)
  dJdu_np = assemble(dJdu)

  dJdm = dJdu_np.inner(dudm_np)
  print "Got dJdm: ", dJdm

  def J(soln):
    return assemble(soln*dx)

  perturbed_ic = Function(ic)
  vec = perturbed_ic.vector()
  vec.axpy(1.0, perturbation_direction.vector())
  perturbed_soln = main(perturbed_ic, annotate=False)

  Jdiff = J(perturbed_soln) - J(soln)
  print "J(ic+perturbation) - J(ic): ", Jdiff
  fail = abs(Jdiff - dJdm) > 1.0e-15
  if fail:
    sys.exit(1)

  def J(ic):
    soln = main(ic, annotate=False)
    return assemble(soln*soln*dx)

  dJ = assemble(derivative(soln*soln*dx, soln))

  minconv = test_initial_condition_tlm(J, dJ, ic, seed=1.0e-4)
  fail = minconv < 1.9
  if fail:
    sys.exit(1)

