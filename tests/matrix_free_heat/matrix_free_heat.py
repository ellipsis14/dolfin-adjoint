import sys

from dolfin import *
from dolfin_adjoint import *

dolfin.parameters["adjoint"]["record_all"] = True

f = Expression("x[0]*(x[0]-1)*x[1]*(x[1]-1)")
mesh = UnitSquare(4, 4)
V = FunctionSpace(mesh, "CG", 1)

def run_forward(initial_condition=None, annotate=True, dump=True):
  u = TrialFunction(V)
  v = TestFunction(V)

  u_0 = Function(V)
  if initial_condition is not None:
    u_0.assign(initial_condition)

  u_1 = Function(V)

  dt = Constant(0.1)

  F = ( (u - u_0)/dt*v + inner(grad(u), grad(v)) + f*v)*dx

  bc = DirichletBC(V, 1.0, "on_boundary")

  a, L = lhs(F), rhs(F)

  solver = AdjointPETScKrylovSolver("default", "none")
  matfree = AdjointKrylovMatrix(a, bcs=bc)

  t = float(dt)
  T = 1.0
  n = 1

  if dump:
    u_out = File("u.pvd", "compressed")
    u_out << u_0

  while t <= T:
      b_rhs = assemble(L)
      bc.apply(b_rhs)
      solver.solve(matfree, down_cast(u_0.vector()), down_cast(b_rhs), annotate=annotate)

      t += float(dt)
      if dump:
        u_out << u_0

  return u_0

def run_replay():
  print "Replaying forward model (will error if differs from correct solution) ..."


  u_out = File("u_replay.pvd", "compressed")

  for i in range(adjointer.equation_count):
      (fwd_var, output) = adjointer.get_forward_solution(i)

      storage = libadjoint.MemoryStorage(output)
      storage.set_compare(tol=0.0)
      storage.set_overwrite(True)
      adjointer.record_variable(fwd_var, storage)

      u_out << output.data

def run_adjoint():
  z_out = File("adjoint.pvd", "compressed")

  for i in range(adjointer.equation_count)[::-1]:

      (adj_var, output) = adjointer.get_adjoint_solution(i, functional)
      
      storage = libadjoint.MemoryStorage(output)
      adjointer.record_variable(adj_var, storage)

      # if we cared about memory, here is where we would put the call to forget
      # any variables we no longer need.

      z_out << output.data

  return output.data

if __name__ == "__main__":

  final_forward = run_forward()

  adj_html("heat_forward.html", "forward")
  adj_html("heat_adjoint.html", "adjoint")

  # The functional is only a function of final state.
  functional=Functional(final_forward*final_forward*dx*dt[FINISH_TIME])
  f_direct = adjointer.evaluate_functional(functional, adjointer.equation_count-1)

  print "Running adjoint model ..."

  for (adjoint, var) in compute_adjoint(functional, forget=False):
    pass
  final_adjoint = adjoint

  def J(ic):
    perturbed_u0 = run_forward(initial_condition=ic, annotate=False, dump=False)
    return assemble(perturbed_u0*perturbed_u0*dx)

  minconv = test_initial_condition_adjoint(J, Function(V), final_adjoint, seed=10.0)

  if minconv < 1.9:
    sys.exit(1)

  dJ = assemble(derivative(final_forward*final_forward*dx, final_forward))

  ic = final_forward
  ic.vector()[:] = 0

  minconv = test_initial_condition_tlm(J, dJ, ic, seed=10.0)

  if minconv < 1.9:
    sys.exit(1)

