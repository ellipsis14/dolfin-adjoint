from dolfin import *
from dolfin_adjoint import *

# Create mesh
def main(dbdt_c, annotate=False):
  mesh = UnitSphere(8)

  # Define function spaces
  PN = FunctionSpace(mesh, "Nedelec 1st kind H(curl)", 1)
  P1 = VectorFunctionSpace(mesh, "CG", 1)

  # Define test and trial functions
  v0 = TestFunction(PN)
  u0 = TrialFunction(PN)
  v1 = TestFunction(P1)
  u1 = TrialFunction(P1)

  # Define functions
  dbdt = as_vector([0.0, 0.0, Constant(dbdt_c, name="dbdt")])
  zero = Expression(("0.0", "0.0", "0.0"), degree=1)
  T = Function(PN)
  J = Function(P1)

  # Dirichlet boundary
  class DirichletBoundary(SubDomain):
      def inside(self, x, on_boundary):
          return on_boundary

  # Boundary condition
  bc = DirichletBC(PN, zero, DirichletBoundary())

  # Solve eddy currents equation (using potential T)
  solve(inner(curl(v0), curl(u0))*dx == -inner(v0, dbdt)*dx, T, bc, annotate=annotate)

  # Solve density equation
  solve(inner(v1, u1)*dx == dot(v1, curl(T))*dx, J, annotate=annotate)

  return J

if __name__ == "__main__":
  J = main(1.0, annotate=True)
  Jc = assemble(inner(J, J)*dx)
  dJdc = compute_gradient(Functional(inner(J, J)*dx*dt[FINISH_TIME]), ScalarParameter("dbdt"))

  def J(c):
    j = main(c)
    return assemble(inner(j, j)*dx)

  minconv = taylor_test(J, ScalarParameter("dbdt"), Jc, dJdc)

  if minconv < 1.9:
    import sys
    sys.exit(1)

