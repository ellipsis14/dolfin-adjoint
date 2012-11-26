from dolfin import *

# Solve the coarse resolution problem
mesh = UnitSquare(2, 2)

V = FunctionSpace(mesh, "CG", 1)
u = TrialFunction(V)
v = TestFunction(V)
sol = Function(V)

s = Constant(1.0)
F = inner(grad(u), grad(v))*dx  - s*v*dx 
bc = DirichletBC(V, Constant(0.0), "on_boundary")

solve(lhs(F) == rhs(F), sol, bc)

# Solve the fine resolution problem
mesh_fine = refine(mesh)

V_fine = FunctionSpace(mesh_fine, "CG", 1)
u_fine = TrialFunction(V_fine)
v_fine = TestFunction(V_fine)
sol_fine = Function(V_fine)

# Generate the fine resolution forms

# Note 1: I get "Expecting the solution variable u to be a member of the trial space.": 
F_fine = replace(F, {u:u_fine, v:v_fine}) 

# Note 2: However, it works fine if I redefine the form: 
F_fine = inner(grad(u_fine), grad(v_fine))*dx  - s*v_fine*dx 

# Note 3: is there a way to "replace" the function space for the boundary condition?
bc_fine = DirichletBC(V_fine, Constant(0.0), "on_boundary") 

solve(lhs(F_fine) == rhs(F_fine), sol_fine, bc_fine)

