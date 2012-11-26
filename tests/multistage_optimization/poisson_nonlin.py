from dolfin import *

# Solve the coarse resolution problem
mesh = UnitSquare(2, 2)

V = FunctionSpace(mesh, "CG", 1)
u = Function(V)
v = TestFunction(V)

s = Constant(1.0)
F = inner(grad(u), grad(v))*dx  - s*v*dx 
bc = DirichletBC(V, Constant(0.0), "on_boundary")

solve(F == 0, u, bc)

# Solve the fine resolution problem
mesh_fine = refine(refine(refine(mesh))) 

V_fine = FunctionSpace(mesh_fine, "CG", 1)
u_fine = Function(V_fine)
v_fine = TestFunction(V_fine)

# Generate the fine resolution forms
F_fine = replace(F, {u: u_fine, v: v_fine}) 
#F_fine = inner(grad(u_fine), grad(v_fine))*dx  - s*v_fine*dx # doing 
bc_fine = DirichletBC(V_fine, Constant(0.0), "on_boundary") # is there a way to "replace" the function space for the boundary condition?

solve(F_fine == 0, u_fine)

