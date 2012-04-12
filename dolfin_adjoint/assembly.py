import dolfin
import copy

dolfin_assemble = dolfin.assemble
def assemble(*args, **kwargs):
  form = args[0]
  output = dolfin_assemble(*args, **kwargs)
  if not isinstance(output, float):
    output.form = form
    output.assemble_system = False

  return output

periodic_bc_apply = dolfin.PeriodicBC.apply
def adjoint_periodic_bc_apply(self, *args, **kwargs):
  for arg in args:
    if not hasattr(arg, 'bcs'):
      arg.bcs = []
    arg.bcs.append(self)
  return periodic_bc_apply(self, *args, **kwargs)
dolfin.PeriodicBC.apply = adjoint_periodic_bc_apply

dirichlet_bc_apply = dolfin.DirichletBC.apply
def adjoint_dirichlet_bc_apply(self, *args, **kwargs):
  for arg in args:
    if not hasattr(arg, 'bcs'):
      arg.bcs = []
    arg.bcs.append(self)
  return dirichlet_bc_apply(self, *args, **kwargs)
dolfin.DirichletBC.apply = adjoint_dirichlet_bc_apply

function_vector = dolfin.Function.vector
def adjoint_function_vector(self):
  vec = function_vector(self)
  vec.function = self
  return vec
dolfin.Function.vector = adjoint_function_vector

def assemble_system(*args, **kwargs):
  lhs = args[0]
  rhs = args[1]
  bcs = args[2]

  if not isinstance(bcs, list):
    bcs = [bcs]

  (lhs_out, rhs_out) = dolfin.assemble_system(*args, **kwargs)
  lhs_out.form = lhs
  lhs_out.bcs = bcs
  lhs_out.assemble_system = True
  rhs_out.form = rhs
  rhs_out.bcs = bcs
  rhs_out.assemble_system = True
  return (lhs_out, rhs_out)
