import dolfin
import copy
import collections

dolfin_assemble = dolfin.assemble
def assemble(*args, **kwargs):
  form = args[0]
  output = dolfin_assemble(*args, **kwargs)
  if not isinstance(output, float):
    output.form = form
  return output

bc_apply = dolfin.DirichletBC.apply
def adjoint_bc_apply(self, *args, **kwargs):
  for arg in args:
    if not hasattr(arg, 'bcs'):
      arg.bcs = []
    arg.bcs.append(self)
  return bc_apply(self, *args, **kwargs)
dolfin.DirichletBC.apply = adjoint_bc_apply

function_vector = dolfin.Function.vector
def adjoint_function_vector(self):
  vec = function_vector(self)
  vec.function = self
  return vec
dolfin.Function.vector = adjoint_function_vector
