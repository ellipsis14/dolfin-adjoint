import dolfin
import copy
import collections

bc_cache = collections.defaultdict(list)

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
    bc_data = copy.copy(bc_cache[arg])
    bc_data.append(self)
    bc_cache[arg] = bc_data
  return bc_apply(self, *args, **kwargs)
dolfin.DirichletBC.apply = adjoint_bc_apply

function_vector = dolfin.Function.vector
def adjoint_function_vector(self):
  vec = function_vector(self)
  vec.function = self
  return vec
dolfin.Function.vector = adjoint_function_vector
