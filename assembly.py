import dolfin
import copy
import collections

assembly_cache = {}
bc_cache = collections.defaultdict(list)

dolfin_assemble = dolfin.assemble
def assemble(*args, **kwargs):
  form = args[0]
  output = dolfin_assemble(*args, **kwargs)
  assembly_cache[output] = form
  return output

bc_apply = dolfin.DirichletBC.apply
def adjoint_bc_apply(self, *args, **kwargs):
  for arg in args:
    bc_data = copy.copy(bc_cache[arg])
    bc_data.append(self)
    bc_cache[arg] = bc_data
  return bc_apply(self, *args, **kwargs)
dolfin.DirichletBC.apply = adjoint_bc_apply

