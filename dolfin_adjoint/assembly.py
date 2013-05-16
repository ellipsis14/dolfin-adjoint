import dolfin
import copy
import utils

dolfin_assemble = dolfin.assemble
def assemble(*args, **kwargs):
  """When a form is assembled, the information about its nonlinear dependencies is lost,
  and it is no longer easy to manipulate. Therefore, dolfin_adjoint overloads the :py:func:`dolfin.assemble`
  function to *attach the form to the assembled object*. This lets the automatic annotation work,
  even when the user calls the lower-level :py:data:`solve(A, x, b)`.
  """
  form = args[0]

  to_annotate = utils.to_annotate(kwargs.pop("annotate", None))

  output = dolfin_assemble(*args, **kwargs)
  if not isinstance(output, float) and to_annotate:
    output.form = form
    output.assemble_system = False

  return output

if hasattr(dolfin, 'PeriodicBC'):
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
  """When a form is assembled, the information about its nonlinear dependencies is lost,
  and it is no longer easy to manipulate. Therefore, dolfin_adjoint overloads the :py:func:`dolfin.assemble_system`
  function to *attach the form to the assembled object*. This lets the automatic annotation work,
  even when the user calls the lower-level :py:data:`solve(A, x, b)`.
  """
  lhs = args[0]
  rhs = args[1]

  if 'bcs' in kwargs:
    bcs = kwargs['bcs']
  elif len(args) > 2:
    bcs = args[2]
  else:
    bcs = []

  if not isinstance(bcs, list):
    bcs = [bcs]

  (lhs_out, rhs_out) = dolfin.assemble_system(*args, **kwargs)

  to_annotate = utils.to_annotate(kwargs.pop("annotate", None))
  if to_annotate:
    lhs_out.form = lhs
    lhs_out.bcs = bcs
    lhs_out.assemble_system = True
    rhs_out.form = rhs
    rhs_out.bcs = bcs
    rhs_out.assemble_system = True

  return (lhs_out, rhs_out)
