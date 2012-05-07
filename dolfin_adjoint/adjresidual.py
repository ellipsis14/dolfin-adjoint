from adjrhs import adj_get_forward_equation
import adjlinalg
import adjglobals

import dolfin
import ufl
import ufl.algorithms

import libadjoint

from dolfin import info_red, info_blue

def get_residual(i):
    (fwd_var, lhs, rhs) = adj_get_forward_equation(i)

    if isinstance(lhs, adjlinalg.IdentityMatrix):
      return None

    fn_space = ufl.algorithms.extract_arguments(lhs)[0].function_space()
    x = dolfin.Function(fn_space)

    if rhs == 0:
      form = lhs
      x = fwd_var.nonlinear_u
    else:
      form = dolfin.action(lhs, x) - rhs

    try:
      y = adjglobals.adjointer.get_variable_value(fwd_var).data
    except libadjoint.exceptions.LibadjointErrorNeedValue:
      info_red("Warning: recomputing forward solution; please report this script on launchpad")
      y = adjglobals.adjointer.get_forward_solution(i)[1].data

    form = dolfin.replace(form, {x: y})

    return form
