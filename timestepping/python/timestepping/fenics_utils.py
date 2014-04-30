#!/usr/bin/env python

# Copyright (C) 2007-2013 Anders Logg and Kristian B. Oelgaard
# Copyright (C) 2008-2013 Martin Sandve Alnes
# Copyright (C) 2011-2012 by Imperial College London
# Copyright (C) 2013 University of Oxford
# Copyright (C) 2014 University of Edinburgh
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, version 3 of the License
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

# Copyright (C) 2007-2013 Anders Logg and Kristian B. Oelgaard from FFC file
# ffc/analysis.py, bzr trunk 1839
# Code first added: 2012-12-06

# Copyright (C) 2008-2013 Martin Sandve Alnes from UFL file
# ufl/algorithms/ad.py, bzr 1.1.x branch 1484
# Code first added: 2013-01-18
  
import copy
  
import dolfin
import ffc
import ufl
  
from exceptions import *
  
__all__ = \
  [
    "QForm",
    "apply_bcs",
    "differentiate_expr",
    "enforce_bcs",
    "evaluate_expr",
    "expand",
    "expand_expr",
    "expand_solver_parameters",
    "extract_form_data",
    "form_quadrature_degree",
    "is_empty_form",
    "is_expand_expr_supported",
    "is_general_constant",
    "is_r0_function",
    "is_r0_function_space",
    "is_self_adjoint_form",
    "is_zero_rhs",
    "lumped_mass"
  ]

class QForm(ufl.form.Form):
  """
  A quadrature degree aware Form. A QForm records the quadrature degree with
  which the Form is to be assembled, and the quadrature degree is considered
  in all rich comparison. Hence two QForm s, which as Form s which would be
  deemed equal, are non-equal if their quadrature degrees differ. Constructor
  arguments are identical to the Form constructor, with the addition of a
  required quadrature_degree keyword argument, equal to the requested quadrature
  degree.
  """

  def __init__(self, arg, quadrature_degree):
    if isinstance(arg, ufl.form.Form):
      arg = arg.integrals()
    if not isinstance(quadrature_degree, int) or quadrature_degree < 0:
      raise InvalidArgumentException("quadrature_degree must be a non-negative integer")

    ufl.form.Form.__init__(self, arg)
    self.__quadrature_degree = quadrature_degree

    return

  def __cmp__(self, other):
    if not isinstance(other, ufl.form.Form):
      raise InvalidArgumentException("other must be a Form")
    comp = self.__quadrature_degree.__cmp__(form_quadrature_degree(other))
    if comp == 0:
      return ufl.form.Form.__cmp__(self, other)
    else:
      return comp
  def __hash__(self):
    return hash((self.__quadrature_degree, ufl.form.Form.__hash__(self)))

  def __add__(self, other):
    if not isinstance(other, ufl.form.Form):
      raise InvalidArgumentException("other must be a Form")
    if not self.__quadrature_degree == form_quadrature_degree(other):
      raise InvalidArgumentException("Unable to add Forms: Quadrature degrees differ")
    return QForm(ufl.form.Form.__add__(self, other), quadrature_degree = self.__quadrature_degree)

  def __sub__(self, other):
    return self.__add__(-other)

  def __mul__(self, other):
    raise NotImplementedException("__mul__ method not implemented")

  def __rmul__(self, other):
    raise NotImplementedException("__rmul__ method not implemented")

  def __neg__(self):
    return QForm(ufl.form.Form.__neg__(self), quadrature_degree = self.__quadrature_degree)

  def quadrature_degree(self):
    """
    Return the quadrature degree.
    """

    return self.__quadrature_degree

  def form_compiler_parameters(self):
    """
    Return a dictionary of form compiler parameters.
    """

    return {"quadrature_degree":self.__quadrature_degree}

def form_quadrature_degree(form):
  """
  Determine the quadrature degree with which the supplied Form is to be
  assembled. If form is a QForm, return the quadrature degree of the QForm.
  Otherwise, return the default quadrature degree if one is set, or return
  the quadrature degree that would be selected by FFC. The final case
  duplicates the internal behaviour of FFC.
  """

  if isinstance(form, QForm):
    return form.quadrature_degree()
  elif isinstance(form, ufl.form.Form):
    if dolfin.parameters["form_compiler"]["quadrature_degree"] > 0:
      quadrature_degree = dolfin.parameters["form_compiler"]["quadrature_degree"]
    else:
      # Update to code above, for FFC 1.2.x
      form_data = extract_form_data(form)
      quadrature_degree = -1
      for integral in form.integrals():
        rep = dolfin.parameters["form_compiler"]["representation"]
        if rep == "auto":
          rep = ffc.analysis._auto_select_representation(integral, form_data.unique_sub_elements, form_data.function_replace_map)
        quadrature_degree = max(quadrature_degree, ffc.analysis._auto_select_quadrature_degree(integral, rep, form_data.unique_sub_elements, form_data.element_replace_map))
    return quadrature_degree
  else:
    raise InvalidArgumentException("form must be a Form")
  
def extract_form_data(form):
  """
  Wrapper for the form.form_data and form.compute_form_data methods of Form s.
  Calls the latter only if the former returns None.
  """
  
  if not isinstance(form, ufl.form.Form):
    raise InvalidArgumentException("form must be a Form")

  form_data = form.form_data()
  if form_data is None:
    form_data = form.compute_form_data()

  return form_data

def is_general_constant(c):
  """
  Return whether the supplied object is a Constant or a ListTensor containing
  Constant s.
  """
  
  if isinstance(c, ufl.tensors.ListTensor):
    for c_c in c:
      if not isinstance(c_c, dolfin.Constant):
        return False
    return True
  else:
    return isinstance(c, dolfin.Constant)

def is_r0_function(fn):
  """
  Return whether the supplied Function is R0 (i.e. a Real over the mesh).
  """
  
  if not isinstance(fn, dolfin.Function):
    raise InvalidArgumentException("fn must be a Function")

  return is_r0_function_space(fn.function_space())

def is_r0_function_space(space):
  """
  Return whether the supplied FunctionSpace is R0 (i.e. a Real over the mesh).
  """
  
  if not isinstance(space, dolfin.FunctionSpaceBase):
    raise InvalidArgumentException("space must be a FunctionSpace")

  e = space.ufl_element()
  return e.family() == "Real" and e.degree() == 0

def evaluate_expr(expr, copy = False):
  """
  Evaluate the supplied expression, and return either a float or GenericVector.
  If copy is False then an existing GenericVector may be returned -- it is
  expected in this case that the return value will never be modified.
  """
  
  if not isinstance(expr, ufl.expr.Expr):
    raise InvalidArgumentException("expr must be an Expr")

  if isinstance(expr, ufl.algebra.Product):
    ops = expr.operands()
    assert(len(ops) > 0)
    val = evaluate_expr(ops[0], copy = copy or len(ops) > 1)
    for op in ops[1:]:
      nval = evaluate_expr(op)
      if not isinstance(nval, float) or not nval == 1.0:
        val *= evaluate_expr(op)
  elif isinstance(expr, ufl.algebra.Sum):
    ops = expr.operands()
    assert(len(ops) > 0)
    val = evaluate_expr(ops[0], copy = copy or len(ops) > 1)
    for op in ops[1:]:
      nval = evaluate_expr(op)
      if not isinstance(nval, float) or not nval == 0.0:
        val += evaluate_expr(op)
  elif isinstance(expr, ufl.algebra.Division):
    ops = expr.operands()
    assert(len(ops) == 2)
    val = evaluate_expr(ops[0]) / evaluate_expr(ops[1])
  elif isinstance(expr, ufl.constantvalue.Zero):
    return 0.0
  elif isinstance(expr, dolfin.Function):
    if is_r0_function(expr):
      val = expr.vector().sum()
    else:
      val = expr.vector()
      if copy:
        val = val.copy()
  elif isinstance(expr, (ufl.constantvalue.FloatValue, ufl.constantvalue.IntValue, dolfin.Constant)):
    val = float(expr)
  elif isinstance(expr, ufl.differentiation.CoefficientDerivative):
    val = evaluate_expr(ufl.algorithms.expand_derivatives(expr))
  else:
    raise NotImplementedException("Expr type %s not implemented" % expr.__class__)

  return val

def differentiate_expr(expr, u):
  """
  Wrapper for the UFL derivative function. This chooses an argument equal to
  Constant(1.0). Form s should be differentiated using the derivative function.
  """
  
  if not isinstance(expr, ufl.expr.Expr):
    raise InvalidArgumentException("expr must be an Expr")
  if not isinstance(u, (dolfin.Constant, dolfin.Function)):
    raise InvalidArgumentException("u must be a Constant or Function")

  if expr is u:
    der = dolfin.Constant(1.0)
  else:
    der = ufl.derivative(expr, u, argument = dolfin.Constant(1.0))

    # Based on code from expand_derivatives1 in UFL file ad.py, (see e.g. bzr
    # 1.1.x branch revision 1484)
    cell = der.cell()
    if cell is None or cell.is_undefined():
      dim = 0
    else:
      dim = der.cell().geometric_dimension()
      
    der = ufl.algorithms.expand_derivatives(der, dim = dim)

  return der

def is_expand_expr_supported(expr):
  """
  Return whether the supplied Expr is supported by expand_expr.
  """
  
  if not isinstance(expr, ufl.expr.Expr):
    raise InvalidArgumentException("expr must be an Expr")

  def multiple_terms(expr):
    if isinstance(expr, (ufl.algebra.Sum,
                         ufl.indexsum.IndexSum)):
      return True
    else:
      for op in expr.operands():
        if multiple_terms(op):
          return True
      # May yield a false negative here
      return False

  if not isinstance(expr, (ufl.algebra.Sum,
                           ufl.algebra.Product,
                           ufl.indexsum.IndexSum,
                           ufl.indexed.Indexed,
                           ufl.tensors.ComponentTensor,
                           ufl.algebra.Division,
                           ufl.restriction.PositiveRestricted,
                           ufl.restriction.NegativeRestricted,
                           ufl.differentiation.Grad,
                           ufl.tensoralgebra.Dot,
                           ufl.tensoralgebra.Inner,
                           ufl.differentiation.CoefficientDerivative,
                           ufl.differentiation.VariableDerivative,
                           ufl.constantvalue.IntValue,
                           ufl.argument.Argument,
                           dolfin.Expression,
                           dolfin.Function,
                           dolfin.Constant,
                           ufl.constantvalue.FloatValue,
                           ufl.geometry.Circumradius,
                           ufl.algebra.Abs,
                           ufl.geometry.FacetNormal,
                           ufl.mathfunctions.Sqrt,
                           ufl.classes.Variable,
                           ufl.mathfunctions.Exp,
                           ufl.constantvalue.Zero,
                           ufl.algebra.Power,
                           ufl.indexing.MultiIndex,
                           ufl.classes.Label)):
    if isinstance(expr, (ufl.tensors.ListTensor,
                         ufl.classes.Conditional)):
      for op in expr.operands():
        if multiple_terms(op):
          return False
    else:
      return False

  for op in expr.operands():
    if not is_expand_expr_supported(op):
      return False

  return True
    
def expand_expr(expr):
  """
  Recursively expand the supplied Expr into the largest possible Sum.
  """
  
  if not isinstance(expr, ufl.expr.Expr):
    raise InvalidArgumentException("expr must be an Expr")
  
  if isinstance(expr, ufl.algebra.Sum):
    terms = []
    for term in expr.operands():
      terms += expand_expr(term)
    return terms
  elif isinstance(expr, ufl.algebra.Product):
    fact1, fact2 = expr.operands()[0], ufl.algebra.Product(*expr.operands()[1:])
    fact1_terms = expand_expr(fact1)
    fact2_terms = expand_expr(fact2)
    terms = []
    for term1 in fact1_terms:
      if isinstance(term1, ufl.algebra.Product):
        term1 = list(term1.operands())
      else:
        term1 = [term1]
      for term2 in fact2_terms:
        if isinstance(term2, ufl.algebra.Product):
          term2 = list(term2.operands())
        else:
          term2 = [term2]
        terms.append(ufl.algebra.Product(*(term1 + term2)))
    return terms
  elif isinstance(expr, ufl.indexsum.IndexSum):
    ops = expr.operands()
    assert(len(ops) == 2)
    return [ufl.indexsum.IndexSum(term, ops[1]) for term in expand_expr(ops[0])]
  elif isinstance(expr, ufl.indexed.Indexed):
    ops = expr.operands()
    assert(len(ops) == 2)
    return [ufl.indexed.Indexed(term, ops[1]) for term in expand_expr(ops[0])]
  elif isinstance(expr, ufl.tensors.ComponentTensor):
    ops = expr.operands()
    assert(len(ops) == 2)
    return [ufl.tensors.ComponentTensor(term, ops[1]) for term in expand_expr(ops[0])]
  elif isinstance(expr, ufl.algebra.Division):
    ops = expr.operands()
    assert(len(ops) == 2)
    return [ufl.algebra.Division(term, ops[1]) for term in expand_expr(ops[0])]
  elif isinstance(expr, ufl.restriction.PositiveRestricted):
    ops = expr.operands()
    assert(len(ops) == 1)
    return [ufl.restriction.PositiveRestricted(term) for term in expand_expr(ops[0])]
  elif isinstance(expr, ufl.restriction.NegativeRestricted):
    ops = expr.operands()
    assert(len(ops) == 1)
    return [ufl.restriction.NegativeRestricted(term) for term in expand_expr(ops[0])]
  elif isinstance(expr, ufl.differentiation.Grad):
    ops = expr.operands()
    assert(len(ops) == 1)
    return [ufl.differentiation.Grad(term) for term in expand_expr(ops[0])]
  elif isinstance(expr, (ufl.tensoralgebra.Dot,
                         ufl.tensoralgebra.Inner,
                         ufl.differentiation.CoefficientDerivative,
                         ufl.differentiation.VariableDerivative)):
    return expand_expr(expand(expr))
  # Expr types white-list. These cannot be expanded.
  elif isinstance(expr, (ufl.constantvalue.IntValue,
                         ufl.argument.Argument,
                         dolfin.Expression,
                         dolfin.Function,
                         dolfin.Constant,
                         ufl.constantvalue.FloatValue,
                         ufl.geometry.Circumradius,
                         ufl.algebra.Abs,
                         ufl.geometry.FacetNormal,
                         ufl.mathfunctions.Sqrt,
                         ufl.classes.Variable,
                         ufl.mathfunctions.Exp,
                         ufl.constantvalue.Zero,
                         ufl.algebra.Power,
                         ufl.indexing.MultiIndex,
                         ufl.classes.Label)):
    return [expr]
  # Expr types grey-list. It might be possible to expand these, but just ignore
  # them at present.
  elif isinstance(expr, (ufl.tensors.ListTensor,
                         ufl.classes.Conditional)):
    return [expr]
  else:
    dolfin.warning("Expr type %s not expanded by expand_expr" % expr.__class__)
    return [expr]

def lumped_mass(space, du = None):
  """
  Return a linear form which can be assembled to yield a lumped mass matrix.
  """
  
  if not isinstance(space, dolfin.FunctionSpaceBase):
    raise InvalidArgumentException("space must be a FunctionSpace")
  n_sub_spaces = space.num_sub_spaces()
  if du is None:
    du = dolfin.TrialFunction(space)
  else:
    if not isinstance(du, ufl.argument.Argument):
      raise InvalidArgumentException("du must be an Argument")
    elif n_sub_spaces > 0:
      if not du.shape() == (n_sub_spaces,):
        raise InvalidArgumentException("Invalid du shape")
    else:
      if not du.shape() == tuple():
        raise InvalidArgumentException("Invalid du shape")        

  c = ufl.coefficient.Coefficient(space.ufl_element())
  if n_sub_spaces > 0:
    masslump = lumped_mass(space.sub(0), du = du[0])
    for i in range(1, n_sub_spaces):
      masslump += lumped_mass(space.sub(i), du = du[i])
    return masslump
  else:
    return ufl.algorithms.expand_derivatives(dolfin.derivative(c * dolfin.dx, c, du = du))

def expand(form, dim = None):
  """
  Expand the supplied Expr or Form. This attempts to yield a canonical form.
  """
  
  if not isinstance(form, (ufl.expr.Expr, ufl.form.Form)):
    raise InvalidArgumentException("form must be an Expr or Form")

  nform = ufl.algorithms.expand_indices(ufl.algorithms.expand_compounds(ufl.algorithms.expand_derivatives(form, dim = dim)))
  if isinstance(form, QForm):
    return QForm(nform, quadrature_degree = form_quadrature_degree(form))
  else:
    return nform

def is_self_adjoint_form(form):
  """
  Return True if the supplied Form is self-adjoint. May return false negatives.
  """
  
  if not isinstance(form, ufl.form.Form):
    raise InvalidArgumentException("form must be a Form")
  
  def arguments(form):
    args = ufl.algorithms.extract_arguments(form)
    assert(len(args) == 2)
    test, trial = args
    if test.count() > trial.count():
      test, trial = trial, test
    return test, trial
  
  a_form = dolfin.adjoint(form)
  
  test, trial = arguments(form)
  a_test, a_trial = arguments(a_form)
  assert(a_test.count() == a_trial.count() - 1)
  
  if not test.element() == a_trial.element():
    return False
  elif not trial.element() == a_test.element():
    return False
  
  a_form = dolfin.replace(a_form, {a_test:trial, a_trial:test})
  
  return expand(form) == expand(a_form)

def apply_bcs(a, bcs, L = None, symmetric_bcs = False):
  """
  Apply boundary conditions to the supplied LHS matrix and (optionally) RHS
  vector. If symmetric_bcs is true then the boundary conditions are applied so as
  to yield a symmetric matrix. If the boundary conditions are not homogeneous
  then a RHS vector should be supplied, although the lack of a RHS in this case
  is not treated as an error.
  """
  
  if not isinstance(a, dolfin.GenericMatrix):
    raise InvalidArgumentException("a must be a square GenericMatrix")
  elif not a.size(0) == a.size(1):
    raise InvalidArgumentException("a must be a square GenericMatrix")
  if not isinstance(bcs, list):
    raise InvalidArgumentException("bcs must be a list of DirichletBC s")
  for bc in bcs:
    if not isinstance(bc, dolfin.cpp.DirichletBC):
      raise InvalidArgumentException("bcs must be a list of DirichletBC s")
  if not L is None and not isinstance(L, dolfin.GenericVector):
    raise InvalidArgumentException("L must be a GenericVector")

  if L is None:
    for bc in bcs:
      bc.apply(a)
    if symmetric_bcs:
      L = a.factory().create_vector()
      L.resize(a.local_range(0))
      for bc in bcs:
        bc.zero_columns(a, L, 1.0)
  else:
    for bc in bcs:
      bc.apply(a, L)
    if symmetric_bcs:
      for bc in bcs:
        bc.zero_columns(a, L, 1.0)

  return

def enforce_bcs(x, bcs):
  """
  Enforce boundary conditions on the supplied GenericVector.
  """
  
  if not isinstance(x, dolfin.GenericVector):
    raise InvalidArgumentException("x must be a GenericVector")
  if not isinstance(bcs, list):
    raise InvalidArgumentException("bcs must be a list of DirichletBC s")
  for bc in bcs:
    if not isinstance(bc, dolfin.cpp.DirichletBC):
      raise InvalidArgumentException("bcs must be a list of DirichletBC s")

  for bc in bcs:
    bc.apply(x)

  return

def is_zero_rhs(rhs):
  """
  Return whether the input can be used to indicate a zero RHS.
  """
  
  if rhs in [0, 0.0]:
    return True
  elif isinstance(rhs, dolfin.Constant):
    return float(rhs) == 0.0
  else:
    return False

def expand_solver_parameters(solver_parameters, default_solver_parameters = {}):
  """
  Return an expanded dictionary of solver parameters with all defaults
  explicitly specified. The optional default_solver_parameters argument can
  be used to override global defaults.
  """
  
  if not isinstance(solver_parameters, dict):
    raise InvalidArgumentException("solver_parameters must be a dictionary")
  if not isinstance(default_solver_parameters, dict):
    raise InvalidArgumentException("default_solver_parameters must be a dictionary")

  def apply(parameters, default):
    lparameters = copy.copy(default)
    for key in parameters:
      if not isinstance(parameters[key], dict):
        lparameters[key] = parameters[key]
      elif key in default:
        lparameters[key] = apply(parameters[key], default[key])
      else:
        lparameters[key] = apply(parameters[key], {})
    return lparameters
  
  if not len(default_solver_parameters) == 0:
    solver_parameters = apply(solver_parameters, default_solver_parameters)
  solver_parameters = apply(solver_parameters, {"linear_solver":"default",
    "lu_solver":dolfin.parameters["lu_solver"].to_dict(),
    "krylov_solver":dolfin.parameters["krylov_solver"].to_dict()})
  
  if solver_parameters["linear_solver"] in ["default", "lu"] or dolfin.has_lu_solver_method(solver_parameters["linear_solver"]):
    del(solver_parameters["krylov_solver"])
  else:
    del(solver_parameters["lu_solver"])
  
  return solver_parameters
  
def is_empty_form(form):
  """
  Return whether the supplied form is "empty" (i.e. contains no terms).
  """
  
  if not isinstance(form, ufl.form.Form):
    raise InvalidArgumentException("form must be a Form")

  if len(form.integrals()) == 0:
    return True
  else:
    return len(extract_form_data(form).integral_data) == 0