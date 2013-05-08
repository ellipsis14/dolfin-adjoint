#!/usr/bin/env python

# Copyright (C) 2008-2013 Martin Sandve Alnes
# Copyright (C) 2011-2012 by Imperial College London
# Copyright (C) 2013 University of Oxford
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

# Based on code from dolfin-adjoint bzr trunk 640

# Copyright (C) 2008-2013 Martin Sandve Alnes from UFL file ufl/form.py, bzr
# branch 1.1.x 1484

import copy
import types
  
import dolfin
import dolfin_adjoint

import libadjoint
import timestepping
import ufl

from timestepping import *
from dolfin_adjoint import *

__doc__ = \
"""
A timestepping abstraction and automatic adjoining library. This library
utilises the FEniCS system for symbolic manipulation and automatic code
generation, and supplements this system with a syntax for the description of
timestepping finite element models.

This version of the library integrates with dolfin-adjoint.
"""

__all__ = timestepping.__all__ + dir(dolfin_adjoint)
for val in copy.copy(__all__):
  if val.startswith("__") and val.endswith("__"):
    __all__.remove(val)
__all__ += [ "__doc__",
  "__license__",
  "__name__",
  "__package__",
  "__version__",
  "dolfin_adjoint",
  "timestepping"]

def system_info():
  """
  Print system information and assorted library versions.
  """
  
  timestepping.system_info()
  dolfin.info("dolfin-adjoint version: %s" % dolfin_adjoint.__version__)
  
  return
dolfin.info("dolfin-adjoint version: %s" % dolfin_adjoint.__version__)

dolfin.parameters["timestepping"]["pre_assembly"]["linear_forms"]["whole_form_optimisation"] = True

# Backwards compatibility for older versions of UFL.
if ufl_version() < (1, 1, 0):
  # Modified version of code from form.py, UFL bzr 1.1.x branch revision 1484
  def Form__mul__(self, coefficient):
    if isinstance(coefficient, ufl.expr.Expr):
      return ufl.formoperators.action(self, coefficient)
    return NotImplemented
  ufl.Form.__mul__ = Form__mul__
  del(Form__mul__)

# dolfin-adjoint internals expect Constant s and Function s to have an
# adj_name attribute. Patch __getattr__ to provide it.
def adj_name__getattr__(self, name):
  if name == "adj_name":
    return self.name()
  else:
    return object.__getattr__(self, name)
dolfin.Constant.__getattr__ = adj_name__getattr__
dolfin.Function.__getattr__ = adj_name__getattr__
del(adj_name__getattr__)

# constant.get_constant expects to be dealing with dolfin-adjoint Constant s.
# Patch it so that it can handle any Constant. Modified version of get_constant
# from constant.py.
def get_constant(a):
  if isinstance(a, dolfin.Constant):
    return a
  else:
    return constant_objects[a]
constant.get_constant.func_code = get_constant.func_code
del(get_constant)

# Resolve namespace clashes
def assemble(*args, **kwargs):
  if isinstance(args[0], TimeSystem):
    return args[0].assemble(*args[1:], **kwargs)
  else:
    return dolfin_adjoint.assemble(*args, **kwargs)

def Constant(value, cell = None, name = "u"):
  if isinstance(value, tuple):
    return dolfin.as_vector([dolfin_adjoint.Constant(val, cell = cell, name = "%s_%i" % (name, i)) for i, val in enumerate(value)])
  else:
    return dolfin_adjoint.Constant(value, cell = cell, name = name)

class TimeFunction(timestepping.TimeFunction):
  def __init__(self, tlevels, space, name = "u"):
    timestepping.TimeFunction.__init__(self, tlevels, space, name = name)
    # Ensure that all time level Function s are defined at all times
    cycle_map = self.final_cycle_map()
    for level in cycle_map:
      self.__lfns[level].wrap(self.__fns[cycle_map[level]])
    return
  
  def cycle(self, extended = True):
    """
    Perform a timestep cycle. The optional extended argument has no effect.
    """
    
    cycle_map = self.cycle_map()
    for level in cycle_map:
      # Annotate the timestep variable cycle
      record = da_annotate_assign(self[cycle_map[level]], self[level])
      assert(not record)
      self[level].vector()[:] = self[cycle_map[level]].vector()  
      
    return

  def final_cycle(self):
    """
    Perform the final cycle.
    """
    
    return

__AssembledTimeSystem_timestep_cycle_orig = timestepping.AssembledTimeSystem.timestep_cycle
def AssembledTimeSystem_timestep_cycle(self, extended = True):
  """
  Perform the timestep cycle. The optional extended argument has no effect.
  """
    
  __AssembledTimeSystem_timestep_cycle_orig(self, extended = False)
  # Signal the end of the timestep
  adj_inc_timestep()
  
  return
timestepping.AssembledTimeSystem.timestep_cycle = AssembledTimeSystem_timestep_cycle
del(AssembledTimeSystem_timestep_cycle)

__AssembledTimeSystem_finalise_orig = timestepping.AssembledTimeSystem.finalise
def AssembledTimeSystem_finalise(self):
  """
  Solve final equations and perform the final variable cycle.
  """
    
  __AssembledTimeSystem_finalise_orig(self)
  # Signal the end of the forward model
  adj_inc_timestep()
  
  return
timestepping.AssembledTimeSystem.finalise = AssembledTimeSystem_finalise
del(AssembledTimeSystem_finalise)

__EquationSolver__init__orig = EquationSolver.__init__
def EquationSolver__init__(self, *args, **kwargs):
  __EquationSolver__init__orig(self, *args, **kwargs)
  solve_orig = self.solve.im_class.solve
  # Equation annotation based on solve in solving.py
  def solve(self, *args, **kwargs):
    # Annotate an equation solve
    record = da_annotate_equation_solve(self)
    # The EquationSolver solve method could do a lot of work, including
    # further solves. Temporarily disable annotation during the solve ...
    annotate = not dolfin.parameters["adjoint"]["stop_annotating"]
    dolfin.parameters["adjoint"]["stop_annotating"] = True
    ret = solve_orig(self, *args, **kwargs)
    # ... and then restore the previous annotation status.
    dolfin.parameters["adjoint"]["stop_annotating"] = not annotate
    if record:
      x = self.x()
      # We still want to wrap functions with WrappedFunction so that we can for
      # example write separate equations for u[0] and u[n]. However
      # dolfin-adjoint needs to treat these as the same Function, so
      # strategically unwrap WrappedFunction s.
      if isinstance(x, WrappedFunction):
        x = x.fn()
      adjglobals.adjointer.record_variable(adjglobals.adj_variables[x], libadjoint.MemoryStorage(adjlinalg.Vector(x)))
    return ret
  solve.__doc__ = solve_orig.__doc__
  self.solve = types.MethodType(solve, self)
  return
EquationSolver.__init__ = EquationSolver__init__
del(EquationSolver__init__)

__AssignmentSolver__init__orig = AssignmentSolver.__init__
def AssignmentSolver__init__(self, *args, **kwargs):
  __AssignmentSolver__init__orig(self, *args, **kwargs)
  solve_orig = self.solve.im_class.solve
  # Equation annotation based on solve in solving.py
  def solve(self, *args, **kwargs):
    # Annotate an assignment solve
    record = da_annotate_equation_solve(self)
    annotate = not dolfin.parameters["adjoint"]["stop_annotating"]
    dolfin.parameters["adjoint"]["stop_annotating"] = True
    ret = solve_orig(self, *args, **kwargs)
    dolfin.parameters["adjoint"]["stop_annotating"] = not annotate
    if record:
      x = self.x()
      if isinstance(x, WrappedFunction):
        x = x.fn()
      adjglobals.adjointer.record_variable(adjglobals.adj_variables[x], libadjoint.MemoryStorage(adjlinalg.Vector(x)))
    return ret
  solve.__doc__ = solve_orig.__doc__
  self.solve = types.MethodType(solve, self)
  return
AssignmentSolver.__init__ = AssignmentSolver__init__
del(AssignmentSolver__init__)

def unwrap_fns(form):
  """
  Return a form with all WrappedFunction s unwrapped.
  """
  
  repl = {}
  for dep in ufl.algorithms.extract_coefficients(form):
    if isinstance(dep, WrappedFunction):
      repl[dep] = dep.fn()
      
  return replace(form, repl)  

# Based on the Functional class in functional.py
class Functional(functional.Functional):    
  def __init__(self, timeform, verbose = False, name = None):
    # Unwrap WrappedFunction s in the Functional
    if isinstance(timeform, ufl.form.Form):
      timeform = unwrap_fns(timeform)
    else:
      for term in timeform.terms:
        term.form = unwrap_fns(term.form)
    functional.Functional.__init__(self, timeform, verbose = verbose, name = name)
    
    return
  
# Based on the ReducedFunctional class in reduced_functional.py
class ReducedFunctional(reduced_functional.ReducedFunctional):
  def eval_array(self, m_array):
    # Clear caches
    clear_caches()
    return reduced_functional.ReducedFunctional.eval_array(self, m_array)

# Based on dolfin_adjoint_assign in function.py.
def da_annotate_assign(y, x):
  """
  Annotate an assignment. Returns whether the variable being solved for should
  be recorded by dolfin-adjoint.
  """
  
  if dolfin.parameters["adjoint"]["stop_annotating"]:
    # Annotation disabled
    return False
  
  if isinstance(x, WrappedFunction):
    x = x.fn()
  if isinstance(y, WrappedFunction):
    y = y.fn()
  if not x == y:
    # ?? What does this do ??
    if not adjglobals.adjointer.variable_known(adjglobals.adj_variables[x]):
      adjglobals.adj_variables.forget(x)
    assign.register_assign(x, y)
    
  return False
  
# A simple cache for use by dolfin-adjoint callbacks.
da_matrix_cache = {}
def clear_caches(*args):
  """
  Clear caches. Constant s or Function s can be supplied, indicating that only
  cached data associated with those coefficients should be cleared.
  """
  
  timestepping.clear_caches(*args)
  da_matrix_cache.clear()
  
  return

# Based on annotate in solving.py
def da_annotate_equation_solve(solve):
  """
  Annotate an equation solve. Returns whether the variable being solved for
  should be recorded by dolfin-adjoint.
  """
  
  if dolfin.parameters["adjoint"]["stop_annotating"]:
    # Annotation disabled
    return False
  
  # The Function being solved for
  x = solve.x()
  if isinstance(x, WrappedFunction):
    x_fn = x.fn()
  else:
    x_fn = x
    
  if isinstance(solve, AssignmentSolver):
    # Assignment solve case
    
    rhs = solve.rhs()
    if isinstance(rhs, (list, tuple)):
      if len(rhs) == 1 and rhs[0][0] == 1.0:
        # This is a direct assignment, so register an assignment
        return da_annotate_assign(rhs[0][1], x_fn)
      nrhs = rhs[0][0] * rhs[0][1]
      for term in rhs[1:]:
        nrhs += term[0] * term[1]
      rhs = nrhs;  del(nrhs)
    if isinstance(rhs, (float, int, ufl.constantvalue.FloatValue, ufl.constantvalue.IntValue, ufl.constantvalue.Zero)):
      # This is a direct assignment, so register an assignment
      return da_annotate_assign(Constant(rhs), x_fn)
    elif isinstance(rhs, (dolfin.Constant, dolfin.Function)):
      # This is a direct assignment, so register an assignment
      return da_annotate_assign(rhs, x_fn)
    # This is a LinearCombination assignment or expression assignment. For now
    # register this as a Galerkin projection.
    eq = dolfin.inner(dolfin.TestFunction(x.function_space()), dolfin.TrialFunction(x.function_space())) * dolfin.dx == \
      dolfin.inner(dolfin.TestFunction(x.function_space()), rhs) * dolfin.dx
    bcs = []
    solver_parameters = {"linear_solver":"lu"}
    adjoint_solver_parameters = solver_parameters
  else:
    # Equation solve case
    assert(isinstance(solve, EquationSolver))
    
    eq = solve.eq()
    bcs = solve.bcs()
    solver_parameters = solve.solver_parameters()
    adjoint_solver_parameters = solve.adjoint_solver_parameters()
  
  # Unwrap WrappedFunction s in the equation
  eq.lhs = unwrap_fns(eq.lhs)
  if not is_zero_rhs(eq.rhs):
    eq.rhs = unwrap_fns(eq.rhs)
  
  if hasattr(x, "_time_level_data"):
    # This is a time level solve. Set up a Matrix class with some caching
    # enabled.
    
    class DAMatrix(adjlinalg.Matrix):
      def __init__(self, *args, **kwargs):
        adjlinalg.Matrix.__init__(self, *args, **kwargs)
        self.__x = x
        self.__x_fn = x_fn
        self.__eq = eq
        self.__bcs = bcs
        self.__solver_parameters = solver_parameters
        self.__adjoint_solver_parameters = adjoint_solver_parameters
        self.parameters = dolfin.Parameters(**dolfin.parameters["timestepping"]["pre_assembly"])
        
        return

      # Based on Matrix.solve in adjlinalg.py
      def solve(self, var, b):
        if isinstance(self.data, adjlinalg.IdentityMatrix) or b.data is None:
          return adjlinalg.Matrix.solve(self, var, b)
        
        # Configure the cache
        if not self.__x in da_matrix_cache:
          da_matrix_cache[self.__x] = {}
        cache = da_matrix_cache[self.__x]
      
        # Boundary conditions
        if var.type in ["ADJ_ADJOINT", "ADJ_SOA", "ADJ_TLM"]:
          bcs = [homogenize(bc) for bc in self.__bcs]
        else:
          bcs = self.__bcs
        static_bcs = n_non_static_bcs(bcs) == 0
        
        if ("pa_a", var.type) in cache:
          # We have cached data for this matrix
          if cache[("pa_a", var.type)] is None:
            # The cache is empty
            static_a = False
            a = assemble(self.data)
            apply_a_bcs = True
          else:              
            # The cache contains a matrix, let's use it
            static_a = True
            a, apply_a_bcs = cache[("pa_a", var.type)]
        else:
          if extract_form_data(self.__eq.lhs).rank == 2:
            assert(not self.__x_fn in ufl.algorithms.extract_coefficients(self.__eq.lhs))
            if not is_zero_rhs(self.__eq.rhs):
              assert(not self.__x_fn in ufl.algorithms.extract_coefficients(self.__eq.rhs))
            # The forward equation is a linear variational problem. Is the
            # forward LHS matrix static?
            static_a = is_static_form(self.__eq.lhs)
          else:
            # Is the matrix static?
            static_a = is_static_form(self.data)
          if static_a:
            # The matrix is static, so we can cache it
            if not self.parameters["equations"]["symmetric_boundary_conditions"] and static_bcs:
              # Cache with boundary conditions
              a = assembly_cache.assemble(self.data, bcs = bcs)
              apply_a_bcs = False
            else:
              # Cache without boundary conditions
              a = assembly_cache.assemble(self.data)
              apply_a_bcs = True
            # Cache
            cache[("pa_a", var.type)] = a, apply_a_bcs
          else:
            # The matrix is not static, so we cannot cache it. Add an empty
            # entry to the cache to prevent repeated processing.
            cache[("pa_a", var.type)] = None
            a = assemble(self.data)
            apply_a_bcs = True
        
        # Assemble the RHS
        if isinstance(b.data, ufl.form.Form):
          b = assemble(b.data)
        else:
          b = b.data.vector()

        # Apply boundary conditions
        if apply_a_bcs:
          apply_bcs(a, bcs, L = b, symmetric_bcs = self.parameters["equations"]["symmetric_boundary_conditions"])
        else:
          enforce_bcs(b, bcs)

        if ("solver", var.type) in cache:
          # Extract a linear solver from the cache
          solver = cache[("solver", var.type)]
        else:
          # Create a new linear solver and cache it
          if var.type in["ADJ_ADJOINT", "ADJ_SOA"]:
            solver_parameters = self.__adjoint_solver_parameters
          else:
            solver_parameters = self.__solver_parameters
          solver = cache[("solver", var.type)] = solver_cache.solver(self.data, solver_parameters, static = static_a and static_bcs, bcs = bcs, symmetric_bcs = self.parameters["equations"]["symmetric_boundary_conditions"])

        # RHS solution vector
        x = adjlinalg.Vector(dolfin.Function(self.__x.function_space()))

        # Solve and return
        solver.set_operator(a)
        solver.solve(x.data.vector(), b)
        return x
  else:
    # This is not a time level solve. Use the default Matrix type.
    DAMatrix = adjlinalg.Matrix
  
  # Annotate the equation
  solving.annotate(eq, x_fn, bcs, solver_parameters = solver_parameters, matrix_class = DAMatrix)
  return True
