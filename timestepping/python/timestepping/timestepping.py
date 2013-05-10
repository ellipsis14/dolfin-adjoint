#!/usr/bin/env python

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

from collections import OrderedDict
import copy
from fractions import Fraction
import os
import pickle

import dolfin
import ffc
import numpy
import scipy
import scipy.optimize
import ufl

from fenics_patches import *

from fenics_overrides import *
from fenics_utils import *
from embedded_cpp import *
from exceptions import *
from quadrature import *
from time_levels import *
from versions import *
from vtu_io import *

__all__ = [
  "AdjoinedTimeSystem",
  "AdjointTimeFunction",
  "AdjointTimeSystem",
  "AdjointVariableMap",
  "AssembledTimeSystem",
  "AssemblyCache",
  "AssignmentSolver",
  "Checkpointer",
  "DiskCheckpointer",
  "EquationSolver",
  "LinearCombination",
  "LinearSolver",
  "MemoryCheckpointer",
  "PAAdjointSolvers",
  "PABilinearForm",
  "PAEquationSolver",
  "PAForm",
  "PALinearForm",
  "SolverCache",
  "StaticConstant",
  "StaticDirichletBC",
  "StaticFunction",
  "TimeFunction",
  "TimeFunctional",
  "TimeSystem",
  "WrappedFunction",
  "add_parameter",
  "assemble",
  "assembly_cache",
  "clear_caches",
  "expand_solver_parameters",
  "extract_non_static_coefficients",
  "is_static_bc",
  "is_static_coefficient",
  "is_static_form",
  "n_non_static_bcs",
  "n_non_static_coefficients",
  "nest_parameters",
  "pa_solve",
  "solver_cache"]

# Enable aggressive compiler optimisations by default.
dolfin.parameters["form_compiler"]["cpp_optimize"] = True
dolfin.parameters["form_compiler"]["cpp_optimize_flags"] = "-O3 -ffast-math -march=native"

def nest_parameters(parameters, key):
  """
  Create a new Parameters object at the specified key in parameters, if one
  does not already exist.
  """
  
  if key in parameters:
    if not isinstance(parameters[key], dolfin.Parameters):
      raise ParameterException("Inconsistent parameter type")
  else:
    p = dolfin.Parameters(key)
    parameters.add(p)
  return

def add_parameter(parameters, key, default_value):
  """
  Add a new parameter at the specified key in parameters. If the parameter
  already exists, check that it is of the same type as default_value. Otherwise,
  set the parameter to be equal to default_value.
  """
  
  if key in parameters:
    if not isinstance(parameters[key], default_value.__class__):
      raise ParameterException("Inconsistent parameter type")
  else:
    parameters.add(key, default_value)
  return

# Configure timestepping parameters.
nest_parameters(dolfin.parameters, "timestepping")
nest_parameters(dolfin.parameters["timestepping"], "pre_assembly")
nest_parameters(dolfin.parameters["timestepping"]["pre_assembly"], "forms")
nest_parameters(dolfin.parameters["timestepping"]["pre_assembly"], "linear_forms")
nest_parameters(dolfin.parameters["timestepping"]["pre_assembly"], "bilinear_forms")
nest_parameters(dolfin.parameters["timestepping"]["pre_assembly"], "equations")
add_parameter(dolfin.parameters["timestepping"]["pre_assembly"]["forms"], "whole_form_optimisation", False)
add_parameter(dolfin.parameters["timestepping"]["pre_assembly"]["linear_forms"], "whole_form_optimisation", False)
add_parameter(dolfin.parameters["timestepping"]["pre_assembly"]["bilinear_forms"], "whole_form_optimisation", True)
add_parameter(dolfin.parameters["timestepping"]["pre_assembly"]["equations"], "symmetric_boundary_conditions", False)
add_parameter(dolfin.parameters["timestepping"]["pre_assembly"], "verbose", True)

class WrappedFunction(dolfin.Function):
  """
  Wraps dolfin Function objects to enable Function aliasing, deferred
  allocation, and Function deallocation. Always has a name and a FunctionSpace.

  Constructor arguments:
    arg: One of:
        1. A FunctionSpace. The WrappedFunction is assigned the given function
           space, but is not associated with any DOLFIN Function.
      or:
        2. A Function. The WrappedFunction is assigned the function space of
           the given Function, and wraps the Function.
    name: A string defining the name of the function.
  """
  
  def __init__(self, arg, name = "u"):
    if not isinstance(name, str):
      raise InvalidArgumentException("name must be a string")
    self.__fn = None
    if isinstance(arg, dolfin.FunctionSpaceBase):
      self.__space = arg
      ufl.coefficient.Coefficient.__init__(self, self.__space.ufl_element())
    elif isinstance(arg, dolfin.Function):
      self.__space = arg.function_space()
      ufl.coefficient.Coefficient.__init__(self, self.__space.ufl_element())
      self.wrap(arg)
    else:
      raise InvalidArgumentException("Require FunctionSpace or Function as first argument")
    self.__name = name

    return

  def allocate(self):
    """
    Wrap a newly allocated Function.
    """
    
    self.wrap(Function(self.__space, name = self.__name))

    return

  def deallocate(self):
    """
    Alias to the unwrap method.
    """
    
    self.unwrap()

    return

  def wrap(self, fn):
    """
    Wrap the supplied Function.
    """
    
    if not isinstance(fn, (dolfin.Function, WrappedFunction)):
      raise InvalidArgumentException("fn must be a Function or WrappedFunction")
    # This comparison is very expensive. Trust that the caller knows what it's
    # doing.
#    elif not fn.function_space() == self.__space:
#      raise InvalidArgumentException("Invalid FunctionSpace")

    if isinstance(fn, WrappedFunction):
      fn = fn.__fn

    self.unwrap()
    self.__fn = fn
    self.this = fn.this

    return

  def unwrap(self):
    """
    Unwrap, so that the WrappedFunction no longer wraps any DOLFIN Function.
    """
    
    if not self.__fn is None:
      del(self.this)
      self.__fn = None

    return

  def is_wrapping(self):
    """
    Return whether the WrappedFunction is currently wrapping any DOLFIN
    Function.
    """
    
    return not self.__fn is None

  def fn(self):
    """
    Return the currently wrapped function, as a Function.
    """
    
    return self.__fn

  def function_space(self):
    """
    Return the function space, as a FunctionSpace.
    """
    
    return self.__space

  def name(self):
    """
    Return the function name, as a string.
    """
    
    return self.__name

  def rename(self, name, label):
    """
    Rename the WrappedFunction.
    """

    if not isinstance(name, str):
      raise InvalidArgumentException("name must be a string")
    
    self.__name = name

    return
      
class TimeFunction(TimeLevels):
  """
  A function defined on a number of time levels. Individual Function s can
  be accessed by indexing directly into the object.

  Constructor arguments:
    tlevels: A TimeLevels prescribing the time levels on which the function is
      defined.
    space: The FunctionSpace on which the function is defined.
    name: A string defining the name of the function.
  """
  
  def __init__(self, tlevels, space, name = "u"):
    if not isinstance(tlevels, TimeLevels):
      raise InvalidArgumentException("tlevels must be a TimeLevels")
    if not isinstance(space, dolfin.FunctionSpaceBase):
      raise InvalidArgumentException("space must be a FunctionSpace")
    if not isinstance(name, str):
      raise InvalidArgumentException("name must be a string")

    fns = {}
    lfns = {}
    for level in tlevels.levels():
      fns[level] = WrappedFunction(Function(space, name = "%s_%s" % (name, level)), name = "%s_%s" % (name, level))
      fns[level]._time_level_data = (self, level)

      nlevel = N + level.offset()
      lfns[nlevel] = WrappedFunction(space, name = "%s_%s" % (name, nlevel))
      lfns[nlevel]._time_level_data = (self, nlevel)

    self._TimeLevels__copy_time_levels(tlevels)
    self.__name = name
    self.__fns = fns
    self.__space = space
    self.__lfns = lfns

    return

  def __getitem__(self, key):
    if isinstance(key, (int, Fraction)):
      if not key in self._TimeLevels__offsets:
        raise InvalidArgumentException("key out of range")
      if not key in self.__lfns:
        self.__lfns[key] = WrappedFunction(self.__fns[n + key], name = "%s_%i" % (self.__name, key))
        self.__lfns[key]._time_level_data = (self, key)
      return self.__lfns[key]
    elif isinstance(key, TimeLevel):
      return self.__fns[key]
    elif isinstance(key, FinalTimeLevel):
      return self.__lfns[key]
    else:
      raise InvalidArgumentException("key must be an integer, Fraction, TimeLevel, or FinalTimeLevel")

  def name(self):
    """
    Return the name of the TimeFunction, as a string.
    """
    
    return self.__name

  def function_space(self):
    """
    Return the function space of the TimeFunction, as a FunctionSpace.
    """
    
    return self.__space

  def all_levels(self):
    """
    Return all levels on which the TimeFunction is defined, as a list of
    integers, Fraction s, TimeLevel s or FinalTimeLevel s.
    """
    
    return list(self._TimeLevels__levels) + self.__lfns.keys()
    
  def has_level(self, level):
    """
    Return whether the TimeFunction is defined on the specified level. level
    may be an integer, Fraction, TimeLevel or FinalTimeLevel.
    """
    
    if not isinstance(level, (int, Fraction, TimeLevel, FinalTimeLevel)):
      raise InvalidArgumentException("level must be an integer, Fraction, TimeLevel, or FinalTimeLevel")

    if isinstance(level, TimeLevel):
      return level in self.__fns
    else:
      return level in self.__lfns

  def fns(self):
    """
    Return all Function s associated with the TimeFunction.
    """
    
    return self.__fns.values()

  def initial_levels(self):
    """
    Return the initial time levels on which the TimeFunction is defined, as a
    list of integers or Fraction s.
    """
    
    levels = []
    for level in self.__lfns:
      if isinstance(level, (int, Fraction)):
        levels.append(level)

    return levels

  def final_levels(self):
    """
    Return the final time levels on which the TimeFunction is defined, as a list
    of FinalTimeLevel s.
    """
    
    levels = []
    for level in self.__lfns:
      if isinstance(level, FinalTimeLevel):
        levels.append(level)

    return levels

  def initial_cycle_map(self):
    """
    Return the initial cycle map, as an OrderedDict with Function keys and
    values.
    """
    
    cycle_map = OrderedDict()
    for level in self.levels():
      if level.offset() in self.__lfns:
        cycle_map[level] = level.offset()

    return cycle_map

  def final_cycle_map(self):
    """
    Return the final cycle map, as an OrderedDict with Function keys and values.
    """
    
    cycle_map = OrderedDict()
    for level in self.levels():
      nlevel = N + level.offset()
      if nlevel in self.__lfns:
        cycle_map[nlevel] = level

    return cycle_map

  def initial_cycle(self):
    """
    Perform the initial cycle. After the initial cycle the Function s on
    TimeLevel s are well-defined, but those on initial levels contain
    arbitrary data.
    """
    
    cycle_map = self.initial_cycle_map()
    for level in cycle_map:
      self.__fns[level].wrap(self.__lfns[cycle_map[level]])

    for level in self.levels():
      if not level in cycle_map:
        self.__fns[level].allocate()
        self.__fns[level].vector().zero()

    return

  def cycle(self, extended = True):
    """
    Perform a timestep cycle. If extended is true, use the extended cycle map
    to perform the cycle, via aliasing. Otherwise, use the cycle map to
    perform the cycle, via copying.
    """
    
    if extended:
      fns = {}
      for level in self.levels():
        fns[level] = self.__fns[level].fn()

      cycle_map = self.extended_cycle_map()
      for level in cycle_map:
        self.__fns[level].wrap(fns[cycle_map[level]])
    else:
      cycle_map = self.cycle_map()
      for level in cycle_map:
        self.__fns[level].assign(self.__fns[cycle_map[level]])

    return

  def final_cycle(self):
    """
    Perform the final cycle. After the final cycle the Function s on
    FinalTimeLevel s are well-defined, but those on TimeLevel s contain
    arbitrary data.
    """
    
    cycle_map = self.final_cycle_map()
    for level in cycle_map:
      self.__lfns[level].wrap(self.__fns[cycle_map[level]])

    return

def StaticConstant(*args, **kwargs):
  """
  Return a Constant which is marked as "static". Arguments are identical to the
  Constant function.
  """
  
  c = Constant(*args, **kwargs)
  if isinstance(c, ufl.tensors.ListTensor):
    for c_c in c:
      assert(isinstance(c_c, dolfin.Constant))
      c_c._time_static = True
  else:
    assert(isinstance(c, dolfin.Constant))
    c._time_static = True

  return c

def StaticFunction(*args, **kwargs):
  """
  Return a Function which is marked as "static". Arguments are identical to the
  Function function.
  """
  
  fn = Function(*args, **kwargs)
  fn._time_static = True

  return fn

class StaticDirichletBC(DirichletBC):
  """
  A DirichletBC which is marked as "static". Constructor arguments are identical
  to the DOLFIN DirichletBC constructor.
  """

  def __init__(self, *args, **kwargs):
    DirichletBC.__init__(self, *args, **kwargs)

    self._time_static = True

    return

def is_static_coefficient(c):
  """
  Return whether the supplied argument is a static Coefficient.
  """
  
  return isinstance(c, (ufl.constantvalue.FloatValue, ufl.constantvalue.IntValue)) or (hasattr(c, "_time_static") and c._time_static)

def extract_non_static_coefficients(form):
  """
  Return all non-static Coefficient s associated with the supplied form.
  """
  
  non_static = []
  for c in ufl.algorithms.extract_coefficients(form):
    if not is_static_coefficient(c):
      non_static.append(c)
  return non_static

def n_non_static_coefficients(form):
  """
  Return the number of non-static Coefficient s associated with the supplied
  form.
  """
  
  non_static = 0
  for c in ufl.algorithms.extract_coefficients(form):
    if not is_static_coefficient(c):
      non_static += 1
  return non_static

def is_static_form(form):
  """
  Return whether the supplied form is "static".
  """
  
  if not isinstance(form, ufl.form.Form):
    raise InvalidArgumentException("form must be a Form")

  for dep in ufl.algorithms.extract_coefficients(form):
    if not is_static_coefficient(dep):
      return False
  return True

def is_static_bc(bc):
  """
  Return whether the supplied DirichletBC is "static".
  """
  
  if not isinstance(bc, dolfin.cpp.DirichletBC):
    raise InvalidArgumentException("bc must be a DirichletBC")

  return hasattr(bc, "_time_static") and bc._time_static

def n_non_static_bcs(bcs):
  """
  Given a list of DirichletBC s, return the number of static DirichletBC s.
  """
  
  if not isinstance(bcs, list):
    raise InvalidArgumentException("bcs must be a list of DirichletBC s")

  n = 0
  for bc in bcs:
    if not is_static_bc(bc):
      n += 1

  return n

class LinearCombination:
  """
  A linear combination.

  Constructor arguments: An arbitrary number of tuples, each of which contains
  two elements:
    (alpha, x)
  where alpha is one of:
      1. A float.
    or:
      2. An arbitrary Expr.
  and x is a Function. The supplied alphas are not checked to ensure that this
  really is a true linear combination.
  """
  
  def __init__(self, *args):
    for arg in args:
      if not isinstance(arg, tuple) or not len(arg) == 2 \
        or not isinstance(arg[0], (float, ufl.expr.Expr)) or not isinstance(arg[1], dolfin.Function):
        raise InvalidArgumentException("Require tuples of (float or Expr, Function) as arguments")

    alpha = [arg[0] for arg in args]
    for i, lalpha in enumerate(alpha):
      if isinstance(lalpha, float):
        alpha[i] = ufl.constantvalue.FloatValue(lalpha)
    y = [arg[1] for arg in args]

    use_exp = False
    for lalpha in alpha:
      if not isinstance(lalpha, (ufl.constantvalue.FloatValue, dolfin.Constant)):
        use_exp = True
        break
    if use_exp:
      fl = ufl.constantvalue.Zero()
      for lalpha, ly in zip(alpha, y):
        fl += lalpha * ly
    else:
      fl = [(lalpha, ly) for lalpha, ly in zip(alpha, y)]

    self.__alpha = alpha
    self.__y = y
    self.__fl = fl

    return

  def dependencies(self, non_symbolic = False):
    """
    Return all dependencies associated with the LinearCombination. The
    optional non_symbolic argument has no effect.
    """
    
    deps = copy.copy(self.__y)
    for alpha in self.__alpha:
      deps += ufl.algorithms.extract_coefficients(alpha)

    return deps

  def nonlinear_dependencies(self):
    """
    Return all non-linear dependencies associated with the LinearCombination.
    """
    
    if isinstance(self.__fl, ufl.expr.Expr):
      nl_deps = []
      for dep in self.dependencies():
        if isinstance(dep, dolfin.Function):
          nl_deps += ufl.algorithms.extract_coefficients(differentiate_expr(self.__fl, dep))
        elif not isinstance(dep, (dolfin.Constant, dolfin.Expression)):
          raise DependencyException("Invalid dependency")
      return nl_deps
    else:
      return []

  def flatten(self):
    """
    Return a flattened version of the LinearCombination. If all alphas are
    floats or Constant s then return a list of (alpha, x) tuples. Otherwise
    return an Expr.
    """
    
    return self.__fl

  def solve(self, x):
    """
    Assign the given Function x to be equal to the value of the
    LinearCombination.
    """
    
    if not isinstance(x, dolfin.Function):
      raise InvalidArgumentException("x must be a Function")
    x = x.vector()

    if isinstance(self.__alpha[0], (ufl.constantvalue.FloatValue, dolfin.Constant)):
      lalpha = float(self.__alpha[0])
    else:
      lalpha = evaluate_expr(self.__alpha[0], copy = False)
    if isinstance(lalpha, float) and lalpha == 1.0:
      x[:] = self.__y[0].vector()[:]
    else:
      x[:] = lalpha * self.__y[0].vector()[:]
    for alpha, y in zip(self.__alpha[1:], self.__y[1:]):
      if isinstance(alpha, (ufl.constantvalue.FloatValue, dolfin.Constant)):
        x.axpy(float(alpha), y.vector())
      else:
        lalpha = evaluate_expr(alpha, copy = False)
        if isinstance(lalpha, dolfin.GenericVector):
          x[:] += lalpha * y.vector()
        else:
          assert(isinstance(lalpha, float))
          x.axpy(lalpha, y.vector())

    return

class AssignmentSolver:
  """
  A "solver" defining a direct assignment.

  Constructor arguments:
    y: The RHS, which must not depend on x. One of:
        1. A float
      or:
        2. A LinearCombination.
      or:
        3. An arbitrary Expr.
    x: A Function, which is assigned equal to the value of y in a "solve".
  """
  
  def __init__(self, y, x):
    if not isinstance(y, (int, float, LinearCombination, ufl.expr.Expr)) \
      and not is_general_constant(y):
      raise InvalidArgumentException("y must be an int, float, LinearCombination or Expr")
    if not isinstance(x, dolfin.Function):
      raise InvalidArgumentException("x must be a Function")

    if isinstance(y, (int, float)):
      y = ufl.constantvalue.FloatValue(y)
    elif isinstance(y, dolfin.Function):
      if y is x:
        raise DependencyException("Assignment is non-linear")
    elif is_general_constant(y):
      pass
    elif isinstance(y, LinearCombination):
      if x in y.dependencies():
        raise DependencyException("Assignment is non-linear")
    else:
      assert(isinstance(y, ufl.expr.Expr))
      if x in ufl.algorithms.extract_coefficients(y):
        raise DependencyException("Assignment is non-linear")
      
    self.__y = y
    self.__x = x

    return

  def is_assembled(self):
    """
    Return whether the AssignmentSolver is assembled. Always true.
    """
    
    return True

  def assemble(self):
    """
    Assemble the AssignmentSolver. Alias to the reassemble method with no
    arguments.
    """
    
    self.reassemble()

    return

  def reassemble(self, *args):
    """
    Reassemble the AssignmentSolver. Has no effect.
    """
    
    return

  def dependencies(self, non_symbolic = False):
    """
    Return all dependencies of the AssignmentSolver, excluding x. The optional
    non_symbolic has no effect.
    """
    
    if isinstance(self.__y, ufl.constantvalue.FloatValue):
      return []
    elif isinstance(self.__y, dolfin.Function) or is_general_constant(self.__y):
      return [self.__y]
    elif isinstance(self.__y, LinearCombination):
      return self.__y.dependencies(non_symbolic = non_symbolic)
    else:
      assert(isinstance(self.__y, ufl.expr.Expr))
      return ufl.algorithms.extract_coefficients(self.__y)

  def nonlinear_dependencies(self):
    """
    Return all non-linear dependencies of the AssignmentSolver.
    """
    
    if isinstance(self.__y, (ufl.constantvalue.FloatValue, dolfin.Function)) or is_general_constant(self.__y):
      return []
    elif isinstance(self.__y, LinearCombination):
      return self.__y.nonlinear_dependencies()
    else:
      assert(isinstance(self.__y, ufl.expr.Expr))
      nl_deps = []
      for dep in ufl.algorithms.extract_coefficients(self.__y):
        if isinstance(dep, dolfin.Function):
          nl_deps += ufl.algorithms.extract_coefficients(differentiate_expr(self.__y, dep))
        elif not isinstance(dep, (dolfin.Constant, dolfin.Expression)):
          raise DependencyException("Invalid dependency")
      return nl_deps
      
  def rhs(self):
    """
    Return the RHS, as either a tuple (FloatValue or Constant, Function) or an
    Expr.
    """
    
    if isinstance(self.__y, dolfin.Function):
      return [(ufl.constantvalue.FloatValue(1.0), self.__y)]
    elif isinstance(self.__y, LinearCombination):
      return self.__y.flatten()
    else:
      assert(isinstance(self.__y, ufl.expr.Expr))
      return self.__y

  def x(self):
    """
    Return the Function being solved for.
    """
    
    return self.__x
  
  def is_linear(self):
    """
    Return whether the AssignmentSolver is linear. Always true.
    """
    
    return True

  def solve(self):
    """
    Solve for x.
    """
    
    if isinstance(self.__y, (ufl.constantvalue.FloatValue, dolfin.Constant)):
      self.__x.vector()[:] = float(self.__y)
    elif isinstance(self.__y, dolfin.Function):
      self.__x.vector()[:] = self.__y.vector()
    elif is_general_constant(self.__y):
      self.__x.assign(dolfin.Constant([y_c for y_c in self.__y]))
    elif isinstance(self.__y, LinearCombination):
      self.__y.solve(self.__x)
    else:
      assert(isinstance(self.__y, ufl.expr.Expr))
      self.__x.vector()[:] = evaluate_expr(self.__y, copy = False)

    return
      
class TimeSystem:
  """
  Used to register timestep equations.
  """
  
  def __init__(self):
    self.__deps = OrderedDict()
    self.__init_solves = OrderedDict()
    self.__solves = OrderedDict()
    self.__final_solves = OrderedDict()

    self.__x_tfns = [set(), True]
    self.__tfns = [set(), True]
    self.__sorted_solves = [[[], [], []], True]

    self.__update = None

    return

  def __process_deps(self, deps, level):
    proc_deps = set()
    for dep in deps:
      if isinstance(dep, dolfin.Function) and hasattr(dep, "_time_level_data"):
        dep_level = dep._time_level_data[1]
        if isinstance(level, (int, Fraction)):
          if not isinstance(dep_level, (int, Fraction)):
            raise DependencyException("Inconsistent dependency type")
        elif not isinstance(level, dep_level.__class__):
          raise DependencyException("Inconsistent dependency type")
        if dep_level > level:
          raise DependencyException("Future dependency")
        proc_deps.add(dep)
    proc_deps = list(proc_deps)
    def cmp(x, y):
      return id(x) - id(y)
    proc_deps.sort(cmp = cmp)

    return proc_deps

  def add_assignment(self, y, x):
    """
    Add an assignment. The arguments match those accepted by the
    AssignmentSolver constructor.
    """

    if self.has_solve(x):
      raise StateException("Solve for %s already registered" % x.name())

    solve = AssignmentSolver(y, x)
    x_deps = solve.dependencies(non_symbolic = True)

    if not hasattr(x, "_time_level_data"):
      raise InvalidArgumentException("Missing time level data")
    last_past_level = x._time_level_data[0].last_past_level()
    level = x._time_level_data[1]
        
    for dep in x_deps:
      if dep is x:
        raise DependencyException("Assignment is non-linear")

    if isinstance(level, (int, Fraction)):
      if level > last_past_level.offset():
        raise TimeLevelException("Must initialise past timestep value")
      deps = self.__process_deps(x_deps, level)
      add_solve = self.__init_solves
    elif isinstance(level, TimeLevel):
      if level <= last_past_level:
        raise TimeLevelException("Must solve for future timestep value")
      deps = self.__process_deps(x_deps, level)
      add_solve = self.__solves
    elif isinstance(level, FinalTimeLevel):
      if level.offset() <= last_past_level.offset():
        raise TimeLevelException("Must solve for future timestep value")
      deps = self.__process_deps(x_deps, level)
      add_solve = self.__final_solves
    else:
      raise InvalidArgumentException("Invalid time level data")

    add_solve[x] = solve
    self.__deps[x] = deps

    self.__x_tfns[0], self.__x_tfns[1] = None, False
    self.__tfns[0], self.__tfns[1] = None, False
    self.__sorted_solves[0], self.__sorted_solves[1] = None, False

    return

  def add_solve(self, *args, **kwargs):
    """
    Add an equation solve.

    Arguments: One of:
        1. Arguments as accepted by the add_assignment method.
      or:
        2. Arguments as accepts by the PAEquationSolver constructor.
      or:
        3. An AssignmentSolver.
      or:
        4. An EquationSolver.
    """
    
    if len(args) == 2 and len(kwargs) == 0 and \
      (isinstance(args[0], (int, float, LinearCombination, ufl.expr.Expr)) or is_general_constant(args[0])) \
      and isinstance(args[1], dolfin.Function):
      self.add_assignment(args[0], args[1])
      return

    if len(args) == 1 and len(kwargs) == 0 and isinstance(args[0], (AssignmentSolver, EquationSolver)):
      solve = args[0]
      x = solve.x()
      if self.has_solve(x):
        raise StateException("Solve for %s already registered" % x.name())
      x_deps = solve.dependencies(non_symbolic = True)
      if isinstance(args[0], AssignmentSolver):
        for dep in x_deps:
          if dep is x:
            raise DependencyException("Invalid non-linear solve")
    else:
      s_kwargs = copy.copy(kwargs)
      if "adjoint_solver_parameters" in s_kwargs:
        if not isinstance(s_kwargs["adjoint_solver_parameters"], dict):
          raise InvalidArgumentException("adjoint_solver_parameters must be a dictionary")
        del(s_kwargs["adjoint_solver_parameters"])
      if "initial_guess" in s_kwargs:
        if not s_kwargs["initial_guess"] is None and not isinstance(s_kwargs["initial_guess"], dolfin.Function):
          raise InvalidArgumentException("initial_guess must be a Function")
        initial_guess = s_kwargs["initial_guess"]
        del(s_kwargs["initial_guess"])
      else:
        initial_guess = None
      if "form_compiler_parameters" in s_kwargs:
        raise NotImplementedException("form_compiler_parameters argument not supported")
      
      eargs = dolfin.fem.solving._extract_args(*args, **s_kwargs)
      x = eargs[1]
      if self.has_solve(x):
        raise StateException("Solve for %s already registered" % x.name())
      eq = eargs[0]
      solve = copy.copy(args), copy.copy(kwargs)
      x_deps = ufl.algorithms.extract_coefficients(eq.lhs)
      if not is_zero_rhs(eq.rhs):
        x_deps += ufl.algorithms.extract_coefficients(eq.rhs)
      if not initial_guess is None:
        x_deps.append(initial_guess)
      eq_lhs = eq.lhs

      lhs_data = extract_form_data(eq_lhs)
      if lhs_data.rank == 2:
        for dep in x_deps:
          if dep is x:
            raise DependencyException("Invalid non-linear solve")
    
    if not hasattr(x, "_time_level_data"):
      raise InvalidArgumentException("Missing time level data")
    last_past_level = x._time_level_data[0].last_past_level()
    level = x._time_level_data[1]
    
    if isinstance(level, (int, Fraction)):
      if level > last_past_level.offset():
        raise TimeLevelException("Must initialise past timestep value")
      deps = self.__process_deps(x_deps, level)
      add_solve = self.__init_solves
    elif isinstance(level, TimeLevel):
      if level <= last_past_level:
        raise TimeLevelException("Must solve for future timestep value")
      deps = self.__process_deps(x_deps, level)
      add_solve = self.__solves
    elif isinstance(level, FinalTimeLevel):
      if level.offset() <= last_past_level.offset():
        raise TimeLevelException("Must solve for future timestep value")
      deps = self.__process_deps(x_deps, level)
      add_solve = self.__final_solves
    else:
      raise InvalidArgumentException("Invalid time level data")

    add_solve[x] = solve
    self.__deps[x] = deps

    self.__x_tfns[0], self.__x_tfns[1] = None, False
    self.__tfns[0], self.__tfns[1] = None, False
    self.__sorted_solves[0], self.__sorted_solves[1] = None, False

    return

  def has_solve(self, x):
    """
    Return whether a solve for the given Function is registered.
    """
    
    if not isinstance(x, dolfin.Function):
      raise InvalidArgumentException("x must be a Function")
    
    if hasattr(x, "_time_level_data"):
      level = x._time_level_data[1]
      if isinstance(level, (int, Fraction)):
        return x in self.__init_solves
      elif isinstance(level, TimeLevel):
        return x in self.__solves
      else:
        assert(isinstance(level, FinalTimeLevel))
        return x in self.__final_solves
    else:
      return x in self.__final_solves

  def remove_solve(self, *args):
    """
    Remove any solve for the given Function.
    """
    
    for x in args:
      if not isinstance(x, dolfin.Function):
        raise InvalidArgumentException("Require Function s as arguments")
        
    for x in args:
      if hasattr(x, "_time_level_data"):
        level = x._time_level_data[1]
        if isinstance(level, (int, Fraction)):
          if x in self.__init_solves:
            del(self.__init_solves[x])
            del(self.__deps[x])
        elif isinstance(level, TimeLevel):
          if x in self.__solves:
            del(self.__solves[x])
            del(self.__deps[x])
        else:
          assert(isinstance(level, FinalTimeLevel))
          if x in self.__final_solves:
            del(self.__final_solves[x])
            del(self.__deps[x])
      elif x in self.__final_solves:
        del(self.__final_solves[x])
        del(self.__deps[x])

    if len(args) > 0:
      self.__x_tfns[0], self.__x_tfns[1] = None, False
      self.__tfns[0], self.__tfns[1] = None, False
      self.__sorted_solves[0], self.__sorted_solves[1] = None, False

    return

  def check_dependencies(self):
    """
    Verify dependencies, including a check for circular dependencies.
    """
    
    checked = set()
    def check(fn, deps, solves, against = []):
      if fn in solves and not fn in checked:
        assert(hasattr(fn, "_time_level_data"))
        level = fn._time_level_data[1]
        for dep in deps[fn]:
          if dep is fn or not dep in solves:
            continue
          dep_level = dep._time_level_data[1]
          if isinstance(level, (int, Fraction)):
            if not isinstance(dep_level, (int, Fraction)):
              raise DependencyException("Inconsistent dependency type")
          elif not isinstance(level, dep_level.__class__):
            raise DependencyException("Inconsistent dependency type")
          if dep_level > level:
            raise DependencyException("Future dependency for Function %s" % fn.name())
          elif dep in against:
            raise DependencyException("Circular dependency for Function %s" % fn.name())
          check(dep, deps, solves, against = against + [fn])
      checked.add(fn)
      return

    for fn in self.__init_solves:
      check(fn, self.__deps, self.__init_solves)
    for fn in self.__solves:
      check(fn, self.__deps, self.__solves)
    for fn in self.__final_solves:
      check(fn, self.__deps, self.__final_solves)
    
    return

  def check_function_levels(self):
    """
    Check that solves for all appropriate Function levels are defined. Emit
    a warning on failure.
    """
    
    for tfn in self.tfns():
      last_past_level = tfn.last_past_level()
      for level in tfn.levels():
        if level > last_past_level:
          if not tfn[level] in self.__solves:
            dolfin.info_red("Missing time level %s solve for TimeFunction %s" % (level, tfn.name()))
        else:
          if not tfn.has_level(level.offset()) or not tfn[level.offset()] in self.__init_solves:
            dolfin.info_red("Missing time level %i solve for TimeFunction %s" % (level.offset(), tfn.name()))

    return

  def set_update(self, update):
    """
    Register an update function. This is a callback:
      def update(s, cs = None):
        ...
        return
    where the update function is called at the start of timetep s and the
    optional cs is a list of Constant s or Function s to be updated. If cs is
    not supplied then all relevant updates should be performed.
    """
    
    if not callable(update):
      raise InvalidArgumentException("update must be a callable")

    self.__update = update

    return

  def x_tfns(self):
    """
    Return a list of Function s which are to be solved for.
    """
    
    if self.__x_tfns[1]:
      tfns = self.__x_tfns[0]
    else:
      tfns = set()
      for fn in self.__init_solves.keys() + self.__solves.keys() + self.__final_solves.keys():
        if hasattr(fn, "_time_level_data"):
          tfns.add(fn._time_level_data[0])
      tfns = sorted(list(tfns))

      self.__x_tfns[0], self.__x_tfns[1] = tfns, True

    return tfns

  def tfns(self):
    """
    Return a list of all registered TimeFunction s.
    """
    
    if self.__tfns[1]:
      tfns = self.__tfns[0]
    else:
      tfns = set()
      for x in self.__deps:
        if hasattr(x, "_time_level_data"):
          tfns.add(x._time_level_data[0])
        for dep in self.__deps[x]:
          if hasattr(dep, "_time_level_data"):
            tfns.add(dep._time_level_data[0])
      tfns = sorted(list(tfns))

      self.__tfns[0], self.__tfns[1] = tfns, True

    return tfns

  def sorted_solves(self):
    """
    Return all solves. Applies dependency resolution. Returns a tuple of
    lists, (initial solves, timestep solves, final solves), where each list
    contains AssignmentSolver s, EquationSolver s, or a tuple of arguments which
    can be used to construct PAEquationSolver s.
    """
    
    if self.__sorted_solves[1]:
      init_solves, solves, final_solves = self.__sorted_solves[0]
    else:
      self.check_function_levels()
      self.check_dependencies()

      def sort_solves(fn, deps, solves, ssolves, added_solves):
        if fn in added_solves:
          return
        for dep in deps[fn]:
          if not dep is fn and dep in solves:
            sort_solves(dep, deps, solves, ssolves, added_solves)
        added_solves.append(fn)
        ssolves.append(solves[fn])
        return

      added_solves = []
      init_solves = []
      solves = []
      final_solves = []
      for x in self.__init_solves:
        sort_solves(x, self.__deps, self.__init_solves, init_solves, added_solves)
      for x in self.__solves:
        sort_solves(x, self.__deps, self.__solves, solves, added_solves)
      for x in self.__final_solves:
        sort_solves(x, self.__deps, self.__final_solves, final_solves, added_solves)

      self.__sorted_solves[0], self.__sorted_solves[1] = [init_solves, solves, final_solves], True

    return copy.copy(init_solves), copy.copy(solves), copy.copy(final_solves)

  def assemble(self, *args, **kwargs):
    """
    Return an AssembledTimeSystem if adjoint is False, and an AdjoinedTimeSystem
    otherwise.

    Valid keyword arguments:
      adjoint (default false): Whether to create an AdjoinedTimeSystem rather
        than an AssembledTimeSystem.
    All other arguments are passed directly to the AssembledTimeSystem or
    AdjoinedTimeSystem constructors.
    """

    kwargs = copy.copy(kwargs)
    if "adjoint" in kwargs:
      adjoint = kwargs["adjoint"]
      del(kwargs["adjoint"])
    else:
      adjoint = False
    
    if adjoint:
      return AdjoinedTimeSystem(self, *args, **kwargs)
    else:
      return AssembledTimeSystem(self, *args, **kwargs)

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
  return apply(solver_parameters, {"linear_solver":"lu", "lu_solver":dolfin.parameters["lu_solver"].to_dict(), "krylov_solver":dolfin.parameters["krylov_solver"].to_dict()})

def LinearSolver(solver_parameters):
  """
  Return an LUSolver or KrylovSolver configured as per the supplied solver
  parameters.
  """
  
  if not isinstance(solver_parameters, dict):
    raise InvalidArgumentException("solver_parameters must be a dictionary")

  solver = "lu"
  pc = None
  kp = {}
  lp = {}
  for key in solver_parameters:
    if key == "linear_solver":
      solver = solver_parameters[key]
    elif key == "preconditioner":
      pc = solver_parameters[key]
    elif key == "krylov_solver":
      kp = solver_parameters[key]
    elif key == "lu_solver":
      lp = solver_parameters[key]
    elif key == "newton_solver":
      pass
    elif key in ["print_matrix", "print_rhs", "reset_jacobian", "symmetric"]:
      raise NotImplementedException("Unsupported solver parameter: %s" % key)
    else:
      raise InvalidArgumentException("Unexpected solver parameter: %s" % key)
  
  if solver == "lu":
    solver = dolfin.LUSolver()
    solver.parameters.update(lp)
  else:
    if pc is None:
      solver = dolfin.KrylovSolver(solver)
    else:
      solver = dolfin.KrylovSolver(solver, pc)
    solver.parameters.update(kp)

  return solver
    
def cache_info(msg, info = dolfin.info):
  """
  Print a message if verbose pre-assembly is enabled.
  """
  
  if dolfin.parameters["timestepping"]["pre_assembly"]["verbose"]:
    info(msg)
  return
    
class AssemblyCache:
  """
  A cache of assembled Form s. The assemble method can be used to assemble a
  given Form. If an assembled version of the Form exists in the cache, then the
  cached result is returned. Note that this does not check that the Form
  dependencies are unchanged between subsequent assemble calls -- that is
  deemed the responsibility of the caller.
  """
  
  def __init__(self):
    self.__cache = {}

    return

  def assemble(self, form, bcs = [], symmetric_bcs = False):
    """
    Return the result of assembling the supplied Form, optionally with boundary
    conditions, which are optionally applied so as to yield a symmetric matrix.
    If an assembled version of the Form exists in the cache, return the cached
    result. Note that this does not check that the Form dependencies are
    unchanged between subsequent assemble calls -- that is deemed the
    responsibility of the caller.
    """
    
    if not isinstance(form, ufl.form.Form):
      raise InvalidArgumentException("form must be a Form")
    if not isinstance(bcs, list):
      raise InvalidArgumentException("bcs must be a list of DirichletBC s")
    for bc in bcs:
      if not isinstance(bc, dolfin.cpp.DirichletBC):
        raise InvalidArgumentException("bcs must be a list of DirichletBC s")

    form_data = extract_form_data(form)
    if len(bcs) == 0:
      key = (expand(form), None, None)
      if not key in self.__cache:
        cache_info("Assembling form with rank %i" % form_data.rank, dolfin.info_red)
        self.__cache[key] = assemble(form)
      else:
        cache_info("Using cached assembled form with rank %i" % form_data.rank, dolfin.info_green)
    else:
      if not form_data.rank == 2:
        raise InvalidArgumentException("form must be rank 2 when applying boundary conditions")

      key = (expand(form), tuple(bcs), symmetric_bcs)
      if not key in self.__cache:
        cache_info("Assembling form with rank 2, with boundary conditions", dolfin.info_red)
        mat = assemble(form)
        apply_bcs(mat, bcs, symmetric_bcs = symmetric_bcs)
        self.__cache[key] = mat
      else:
        cache_info("Using cached assembled form with rank 2, with boundary conditions", dolfin.info_green)

    return self.__cache[key]

  def info(self):
    """
    Print some cache status information.
    """
    
    counts = [0, 0, 0]
    for key in self.__cache.keys():
      counts[extract_form_data(key[0]).rank] += 1

    dolfin.info_blue("Assembly cache status:")
    for i in range(3):
      dolfin.info_blue("Pre-assembled rank %i forms: %i" % (i, counts[i]))

    return

  def clear(self, *args):
    """
    Clear the cache. If arguments are supplied, clear only the cached assembled
    Form s which depend upon the supplied Constant s or Function s.
    """
    
    if len(args) == 0:
      self.__cache = {}
    else:
      for dep in args:
        if not isinstance(dep, (dolfin.Constant, dolfin.Function)):
          raise InvalidArgumentException("Arguments must be Constant s or Function s")

      for dep in args:
        for key in copy.copy(self.__cache.keys()):
          form = key[0]
          if dep in ufl.algorithms.extract_coefficients(form):
            del(self.__cache[key])

    return

class SolverCache:
  """
  A cache of LUSolver s and KrylovSolver s. The solver method can be used to
  return an LUSolver or KrylovSolver suitable for solving an equation with the
  supplied rank 2 Form defining the LHS matrix.
  """
  
  def __init__(self):
    self.__cache = OrderedDict()

    return

  def __del__(self):
    for key in self.__cache:
      del(self.__cache[key])

    return

  def solver(self, form, solver_parameters, static = False, bcs = [], symmetric_bcs = False):
    """
    Return an LUSolver or KrylovSolver suitable for solving an equation with the
    supplied rank 2 Form defining the LHS. If such a solver exists in the cache,
    return the cached solver. Optionally accepts boundary conditions which
    are optionally applied so as to yield a symmetric matrix. If static is true
    then it is assumed that the Form is "static", and solver caching
    options are enabled. The appropriate value of the static argument is
    deemed the responsibility of the caller.
    """
    
    if not isinstance(form, ufl.form.Form):
      raise InvalidArgumentException("form must be a rank 2 Form")
    elif not extract_form_data(form).rank == 2:
      raise InvalidArgumentException("form must be a rank 2 Form")
    if not isinstance(solver_parameters, dict):
      raise InvalidArgumentException("solver_parameters must be a dictionary")
    if not isinstance(bcs, list):
      raise InvalidArgumentException("bcs must be a list of DirichletBC s")
    for bc in bcs:
      if not isinstance(bc, dolfin.cpp.DirichletBC):
        raise InvalidArgumentException("bcs must be a list of DirichletBC s")

    def flatten_parameters(opts):
      assert(isinstance(opts, dict))
      fopts = []
      for key in opts.keys():
        if isinstance(opts[key], dict):
          fopts.append(flatten_parameters(opts[key]))
        else:
          fopts.append((key, opts[key]))
      return tuple(fopts)

    if static:
      if not "linear_solver" in solver_parameters or solver_parameters["linear_solver"] == "lu":
        solver_parameters = expand_solver_parameters(solver_parameters, default_solver_parameters = {"lu_solver":{"reuse_factorization":True, "same_nonzero_pattern":True}})
      else:
        solver_parameters = expand_solver_parameters(solver_parameters, default_solver_parameters = {"krylov_solver":{"preconditioner":{"reuse":True}}})
    else:
      solver_parameters = expand_solver_parameters(solver_parameters)

      static_parameters = False
      if solver_parameters["linear_solver"] == "lu":
        static_parameters = solver_parameters["lu_solver"]["reuse_factorization"] or solver_parameters["lu_solver"]["same_nonzero_pattern"]
      else:
        static_parameters = solver_parameters["krylov_solver"]["preconditioner"]["reuse"]
      if static_parameters:
        raise ParameterException("Non-static solve supplied with static solver parameters")

    if static:
      if len(bcs) == 0:
        key = (expand(form), flatten_parameters(solver_parameters))
      else:
        key = (expand(form), tuple(bcs), symmetric_bcs, flatten_parameters(solver_parameters))
    else:
      args = ufl.algorithms.extract_arguments(form)
      assert(len(args) == 2)
      test, trial = args
      if test.count() > trial.count():
        test, trial = trial, test
      if len(bcs) == 0:
        key = ((test, trial), flatten_parameters(solver_parameters))
      else:
        key = ((test, trial), tuple(bcs), symmetric_bcs, flatten_parameters(solver_parameters))

    if not key in self.__cache:
      if static:
        cache_info("Creating new static linear solver", dolfin.info_red)
      else:
        cache_info("Creating new non-static linear solver", dolfin.info_red)
      self.__cache[key] = LinearSolver(solver_parameters)
    else:
      cache_info("Using cached linear solver", dolfin.info_green)
    return self.__cache[key]

  def clear(self, *args):
    """
    Clear the cache. If arguments are supplied, clear only the solvers
    associated with Form s which depend upon the supplied Constant s or
    Function s.
    """
    
    if len(args) == 0:
      for key in self.__cache.keys():
        del(self.__cache[key])
    else:
      for dep in args:
        if not isinstance(dep, (dolfin.Constant, dolfin.Function)):
          raise InvalidArgumentException("Arguments must be Constant s or Function s")

      for key in self.__cache:
        form = key[0]
        if isinstance(form, ufl.form.Form) and dep in ufl.algorithms.extract_coefficients(form):
          del(self.__cache[key])

    return

# Default assembly and solver caches.
assembly_cache = AssemblyCache()
solver_cache = SolverCache()
def clear_caches(*args):
  """
  Clear the default assembly and solver caches. If arguments are supplied, clear
  only cached data associated with Form s which depend upon the supplied
  Constant s or Function s.
  """

  assembly_cache.clear(*args)
  solver_cache.clear(*args)
  
  return
    
class PAForm:
  """
  A pre-assembled form. Given a form of arbitrary rank, this finds and
  pre-assembles static terms.

  Constructor arguments:
    form: The Form to be pre-assembled.
    parameters: Parameters defining detailed optimisation options.
  """
  
  def __init__(self, form, parameters = dolfin.parameters["timestepping"]["pre_assembly"]["forms"]):
    if isinstance(parameters, dict):
      self.parameters = dolfin.Parameters(**parameters)
    else:
      self.parameters = dolfin.Parameters(parameters)

    self.__set(form)

    return

  def __set(self, form):
    if not isinstance(form, ufl.form.Form) or is_empty_form(form):
      raise InvalidArgumentException("form must be a non-empty Form")

    if self.parameters["whole_form_optimisation"]:
      if is_static_form(form):
        self.__set_pa([form], [])
      else:
        self.__set_pa([], [form])
    else:
      self.__set_split(form)

    self.__rank = extract_form_data(form).rank
    self.__deps = ufl.algorithms.extract_coefficients(form)

    return

  if ufl_version() < (1, 2, 0):
    def __form_integrals(self, form):
      form_data = extract_form_data(form)
      integrals = []
      for integral_data in form_data.integral_data:
        integrals += integral_data[2]
      return integrals
    
    def __preprocess_integral(self, form, integral):
      form_data = extract_form_data(form)
      integrand, measure = integral.integrand(), integral.measure()
      repl = {}
      for old, new in zip(form_data.arguments + form_data.coefficients, form_data.original_arguments + form_data.original_coefficients):
        repl[old] = new
      integrand = replace(integrand, repl)
      return integrand, [measure]
  else:
    def __form_integrals(self, form):
      return form.integrals()
    
    def __preprocess_integral(self, form, integral):
      integrand = integral.integrand()
      domain_type, domain_description, compiler_data, domain_data = \
        integral.domain_type(), integral.domain_description(), integral.compiler_data(), integral.domain_data()
      return integrand, [domain_type, domain_description, compiler_data, domain_data]

  def __set_split(self, form):
    quadrature_degree = form_quadrature_degree(form)

    pre_assembled_L = []
    non_pre_assembled_L = []
    
    for integral in self.__form_integrals(form):
      integrand, iargs = self.__preprocess_integral(form, integral)
      integrand = ufl.algebra.Sum(*expr_terms(integrand))
      if isinstance(integrand, ufl.algebra.Sum):
        terms = integrand.operands()
      else:
        terms = [integrand]
      for term in terms:
        tform = QForm([ufl.Integral(term, *iargs)], quadrature_degree = quadrature_degree)
        if is_static_form(tform):
          pre_assembled_L.append(tform)
        else:
          non_pre_assembled_L.append(tform)

    self.__set_pa(pre_assembled_L, non_pre_assembled_L)

    return

  def __set_pa(self, pre_assembled_L, non_pre_assembled_L):
    if len(pre_assembled_L) == 0:
      self.__pre_assembled_L = None
    else:
      l_L = pre_assembled_L[0]
      for L in pre_assembled_L[1:]:
        l_L += L
      self.__pre_assembled_L = assembly_cache.assemble(l_L)

    if len(non_pre_assembled_L) == 0:
      self.__non_pre_assembled_L = None
    else:
      self.__non_pre_assembled_L = non_pre_assembled_L[0]
      for L in non_pre_assembled_L[1:]:
        self.__non_pre_assembled_L += L
        
    self.__n_pre_assembled = len(pre_assembled_L)
    self.__n_non_pre_assembled = len(non_pre_assembled_L)

    return

  def assemble(self, copy = False):
    """
    Return the result of assembling the Form associated with the PAForm. If
    copy is False then existing data may be returned -- it is expected in this
    case that the return value will never be modified.
    """
    
    if self.__non_pre_assembled_L is None:
      if self.__pre_assembled_L is None:
        raise StateException("Cannot assemble empty form")
      else:
        if copy:
          L = self.__pre_assembled_L.copy()
        else:
          L = self.__pre_assembled_L
    else:
      if hasattr(self, "_PAForm__non_pre_assembled_L_tensor"):
        L = self.__non_pre_assembled_L_tensor
        assemble(self.__non_pre_assembled_L, tensor = L, reset_sparsity = False)
      else:
        L = self.__non_pre_assembled_L_tensor = assemble(self.__non_pre_assembled_L)
      if not self.__pre_assembled_L is None:
        L += self.__pre_assembled_L

    return L
  
  def rank(self):
    """
    Return the Form rank.
    """
    
    return self.__rank
    
  def n_pre_assembled(self):
    """
    Return the number of pre-assembled terms.
    """
    
    return self.__n_pre_assembled

  def n_non_pre_assembled(self):
    """
    Return the number of non-pre-assembled terms.
    """
    
    return self.__n_non_pre_assembled

  def n(self):
    """
    Return the total number of terms.
    """
    
    return self.n_pre_assembled() + self.n_non_pre_assembled()

  def dependencies(self, non_symbolic = False):
    """
    Return Form dependencies. The optional non_symbolic has no effect.
    """
    
    return self.__deps
  
  def replace(self, mapping):
    """
    Replace coefficients.
    """
    
    if not self.__non_pre_assembled_L is None:
      self.__non_pre_assembled_L = replace(self.__non_pre_assembled_L, mapping)
    
    return
    
class PABilinearForm(PAForm):
  """
  A pre-assembled bi-linear form. This is identical to PAForm, but with
  different default parameters.
  """
  
  def __init__(self, form, parameters = dolfin.parameters["timestepping"]["pre_assembly"]["bilinear_forms"]):
    PAForm.__init__(self, form, parameters = parameters)

    return
    
class PALinearForm(PAForm):
  """
  A pre-assembled linear form. This is similar to PAForm, but applies additional
  optimisations specific to linear forms. Also has different default parameters.
  """
  
  def __init__(self, form, parameters = dolfin.parameters["timestepping"]["pre_assembly"]["linear_forms"]):
    PAForm.__init__(self, form, parameters = parameters)

    return

  def _PAForm__set(self, form):
    if not isinstance(form, ufl.form.Form) or is_empty_form(form):
      raise InvalidArgumentException("form must be a non-empty Form")

    if self.parameters["whole_form_optimisation"]:
      if is_static_form(form):
        self.__set_pa([form], [], [])
      else:
        self.__set_pa([], [], [form])
    else:
      self.__set_split(form)

    self._PAForm__rank = extract_form_data(form).rank
    self._PAForm__deps = ufl.algorithms.extract_coefficients(form)

    return

  def __set_split(self, form):
    quadrature_degree = form_quadrature_degree(form)
    
    pre_assembled_L = []
    mult_assembled_L = OrderedDict()
    non_pre_assembled_L = []
    
    def matmul_optimisation(tform):
      tcs = extract_non_static_coefficients(term)
      if not len(tcs) == 1 or not isinstance(tcs[0], dolfin.Function) or \
        (not dolfin.MPI.num_processes() == 1 and not tcs[0].function_space().num_sub_spaces() == 0):
        return False
      fn = tcs[0]
        
      mat_form = derivative(tform, fn)
      if n_non_static_coefficients(mat_form) > 0:
        return False

      if fn in mult_assembled_L:
        mult_assembled_L[fn].append(mat_form)
      else:
        mult_assembled_L[fn] = [mat_form]
      
      return True
    
    for integral in self._PAForm__form_integrals(form):
      integrand, iargs = self._PAForm__preprocess_integral(form, integral)
      integrand = ufl.algebra.Sum(*expr_terms(integrand))
      if isinstance(integrand, ufl.algebra.Sum):
        terms = integrand.operands()
      else:
        terms = [integrand]
      for term in terms:
        tform = QForm([ufl.Integral(term, *iargs)], quadrature_degree = quadrature_degree)
        if is_static_form(tform):
          pre_assembled_L.append(tform)
        elif not matmul_optimisation(tform):
          non_pre_assembled_L.append(tform)

    self.__set_pa(pre_assembled_L, mult_assembled_L, non_pre_assembled_L)
          
    return

  def __set_pa(self, pre_assembled_L, mult_assembled_L, non_pre_assembled_L):
    if len(pre_assembled_L) == 0:
      self._PAForm__pre_assembled_L = None
    else:
      l_L = pre_assembled_L[0]
      for L in pre_assembled_L[1:]:
        l_L += L
      self._PAForm__pre_assembled_L = assembly_cache.assemble(l_L)

    n_mult_assembled_L = 0
    if len(mult_assembled_L) == 0:
      self.__mult_assembled_L = None
    else:
      self.__mult_assembled_L = []
      for fn in mult_assembled_L:
        mat_forms = mult_assembled_L[fn]
        n_mult_assembled_L += len(mat_forms)
        mat_form = mat_forms[0]
        for lmat_form in mat_forms[1:]:
          mat_form += lmat_form
        self.__mult_assembled_L.append([assembly_cache.assemble(mat_form), fn])

    if len(non_pre_assembled_L) == 0:
      self._PAForm__non_pre_assembled_L = None
    else:
      self._PAForm__non_pre_assembled_L = non_pre_assembled_L[0]
      for L in non_pre_assembled_L[1:]:
        self._PAForm__non_pre_assembled_L += L

    self._PAForm__n_pre_assembled = len(pre_assembled_L) + n_mult_assembled_L
    self._PAForm__n_non_pre_assembled = len(non_pre_assembled_L)

    return

  def assemble(self, copy = False):
    """
    Return the result of assembling the Form associated with this PALinearForm.
    If copy is False then an existing GenericVector may be returned -- it is
    expected in this case that the return value will never be modified.
    """
    
    if self._PAForm__non_pre_assembled_L is None:
      if self.__mult_assembled_L is None:
        if self._PAForm__pre_assembled_L is None:
          raise StateException("Cannot assemble empty form")
        else:
          if copy:
            L = self._PAForm__pre_assembled_L.copy()
          else:
            L = self._PAForm__pre_assembled_L
      else:
        L = self.__mult_assembled_L[0][0] * self.__mult_assembled_L[0][1].vector()
        for i in range(1, len(self.__mult_assembled_L)):
          L += self.__mult_assembled_L[i][0] * self.__mult_assembled_L[i][1].vector()
        if not self._PAForm__pre_assembled_L is None:
          L += self._PAForm__pre_assembled_L
    else:
      if hasattr(self, "_PAForm__non_pre_assembled_L_tensor"):
        L = self._PAForm__non_pre_assembled_L_tensor
        assemble(self._PAForm__non_pre_assembled_L, tensor = L, reset_sparsity = False)
      else:
        L = self._PAForm__non_pre_assembled_L_tensor = assemble(self._PAForm__non_pre_assembled_L)
      if not self.__mult_assembled_L is None:
        for i in range(len(self.__mult_assembled_L)):
          L += self.__mult_assembled_L[i][0] * self.__mult_assembled_L[i][1].vector()
      if not self._PAForm__pre_assembled_L is None:
        L += self._PAForm__pre_assembled_L

    return L
  
  def replace(self, mapping):
    """
    Replace coefficients.
    """
    
    PAForm.replace(self, mapping)
    if not self.__mult_assembled_L is None:
      for i in range(len(self.__mult_assembled_L)):
        mat, fn = self.__mult_assembled_L[i]
        if fn in mapping:
          self.__mult_assembled_L[i] = mat, mapping[fn]
    
    return

class EquationSolver:
  """
  A generic linear or non-linear equation solver.

  Constructor arguments:
    eq: The Equation being solved.
    x: The Function being solved for.
    bcs: A list of DirichletBC s.
    solver_parameters: A dictionary of solver parameters.
    adjoint_solver_parameters: A dictionary of solver parameters for an adjoint
      solve.
  """
  
  def __init__(self, eq, x, bcs = [], solver_parameters = {}, adjoint_solver_parameters = None):
    if not isinstance(eq, ufl.equation.Equation):
      raise InvalidArgumentException("eq must be an Equation")
    if not isinstance(x, dolfin.Function):
      raise InvalidArgumentException("x must be a Function")
    if isinstance(bcs, dolfin.cpp.DirichletBC):
      bcs = [bcs]
    if not isinstance(bcs, list):
      raise InvalidArgumentException("bcs must be a DirichletBC or a list of DirichletBC s")
    for bc in bcs:
      if not isinstance(bc, dolfin.cpp.DirichletBC):
        raise InvalidArgumentException("bcs must be a DirichletBC or a list of DirichletBC s")
    if not isinstance(solver_parameters, dict):
      raise InvalidArgumentException("solver_parameters must be a dictionary")
    if not adjoint_solver_parameters is None and not isinstance(adjoint_solver_parameters, dict):
      raise InvalidArgumentException("adjoint_solver_parameters must be a dictionary")

    solver_parameters = copy.copy(solver_parameters)
    if adjoint_solver_parameters is None:
      adjoint_solver_parameters = solver_parameters
    else:
      adjoint_solver_parameters = copy.copy(adjoint_solver_parameters)

    x_deps = ufl.algorithms.extract_coefficients(eq.lhs)
    if not is_zero_rhs(eq.rhs):
      x_deps += ufl.algorithms.extract_coefficients(eq.rhs)

    is_linear = not x in x_deps
      
    self.__eq = eq
    self.__x = x
    self.__bcs = copy.copy(bcs)
    self.__J = None
    self.__hbcs = None
    self.__solver_parameters = solver_parameters
    self.__adjoint_solver_parameters = adjoint_solver_parameters
    self.__x_deps = x_deps
    self.__is_linear = is_linear
    
    self.__tl = [None, False]

    return

  def is_assembled(self):
    """
    Return whether the EquationSolver is assembled. Always true.
    """
    
    return True

  def assemble(self):
    """
    Assemble the EquationSolver. Alias to the reassemble method with no
    arguments.
    """
    
    self.reassemble()

    return

  def reassemble(self, *args):
    """
    Reassemble the equation solver. Has no effect.
    """
    
    return

  def dependencies(self, non_symbolic = False):
    """
    Return Equation dependencies. The optional non_symbolic argument has no
    effect.
    """
    
    return self.__x_deps

  def nonlinear_dependencies(self):
    """
    Return non-linear dependencies of the equation, by returning the
    dependencies of the tangent linear equation.
    """
    
    d, od = self.tangent_linear()
    nl_deps = ufl.algorithms.extract_coefficients(d)
    for od_form in od.values():
      nl_deps += ufl.algorithms.extract_coefficients(od_form)

    return nl_deps

  def x(self):
    """
    Return the Function being solved for.
    """
    
    return self.__x

  def eq(self):
    """
    Return the Equation.
    """
    
    return self.__eq.lhs == self.__eq.rhs

  def bcs(self):
    """
    Return the DirichletBC s.
    """
    
    return self.__bcs

  def J(self):
    """
    Return the derivative of the residual with respect to x (the Jacobian).
    """

    if self.__J is None:
      if self.is_linear():
        form = action(self.__eq.lhs, self.__x)
      else:
        form = self.__eq.lhs
      if not is_zero_rhs(self.__eq.rhs):
        form -= self.__eq.rhs
      self.__J = derivative(form, self.__x)

    return self.__J

  def hbcs(self):
    """
    Return a homogenised version of the DirichletBC s associated with the
    EquationSolver.
    """
    
    if self.__hbcs is None:
      self.__hbcs = [homogenize(bc) for bc in self.__bcs]

    return self.__hbcs

  def solver_parameters(self):
    """
    Return solver parameters.
    """
    
    return self.__solver_parameters

  def adjoint_solver_parameters(self):
    """
    Return adjoint solver parameters.
    """
    
    return self.__adjoint_solver_parameters

  def is_linear(self):
    """
    Return whether the Equation is written as a linear variational problem.
    """
    
    return self.__is_linear

  def __add_tangent_linear(self):
    a_d = None
    a_od = {}

    def add_lhs_dep(form, dep, x):
      add_rhs_dep(action(-form, x), dep, x)
      return
    def add_rhs_dep(form, dep, x):
      if isinstance(dep, dolfin.Function):
        mat = derivative(-form, dep)
        if dep in a_od:
          a_od[dep] += mat
        else:
          a_od[dep] = mat
      elif not isinstance(dep, (dolfin.Constant, dolfin.Expression)):
        raise DependencyException("Invalid dependency")
      return

    x = self.__x
    lhs, rhs = self.__eq.lhs, self.__eq.rhs

    if self.is_linear():
      a_d = lhs
      for dep in ufl.algorithms.extract_coefficients(lhs):
        if dep is x:
          raise DependencyException("Invalid non-linear solve")
        add_lhs_dep(lhs, dep, x)
      if not is_zero_rhs(rhs):
        for dep in ufl.algorithms.extract_coefficients(rhs):
          if dep is x:
            raise DependencyException("Invalid non-linear solve")
          add_rhs_dep(rhs, dep, x)
    else:
      for dep in ufl.algorithms.extract_coefficients(lhs):
        add_rhs_dep(-lhs, dep, x)
      if not is_zero_rhs(rhs):
        for dep in ufl.algorithms.extract_coefficients(rhs):
          add_rhs_dep(rhs, dep, x)

      if not x in a_od:
        raise DependencyException("Missing LHS")
      a_d = a_od[x]
      del(a_od[x])

    self.__tl = [(a_d, a_od), True]

    return

  def tangent_linear(self):
    """
    Return the tangent linear equation. The tangent linear is returned in
    terms of blocks, where each block is a Form that can be used to assemble
    a matrix block associated with the given dependency. The return value is
    a tuple (diagonal Form, off diagonal Form s), where the off diagonal Form s
    are a dictionary with dependencies as keys and Form s as values.
    """
    
    if not self.__tl[1]:
      self.__add_tangent_linear()

    return self.__tl[0]

  def derivative(self, parameter):
    """
    Return the derivative of the residual with respect to the supplied Constant
    or Function.
    """
    
    if not isinstance(parameter, (dolfin.Constant, dolfin.Function)):
      raise InvalidArgumentException("parameter must be a Constant or Function")

    if self.is_linear():
      form = action(self.__eq.lhs, self.__x)
    else:
      form = self.__eq.lhs
    if not is_zero_rhs(self.__eq.rhs):
      form -= self.__eq.rhs

    return derivative(form, parameter)

  def solve(self):
    """
    Solve the equation. This calls the DOLFIN solve function.
    """

    if self.is_linear() and extract_form_data(self.__eq.lhs).rank == 1:
      raise NotImplementedException("Solve for linear variational problem with rank 1 LHS not implemented")
    
    if self.__J is None:
      dolfin.solve(self.__eq, self.__x, self.__bcs, solver_parameters = self.__solver_parameters)
    else:
      dolfin.solve(self.__eq, self.__x, self.__bcs, J = self.__J, solver_parameters = self.__solver_parameters)
    return
    
class PAEquationSolver(EquationSolver):
  """
  An EquationSolver applying additional pre-assembly and solver caching
  optimisations. This utilises pre-assembly of static terms. The arguments match
  those accepted by the DOLFIN solve function, with the following differences:

    Argument 1: May be a general equation. Linear systems are detected
      automatically.
    initial_guess: The initial guess for an iterative solver.
    adjoint_solver_parameters: A dictionary of solver parameters for an adjoint
      equation solve.
  """
  
  def __init__(self, *args, **kwargs):
    args, kwargs = copy.copy(args), copy.copy(kwargs)

    # Process arguments not to be passed to _extract_args
    if "parameters" in kwargs:
      parameters = kwargs["parameters"]
      del(kwargs["parameters"])
    else:
      parameters = dolfin.parameters["timestepping"]["pre_assembly"]
    if isinstance(parameters, dict):
      parameters = dolfin.Parameters(**parameters)
    else:
      parameters = dolfin.Parameters(parameters)
    if "initial_guess" in kwargs:
      if not kwargs["initial_guess"] is None and not isinstance(kwargs["initial_guess"], dolfin.Function):
        raise InvalidArgumentException("initial_guess must be a Function")
      initial_guess = kwargs["initial_guess"]
      del(kwargs["initial_guess"])
    else:
      initial_guess = None
    if "adjoint_solver_parameters" in kwargs:
      if not kwargs["adjoint_solver_parameters"] is None and not isinstance(kwargs["adjoint_solver_parameters"], dict):
        raise InvalidArgumentException("adjoint_solver_parameters must be a dictionary")
      adjoint_solver_parameters = kwargs["adjoint_solver_parameters"]
      del(kwargs["adjoint_solver_parameters"])
    else:
      adjoint_solver_parameters = None

    # Process remaining arguments
    if "form_compiler_parameters" in kwargs:
      raise NotImplementedException("form_compiler_parameters argument not supported")
    eq, x, bcs, J, tol, goal, form_parameters, solver_parameters = dolfin.fem.solving._extract_args(*args, **kwargs)

    # Relax requirements on equation syntax
    eq_lhs_rank = extract_form_data(eq.lhs).rank
    if eq_lhs_rank == 1:
      form = eq.lhs
      if not is_zero_rhs(eq.rhs):
        form -= eq.rhs
      if x in ufl.algorithms.extract_coefficients(form):
        if J is None:
          J = derivative(form, x)
        if x in ufl.algorithms.extract_coefficients(J):
          # Non-linear solve
          is_linear = False
        else:
          # Linear solve, rank 2 LHS
          cache_info("Detected that solve for %s is linear" % x.name(), dolfin.info_blue)
          form = replace(form, {x:dolfin.TrialFunction(x.function_space())})
          eq = lhs(form) == rhs(form)
          eq_lhs_rank = extract_form_data(eq.lhs).rank
          assert(eq_lhs_rank == 2)
          is_linear = True
      else:
        # Linear solve, rank 1 LHS
        is_linear = True
    elif eq_lhs_rank == 2:
      form = eq.lhs
      if not is_zero_rhs(eq.rhs):
        form -= eq.rhs
      if not x in ufl.algorithms.extract_coefficients(form):
        # Linear solve, rank 2 LHS
        eq = lhs(form) == rhs(form)
        eq_lhs_rank = extract_form_data(eq.lhs).rank
        assert(eq_lhs_rank == 2)
        is_linear = True
      else:
        # ??
        raise InvalidArgumentException("Invalid equation")
        
    # Initial guess sanity checking
    if not initial_guess is None:
      if is_linear and eq_lhs_rank == 1:
        # Supplied an initial guess for a linear solve with a rank 1 LHS -
        # ignore it
        initial_guess = None
      elif "linear_solver" in solver_parameters and not solver_parameters["linear_solver"] == "lu":
        # Supplied an initial guess with a Krylov solver - check the
        # initial_guess solver parameter
        if not "krylov_solver" in solver_parameters:
          solver_parameters["krylov_solver"] = {}
        if not "nonzero_initial_guess" in solver_parameters["krylov_solver"]:
          solver_parameters["krylov_solver"]["nonzero_initial_guess"] = True
        elif not solver_parameters["krylov_solver"]["nonzero_initial_guess"]:
          raise ParameterException("initial_guess cannot be set if nonzero_initial_guess solver parameter is False")
      elif is_linear:
        # Supplied an initial guess for a linear solve with an LU solver -
        # ignore it
        initial_guess = None

    # Initialise
    EquationSolver.__init__(self, eq, x, bcs, solver_parameters = solver_parameters, adjoint_solver_parameters = adjoint_solver_parameters)
    self.__args = args
    self.__kwargs = kwargs
    self.__J = J
    self.__tol = tol
    self.__goal = goal
    self.__form_parameters = form_parameters
    self.__initial_guess = initial_guess
    self.parameters = parameters

    # Assemble
    self.reassemble()

    return
  
  def reassemble(self, *args):
    """
    Reassemble the PAEquationSolver. If no arguments are supplied, reassemble
    both the LHS and RHS. Otherwise, only reassemble the LHS or RHS if they
    depend upon the supplied Constant s or Function s. Note that this does
    not clear the assembly or solver caches -- hence if a static Constant,
    Function, or DicichletBC is modified then one should clear the caches before
    calling reassemble on the PAEquationSolver.
    """
    
    x, eq, bcs, solver_parameters = self.x(), self.eq(), self.bcs(), self.solver_parameters()
    x_deps = self.dependencies()
    if self.is_linear():
      for dep in x_deps:
        if dep is x:
          raise DependencyException("Invalid non-linear solve")

      def assemble_lhs():
        eq_lhs_rank = extract_form_data(eq.lhs).rank
        if eq_lhs_rank == 2:
          static_bcs = n_non_static_bcs(bcs) == 0
          static_form = is_static_form(eq.lhs)
          if not self.parameters["equations"]["symmetric_boundary_conditions"] and len(bcs) > 0 and static_bcs and static_form:
            a = assembly_cache.assemble(eq.lhs, bcs = bcs, symmetric_bcs = False)
            cache_info("Pre-assembled LHS terms in solve for %s    : 1" % x.name(), dolfin.info_blue)
            cache_info("Non-pre-assembled LHS terms in solve for %s: 0" % x.name(), dolfin.info_blue)
            solver = solver_cache.solver(eq.lhs, solver_parameters, static = True, bcs = bcs, symmetric_bcs = False)
          else:
            a = PABilinearForm(eq.lhs, parameters = self.parameters["bilinear_forms"])
            cache_info("Pre-assembled LHS terms in solve for %s    : %i" % (x.name(), a.n_pre_assembled()), dolfin.info_blue)
            cache_info("Non-pre-assembled LHS terms in solve for %s: %i" % (x.name(), a.n_non_pre_assembled()), dolfin.info_blue)
            solver = solver_cache.solver(eq.lhs, solver_parameters, static = a.n_non_pre_assembled() == 0 and static_bcs and static_form, bcs = bcs, symmetric_bcs = self.parameters["equations"]["symmetric_boundary_conditions"])
        else:
          assert(eq_lhs_rank == 1)
          a = PALinearForm(eq.lhs, parameters = self.parameters["linear_forms"])
          cache_info("Pre-assembled LHS terms in solve for %s    : %i" % (x.name(), a.n_pre_assembled()), dolfin.info_blue)
          cache_info("Non-pre-assembled LHS terms in solve for %s: %i" % (x.name(), a.n_non_pre_assembled()), dolfin.info_blue)
          solver = None
        return a, solver
      def assemble_rhs():
        L = PALinearForm(eq.rhs, parameters = self.parameters["linear_forms"])
        cache_info("Pre-assembled RHS terms in solve for %s    : %i" % (x.name(), L.n_pre_assembled()), dolfin.info_blue)
        cache_info("Non-pre-assembled RHS terms in solve for %s: %i" % (x.name(), L.n_non_pre_assembled()), dolfin.info_blue)
        return L

      if len(args) == 0:
        a, solver = assemble_lhs()
        L = assemble_rhs()
      else:
        a, solver = self.__a, self.__solver
        L = self.__L
        lhs_cs = ufl.algorithms.extract_coefficients(eq.lhs)
        rhs_cs = ufl.algorithms.extract_coefficients(eq.rhs)
        for dep in args:
          if dep in lhs_cs:
            a, solver = assemble_lhs()
            break
        for dep in args:
          if dep in rhs_cs:
            L = assemble_rhs()
            break
    else:
      J, hbcs = self.J(), self.hbcs()

      def assemble_lhs():
        a = PABilinearForm(J, parameters = self.parameters["bilinear_forms"])
        cache_info("Pre-assembled LHS terms in solve for %s    : %i" % (x.name(), a.n_pre_assembled()), dolfin.info_blue)
        cache_info("Non-pre-assembled LHS terms in solve for %s: %i" % (x.name(), a.n_non_pre_assembled()), dolfin.info_blue)
        solver = solver_cache.solver(J, solver_parameters, static = False, bcs = hbcs, symmetric_bcs = self.parameters["equations"]["symmetric_boundary_conditions"])
        return a, solver
      def assemble_rhs():
        L = -eq.lhs
        if not is_zero_rhs(eq.rhs):
          L += eq.rhs
        L = PALinearForm(L, parameters = self.parameters["linear_forms"])
        cache_info("Pre-assembled RHS terms in solve for %s    : %i" % (x.name(), L.n_pre_assembled()), dolfin.info_blue)
        cache_info("Non-pre-assembled RHS terms in solve for %s: %i" % (x.name(), L.n_non_pre_assembled()), dolfin.info_blue)
        return L

      if len(args) == 0:
        a, solver = assemble_lhs()
        L = assemble_rhs()
      else:
        a, solver = self.__a, self.__solver
        L = self.__L
        lhs_cs = ufl.algorithms.extract_coefficients(J)
        rhs_cs = ufl.algorithms.extract_coefficients(eq.lhs)
        if not is_zero_rhs(eq.rhs):
          rhs_cs += ufl.algorithms.extract_coefficients(eq.rhs)
        for dep in args:
          if dep in lhs_cs:
            a, solver = assemble_lhs()
            break
        for dep in args:
          if dep in rhs_cs:
            L = assemble_rhs()
            break
       
      self.__dx = x.vector().copy()

    self.__a, self.__solver = a, solver
    self.__L = L

    return

  def dependencies(self, non_symbolic = False):
    """
    Return equation dependencies. If non_symbolic is true, also return any
    other dependencies which could alter the result of a solve, such as the
    initial guess.
    """
    
    def uses_x_as_initial_guess():
      if not self.is_linear():
        return self.__initial_guess is None
      solver = self.solver()
      if solver is None:
        return False
      else:
        return self.__initial_guess is None and hasattr(solver.parameters, "nonzero_initial_guess") and solver.parameters["nonzero_initial_guess"]
    
    if not non_symbolic:
      return EquationSolver.dependencies(self, non_symbolic = False)
    elif not self.__initial_guess is None:
      return EquationSolver.dependencies(self, non_symbolic = True) + [self.__initial_guess]
    elif uses_x_as_initial_guess():
      return EquationSolver.dependencies(self, non_symbolic = True) + [self.x()]
    else:
      return EquationSolver.dependencies(self, non_symbolic = True)

  def solver(self):
    """
    Return the linear solver.
    """
    
    return self.__solver

  def solve(self):
    """
    Solve the equation. This utilises a custom Newton solver for non-linear
    equations. The Newton solver is intended to have near identical behaviour
    to the Newton solver supplied with DOLFIN, but utilises pre-assembly.
    """
    
    x = self.x()
    if not self.__initial_guess is None:
      x.assign(self.__initial_guess)
    
    if self.is_linear():
      bcs, solver = self.bcs(), self.solver()

      if isinstance(self.__a, dolfin.GenericMatrix):
        a = self.__a
        L = assemble(self.__L, copy = len(bcs) > 0)
        enforce_bcs(L, bcs)

        solver.set_operator(a)
        solver.solve(x.vector(), L)
      elif self.__a.rank() == 2:
        a = assemble(self.__a, copy = len(bcs) > 0)
        L = assemble(self.__L, copy = len(bcs) > 0)
        apply_bcs(a, bcs, L = L, symmetric_bcs = self.parameters["equations"]["symmetric_boundary_conditions"])

        solver.set_operator(a)
        solver.solve(x.vector(), L)
      else:
        assert(self.__a.rank() == 1)
        assert(solver is None)
        a = assemble(self.__a, copy = False)
        L = assemble(self.__L, copy = False)

        x.vector().set_local(L.array() / a.array())
        x.vector().apply("insert")
        enforce_bcs(x.vector(), bcs)
    else:
      # Newton solver, intended to have near identical behaviour to the Newton
      # solver supplied with DOLFIN. See
      # http://fenicsproject.org/documentation/tutorial/nonlinear.html for
      # further details.
      
      default_parameters = dolfin.NewtonSolver.default_parameters()
      solver_parameters = self.solver_parameters()
      if "newton_solver" in solver_parameters:
        parameters = solver_parameters["newton_solver"]
      else:
        parameters = {}
      solver = self.solver()

      atol = default_parameters["absolute_tolerance"]
      rtol = default_parameters["relative_tolerance"]
      max_its = default_parameters["maximum_iterations"]
      omega = default_parameters["relaxation_parameter"]
      err = default_parameters["error_on_nonconvergence"]
      r_def = default_parameters["convergence_criterion"]
      for key in parameters.keys():
        if key == "absolute_tolerance":
          atol = parameters[key]
        elif key == "convergence_criterion":
          r_def = parameters[key]
        elif key == "error_on_nonconvergence":
          err = parameters[key]
        elif key == "maximum_iterations":
          max_its = parameters[key]
        elif key == "relative_tolerance":
          rtol = parameters[key]
        elif key == "relaxation_parameter":
          omega = parameters[key]
        elif key in ["method", "report"]:
          raise NotImplementedException("Unsupported solver parameter: %s" % key)
        else:
          raise InvalidArgumentException("Unexpected solver parameter: %s" % key)

      eq, bcs, hbcs = self.eq(), self.bcs(), self.hbcs()
      a, L = self.__a, self.__L

      x_name = x.name()
      x = x.vector()
      enforce_bcs(x, bcs)

      dx = self.__dx
      if not isinstance(solver, dolfin.GenericLUSolver):
        dx.zero()
        
      if r_def == "residual":
        l_L = assemble(L, copy = len(hbcs) > 0)
        enforce_bcs(l_L, hbcs)
        r_0 = l_L.norm("l2")
        it = 0
        if r_0 >= atol:
          l_a = assemble(a, copy = len(hbcs) > 0)
          apply_bcs(l_a, hbcs, symmetric_bcs = self.parameters["equations"]["symmetric_boundary_conditions"])
          solver.set_operator(l_a)
          solver.solve(dx, l_L)
          x.axpy(omega, dx)
          it += 1
          atol = max(atol, r_0 * rtol)
          while it < max_its:
            l_L = assemble(L, copy = len(hbcs) > 0)
            enforce_bcs(l_L, hbcs)
            r = l_L.norm("l2")
            if r < atol:
              break
            l_a = assemble(a, copy = len(hbcs) > 0)
            apply_bcs(l_a, hbcs, symmetric_bcs = self.parameters["equations"]["symmetric_boundary_conditions"])
            solver.set_operator(l_a)
            solver.solve(dx, l_L)
            x.axpy(omega, dx)
            it += 1
      elif r_def == "incremental":
        l_a = assemble(a, copy = len(hbcs) > 0)
        l_L = assemble(L, copy = len(hbcs) > 0)
        apply_bcs(l_a, hbcs, L = l_L, symmetric_bcs = self.parameters["equations"]["symmetric_boundary_conditions"])
        solver.set_operator(l_a)
        solver.solve(dx, l_L)
        x.axpy(omega, dx)
        it = 1
        r_0 = dx.norm("l2")
        if r_0 >= atol:
          atol = max(atol, rtol * r_0)
          while it < max_its:
            l_a = assemble(a, copy = len(hbcs) > 0)
            l_L = assemble(L, copy = len(hbcs) > 0)
            apply_bcs(l_a, hbcs, L = l_L, symmetric_bcs = self.parameters["equations"]["symmetric_boundary_conditions"])
            solver.set_operator(l_a)
            solver.solve(dx, l_L)
            x.axpy(omega, dx)
            it += 1
            if dx.norm("l2") < atol:
              break
      else:
        raise ParameterException("Invalid convergence criterion: %s" % r_def)
      if it == max_its:
        if err:
          raise StateException("Newton solve for %s failed to converge after %i iterations" % (x_name, it))
        else:
          dolfin.info_red("Warning: Newton solve for %s failed to converge after %i iterations" % (x_name, it))
#      dolfin.info("Newton solve for %s converged after %i iterations" % (x_name, it))

    return

def pa_solve(*args, **kwargs):
  """
  Instantiate a PAEquationSolver using the supplied arguments and call its solve
  method.
  """
  
  PAEquationSolver(*args, **kwargs).solve()

  return
  
class AssembledTimeSystem:
  """
  Used to solve timestep equations with timestep specific optimisations applied.

  Constructor arguments:
    tsystem: A TimeSystem defining the timestep equations.
    initialise: Whether the initialise method is to be called.
    reassemble: Whether the reassemble methods of solvers defined by the
      TimeSystem should be called.
  """
  
  def __init__(self, tsystem, initialise = True, reassemble = False):
    if not isinstance(tsystem, TimeSystem):
      raise InvalidArgumentException("tsystem must be a TimeSystem")
    
    tfns = tsystem.tfns()
    init_solves, solves, final_solves = tsystem.sorted_solves()

    for i, solve in enumerate(init_solves):
      if isinstance(solve, (AssignmentSolver, EquationSolver)):
        if not solve.is_assembled():
          solve.assemble()
        elif reassemble:
          solve.reassemble()
      else:
        init_solves[i] = PAEquationSolver(*solve[0], **solve[1])
    for i, solve in enumerate(solves):
      if isinstance(solve, (AssignmentSolver, EquationSolver)):
        if not solve.is_assembled():
          solve.assemble()
        elif reassemble:
          solve.reassemble()
      else:
        solves[i] = PAEquationSolver(*solve[0], **solve[1])
    for i, solve in enumerate(final_solves):
      if isinstance(solve, (AssignmentSolver, EquationSolver)):
        if not solve.is_assembled():
          solve.assemble()
        elif reassemble:
          solve.reassemble()
      else:
        final_solves[i] = PAEquationSolver(*solve[0], **solve[1])

    self.__tfns = tfns
    self.__init_solves = init_solves
    self.__solves = solves
    self.__final_solves = final_solves
    self.__update = tsystem._TimeSystem__update
    self.__s = 0

    if initialise:
      self.initialise()

    return

  def tfns(self):
    """
    Return a list of all registered TimeFunction s.
    """
    
    return self.__tfns
    
  def initialise(self):
    """
    Solve initial equations and perform the initial variable cycle. Also reset
    the timestep counter.
    """
    
    self.timestep_update(s = 0)
    
#    dolfin.info("Performing forward initial solves")
    for solve in self.__init_solves:
      solve.solve()

#    dolfin.info("Performing forward initial cycle")
    for tfn in self.__tfns:
      tfn.initial_cycle()
      
    return

  def timestep_update(self, s = None, cs = None):
    """
    Call the update callback. s can be used to set the value of the timestep
    counter. Otherwise the timestep counter is incremented. The optional cs is a
    list of Constant s or Function s to be passed to the callback for update.
    """
    
    if not s is None:
      if not isinstance(s, int) or s < 0:
        raise InvalidArgumentException("s must be a non-negative integer")
    if not cs is None:
      if not isinstance(cs, (list, set)):
        raise InvalidArgumentException("cs must be a list of Constant s or Function s")
      for c in cs:
        if not isinstance(c, (dolfin.Constant, dolfin.Function)):
          raise InvalidArgumentException("cs must be a list of Constant s or Function s")
    
    if not s is None:
      self.__s = s
    else:
      self.__s += 1
    if not self.__update is None:
      self.__update(self.__s, cs = cs)

    return

  def timestep_solve(self):
    """
    Perform the timestep solves.
    """
    
#    dolfin.info("Performing forward timestep")
    for solve in self.__solves:
      solve.solve()

    return

  def timestep_cycle(self, extended = True):
    """
    Perform the timestep cycle. If extended is true, use the extended cycle
    with Function aliasing. Otherwise use Function copying.
    """
    
#    dolfin.info("Performing forward timestep cycle")
    for tfn in self.__tfns:
      tfn.cycle(extended = extended)

    return
      
  def timestep(self, ns = 1):
    """
    Perform ns timesteps.
    """
    
    for i in range(ns):
      self.timestep_update()
      self.timestep_solve()
      self.timestep_cycle()

    return

  def finalise(self):
    """
    Solve final equations and perform the final variable cycle.
    """
    
    self.timestep_update(s = self.__s + 1)
    
#    dolfin.info("Performing forward final cycle")
    for tfn in self.__tfns:
      tfn.final_cycle()

#    dolfin.info("Performing forward final solves")
    for solve in self.__final_solves:
      solve.solve()

    return

  def reassemble(self, *args, **kwargs):
    """
    Call the reassemble method of all solves, with the supplied arguments. This
    first clears the relevant cache data if clear_caches is true.

    Valid keyword arguments:
      clear_caches (default true): Whether to clear caches.
    """

    lclear_caches = True
    for key in kwargs:
      if key == "clear_caches":
        lclear_caches = kwargs["clear_caches"]
      else:
        raise InvalidArgumentException("Unexpected keyword argument: %s" % key)
    if lclear_caches:
      clear_caches(*args)
    for solve in self.__init_solves + self.__solves + self.__final_solves:
      solve.reassemble(*args)
      
    return
    
def assemble(*args, **kwargs):
  """
  Wrapper for the DOLFIN assemble function. Correctly handles PAForm s,
  TimeSystem s and QForm s.
  """
  
  if isinstance(args[0], PAForm):
    return args[0].assemble(*args[1:], **kwargs)
  elif isinstance(args[0], TimeSystem):
    return args[0].assemble(*args[1:], **kwargs)
  elif isinstance(args[0], QForm):
    if "form_compiler_parameters" in kwargs:
      raise InvalidArgumentException("Cannot supply form_compiler_parameters argument when assembling a QForm")
    return dolfin.assemble(form_compiler_parameters = args[0].form_compiler_parameters(), *args, **kwargs)
  else:
    return dolfin.assemble(*args, **kwargs)

class AdjointTimeFunction(TimeLevels):
  """
  An adjoint function defined on a number of time levels.

  Constructor arguments:
    tfn: The associated forward TimeFunction.
  """
  
  def __init__(self, tfn):
    if not isinstance(tfn, TimeFunction):
      raise InvalidArgumentException("tfn must be a TimeFunction")

    name = tfn.name()

    fns = {}
    for level in tfn.levels():
      fns[level] = Function(name = "%s_%s_adjoint" % (name, level), *[tfn.function_space()])
      fns[level]._time_level_data = (self, level)
      fns[level]._adjoint_data = [tfn[level]]
    for level in tfn.initial_levels():
      fns[level] = WrappedFunction(fns[n + level], name = "%s_%s_adjoint" % (name, level))
      fns[level]._time_level_data = (self, level)
      fns[level]._adjoint_data = [tfn[level]]
    for level in tfn.final_levels():
      fns[level] = WrappedFunction(fns[n + level.offset()], name = "%s_%s_adjoint" % (name, level))
      fns[level]._time_level_data = (self, level)
      fns[level]._adjoint_data = [tfn[level]]
      
    self._TimeLevels__copy_time_levels(tfn)
    self.__name = name
    self.__fns = fns
    self.__space = tfn.function_space()
    self.__tfn = tfn

    return

  def __getitem__(self, key):
    if isinstance(key, (int, Fraction, TimeLevel, FinalTimeLevel)):
      return self.__fns[key]
    else:
      raise InvalidArgumentException("key must be an integer, Fraction, TimeLevel, or FinalTimeLevel")

  def name(self):
    """
    Return the name of the AdjointTimeFunction, as a string.
    """
    
    return self.__name

  def has_level(self, level):
    """
    Return whether the AdjointTimeFunction is defined on the specified level.
    level may be an integer, Fraction, TimeLevel or FinalTimeLevel.
    """
    
    if not isinstance(level, (int, Fraction, TimeLevel, FinalTimeLevel)):
      raise InvalidArgumentException("level must be an integer, Fraction, TimeLevel, or FinalTimeLevel")

    return level in self.__fns

  def forward(self):
    """
    Return the forward TimeFunction associated with the AdjointTimeFunction.
    """
    
    return self.__tfn
    
class AdjointVariableMap:
  """
  A map between forward and adjoint variables. Indexing into the
  AdjointVariableMap with a forward Function yields an associated adjoint
  function, and similarly indexing into the AdjointVariableMap with an adjoint
  Function yields an associated forward function. Allocates adjoint Function s
  as required.
  """
  
  def __init__(self):
    self.__a_tfns = {}
    self.__f_tfns = {}
    self.__a_fns = {}
    self.__f_fns = {}

    return

  def __getitem__(self, key):
    return self.__add(key)

  def __add(self, var):    
    if isinstance(var, TimeFunction):
      if not var in self.__a_tfns:
        f_tfn = var
        a_tfn = AdjointTimeFunction(f_tfn)
        self.__a_tfns[f_tfn] = a_tfn
        self.__f_tfns[a_tfn] = f_tfn
        for level in f_tfn.all_levels():
          self.__a_fns[f_tfn[level]] = a_tfn[level]
          self.__f_fns[a_tfn[level]] = f_tfn[level]
      return self.__a_tfns[var]
    elif isinstance(var, AdjointTimeFunction):
      if not var in self.__f_tfns:
        f_tfn = var.forward()
        a_tfn = var
        self.__a_tfns[f_tfn] = a_tfn
        self.__f_tfns[a_tfn] = f_tfn
        for level in f_tfn.all_levels():
          self.__a_fns[f_tfn[level]] = a_tfn[level]
          self.__f_fns[a_tfn[level]] = f_tfn[level]
      return self.__f_tfns[var]
    elif isinstance(var, dolfin.Function):
      if is_static_coefficient(var):
        return var
      elif hasattr(var, "_time_level_data"):
        return self.__add(var._time_level_data[0])[var._time_level_data[1]]
      elif hasattr(var, "_adjoint_data"):
        if not var in self.__f_fns:
          self.__a_fns[var._adjoint_data[0]] = var
          self.__f_fns[var] = var._adjoint_data[0]
        return var._adjoint_data[0]
      else:
        if not var in self.__a_fns:
          a_fn = Function(name = "%s_adjoint" % var.name(), *[var.function_space()])
          a_fn._adjoint_data = [var]
          self.__a_fns[var] = a_fn
          self.__f_fns[a_fn] = var
        return self.__a_fns[var]
    elif isinstance(var, dolfin.Constant):
      return var
    else:
      raise InvalidArgumentException("Argument must be an AdjointTimeFunction, TimeFunction, Function, or Constant")

  def zero_adjoint(self):
    """
    Zero all adjoint Function s,
    """

    for a_fn in self.__f_fns:
      if not hasattr(a_fn, "_time_level_data") or isinstance(a_fn._time_level_data[1], TimeLevel):
        a_fn.vector().zero()

    return

class TimeFunctional:
  """
  A template for a functional with an explicit time dependence.
  """
  
  def __init__(self):
    return

  def initialise(self, val = 0.0):
    """
    Initialise, with an initial functional value of val.
    """
    
    raise AbstractMethodException("initialise method not overridden")

  def addto(self, s):
    """
    Add to the functional at the end of timestep number s.
    """
    
    raise AbstractMethodException("addto method not overridden")

  def value(self):
    """
    Return the functional value.
    """
    
    raise AbstractMethodException("value method not overridden")

  def dependencies(self, s = None, non_symbolic = False):
    """
    Return the functional dependencies at the end of timestep number s. If
    non_symbolic is true, also return any other dependencies on which the value
    of the functional could depend at the end of timestep number s.
    """
    
    raise AbstractMethodException("dependencies method not overridden")

  def derivative(self, parameter, s):
    """
    Return the derivative of the functional with respect to the specified
    Constant of Function at the end of the timestep number s.
    """
    
    raise AbstractMethodException("derivative method not overridden")
  
class PAAdjointSolvers:
  """
  Defines a set of solves for adjoint equations, applying pre-assembly and
  solver caching optimisations. Expects as input a list of earlier forward
  equations and a list of later forward equations. If the earlier equations
  solve for {x_1, x_2, ...}, then the Function s on which the later equations
  depend should all be static or in the {x_1, x_2, ...}, although the failure
  of this requirement is not treated as an error.

  Constructor arguments:
    f_solves_a: Earlier time forward equations, as a list of AssignmentSolver s
      or EquationSolver s.
    f_solves_b: Later time forward equations, as a list of AssignmentSolver s
      or EquationSolver s.
    a_map: The AdjointVariableMap used to convert between forward and adjoint
      Function s.
    parameters: Parameters defining detailed optimisation options.
  """
  
  def __init__(self, f_solves_a, f_solves_b, a_map, parameters = dolfin.parameters["timestepping"]["pre_assembly"]):
    if not isinstance(f_solves_a, list):
      raise InvalidArgumentException("f_solves_a must be a list of AssignmentSolver s or EquationSolver s")
    for f_solve in f_solves_a:
      if not isinstance(f_solve, (AssignmentSolver, EquationSolver)):
        raise InvalidArgumentException("f_solves_a must be a list of AssignmentSolver s or EquationSolver s")
    if not isinstance(f_solves_b, list):
      raise InvalidArgumentException("f_solves_b must be a list of AssignmentSolver s or EquationSolver s")
    for f_solve in f_solves_b:
      if not isinstance(f_solve, (AssignmentSolver, EquationSolver)):
        raise InvalidArgumentException("f_solves_b must be a list of AssignmentSolver s or EquationSolver s")
    if not isinstance(a_map, AdjointVariableMap):
      raise InvalidArgumentException("a_map must be an AdjointVariableMap")
    if isinstance(parameters, dict):
      parameters = dolfin.Parameters(**parameters)
    else:
      parameters = dolfin.Parameters(parameters)

    # Reverse causality
    f_solves_a = copy.copy(f_solves_a);  f_solves_a.reverse()
    f_solves_b = copy.copy(f_solves_b);  f_solves_b.reverse()

    la_a_forms = []
    la_x = []
    la_L_forms = []
    la_L_as = []
    la_bcs = []
    la_solver_parameters = []
    la_keys = {}

    # Create an adjoint solve for each forward solve in f_solves_a, and add
    # the adjoint LHS
    for f_solve in f_solves_a:
      f_x = f_solve.x()
      a_x = a_map[f_x]
      a_space = a_x.function_space()
      assert(not a_x in la_keys)
      if isinstance(f_solve, AssignmentSolver):
        la_a_forms.append(None)
        la_bcs.append([])
        la_solver_parameters.append({})
      else:
        assert(isinstance(f_solve, EquationSolver))
        f_a = f_solve.tangent_linear()[0]
        f_a_rank = extract_form_data(f_a).rank
        if f_a_rank == 2:
          a_test, a_trial = dolfin.TestFunction(a_space), dolfin.TrialFunction(a_space)
          a_a = adjoint(f_a, adjoint_arguments = (a_test, a_trial))
          la_a_forms.append(a_a)
          la_bcs.append(f_solve.hbcs())
          la_solver_parameters.append(copy.copy(f_solve.adjoint_solver_parameters()))
#          if not "krylov_solver" in la_solver_parameters[-1]:
#            la_solver_parameters[-1]["krylov_solver"] = {}
#          la_solver_parameters[-1]["krylov_solver"]["nonzero_initial_guess"] = False
        else:
          assert(f_a_rank == 1)
          a_a = f_a
          la_a_forms.append(a_a)
          la_bcs.append(f_solve.hbcs())
          la_solver_parameters.append(None)
      la_x.append(a_x)
      la_L_forms.append(None)
      la_L_as.append([])
      la_keys[a_x] = len(la_x) - 1

    # Add adjoint RHS terms corresponding to terms in each forward solve in
    # f_solves_a and f_solves_b
    for f_solve in f_solves_a + f_solves_b:
      f_x = f_solve.x()
      a_dep = a_map[f_x]
      if isinstance(f_solve, AssignmentSolver):
        f_rhs = f_solve.rhs()
        if isinstance(f_rhs, ufl.expr.Expr):
          # Adjoin an expression assignment RHS
          for f_dep in ufl.algorithms.extract_coefficients(f_rhs):
            if isinstance(f_dep, dolfin.Function):
              a_x = a_map[f_dep]
              a_rhs = differentiate_expr(f_rhs, f_dep) * a_dep
              if a_x in la_keys and not isinstance(a_rhs, ufl.constantvalue.Zero):
                la_L_as[la_keys[a_x]].append(a_rhs)
        else:
          # Adjoin a linear combination assignment RHS
          for alpha, f_dep in f_rhs:
            a_x = a_map[f_dep]
            if a_x in la_keys:
              la_L_as[la_keys[a_x]].append((alpha, a_dep))
      else:
        # Adjoin an equation RHS
        assert(isinstance(f_solve, EquationSolver))
        a_trial = dolfin.TrialFunction(a_dep.function_space())
        f_a_od = f_solve.tangent_linear()[1]
        for f_dep in f_a_od:
          a_x = a_map[f_dep]
          if a_x in la_keys:
            a_test = dolfin.TestFunction(a_x.function_space())
            a_key = la_keys[a_x]
            a_form = -action(adjoint(f_a_od[f_dep], adjoint_arguments = (a_test, a_trial)), a_dep)
            if la_L_forms[a_key] is None:
              la_L_forms[a_key] = a_form
            else:
              la_L_forms[a_key] += a_form
              
    self.__a_map = a_map
    self.__a_a_forms = la_a_forms
    self.__a_x = la_x
    self.__a_L_forms = la_L_forms
    self.__a_L_as = la_L_as
    self.__a_bcs = la_bcs
    self.__a_solver_parameters = la_solver_parameters
    self.__a_keys = la_keys
    self.parameters = parameters
    
    self.__functional = None
    self.reassemble()
      
    return

  def reassemble(self, *args):
    """
    Reassemble the adjoint solvers. If no arguments are supplied then all
    equations are re-assembled. Otherwise, only the LHSs or RHSs which depend
    upon the supplied Constant s or Function s are reassembled. Note that this
    does not clear the assembly or solver caches -- hence if a static
    Constant, Function, or DirichletBC is modified then one should clear the
    caches before calling reassemble on the PAAdjointSolvers.
    """

    def assemble_lhs(i):
      if self.__a_a_forms[i] is None:
        a_a = None
        a_solver = None
      else:
        a_a_rank = extract_form_data(self.__a_a_forms[i]).rank
        if a_a_rank == 2:
          static_bcs = n_non_static_bcs(self.__a_bcs[i]) == 0
          static_form = is_static_form(self.__a_a_forms[i])
          if len(self.__a_bcs[i]) > 0 and static_bcs and static_form:
            a_a = assembly_cache.assemble(self.__a_a_forms[i], bcs = self.__a_bcs[i], symmetric_bcs = self.parameters["equations"]["symmetric_boundary_conditions"])
            a_solver = solver_cache.solver(self.__a_a_forms[i], self.__a_solver_parameters[i], static = True, bcs = self.__a_bcs[i], symmetric_bcs = self.parameters["equations"]["symmetric_boundary_conditions"])
          else:
            a_a = PABilinearForm(self.__a_a_forms[i], parameters = self.parameters["bilinear_forms"])
            a_solver = solver_cache.solver(self.__a_a_forms[i], self.__a_solver_parameters[i], static = a_a.n_non_pre_assembled() == 0 and static_bcs and static_form, bcs = self.__a_bcs[i], symmetric_bcs = self.parameters["equations"]["symmetric_boundary_conditions"])
        else:
          assert(a_a_rank == 1)
          assert(self.__a_solver_parameters[i] is None)
          a_a = PALinearForm(self.__a_a_forms[i], parameters = self.parameters["linear_forms"])
          a_solver = None
      return a_a, a_solver
    def assemble_rhs(i):
      if self.__a_L_forms[i] is None:
        return None
      else:
        return PALinearForm(self.__a_L_forms[i], parameters = self.parameters["linear_forms"])
    
    if len(args) == 0:
      la_a, la_solvers = [], []
      la_L = []
      for i in range(len(self.__a_x)):
        a_a, a_solver = assemble_lhs(i)
        a_L = assemble_rhs(i)
        la_a.append(a_a)
        la_solvers.append(a_solver)
        la_L.append(a_L)

      self.set_functional(self.__functional)
    else:
      la_a, la_solvers = copy.copy(self.__a_a), copy.copy(self.__a_solvers)
      la_L = copy.copy(self.__a_L)
      for i in range(len(self.__a_x)):
        for dep in args:
          if not self.__a_a_forms[i] is None and dep in ufl.algorithms.extract_coefficients(self.__a_a_forms[i]):
            la_a[i], la_solvers[i] = assemble_lhs(i)
            break
        for dep in args:
          if not self.__a_L_forms[i] is None and dep in ufl.algorithms.extract_coefficients(self.__a_L_forms[i]):
            la_L[i] = assemble_rhs(i)
            break

      if isinstance(self.__functional, ufl.form.Form):
        for dep in args:
          if dep in ufl.algorithms.extract_coefficients(self.__functional):
            self.set_functional(self.__functional)
            break
      else:
        self.set_functional(self.__functional)

    self.__a_a, self.__a_solvers = la_a, la_solvers
    self.__a_L = la_L

    return

  def a_x(self):
    """
    Return the adjoint Function s being solved for.
    """
    
    return self.__a_x

  def solve(self):
    """
    Solve all adjoint equations.
    """
    
    for i in range(len(self.__a_x)):
      a_a = self.__a_a[i]
      a_x = self.__a_x[i]
      a_L = self.__a_L[i]
      a_L_as = self.__a_L_as[i]
      a_L_rhs = self.__a_L_rhs[i]
      a_bcs = self.__a_bcs[i]
      a_solver = self.__a_solvers[i]

      def evaluate_a_L_as(i):
        if isinstance(a_L_as[i], ufl.expr.Expr):
          if is_r0_function(a_x):
            L = evaluate_expr(a_L_as[i], copy = False)
            if isinstance(L, dolfin.GenericVector):
              l_L = L.sum()
            else:
              assert(isinstance(L, float))
              l_L = L
            L = a_x.vector().copy()
            L[:] = l_L
          else:
            L = evaluate_expr(a_L_as[i], copy = True)
            if isinstance(L, float):
              assert(isinstance(L, float))
              l_L = L
              L = a_x.vector().copy()
              L[:] = l_L
            else:
              assert(isinstance(L, dolfin.GenericVector))
        else:
          L = float(a_L_as[i][0]) * a_L_as[i][1].vector()
        return L
      def add_a_L_as(i, L):
        if isinstance(a_L_as[i], ufl.expr.Expr):
          l_L = evaluate_expr(a_L_as[i], copy = False)
          if is_r0_function(a_x):
            if isinstance(l_L, dolfin.GenericVector):
              l_L = l_L.sum()
            else:
              assert(isinstance(l_L, float))
          if isinstance(l_L, dolfin.GenericVector):
            L += l_L
          else:
            L.add_local(l_L * numpy.ones(L.local_range(0)[1] - L.local_range(0)[0]))
            L.apply("insert")
        else:
          L.axpy(float(a_L_as[i][0]), a_L_as[i][1].vector())
        return

      if a_L_rhs is None:
        if len(a_L_as) == 0:
          if a_L is None:
            if a_a is None or len(a_bcs) == 0:
              a_x.vector().zero()
              continue
            else:
              L = a_x.vector().copy()
              L.zero()
          else:
            L = assemble(a_L, copy = len(a_bcs) > 0)
        else:
          L = evaluate_a_L_as(0)
          for i in range(1, len(a_L_as)):
            add_a_L_as(i, L)
          if not a_L is None:
            L += assemble(a_L, copy = False)
      else:
        if isinstance(a_L_rhs, PAForm):
          L = assemble(a_L_rhs, copy = len(a_bcs) > 0 or not a_L is None or len(a_L_as) > 0)
        else:
          L = assemble(a_L_rhs)
        if not a_L is None:
          L += assemble(a_L, copy = False)
        for i in range(len(a_L_as)):
          add_a_L_as(i, L)

      if a_a is None:
        assert(len(a_bcs) == 0)
        assert(a_solver is None)
        a_x.vector()[:] = L
      elif a_solver is None:
        assert(a_a.rank() == 1)
        a_a = assemble(a_a, copy = False)
        a_x.vector().set_local(L.array() / a_a.array())
        a_x.vector().apply("insert")
        enforce_bcs(a_x.vector(), a_bcs)
      else:
        if isinstance(a_a, dolfin.GenericMatrix):
          enforce_bcs(L, a_bcs)
        else:
          a_a = assemble(a_a, copy = len(a_bcs) > 0)
          apply_bcs(a_a, a_bcs, L = L, symmetric_bcs = self.parameters["equations"]["symmetric_boundary_conditions"])
        a_solver.set_operator(a_a)
        a_solver.solve(a_x.vector(), L)

    return

  def set_functional(self, functional):
    """
    Set a functional, defining associated adjoint RHS terms.
    """
    
    if functional is None:
      self.__a_L_rhs = [None for i in range(len(self.__a_x))]
      self.__functional = None
    elif isinstance(functional, ufl.form.Form):
      if not extract_form_data(functional).rank == 0:
        raise InvalidArgumentException("functional must be rank 0")

      a_rhs = OrderedDict()
      for f_dep in ufl.algorithms.extract_coefficients(functional):
        if is_static_coefficient(f_dep):
          pass
        elif isinstance(f_dep, dolfin.Function):
          a_x = self.__a_map[f_dep]
          a_rhs[a_x] = derivative(functional, f_dep)
        elif isinstance(f_dep, (dolfin.Constant, dolfin.Expression)):
          pass
        else:
          raise DependencyException("Invalid dependency")

      self.__a_L_rhs = [None for i in range(len(self.__a_x))]
      for a_x in a_rhs:
        if a_x in self.__a_keys:
          self.__a_L_rhs[self.__a_keys[a_x]] = PALinearForm(a_rhs[a_x], parameters = self.parameters["linear_forms"])
      self.__functional = functional
    elif isinstance(functional, TimeFunctional):
      self.__a_L_rhs = [None for i in range(len(self.__a_x))]
      self.__functional = functional
    else:
      raise InvalidArgumentException("functional must be a Form or a TimeFunctional")

    return

  def update_functional(self, s):
    """
    Update the adjoint RHS associated with the functional at the end of timestep
    s.
    """
    
    if not isinstance(s, int) or s < 0:
      raise InvalidArgumentException("s must be a non-negative integer")

    if not isinstance(self.__functional, TimeFunctional):
      return

    a_rhs = OrderedDict()
    for f_dep in self.__functional.dependencies(s):
      if is_static_coefficient(f_dep):
        pass
      elif isinstance(f_dep, dolfin.Function):
        a_x = self.__a_map[f_dep]
        a_rhs[a_x] = self.__functional.derivative(f_dep, s)
      elif isinstance(f_dep, dolfin.Constant):
        pass
      else:
        raise DependencyException("Invalid dependency")

    self.__a_L_rhs = [None for i in range(len(self.__a_x))]
    for a_x in a_rhs:
      if not a_x in self.__a_keys:
        dolfin.info_red("Warning: Missing functional dependency %s" % a_x.name())
      else:
        self.__a_L_rhs[self.__a_keys[a_x]] = a_rhs[a_x]

    return
    
class AdjointTimeSystem:
  """
  Used to solve adjoint timestep equations with timestep specific optimisations
  applied. This assumes that forward model data is updated externally.

  Constructor arguments:
    tsystem: A TimeSystem defining the timestep equations.
    functional: A rank 0 form or a TimeFunctional defining the functional.
  """
  
  def __init__(self, tsystem, functional = None):
    if not isinstance(tsystem, AssembledTimeSystem):
      raise InvalidArgumentException("tsystem must be an AssembledTimeSystem")

    # Step 1: Set up the adjoint variable map
    a_map = AdjointVariableMap()

    # Step 2: Set up the forward variable cycle "solvers"
    f_tfns = tsystem.tfns()
    f_as_init_solves = []
    f_as_solves = []
    # A block identity operator. In the first adjoint timestep cycle levels
    # n + i and N + i are logically distinct, but the Function s are aliased to
    # each other. Replacing f_as_solves with f_as_id_solves effectively zeros
    # the action of the off-diagonal blocks associated with levels n + i.
    f_as_id_solves = []
    f_as_final_solves = []
    for f_tfn in f_tfns:
      init_cycle_map = f_tfn.initial_cycle_map()
      cycle_map = f_tfn.cycle_map()
      final_cycle_map = f_tfn.final_cycle_map()
      for level in init_cycle_map:
        f_as_init_solves.append(AssignmentSolver(f_tfn[init_cycle_map[level]], f_tfn[level]))
      for level in cycle_map:
        f_as_solves.append(AssignmentSolver(f_tfn[cycle_map[level]], f_tfn[level]))
        f_as_id_solves.append(AssignmentSolver(0.0, f_tfn[level]))
      for level in final_cycle_map:
        f_as_final_solves.append(AssignmentSolver(f_tfn[final_cycle_map[level]], f_tfn[level]))
        
    # Step 3: Adjoin the final solves
#    dolfin.info("Initialising adjoint initial solves")
    a_initial_solves = PAAdjointSolvers(tsystem._AssembledTimeSystem__final_solves, [], a_map)
#    dolfin.info("Initialising adjoint initial cycle 1")
    a_initial_cycle1 = PAAdjointSolvers(f_as_final_solves, tsystem._AssembledTimeSystem__final_solves, a_map)
#    dolfin.info("Initialising adjoint initial cycle 2")
    a_initial_cycle2 = PAAdjointSolvers(f_as_id_solves, f_as_final_solves, a_map)
#    dolfin.info("Initialising adjoint initial cycle 3")
    a_initial_cycle3 = PAAdjointSolvers(f_as_init_solves, f_as_final_solves, a_map)

    # Step 4: Adjoin the timestep
#    dolfin.info("Initialising adjoint timestep cycle")
    a_cycle = PAAdjointSolvers(f_as_solves, tsystem._AssembledTimeSystem__solves, a_map)
#    dolfin.info("Initialising adjoint timestep")
    a_solves = PAAdjointSolvers(tsystem._AssembledTimeSystem__solves, f_as_solves, a_map)
    
    # Step 5: Adjoin the initial solves
#    dolfin.info("Initialising adjoint final cycle")
    # For the final adjoint cycle and solves we need to know the first forward
    # solves for all time levels. If no forward timestep "solve" is specified
    # for a given level, then we need to know the forward timestep "cycle"
    # instead.
    f_init_dep_solves = copy.copy(tsystem._AssembledTimeSystem__solves)  
    f_x = set()
    for solve in tsystem._AssembledTimeSystem__solves:
      f_x.add(solve.x())
    for f_tfn in f_tfns:
      cycle_map = f_tfn.cycle_map()
      for level in cycle_map:
        if not f_tfn[level] in f_x:
          f_init_dep_solves.append(AssignmentSolver(f_tfn[cycle_map[level]], f_tfn[level]))
    a_final_cycle = PAAdjointSolvers(f_as_init_solves, f_init_dep_solves, a_map)
#    dolfin.info("Initialising adjoint final solves")
    a_final_solves = PAAdjointSolvers(tsystem._AssembledTimeSystem__init_solves, f_as_init_solves, a_map)

    self.__tsystem = tsystem
    self.__a_map = a_map
    self.__a_initial_solves = a_initial_solves
    self.__a_initial_cycle1 = a_initial_cycle1
    self.__a_initial_cycle2 = a_initial_cycle2
    self.__a_initial_cycle3 = a_initial_cycle3
    self.__a_cycle = a_cycle
    self.__a_solves = a_solves
    self.__a_final_cycle = a_final_cycle
    self.__a_final_solves = a_final_solves

    self.__s = 0

    self.set_functional(functional)
        
    return

  def a_map(self):
    """
    Return the AdjointVariableMap used by the AdjointTimeSystem.
    """
    
    return self.__a_map

  def initialise(self):
    """
    Solve initial adjoint equations.
    """
    
    if self.__functional is None:
      dolfin.info_red("Warning: Running adjoint model with no functional defined")

    self.__a_map.zero_adjoint()

#    dolfin.info("Performing adjoint initial solves")
    self.__a_initial_solves.solve()
#    dolfin.info("Performing adjoint initial cycle 1")
    self.__a_initial_cycle1.solve()

    self.__s = 0
    
    return

  def timestep_cycle(self):
    """
    Perform the adjoint timestep cycle.
    """
    
    if self.__s == 0:
#      dolfin.info("Performing adjoint initial cycle 2")
      self.__a_initial_cycle2.solve()
    else:
#      dolfin.info("Performing adjoint timestep cycle")
      self.__a_cycle.solve()

    return

  def timestep_solve(self):
    """
    Perform the adjoint timestep solve.
    """
    
#    dolfin.info("Performing adjoint timestep")
    self.__a_solves.solve()
    self.__s += 1

    return

  def timestep(self):
    """
    Perform an adjoint timestep.
    """
    
    self.timestep_cycle()
    self.timesep_solve()

    return

  def final_cycle(self):
    """
    Perform the final adjoint cycle.
    """
    
    if self.__s == 0:
#      dolfin.info("Performing adjoint initial cycle 3")
      self.__a_initial_cycle3.solve()
    else:
#      dolfin.info("Performing adjoint final cycle")
      self.__a_final_cycle.solve()

    return

  def finalise(self):
    """
    Solve final adjoint equations.
    """
    
#    dolfin.info("Performing adjoint final solves")
    self.__a_final_solves.solve()
    
    return

  def set_functional(self, functional):
    """
    Set a functional, defining associated adjoint RHS terms.
    """
    
    if functional is None:
      self.__a_initial_solves.set_functional(None)
      self.__a_initial_cycle1.set_functional(None)
      self.__a_initial_cycle2.set_functional(None)
      self.__a_cycle.set_functional(None)
    elif isinstance(functional, ufl.form.Form):
      if not extract_form_data(functional).rank == 0:
        raise InvalidArgumentException("functional must be rank 0")

      for f_dep in ufl.algorithms.extract_coefficients(functional):
        if isinstance(f_dep, dolfin.Function) and not is_static_coefficient(f_dep):
          if not hasattr(f_dep, "_time_level_data"):
            raise DependencyException("Missing time level data")
          f_ftn, level = f_dep._time_level_data
          if not isinstance(level, FinalTimeLevel):
            raise DependencyException("Functional must depend only upon final time levels or static coefficients")

      self.__a_initial_solves.set_functional(functional)
      self.__a_initial_cycle1.set_functional(functional)
      self.__a_initial_cycle2.set_functional(None)
      self.__a_cycle.set_functional(None)
    elif isinstance(functional, TimeFunctional):
      self.__a_initial_solves.set_functional(None)
      self.__a_initial_cycle1.set_functional(None)
      self.__a_initial_cycle2.set_functional(functional)
      self.__a_cycle.set_functional(functional)
    else:
      raise InvalidArgumentException("functional must be a Form or a TimeFunctional")
      
    self.__functional = functional
    
    return

  def update_functional(self, s):
    """
    Update the adjoint RHS associated with the functional at the end of timestep
    s.
    """
    
    if not isinstance(s, int) or s < 0:
      raise InvalidArgumentException("s must be a non-negative integer")
    
    if isinstance(self.__functional, TimeFunctional):
      if self.__s == 0:
        self.__a_initial_cycle2.update_functional(s)
      else:
        self.__a_cycle.update_functional(s)

    return

  def reassemble(self, *args, **kwargs):
    """
    Reassemble the adjoint solvers. If no arguments are supplied then all
    equations are re-assembled. Otherwise, only the LHSs or RHSs which depend
    upon the supplied Constant s or Function s are reassembled. This
    first clears the relevant cache data if clear_caches is true.

    Valid keyword arguments:
      clear_caches (default true): Whether to clear caches.
    """

    lclear_caches = True
    for key in kwargs:
      if key == "clear_caches":
        lclear_caches = kwargs["clear_caches"]
      else:
        raise InvalidArgumentException("Unexpected keyword argument: %s" % key)
    if lclear_caches:
      clear_caches(*args)
    self.__a_initial_solves.reassemble(*args)
    self.__a_initial_cycle1.reassemble(*args)
    self.__a_initial_cycle2.reassemble(*args)
    self.__a_initial_cycle3.reassemble(*args)
    self.__a_cycle.reassemble(*args)
    self.__a_solves.reassemble(*args)
    self.__a_final_cycle.reassemble(*args)
    self.__a_final_solves.reassemble(*args)

    return
    
class Checkpointer:
  """
  A template for Constant and Function storage.
  """
  
  def __init__(self):
    return

  def __pack(self, c):
    if isinstance(c, dolfin.Constant):
      return float(c)
    else:
      assert(isinstance(c, dolfin.Function))
      return c.vector().array()

  def __unpack(self, c, c_c):
    if isinstance(c, dolfin.Constant):
      c.assign(c_c)
    else:
      assert(isinstance(c, dolfin.Function))
      c.vector().set_local(c_c)
      c.vector().apply("insert")

    return

  def __verify(self, c, c_c, tolerance = 0.0):
    if isinstance(c, dolfin.Constant):
      err = abs(float(c) - c_c)
      if err > tolerance:
        raise CheckpointException("Invalid checkpoint data for Constant with value %.6g, error %.6g" % (float(c), err))
    else:
      assert(isinstance(c, dolfin.Function))
      err = abs(c.vector().array() - c_c).max()
      if err > tolerance:
        raise CheckpointException("Invalid checkpoint data for Function %s, error %.6g" % (c.name(), err))

    return
        
  def __check_cs(self, cs):
    if not isinstance(cs, (list, set)):
      raise InvalidArgumentException("cs must be a list of Constant s or Function s")
    for c in cs:
      if not isinstance(c, (dolfin.Constant, dolfin.Function)):
        raise InvalidArgumentException("cs must be a list of Constant s or Function s")

    if not isinstance(cs, set):
      cs = set(cs)
    return cs
    
  def checkpoint(self, key, cs):
    """
    Store, with the supplied key, the supplied Constant s and Function s.
    """
    
    raise AbstractMethodException("checkpoint method not overridden")

  def restore(self, key, cs = None):
    """
    Restore Constant s and Function s with the given key. If cs is supplied,
    only restore Constant s and Function s found in cs.
    """
    
    raise AbstractMethodException("restore method not overridden")

  def has_key(key):
    """
    Return whether any data is associated with the given key.
    """
    
    raise AbstractMethodException("has_key method not overridden")
    
  def verify(self, key, tolerance = 0.0):
    """
    Verify data associated with the given key, with the specified tolerance.
    """
    
    raise AbstractMethodException("verify method not overridden")

  def remove(self, key):
    """
    Remove data associated with the given key.
    """
    
    raise AbstractMethodException("remove method not overridden")

  def clear(self, keep = []):
    """
    Clear all stored data, except for those with keys in keep.
    """
    
    raise AbstractMethodException("clear method not overridden")
    
class MemoryCheckpointer(Checkpointer):
  """
  Constant and Function storage in memory.
  """
  
  def __init__(self):
    Checkpointer.__init__(self)
    
    self.__cache = {}

    return

  def checkpoint(self, key, cs):
    """
    Store, with the supplied key, the supplied Constant s and Function s.
    """
    
    if key in self.__cache:
      raise CheckpointException("Attempting to overwrite checkpoint with key %s" % str(key))
    cs = self._Checkpointer__check_cs(cs)
  
    c_cs = {}
    for c in cs:
      c_cs[c] = self._Checkpointer__pack(c)

    self.__cache[key] = c_cs

    return

  def restore(self, key, cs = None):
    """
    Restore Constant s and Function s with the given key. If cs is supplied,
    only restore Constant s and Function s found in cs.
    """
    
    if not key in self.__cache:
      raise CheckpointException("Missing checkpoint with key %s" % str(key))
    if not cs is None:
      cs = self._Checkpointer__check_cs(cs)

    c_cs = self.__cache[key]
    if cs is None:
      cs = c_cs.keys()
      
    for c in cs:
      self._Checkpointer__unpack(c, c_cs[c])

    return

  def has_key(self, key):
    """
    Return whether any data is associated with the given key.
    """
    
    return key in self.__cache

  def verify(self, key, tolerance = 0.0):
    """
    Verify data associated with the given key, with the specified tolerance.
    """
    
    if not key in self.__cache:
      raise CheckpointException("Missing checkpoint with key %s" % str(key))
    if not isinstance(tolerance, float) or tolerance < 0.0:
      raise InvalidArgumentException("tolerance must be a non-negative float")
    c_cs = self.__cache[key]

    try:
      for c in c_cs:
        self._Checkpointer__verify(c, c_cs[c], tolerance = tolerance)
      dolfin.info_green("Verified checkpoint with key %s" % str(key))
    except CheckpointException as e:
      dolfin.info_red(str(e))
      raise CheckpointException("Failed to verify checkpoint with key %s" % str(key))

    return

  def remove(self, key):
    """
    Remove data associated with the given key.
    """
    
    if not key in self.__cache:
      raise CheckpointException("Missing checkpoint with key %s" % str(key))

    del(self.__cache[key])

    return

  def clear(self, keep = []):
    """
    Clear all stored data, except for those with keys in keep.
    """
    
    if not isinstance(keep, list):
      raise InvalidArgumentException("keep must be a list")
    
    if len(keep) == 0:
      self.__cache = {}
    else:
      for key in copy.copy(self.__cache.keys()):
        if not key in keep:
          del(self.__cache[key])

    return

class DiskCheckpointer(Checkpointer):
  """
  Constant and Function storage on disk. All keys handled by a DiskCheckpointer
  are internally cast to strings.

  Constructor arguments:
    dirname: The directory in which data is to be stored.
  """
  
  def __init__(self, dirname = "checkpoints~"):
    if not isinstance(dirname, str):
      raise InvalidArgumentException("dirname must be a string")
    
    Checkpointer.__init__(self)

    if dolfin.MPI.process_number() == 0:    
      if not os.path.exists(dirname):
        os.mkdir(dirname)
    dolfin.MPI.barrier()
    
    self.__dirname = dirname
    self.__filenames = {}
    self.__id_map = {}

    return

  def __filename(self, key):
    return os.path.join(self.__dirname, "checkpoint_%s_%i" % (str(key), dolfin.MPI.process_number()))

  def checkpoint(self, key, cs):
    """
    Store, with the supplied key, the supplied Constant s and Function s. The
    key is internally cast to a string.
    """
    
    key = str(key)
    if key in self.__filenames:
      raise CheckpointException("Attempting to overwrite checkpoint with key %s" % key)
    cs = self._Checkpointer__check_cs(cs)

    c_cs = {}
    id_map = {}
    for c in cs:
      c_id = c.id()
      c_cs[c_id] = self._Checkpointer__pack(c)
      id_map[c_id] = c

    filename = self.__filename(key)
    handle = open(filename, "wb")
    pickler = pickle.Pickler(handle, -1)
    pickler.dump(c_cs)

    self.__filenames[key] = filename
    self.__id_map[key] = id_map

    return

  def restore(self, key, cs = None):
    """
    Restore Constant s and Function s with the given key. If cs is supplied,
    only restore Constant s and Function s found in cs. The key is internally
    cast to a string.
    """
    
    key = str(key)
    if not key in self.__filenames:
      raise CheckpointException("Missing checkpoint with key %s" % key)
    if not cs is None:
      cs = self._Checkpointer__check_cs(cs)
      cs = [c.id() for c in cs]

    handle = open(self.__filename(key), "rb")
    pickler = pickle.Unpickler(handle)
    c_cs = pickler.load()
    if cs is None:
      cs = c_cs.keys()

    id_map = self.__id_map[key]
    for c_id in cs:
      c = id_map[c_id]
      self._Checkpointer__unpack(c, c_cs[c_id])

    return

  def has_key(self, key):
    """
    Return whether any data is associated with the given key. The key is
    internally cast to a string.
    """
    
    key = str(key)
    return key in self.__filenames

  def verify(self, key, tolerance = 0.0):
    """
    Verify data associated with the given key, with the specified tolerance. The
    key is internally cast to a string.
    """
    
    key = str(key)
    if not key in self.__filenames:
      raise CheckpointException("Missing checkpoint with key %s" % key)
    if not isinstance(tolerance, float) or tolerance < 0.0:
      raise InvalidArgumentException("tolerance must be a non-negative float")
    handle = open(self.__filename(key), "rb")
    pickler = pickle.Unpickler(handle)
    c_cs = pickler.load()

    try:
      id_map = self.__id_map[key]
      for c_id in c_cs:
        c = id_map[c_id]
        self._Checkpointer__verify(c, c_cs[c_id], tolerance = tolerance)
      dolfin.info_green("Verified checkpoint with key %s" % key)
    except CheckpointException as e:
      dolfin.info_red(str(e))
      raise CheckpointException("Failed to verify checkpoint with key %s" % key)

    return

  def remove(self, key):
    """
    Remove data associated with the given key. The key is internally cast to a
    string.
    """
    
    key = str(key)
    if not key in self.__filenames:
      raise CheckpointException("Missing checkpoint with key %s" % key)

#    os.remove(self.__filenames[key])
    del(self.__filenames[key])
    del(self.__id_map[key])

    return

  def clear(self, keep = []):
    """
    Clear all stored data, except for those with keys in keep. The keys are
    internally cast to strings.
    """
    
    if not isinstance(keep, list):
      raise InvalidArgumentException("keep must be a list")

    if len(keep) == 0:
#      for key in self.__filenames:
#        os.remove(self.__filenames[key])
      self.__filenames = {}
      self.__id_map = {}
    else:
      keep = [str(key) for key in keep]
      for key in copy.copy(self.__filenames.keys()):
        if not key in keep:
 #         os.remove(self.__filenames[key])
          del(self.__filenames[key])
          del(self.__id_map[key])

    return

class AdjoinedTimeSystem:
  """
  Used to solve forward and adjoint timestep equations. This performs necessary
  storage and recovery management.

  Constructor arguments:
    tsystem: A TimeSystem defining the timestep equations.
    functional: A rank 0 form or a TimeFunctional defining the functional.
    disk_period: Data is written to disk every disk_period timesteps. In order
      to compute an adjoint solution data is restored from disk and the forward
      solution recomputed between the storage points. If disk_period is equal to
      None then the entire forward solution is stored in memory.
    initialise: Whether the initialise method is to be called.
    reassemble: Whether the reassemble methods of solvers defined by the
      TimeSystem should be called.
  """
  
  def __init__(self, tsystem, functional = None, disk_period = None, initialise = True, reassemble = False):
    if not isinstance(tsystem, TimeSystem):
      raise InvalidArgumentException("tsystem must be a TimeSystem")
    if not functional is None and not isinstance(functional, (ufl.form.Form, TimeFunctional)):
      raise InvalidArgumentException("functional must be a Form or a TimeFunctional")
    if not disk_period is None:
      if not isinstance(disk_period, int) or disk_period <= 0:
        raise InvalidArgumentException("disk_period must be a positive integer")
      
    forward = assemble(tsystem, adjoint = False, initialise = False, reassemble = reassemble)
    adjoint = AdjointTimeSystem(forward)

    init_cp_cs1 = set()
    init_cp_cs2 = set()
    for solve in forward._AssembledTimeSystem__init_solves:
      init_cp_cs1.add(solve.x())
      for dep in solve.dependencies(non_symbolic = True):
        if isinstance(dep, dolfin.Function) and hasattr(dep, "_time_level_data"):
          init_cp_cs1.add(dep)
    cp_cs = set()
    cp_f_tfn = {}
    for solve in forward._AssembledTimeSystem__solves:
      cp_cs.add(solve.x())
      for dep in solve.dependencies(non_symbolic = True):
        if isinstance(dep, dolfin.Function) and hasattr(dep, "_time_level_data"):
          cp_cs.add(dep)
          f_tfn, level = dep._time_level_data
          if f_tfn in cp_f_tfn:
            cp_f_tfn[f_tfn] = min(cp_f_tfn[f_tfn], level)
          else:
            cp_f_tfn[f_tfn] = level
    nl_cp_cs = set()
    update_cs = set()
    for solve in forward._AssembledTimeSystem__solves:
      for c in solve.nonlinear_dependencies():
        if isinstance(c, dolfin.Function) and hasattr(c, "_time_level_data"):
          nl_cp_cs.add(c)
        elif isinstance(c, (dolfin.Constant, dolfin.Function)) and not is_static_coefficient(c):
          update_cs.add(c)
    final_cp_cs = set()
        
    for f_tfn in forward.tfns():
      for level in f_tfn.initial_levels():
        init_cp_cs1.add(f_tfn[level])
        init_cp_cs2.add(f_tfn[level])
      for level in f_tfn.cycle_map().values():
        if f_tfn in cp_f_tfn and level >= cp_f_tfn[f_tfn]:
          cp_cs.add(f_tfn[level])
      for level in f_tfn.final_levels():
        final_cp_cs.add(f_tfn[level])
        
    self.__forward = forward
    self.__adjoint = adjoint
    self.__a_map = adjoint.a_map()
    self.__memory_checkpointer = MemoryCheckpointer()
    if disk_period is None:
      self.__disk_checkpointer = None
    else:
      self.__disk_checkpointer = DiskCheckpointer()
    self.__disk_period = disk_period
    self.__disk_m = -1
    self.__init_cp_cs1 = init_cp_cs1
    self.__init_cp_cs2 = init_cp_cs2
    self.__cp_cs = cp_cs
    self.__nl_cp_cs = nl_cp_cs
    self.__update_cs = update_cs
    self.__final_cp_cs = final_cp_cs
    self.__s = None
    self.__S = None

    self.set_functional(functional)

    if initialise:
      self.initialise()

    return

  def __timestep_checkpoint(self, s, cp_cs):
    if self.__disk_period is None:
      self.__memory_checkpointer.checkpoint(s, cp_cs)
    elif s < 0:
      self.__disk_checkpointer.checkpoint(s, cp_cs)
    else:
      m = s // self.__disk_period
      i = s % self.__disk_period
      if not m == self.__disk_m:
        self.__disk_checkpointer.checkpoint(m, cp_cs)
        self.__disk_m = m
    
    return

  def __restore_timestep_checkpoint(self, s, cp_cs = None, update_cs = None):
    if update_cs is None:
      def lupdate_cs(s):
        return None
    elif isinstance(update_cs, (list, set)):
      def lupdate_cs(s):
        return update_cs
    else:
      assert(callable(update_cs))
      lupdate_cs = update_cs
      
    if cp_cs is None:
      def lcp_cs(s):
        return self.__cp_cs
    elif isinstance(cp_cs, (list, set)):
      def lcp_cs(s):
        return cp_cs
    else:
      assert(callable(cp_cs))
      lcp_cs = cp_cs
          
    if self.__disk_period is None:
      if s >= 0:
        self.__forward.timestep_update(s = s, cs = lupdate_cs(s))
        self.__memory_checkpointer.restore(s, cs = lcp_cs(s))
      else:
        self.__memory_checkpointer.restore(s)
    elif s < 0:
      self.__disk_checkpointer.restore(s)
    else:
      m = s // self.__disk_period
      i = s % self.__disk_period

      if self.__memory_checkpointer.has_key((m, i)):
        self.__forward.timestep_update(s = s, cs = lupdate_cs(s))
        self.__memory_checkpointer.restore((m, i))
      else:
        self.__clear_rerun_checkpoints()
        self.__disk_checkpointer.restore(m)
        self.__disk_m = m
        
        self.__memory_checkpointer.checkpoint((m, 0), lcp_cs(m * self.__disk_period))
        if m == self.__S // self.__disk_period:
          for j in range(1, (self.__S % self.__disk_period) + 1):
            ls = m * self.__disk_period + j
            if m > 0 or j > 1:
              self.__forward.timestep_cycle()
            self.__forward.timestep_update(s = ls)
            self.__forward.timestep_solve()
            self.__memory_checkpointer.checkpoint((m, j), lcp_cs(ls))

          if not i == self.__S % self.__disk_period:
            self.__forward.timestep_update(s = s, cs = lupdate_cs(s))
            self.__memory_checkpointer.restore((m, i))
        else:
          for j in range(1, self.__disk_period):
            ls = m * self.__disk_period + j
            if m > 0 or j > 1:
              self.__forward.timestep_cycle()
            self.__forward.timestep_update(s = ls)
            self.__forward.timestep_solve()
            self.__memory_checkpointer.checkpoint((m, j), lcp_cs(ls))
          
          if not i == self.__disk_period - 1:
            self.__forward.timestep_update(s = s, cs = lupdate_cs(s))
            self.__memory_checkpointer.restore((m, i))

    return

  def __clear_rerun_checkpoints(self):
    if not self.__disk_period is None:
      self.__memory_checkpointer.clear()
      self.__disk_m = -1

    return

  def __clear_timestep_checkpoints(self, keep = []):
    if self.__disk_period is None:
      self.__memory_checkpointer.clear(keep = keep)
    else:
      self.__disk_checkpointer.clear(keep = keep)
      self.__memory_checkpointer.clear()

    return

  def __verify_timestep_checkpoint(self, s, tolerance = 0.0):
    if self.__disk_period is None:
      self.__memory_checkpointer.verify(s, tolerance = tolerance)
    elif s < 0:
      self.__disk_checkpointer.verify(s, tolerance = tolerance)
    else:
      m = s // self.__disk_period
      i = s % self.__disk_period
      if i == 0:
        self.__disk_checkpointer.verify(m, tolerance = tolerance)
      if self.__memory_checkpointer.has_key((m, i)):
        self.__memory_checkpointer.verify((m, i), tolerance = tolerance)
      
    return

  def a_map(self):
    """
    Return the AdjointVariableMap associated with the AdjoinedTimeSystem.
    """
    
    return self.__a_map

  def initialise(self):
    """
    Solve initial equations.
    """
    
    if not self.__S is None:
      raise StateException("Initialisation after finalisation")
    elif not self.__s is None:
      raise StateException("Multiple initialisations")

    self.__timestep_checkpoint(-1, self.__init_cp_cs1)
    self.__s = 0
    self.__forward.initialise()
    self.__timestep_checkpoint(-2, self.__init_cp_cs2)

#    self.__timestep_checkpoint(0, self.__cp_cs)

    if isinstance(self.__functional, TimeFunctional):
      self.__functional.initialise()
      self.__functional.addto(0)

    return
    
  def timestep(self, ns = 1):
    """
    Perform ns timesteps.
    """
    
    if self.__s is None:
      raise StateException("Timestep before initialisation")
    elif not self.__S is None:
      raise StateException("Timestep after finalisation")
    
    for i in range(ns):
      self.__s += 1
      self.__forward.timestep_update()
      self.__forward.timestep_solve()
      self.__timestep_checkpoint(self.__s, self.__cp_cs)
      self.__forward.timestep_cycle()
      if isinstance(self.__functional, TimeFunctional):
        self.__functional.addto(self.__s)

    return
    
  def finalise(self):
    """
    Solve final equations.
    """
    
    if self.__s is None:
      raise StateException("Finalisation before initialisation")
    elif not self.__S is None:
      raise StateException("Multiple finalisations")

    self.__timestep_checkpoint(-3, self.__cp_cs)
    self.__forward.finalise()
    self.__S = self.__s
    self.__timestep_checkpoint(-4, self.__final_cp_cs)

    return

  def verify_checkpoints(self, tolerance = 0.0):
    """
    Verify the reproducibility of stored forward model data to within the
    specified tolerance.
    """
    
    if self.__S is None:
      raise StateException("Attempting to rerun model before finalisation")

    self.__restore_timestep_checkpoint(-1)
    self.__forward.initialise()
    self.__verify_timestep_checkpoint(-2, tolerance = tolerance)

#    self.__verify_timestep_checkpoint(0, tolerance = tolerance)
    for i in range(self.__S):
      self.__forward.timestep_update()
      self.__forward.timestep_solve()
      self.__verify_timestep_checkpoint(i + 1, tolerance = tolerance)
      self.__forward.timestep_cycle()

    self.__verify_timestep_checkpoint(-3, tolerance = tolerance)
    self.__forward.finalise()
    self.__verify_timestep_checkpoint(-4, tolerance = tolerance)

    return

  def set_functional(self, functional):
    """
    Set a functional, defining associated adjoint RHS terms.
    """
    
    self.__adjoint.set_functional(functional)
    self.__functional = functional
    
    return

  def compute_functional(self, rerun_forward = False, recheckpoint = False):
    """
    Return the functional value, optionally re-running the forward model and
    optionally regenerating the forward model storage data.
    """
    
    if self.__functional is None:
      raise StateException("No functional defined")

    if rerun_forward:
      self.rerun_forward(recheckpoint = recheckpoint)

    if isinstance(self.__functional, ufl.form.Form):
      return assemble(self.__functional)
    else:
      return self.__functional.value()

  def rerun_forward(self, recheckpoint = False):
    """
    Re-run the forward model, optionally regenerating the forward model storage
    data.
    """
    
    if self.__S is None:
      raise StateException("Attempting to rerun model before finalisation")

    self.__restore_timestep_checkpoint(-1)
    if recheckpoint:
      self.__clear_timestep_checkpoints(keep = [-1])

    self.__forward.initialise()
    if recheckpoint:
      self.__timestep_checkpoint(-2, self.__init_cp_cs2)
#      self.__timestep_checkpoint(0, self.__cp_cs)
    if isinstance(self.__functional, TimeFunctional):
      self.__functional.initialise()
      self.__functional.addto(0)
      
    for i in range(self.__S):
      self.__forward.timestep_update()
      self.__forward.timestep_solve()
      if recheckpoint:
        self.__timestep_checkpoint(i + 1, self.__cp_cs)
      self.__forward.timestep_cycle()
      if isinstance(self.__functional, TimeFunctional):
        self.__functional.addto(i + 1)
        
    if recheckpoint:
      self.__timestep_checkpoint(-3, self.__cp_cs)
    self.__forward.finalise()
    if recheckpoint:
      self.__timestep_checkpoint(-4, self.__final_cp_cs)

    return

  def reassemble_forward(self, *args, **kwargs):
    """
    Reassemble the forward model. This first clears the relevant cache data if
    clear_caches is true.

    Valid keyword arguments:
      clear_caches (default true): Whether to clear caches.
    """
    
    self.__forward.reassemble(*args, **kwargs)

    return

  def reassemble_adjoint(self, *args, **kwargs):
    """
    Reassemble the adjoint model. This first clears the relevant cache data if
    clear_caches is true.

    Valid keyword arguments:
      clear_caches (default true): Whether to clear caches.
    """
    
    self.__adjoint.reassemble(*args, **kwargs)

    return

  def reassemble(self, *args, **kwargs):
    """
    Reassemble the forward and adjoint models. This first clears the relevant
    cache data if clear_caches is true.

    Valid keyword arguments:
      clear_caches (default true): Whether to clear caches.
    """
    
    lclear_caches = True
    for key in kwargs:
      if key == "clear_caches":
        lclear_caches = kwargs["clear_caches"]
      else:
        raise InvalidArgumentException("Unexpected keyword argument: %s" % key)
    if lclear_caches:
      clear_caches(*args)
    self.reassemble_forward(clear_caches = False, *args)
    self.reassemble_adjoint(clear_caches = False, *args)

    return

  def compute_gradient(self, parameters, callback = None, project = False, project_solver_parameters = {}):
    """
    Compute the model constrained derivative of the registered functional with
    respect to the supplied Constant or Function, or list of Constant s or
    Function s. The optional callback function has the form:
      def callback(s):
        ...
        return
    and is called at the end of forward timestep s.

    The return value has the following type:
      1. parameters is a Constant: The derivative is returned as a Constant.
      2. parameters is a Function, and
         a. project is False: The derivative is returned as a GenericVector.
         b. project is True: The derivative is returned as a (GenericVector,
            Function) pair, where the Function contains the derivative projected
            onto the trial space of parameters. project_solver_parameters
            defines solver options for the associated mass matrix inversion.
      3. parameters is a list of Constant s or Function s: The derivative is
         returned as a list. If parameters[i] is a Constant, then element i
         of the returned list has a type as given by 1. above. If parameters[i]
         is a Function then element i of the returned list has a type given by
         2. above.
    """
    
    if isinstance(parameters, list):
      for parameter in parameters:
        if not isinstance(parameter, (dolfin.Constant, dolfin.Function)):
          raise InvalidArgumentException("parameters must be a Constant or Function, or a list of Constant s or Function s")
    elif isinstance(parameters, (dolfin.Constant, dolfin.Function)):
      return self.compute_gradient([parameters], callback = callback, project = project, project_solver_parameters = project_solver_parameters)[0]
    else:
      raise InvalidArgumentException("parameters must be a Constant or Function, or a list of Constant s or Function s")
    if self.__S is None:
      raise StateException("Attempting to compute gradient before finalisation")
    if not callback is None and not callable(callback):
      raise InvalidArgumentException("callback must be a callable")
    if not project_solver_parameters is None and not isinstance(project_solver_parameters, dict):
      raise InvalidArgumentException("project_solver_parameters must be a dict")

    for parameter in parameters:
      if not is_static_coefficient(parameter):
        raise InvalidArgumentException("Differentiating with respect to non-static parameter")

    f_init_solves = self.__forward._AssembledTimeSystem__init_solves
    f_solves = self.__forward._AssembledTimeSystem__solves
    f_final_solves = self.__forward._AssembledTimeSystem__final_solves

    dFdm = {j:[OrderedDict() for i in range(len(parameters))] for j in range(-1, 2)}
    for i, parameter in enumerate(parameters):
      for j, solves in zip(range(-1, 2), [f_init_solves, f_solves, f_final_solves]):
        for f_solve in solves:
          f_x = f_solve.x()
          if isinstance(f_solve, AssignmentSolver):
            f_rhs = f_solve.rhs()
            if isinstance(f_rhs, ufl.expr.Expr):
              f_der = differentiate_expr(f_rhs, parameter)
              if not isinstance(f_der, ufl.constantvalue.Zero):
                dFdm[j][i][f_x] = f_der
            else:
              f_der = []
              for alpha, f_dep in f_rhs:
                if parameter is alpha:
                  f_der.append(f_dep)
                elif parameter is f_dep:
                  f_der.append(alpha)
              if len(f_der) > 0:
                dFdm[j][i][f_x] = f_der
          else:
            assert(isinstance(f_solve, EquationSolver))
            f_der = f_solve.derivative(parameter)
            if not is_empty_form(f_der):
              rank = extract_form_data(f_der).rank
              if rank == 1:
                f_der = PALinearForm(f_der)
              else:
                assert(rank == 2)
                f_der = PABilinearForm(adjoint(f_der, \
                  adjoint_arguments = (dolfin.TestFunction(parameter.function_space()), dolfin.TrialFunction(f_x.function_space()))))
              dFdm[j][i][f_x] = f_der

    cp_cs = copy.copy(self.__nl_cp_cs)
    update_cs = copy.copy(self.__update_cs)
    for i in range(len(parameters)):
      for f_der in dFdm[0][i].values():
        if isinstance(f_der, PAForm):
          for c in f_der.dependencies(non_symbolic = True):
            if isinstance(c, dolfin.Function) and hasattr(c, "_time_level_data"):
              cp_cs.add(c)
            elif not is_static_coefficient(c):
              update_cs.add(c)
        elif isinstance(f_der, ufl.expr.Expr):
          for c in ufl.algorithms.extract_coefficients(f_der):
            if isinstance(c, dolfin.Function) and hasattr(c, "_time_level_data"):
              cp_cs.add(c)
            elif not is_static_coefficient(c):
              update_cs.add(c)
        else:
          for c in f_der:
            if isinstance(c, dolfin.Function) and hasattr(c, "_time_level_data"):
              cp_cs.add(c)
            elif not is_static_coefficient(c):
              update_cs.add(c)
    if isinstance(self.__functional, TimeFunctional):
      s_cp_cs = cp_cs
      s_update_cs = update_cs
      def cp_cs(s):
        cp_cs = copy.copy(s_cp_cs)
        for c in self.__functional.dependencies(s, non_symbolic = True):
          if isinstance(c, dolfin.Function) and hasattr(c, "_time_level_data"):
            if not isinstance(c._time_level_data[1], TimeLevel):
              raise DependencyException("Unexpected initial or final time level functional dependency")
            cp_cs.add(c)
        return cp_cs
      def update_cs(s):
        update_cs = copy.copy(s_update_cs)
        for c in self.__functional.dependencies(s, non_symbolic = True):
          if isinstance(c, dolfin.Function) and hasattr(c, "_time_level_data"):
            pass
          elif is_static_coefficient(c):
            update_cs.add(c)
        return update_cs

    grad = [None for i in range(len(parameters))]
    for i, parameter in enumerate(parameters):
      if self.__functional is None or isinstance(self.__functional, TimeFunctional):
        der = ufl.form.Form([])
      else:
        der = derivative(self.__functional, parameter)
      if is_empty_form(der):
        if isinstance(parameter, dolfin.Constant):
          grad[i] = 0.0
        else:
          assert(isinstance(parameter, dolfin.Function))
          grad[i] = parameter.vector().copy()
          grad[i].zero()
      else:
        grad[i] = assemble(der)

    def addto_grad(j, a_solves, a_map):
      for i, parameter in enumerate(parameters):
        for a_x in a_solves.a_x():
          f_x = self.__a_map[a_x]
          if not f_x in dFdm[j][i]:
            continue
          f_der = dFdm[j][i][f_x]
          if isinstance(f_der, PAForm):
            f_der_a = assemble(f_der, copy = False)
            if isinstance(f_der_a, dolfin.GenericMatrix):
              grad[i] -= f_der_a * a_x.vector()
            else:
              assert(isinstance(f_der_a, dolfin.GenericVector))
              grad[i] -= f_der_a.inner(a_x.vector())
          elif isinstance(f_der, ufl.expr.Expr):
            f_der = evaluate_expr(f_der, copy = False)
            if isinstance(f_der, float):
              if isinstance(grad[i], float):
                grad[i] += f_der * a_x.vector().sum()
              else:
                grad[i].axpy(f_der, a_x.vector())
            else:
              assert(isinstance(f_der, dolfin.GenericVector))
              grad[i] += f_der.inner(a_x.vector())
          else:
            for term in f_der:
              if isinstance(term, (ufl.constantvalue.FloatValue, dolfin.Constant)):
                if isinstance(grad[i], float):
                  grad[i] += float(term) * a_x.vector().sum()
                else:
                  grad[i].axpy(float(term), a_x.vector())
              else:
                assert(isinstance(term, dolfin.Function))
                grad[i] += term.vector().inner(a_x.vector())
      return

    self.__restore_timestep_checkpoint(-1)
    
    self.__restore_timestep_checkpoint(-4)
    self.__forward.timestep_update(s = self.__S + 1)
    self.__adjoint.initialise()
    addto_grad(1, self.__adjoint._AdjointTimeSystem__a_initial_solves, self.__a_map)
    if not callback is None:
      callback(s = self.__S + 1)

    self.__restore_timestep_checkpoint(-3)
    for i in range(self.__S):
      self.__adjoint.update_functional(self.__S - i)
      self.__adjoint.timestep_cycle()
      self.__restore_timestep_checkpoint(self.__S - i, cp_cs = cp_cs, update_cs = update_cs)
      self.__adjoint.timestep_solve()
      addto_grad(0, self.__adjoint._AdjointTimeSystem__a_solves, self.__a_map)
      if not callback is None:
        callback(s = self.__S - i)

    self.__adjoint.final_cycle()
    self.__restore_timestep_checkpoint(-2)
    self.__adjoint.finalise()
    addto_grad(-1, self.__adjoint._AdjointTimeSystem__a_final_solves, self.__a_map)
    if not callback is None:
      callback(s = 0)

    self.__clear_rerun_checkpoints()
    self.__restore_timestep_checkpoint(-4)

    ngrad = []
    for i, parameter in enumerate(parameters):
      if isinstance(grad[i], float):
        ngrad.append(Constant(grad[i]))
      else:
        assert(isinstance(grad[i], dolfin.GenericVector))
        if project:
          space = parameter.function_space()
          ngrad.append((grad[i], Function(space, name = "gradient_%s" % parameter.name())))
          mass = dolfin.inner(dolfin.TestFunction(space), dolfin.TrialFunction(space)) * dolfin.dx
          solver = solver_cache.solver(mass, solver_parameters = project_solver_parameters, static = True)
          solver.solve(assembly_cache.assemble(mass), ngrad[-1][1].vector(), ngrad[-1][0])
        else:
          ngrad.append(grad[i])
    return ngrad
  
  def taylor_test(self, parameter, ntest = 6, fact = 1.0e-4, J = None, grad = None):
    """
    Perform a Taylor remainder convergence test of a model constrained
    derivative calculation.

    Arguments:
      parameter: A static Constant or Function.
      ntest: The number of forward model runs to perform. This must be >= 2.
      fact: The relative size of the perturbation to be applied to the model
        parameter. If the parameter is a Constant then rfact is the magnitude of
        the smallest applied permutation. If the parameter is a Function then
        the smallest perturbation is a Function with a vector containing random
        values in the range [-1/2 rfact, +1/2 rfact) with a uniform probability
        distribution, where rfact = fact * || v ||_inf if || v ||_inf > 0, and
        rfact = fact otherwise, and where v is the vector associated with the
        parameter. In each forward model run the perturbation is multiplied by
        2 ** i, where i = ntest - 1, ntest - 2, ..., 1, 0.
      J: The value of the functional. Note that, if the functional is a
        TimeFunctional, the TimeFunctional is initialised with this value at the
        end of the test.
      grad: The derivative to verify. If not supplied, a derivative is computed
        using the adjoint model.

    Returns the relative orders of convergence. A successful verification will
    return orders close to 2.
    """
    
    if not isinstance(parameter, (dolfin.Constant, dolfin.Function)):
      raise InvalidArgumentException("parameter must be a Constant or Function")
    if not isinstance(ntest, int) or ntest <= 1:
      raise InvalidArgumentException("ntest must be an integer greater than or equal to 2")
    if not isinstance(fact, float) or fact <= 0.0:
      raise InvalidArgumentException("fact must be a positive float")
    if self.__S is None:
      raise StateException("Attempting to run Taylor remainder convergence test before finalisation")

    dolfin.info("Running Taylor remainder convergence test")

    if not grad is None:
      if isinstance(parameter, dolfin.Constant):
        if not isinstance(grad, dolfin.Constant):
          raise InvalidArgumentException("Invalid grad type")
      else:
        assert(isinstance(parameter, dolfin.Function))
        if isinstance(grad, tuple):
          if not len(grad) == 2 or not isinstance(grad[0], dolfin.GenericVector) or not isinstance(grad[1], dolfin.Function):
            raise InvalidArgumentException("Invalid grad type")
          elif not grad[0].local_size() == parameter.vector().local_size():
            raise InvalidArgumentException("Invalid grad size")
        elif not isinstance(grad, dolfin.GenericVector):
          raise InvalidArgumentException("Invalid grad type")
        elif not grad.local_size() == parameter.vector().local_size():
          raise InvalidArgumentException("Invalid grad size")
    else:
      grad = self.compute_gradient(parameter)
    if isinstance(grad, dolfin.Constant):
      grad = float(grad)
    elif isinstance(grad, tuple):
      grad = grad[0]
    if not J is None:
      if not isinstance(J, float):
        raise InvalidArgumentException("J must be a float")
    else:
      J = self.compute_functional()

    if isinstance(parameter, dolfin.Constant):
      parameter_orig = float(parameter)
      perturb = fact * abs(parameter_orig)
      if perturb == 0.0:
        perturb = fact
    else:
      assert(isinstance(parameter, dolfin.Function))
      parameter_orig = Function(parameter, name = "%s_original" % parameter.name())
      parameter_orig_arr = parameter_orig.vector().array()
      perturb = parameter_orig.vector().copy()
      shape = parameter_orig_arr.shape
      rfact = fact * abs(parameter_orig_arr).max()
      if rfact == 0.0:
        rfact = fact
      numpy.random.seed(0)
      perturb.set_local(rfact * numpy.random.random(shape) - (0.5 * rfact))
      numpy.random.seed()
      perturb.apply("insert")
      
    errs_1 = []
    errs_2 = []
    for i in range(ntest - 1, -1, -1):
      if isinstance(parameter, dolfin.Constant):
        parameter.assign(Constant(parameter_orig + (2 ** i) * perturb))

        self.reassemble_forward(parameter)
        Jp = self.compute_functional(rerun_forward = True)
        
        errs_1.append(abs(Jp - J))
        errs_2.append(abs(Jp - J - (2 ** i) * perturb * grad))
      else:
        assert(isinstance(parameter, dolfin.Function))
        parameter.vector()[:] = parameter_orig.vector() + (2 ** i) * perturb

        self.reassemble_forward(parameter)
        Jp = self.compute_functional(rerun_forward = True)

        errs_1.append(abs(Jp - J))
        errs_2.append(abs(Jp - J - (2 ** i) * perturb.inner(grad)))

    parameter.assign(parameter_orig)
    self.reassemble_forward(parameter)
    self.__restore_timestep_checkpoint(-4)
    if isinstance(self.__functional, TimeFunctional):
      self.__functional.initialise(val = J)

    dolfin.info("Taylor remainder test, no adjoint errors  : %s" % str(errs_1))
    dolfin.info("Taylor remainder test, with adjoint errors: %s" % str(errs_2))
    
    for i, err in enumerate(errs_1):
      if err == 0.0:
        errs_1[i] = numpy.NAN
    for i, err in enumerate(errs_2):
      if err == 0.0:
        errs_2[i] = numpy.NAN
    orders_1 = numpy.empty(len(errs_1) - 1)
    for i in range(1, len(errs_1)):
      orders_1[i - 1] = -numpy.log(errs_1[i] / errs_1[i - 1]) / numpy.log(2.0)
    orders_2 = numpy.empty(len(errs_2) - 1)
    for i in range(1, len(errs_2)):
      orders_2[i - 1] = -numpy.log(errs_2[i] / errs_2[i - 1]) / numpy.log(2.0)

    if any(orders_1 < 0.9):
      dolfin.info_red("Taylor remainder test, no adjoint orders  : %s" % str(orders_1))
    else:
      dolfin.info_green("Taylor remainder test, no adjoint orders  : %s" % str(orders_1))
    if any(orders_2 < 1.9):
      dolfin.info_red("Taylor remainder test, with adjoint orders: %s" % str(orders_2))
    else:
      dolfin.info_green("Taylor remainder test, with adjoint orders: %s" % str(orders_2))

    return orders_2

  def minimise_functional(self, parameters, tolerance, parameters_0 = None, bounds = None, method = None, options = None):
    """
    Minimise the registered functional subject to variations in the supplied
    static Constant s or Function s, to within the specified tolerance. This is
    a wrapper for SciPy optimisation functions. Currently only works with a
    single MPI process.

    Arguments:
      parameters: The Constant s or Function s with respect to which the
        functional is minimised.
      tolerance: Minimisation tolerance.
      parameters_0: A list of initial guesses for the parameters. For Constant
        parameters the associated elements of parameters_0 may be None, a float,
        or a Constant. For Function parameters the associated elements of
        parameters_0 may be None, a float, or a Function. A None element
        indicates that the input value of the associated parameter is to be used
        as an initial guess. 
      bounds: A list of (min, max) bounds for the parameters. The bounds have
        the same types as the elements of parameters_0. A None bound indicates
        no lower or upper bound.
      method: Optimisation method.
      options: A dictionary of additional keyword arguments to pass to the SciPy
        optimisation function.
    """
    
    if isinstance(parameters, list):
      for parameter in parameters:
        if not isinstance(parameter, (dolfin.Constant, dolfin.Function)):
          raise InvalidArgumentException("parameters must be a Constant or Function, or a list of Constant s or Function s")
    elif isinstance(parameters, (dolfin.Constant, dolfin.Function)):
      kwargs = {"method":method, "options":options}
      if bounds is None:
        kwargs["bounds"] = None
      else:
        kwargs["bounds"] = [bounds]
      if parameters_0 is None:
        kwargs["parameters_0"] = None
      else:
        kwargs["parameters_0"] = [parameters_0]
      return self.minimise_functional([parameters], tolerance, **kwargs)
    else:
      raise InvalidArgumentException("parameters must be a Constant or Function, or a list of Constant s or Function s")

    if not parameters_0 is None:
      if not isinstance(parameters_0, list):
        raise InvalidArgumentException("parameters_0 must be a list of Nones, floats, Constant s or Function s")
      elif not len(parameters_0) == len(parameters):
        raise InvalidArgumentException("Invalid parameters_0 size")
      for i, parameter_0 in enumerate(parameters_0):
        if parameter_0 is None:
          pass
        elif isinstance(parameters[i], dolfin.Constant):
          if not isinstance(parameter_0, (float, dolfin.Constant)):
            raise InvalidArgumentException("Invalid parameters_0 element type")
        else:
          assert(isinstance(parameters[i], dolfin.Function))
          if isinstance(parameter_0, float):
            pass
          elif isinstance(parameter_0, dolfin.Function):
            if not parameter_0.vector().local_size() == parameters[i].vector().local_size():
              raise InvalidArgumentException("Invalid parameters_0 element size")
          else:
            raise InvalidArgumentException("Invalid parameters_0 element type")

    if not bounds is None:
      if not isinstance(bounds, list):
        raise InvalidArgumentException("bounds must be a list of tuples of lower and upper bounds")
      elif not len(bounds) == len(parameters):
        raise InvalidArgumentException("Invalid bounds size")
      for i, bound in enumerate(bounds):
        if not isinstance(bound, tuple) or not len(bound) == 2:
          raise InvalidArgumentException("bounds must be a list of tuples of lower and upper bounds")
        for j in range(2):
          if bound[j] is None:
            pass
          elif isinstance(parameters[i], dolfin.Constant):
            if not isinstance(bound[j], (float, dolfin.Constant)):
              raise InvalidArgumentException("Invalid bound type")
          else:
            assert(isinstance(parameters[i], dolfin.Function))
            if isinstance(bound[j], float):
              pass
            elif isinstance(bound[j], dolfin.Function):
              if not bound[j].vector().local_size() == parameters[i].vector().local_size():
                raise InvalidArgumentException("Invalid bound size")
            else:
              raise InvalidArgumentException("Invalid bound type")

    if not isinstance(tolerance, float) or tolerance < 0.0:
      raise InvalidArgumentException("tolerance must be a non-negative float")
    if method is None:
      if bounds is None:
        method = "BFGS"
      else:
        method = "L-BFGS-B"
    elif not isinstance(method, str):
      raise InvalidArgumentException("method must be a string")
    if not options is None and not isinstance(options, dict):
      raise InvalidArgumentException("options must be a dictionary")
    if dolfin.MPI.num_processes() > 1:
      raise NotImplementedException("minimise_functional cannot be used with more than one MPI process")
    if self.__functional is None:
      raise StateException("No functional defined")
 
    dolfin.info("Running functional minimisation")

    N = 0
    indices = []
    for parameter in parameters:
      if isinstance(parameter, dolfin.Constant):
        n = 1
      else:
        n = parameter.vector().local_size()
      indices.append((N, N + n))
      N += n
     
    def pack(parameters, arr):
      for i, parameter in enumerate(parameters):
        start, end = indices[i]
        if isinstance(parameter, dolfin.Constant):
          arr[start] = float(parameter)
        elif isinstance(parameter, dolfin.Function):
          arr[start:end] = parameter.vector().array()
        else:
          assert(isinstance(parameter, dolfin.GenericVector))
          arr[start:end] = parameter.array()
      return
    def unpack(parameters, arr):
      for i, parameter in enumerate(parameters):
        start, end = indices[i]
        if isinstance(parameter, dolfin.Constant):
          parameter.assign(arr[start])
        else:
          assert(isinstance(parameter, dolfin.Function))
          parameter.vector().set_local(arr[start:end])
          parameter.vector().apply("insert")
      return
  
    if bounds is None:
      bound_arrs = None
    else:
      nbounds = []
      for i, parameter in enumerate(parameters):
        if isinstance(parameter, dolfin.Constant):
          n = 1
        else:
          n = parameter.vector().local_size()
        if bounds[i][0] is None:
          p_l_bounds = [None for j in range(n)]
        elif isinstance(bounds[i][0], float):
          if isinstance(parameter, dolfin.Constant):
            p_l_bounds = [bounds[i][0]]
          else:
            p_l_bounds = numpy.array([bounds[i][0] for j in range(*parameter.vector().local_range(0))])
        elif isinstance(bounds[i][0], dolfin.Constant):
          p_l_bounds = [float(bounds[i][0])]
        else:
          p_l_bounds = bounds[i][0].vector().array()
        if bounds[i][1] is None:
          p_u_bounds = [None for j in range(n)]
        elif isinstance(bounds[i][1], float):
          if isinstance(parameter, dolfin.Constant):
            p_u_bounds = [bounds[i][1]]
          else:
            p_u_bounds = numpy.array([bounds[i][1] for j in range(*parameter.vector().local_range(0))])
        elif isinstance(bounds[i][1], dolfin.Constant):
          p_u_bounds = [float(bounds[i][1])]
        else:
          p_u_bounds = bounds[i][1].vector().array()
        for j in range(n):
          nbounds.append((p_l_bounds[j], p_u_bounds[j]))
      bounds = nbounds

    fun_x = numpy.empty(N)
    pack(parameters, fun_x)
    rerun_forward = [False]
    reassemble_adjoint = [False]
    rerun_adjoint = [True]
    def reassemble_forward(x):
      if not abs(fun_x - x).max() == 0.0:
        fun_x[:] = x
        unpack(parameters, x)
        self.reassemble_forward(*parameters)
        rerun_forward[0] = True
        reassemble_adjoint[0] = True
        rerun_adjoint[0] = True
      return

    n_fun = [0]
    def fun(x):
      reassemble_forward(x)
      if rerun_forward[0]:
        fn = self.compute_functional(rerun_forward = True, recheckpoint = True)
        rerun_forward[0] = False
      else:
        fn = self.compute_functional()
      n_fun[0] += 1
      dolfin.info("Functional evaluation %i       : %.17e" % (n_fun[0], fn))
      
      return fn

    garr = numpy.empty(N)
    n_jac = [0]
    def jac(x):
      reassemble_forward(x)
      if rerun_forward[0]:
        dolfin.info_red("Warning: Re-running forward model")
        self.rerun_forward(recheckpoint = True)
        rerun_forward[0] = False
      if reassemble_adjoint[0]:
        self.reassemble_adjoint(clear_caches = False, *parameters)
        reassemble_adjoint[0] = False
      if rerun_adjoint[0]:
        grad = self.compute_gradient(parameters)
        pack(grad, garr)
        rerun_adjoint[0] = False
        
      n_jac[0] += 1
      grad_norm = abs(garr).max()
      dolfin.info("Gradient evaluation %i inf norm: %.17e" % (n_jac[0], grad_norm))

      return garr

    if not parameters_0 is None:
      for i, parameter_0 in enumerate(parameters_0):
        if parameter_0 is None:
          pass
        elif isinstance(parameters[i], dolfin.Function) and isinstance(parameter_0, float):
          parameters[i].vector()[:] = parameter_0
        else:
          parameters[i].assign(parameter_0)
    x = numpy.empty(N)
    pack(parameters, x)

    if hasattr(scipy.optimize, "minimize"):
      res = scipy.optimize.minimize(fun, x, method = method, jac = jac, bounds = bounds, tol = tolerance, options = options)
      if not res["success"]:
        raise StateException("scipy.optimize.minimize failure")
      dolfin.info("scipy.optimize.minimize success with %i functional evaluation(s)" % res["nfev"])
      x = res["x"]
    else:
      if options is None:
        options = {}
      if method == "BFGS":
        res = scipy.optimize.fmin_bfgs(fun, x, fprime = jac, gtol = tolerance, full_output = True, **options)
        if not res[6] == 0:
          raise StateException("scipy.optimize.fmin_bfgs failure")
        dolfin.info("scipy.optimize.fmin_bfgs success with %i functional evaluation(s) and %i gradient calculation(s)" % (res[4], res[5]))
        x = res[0]
      elif method == "L-BFGS-B":
        res = scipy.optimize.fmin_l_bfgs_b(fun, x, fprime = jac, bounds = bounds, pgtol = tolerance, **options)
        if not res[2]["warnflag"] == 0:
          raise StateException("scipy.optimize.fmin_l_bfgs_b failure")
        dolfin.info("scipy.optimize.fmin_l_bfgs_b success with %i functional evaluation(s)" % res[2]["funcalls"])
        x = res[0]
      else:
        raise NotImplementedException("%s optimisation method not supported with SciPy version %s" % (method, scipy.__version__))

    reassemble_forward(x)
    if rerun_forward[0]:
      dolfin.info_red("Warning: Re-running forward model")
      self.rerun_forward(recheckpoint = True)
    if reassemble_adjoint[0]:
      self.reassemble_adjoint(clear_caches = False, *parameters)
    
    return res