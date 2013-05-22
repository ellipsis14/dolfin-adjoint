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

import dolfin
import ufl

from exceptions import *
from fenics_overrides import *
from fenics_utils import *

__all__ = \
  [
    "AssemblyCache",
    "SolverCache",
    "assembly_cache",
    "solver_cache",
    "cache_info",
    "clear_caches"
  ]

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
      if not "linear_solver" in solver_parameters or solver_parameters["linear_solver"] in ["direct", "lu"] or dolfin.has_lu_solver_method(solver_parameters["linear_solver"]):
        solver_parameters = expand_solver_parameters(solver_parameters, default_solver_parameters = {"lu_solver":{"reuse_factorization":True, "same_nonzero_pattern":True}})
      else:
        solver_parameters = expand_solver_parameters(solver_parameters, default_solver_parameters = {"krylov_solver":{"preconditioner":{"reuse":True}}})
    else:
      solver_parameters = expand_solver_parameters(solver_parameters)

      static_parameters = False
      if solver_parameters["linear_solver"] in ["direct", "lu"] or dolfin.has_lu_solver_method(solver_parameters["linear_solver"]):
        static_parameters = solver_parameters["lu_solver"]["reuse_factorization"] or solver_parameters["lu_solver"]["same_nonzero_pattern"]
      else:
        static_parameters = solver_parameters["krylov_solver"]["preconditioner"]["reuse"]
      if static_parameters:
        raise ParameterException("Non-static solve supplied with static solver parameters")

    if static:
      if len(bcs) == 0:
        key = (expand(form), None, None, flatten_parameters(solver_parameters))
      else:
        key = (expand(form), tuple(bcs), symmetric_bcs, flatten_parameters(solver_parameters))
    else:
      args = ufl.algorithms.extract_arguments(form)
      assert(len(args) == 2)
      test, trial = args
      if test.count() > trial.count():
        test, trial = trial, test
      if len(bcs) == 0:
        key = ((test, trial), None, None, flatten_parameters(solver_parameters))
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