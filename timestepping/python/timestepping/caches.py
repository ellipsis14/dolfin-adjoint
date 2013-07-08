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
from versions import *

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

  def assemble(self, form, bcs = [], symmetric_bcs = False, compress = False):
    """
    Return the result of assembling the supplied Form.

    Arguments:
      form: The form.
      bcs: Dirichlet BCs applied to a matrix.
      symmetric_bcs: Whether Dirichlet BCs should be applied so as to yield a
        symmetric matrix.
      compress: Whether a matrix should be compressed.
    """
    
    if not isinstance(form, ufl.form.Form):
      raise InvalidArgumentException("form must be a Form")
    if not isinstance(bcs, list):
      raise InvalidArgumentException("bcs must be a list of DirichletBC s")
    for bc in bcs:
      if not isinstance(bc, dolfin.cpp.DirichletBC):
        raise InvalidArgumentException("bcs must be a list of DirichletBC s")

    rank = extract_form_data(form).rank
    if len(bcs) == 0:
      if not rank == 2:
        compress = None
      key = (expand(form), None, None, compress)
      if not key in self.__cache:
        cache_info("Assembling form with rank %i" % rank, dolfin.info_red)
        self.__cache[key] = assemble(form)
        if rank == 2 and compress:
          self.__cache[key].compress()
      else:
        cache_info("Using cached assembled form with rank %i" % rank, dolfin.info_green)
    else:
      if not rank == 2:
        raise InvalidArgumentException("form must be rank 2 when applying boundary conditions")

      key = (expand(form), tuple(bcs), symmetric_bcs, compress)
      if not key in self.__cache:
        cache_info("Assembling form with rank 2, with boundary conditions", dolfin.info_red)
        mat = assemble(form)
        apply_bcs(mat, bcs, symmetric_bcs = symmetric_bcs)
        self.__cache[key] = mat
        if compress:
          mat.compress()
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

    dolfin.info("Assembly cache status:")
    for i in range(3):
      dolfin.info("Pre-assembled rank %i forms: %i" % (i, counts[i]))

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

  def solver(self, form, solver_parameters,
    pre_assembly_parameters = None,
    static = None,
    bcs = [], symmetric_bcs = False,
    a = None):
    """
    Return a linear solver suitable for solving an equation with the supplied
    rank 2 Form defining the LHS. If such a solver exists in the cache, return
    the cached solver.

    Arguments:
      form: The form defining the matrix.
      solver_parameters: Linear solver parameters.
      bcs: Dirichlet BCs applied to the matrix.
      symmetric_bcs: Whether the Dirichlet BCs are applied so as to yield a
        symmetric matrix.
    and then either:
      pre_assembly_parameters: Pre-assembly parameters used to pre-assemble the
        form. Optional. Use an empty dictionary to denote the use of default
        pre-assembly parameters.
      static: Whether the form is static. Defaults to False if not supplied.
    or:
      a: A GenericMatrix resulting from the assembly of the form.
      
    This has three intended use cases:
      1. Pre-assembled form:
           solver = solver_cache.solver(form,
             solver_parameters, pre_assembly_parameters,
             static = static,
             bcs = bcs, symmetric_bcs = symmetric_bcs)
         If the default pre-assembly parameters are used, then an empty
         dictionary should be passed as the third argument. Assemble the form
         using:
           pa_form = PABilinearForm(form,
             pre_assembly_parameters = pre_assembly_parameters)
           a = assemble(pa_form, ...)
           apply_bcs(a, bcs, ..., symmetric_bcs = symmetric_bcs)
      2. Cached matrix:
           a = assembly_cache.assemble(form, ...)
           apply_bcs(a, bcs, ..., symmetric_bcs = symmetric_bcs)
           solver = solver_cache.solver(form,
             solver_parameters,
             bcs = bcs, symmetric_bcs = symmetric_bcs,
             a = a)
      3. Custom assembled form:
           solver = solver_cache.solver(form,
             solver_parameters,
             bcs = bcs, symmetric_bcs = symmetric_bcs)
         and then assemble the form using:
           a = assemble(form, ...)
           apply_bcs(a, bcs, ..., symmetric_bcs = symmetric_bcs)
    """
    
    def expanded_solver_parameters(form, solver_parameters, static, bcs, symmetric_bcs):
      if static:
        default = {"lu_solver":{"reuse_factorization":True, "same_nonzero_pattern":True},
                   "krylov_solver":{"preconditioner":{"reuse":True}}}
        if dolfin_version() >= (1, 1, 0) and (len(bcs) == 0 or symmetric_bcs) and is_self_adjoint_form(form):
          default["lu_solver"]["symmetric_operator"] = True
        solver_parameters = expand_solver_parameters(solver_parameters,
          default_solver_parameters = default)
      else:
        default = {"lu_solver":{"reuse_factorization":False, "same_nonzero_pattern":False},
                   "krylov_solver":{"preconditioner":{"reuse":False}}}
        if dolfin_version() >= (1, 1, 0) and (len(bcs) == 0 or symmetric_bcs) and is_self_adjoint_form(form):
          default["lu_solver"]["symmetric_operator"] = True
        solver_parameters = expand_solver_parameters(solver_parameters,
          default_solver_parameters = default)
        
        static_parameters = False
        if solver_parameters["linear_solver"] in ["direct", "lu"] or dolfin.has_lu_solver_method(solver_parameters["linear_solver"]):
          static_parameters = solver_parameters["lu_solver"]["reuse_factorization"] or \
                              solver_parameters["lu_solver"]["same_nonzero_pattern"]
        else:
          static_parameters = solver_parameters["krylov_solver"]["preconditioner"]["reuse"]
        if static_parameters:
          raise ParameterException("Non-static solve supplied with static solver parameters")
        
      return solver_parameters
    
    def form_key(form, static):
      if static:
        return expand(form)
      else:
        args = ufl.algorithms.extract_arguments(form)
        assert(len(args) == 2)
        test, trial = args
        if test.count() > trial.count():
          test, trial = trial, test
        return test, trial
    
    def bc_key(bcs, symmetric_bcs):
      if len(bcs) == 0:
        return None
      else:
        return tuple(bcs), symmetric_bcs

    def parameters_key(opts):
      if opts is None:
        return None
      assert(isinstance(opts, (dolfin.Parameters, dict)))
      fopts = []
      for key in opts.keys():
        if isinstance(opts[key], (dolfin.Parameters, dict)):
          fopts.append(parameters_key(opts[key]))
        else:
          fopts.append((key, opts[key]))
      return tuple(fopts)
    
    if not isinstance(form, ufl.form.Form):
      raise InvalidArgumentException("form must be a rank 2 Form")
    elif not extract_form_data(form).rank == 2:
      raise InvalidArgumentException("form must be a rank 2 Form")
    if not isinstance(solver_parameters, dict):
      raise InvalidArgumentException("solver_parameters must be a dictionary")
      
    if a is None:
      if static is None:
        static = False
        
      if not pre_assembly_parameters is None and not isinstance(pre_assembly_parameters, (dolfin.Parameters, dict)):
        raise InvalidArgumentException("pre_assembly_parameters must be None, a Parameters, or dictionary")
      if not isinstance(bcs, list):
        raise InvalidArgumentException("bcs must be a list of DirichletBC s")
      for bc in bcs:
        if not isinstance(bc, dolfin.cpp.DirichletBC):
          raise InvalidArgumentException("bcs must be a list of DirichletBC s")
      
      solver_parameters = expanded_solver_parameters(form, solver_parameters, static, bcs, symmetric_bcs)
      if not pre_assembly_parameters is None:
        npre_assembly_parameters = dolfin.parameters["timestepping"]["pre_assembly"]["bilinear_forms"].copy()
        npre_assembly_parameters.update(pre_assembly_parameters)
        pre_assembly_parameters = npre_assembly_parameters;  del(npre_assembly_parameters)
              
      key = (form_key(form, static),
             parameters_key(solver_parameters),
             parameters_key(pre_assembly_parameters),
             bc_key(bcs, symmetric_bcs),
             None)
    else:
      if not isinstance(a, dolfin.GenericMatrix):
        raise InvalidArgumentException("a must be a GenericMatrix")
      
      if not pre_assembly_parameters is None:
        raise InvalidArgumentException("Cannot supply pre_assembly_parameters argument if a GenericMatrix is supplied")
      if not static is None:
        raise InvalidArgumentException("Cannot supply static argument if a GenericMatrix is supplied")
      
      static = True      
      
      solver_parameters = expanded_solver_parameters(form, solver_parameters, True, bcs, symmetric_bcs)    
      
      key = (form_key(form, True),
             parameters_key(solver_parameters),
             None,
             bc_key(bcs, symmetric_bcs),
             a.id())

    if not key in self.__cache:
      if static:
        cache_info("Creating new static linear solver", dolfin.info_red)
      else:
        cache_info("Creating new non-static linear solver", dolfin.info_red)
      self.__cache[key] = LinearSolver(solver_parameters)
    else:
      if static:
        cache_info("Using cached static linear solver", dolfin.info_green)
      else:
        cache_info("Using cached non-static linear solver", dolfin.info_green)
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