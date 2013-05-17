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

from caches import *
from exceptions import *
from fenics_overrides import *
from fenics_utils import *
from quadrature import *
from statics import *
from versions import *

__all__ = \
  [
    "PABilinearForm",
    "PAForm",
    "PALinearForm"
  ]

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

    if is_static_form(form):
      self.__set_pa([form], [])
    elif self.parameters["whole_form_optimisation"]:
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
  
  def is_static(self):
    """
    Return whether the Form is static.
    """
    
    return self.__n_non_pre_assembled == 0
    
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
  
_assemble_classes.append(PAForm)
    
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

    if is_static_form(form):
      self.__set_pa([form], [], [])
    elif self.parameters["whole_form_optimisation"]:
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
    self.__static = (self._PAForm__n_non_pre_assembled == 0) and (n_mult_assembled_L == 0)

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
  
  def is_static(self):
    """
    Return whether the Form is static.
    """
    
    return self.__static
  
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