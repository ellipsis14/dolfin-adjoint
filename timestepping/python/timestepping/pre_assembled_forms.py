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

if ufl_version() < (1, 2, 0):
  def form_integrals(form):
    """
    Return Integral s associated with the given Form.
    """
    
    form_data = extract_form_data(form)
    integrals = []
    for integral_data in form_data.integral_data:
      integrals += integral_data[2]
    return integrals
  
  def preprocess_integral(form, integral):
    """
    Given an Integral associated with the given Form, return the integrand and
    a list of arguments which can be used to construct an Integral from the
    integrand.
    """
    
    form_data = extract_form_data(form)
    integrand, measure = integral.integrand(), integral.measure()
    repl = {}
    for old, new in zip(form_data.arguments + form_data.coefficients, form_data.original_arguments + form_data.original_coefficients):
      repl[old] = new
    integrand = replace(integrand, repl)
    return integrand, [measure]
else:
  def form_integrals(form):
    """
    Return Integral s associated with the given Form.
    """
    
    return form.integrals()
  
  def preprocess_integral(form, integral):
    """
    Given an Integral associated with the given Form, return the integrand and
    a list of arguments which can be used to construct an Integral from the
    integrand.
    """
    
    integrand = integral.integrand()
    domain_type, domain_description, compiler_data, domain_data = \
      integral.domain_type(), integral.domain_description(), integral.compiler_data(), integral.domain_data()
    return integrand, [domain_type, domain_description, compiler_data, domain_data]

class PAForm:
  """
  A pre-assembled form. Given a form of arbitrary rank, this finds and
  pre-assembles static terms.

  Constructor arguments:
    form: The Form to be pre-assembled.
    parameters: Parameters defining detailed optimisation options.
  """
  
  def __init__(self, form, parameters = {}, default_parameters = dolfin.parameters["timestepping"]["pre_assembly"]["forms"]):
    if not isinstance(default_parameters, dolfin.Parameters):
      raise InvalidArgumentException("default_parameters must be a Parameters")
    
    nparameters = default_parameters.copy()
    nparameters.update(parameters)
    parameters = nparameters;  del(nparameters)

    self.parameters = parameters
    self.__set(form)

    return

  def __set(self, form):
    if not isinstance(form, ufl.form.Form) or is_empty_form(form):
      raise InvalidArgumentException("form must be a non-empty Form")

    self._set_optimise(form)

    self.__rank = extract_form_data(form).rank
    self.__deps = ufl.algorithms.extract_coefficients(form)

    return

  def _set_optimise(self, form):
    if is_static_form(form):
      self._set_pa([form], [])
    elif self.parameters["term_optimisation"]:
      quadrature_degree = form_quadrature_degree(form)

      pre_assembled_L = []
      non_pre_assembled_L = []
      
      for integral in form_integrals(form):
        integrand, iargs = preprocess_integral(form, integral)
        if self.parameters["expand_form"]:
          if isinstance(integrand, ufl.algebra.Sum):
            terms = [(term, expand_expr(term)) for term in integrand.operands()]
          else:
            terms = [(integrand, expand_expr(integrand))]
        else:
          if isinstance(integrand, ufl.algebra.Sum):
            terms = [[(term, term)] for term in integrand.operands()]
          else:
            terms = [[(term, term)]]        
        for term in terms:
          pterm = [], []
          for sterm in term[1]:
            stform = QForm([ufl.Integral(sterm, *iargs)], quadrature_degree = quadrature_degree)
            if is_static_form(stform):
              pterm[0].append(stform)
            else:
              pterm[1].append(stform)
          if len(pterm[0]) == 0:
            tform = QForm([ufl.Integral(term[0], *iargs)], quadrature_degree = quadrature_degree)
            non_pre_assembled_L.append(tform)
          else:
            pre_assembled_L += pterm[0]
            non_pre_assembled_L += pterm[1]

      self._set_pa(pre_assembled_L, non_pre_assembled_L)
    else:
      self._set_pa([], [form])

    return

  def _set_pa(self, pre_assembled_L, non_pre_assembled_L):
    if len(pre_assembled_L) == 0:
      self._pre_assembled_L = None
    else:
      l_L = pre_assembled_L[0]
      for L in pre_assembled_L[1:]:
        l_L += L
      self._pre_assembled_L = assembly_cache.assemble(l_L)

    if len(non_pre_assembled_L) == 0:
      self._non_pre_assembled_L = None
    else:
      self._non_pre_assembled_L = non_pre_assembled_L[0]
      for L in non_pre_assembled_L[1:]:
        self._non_pre_assembled_L += L
        
    self._n_pre_assembled = len(pre_assembled_L)
    self._n_non_pre_assembled = len(non_pre_assembled_L)

    return

  def assemble(self, copy = False):
    """
    Return the result of assembling the Form associated with the PAForm. If
    copy is False then existing data may be returned -- it is expected in this
    case that the return value will never be modified.
    """
    
    if self._non_pre_assembled_L is None:
      if self._pre_assembled_L is None:
        raise StateException("Cannot assemble empty form")
      else:
        if copy:
          L = self._pre_assembled_L.copy()
        else:
          L = self._pre_assembled_L
    else:
      L = assemble(self._non_pre_assembled_L)
      if not self._pre_assembled_L is None:
        L += self._pre_assembled_L

    return L
  
  def rank(self):
    """
    Return the Form rank.
    """
    
    return self.__rank
  
  def is_static(self):
    """
    Return whether the PAForm is static.
    """
    
    return self._n_non_pre_assembled == 0
    
  def n_pre_assembled(self):
    """
    Return the number of pre-assembled terms.
    """
    
    return self._n_pre_assembled

  def n_non_pre_assembled(self):
    """
    Return the number of non-pre-assembled terms.
    """
    
    return self._n_non_pre_assembled

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
    
    for i, dep in enumerate(self.__deps):
      if dep in mapping:
        self.__deps[i] = mapping[dep]
    if not self._non_pre_assembled_L is None:
      self._non_pre_assembled_L = replace(self._non_pre_assembled_L, mapping)
    
    return
  
_assemble_classes.append(PAForm)
    
class PABilinearForm(PAForm):
  """
  A pre-assembled bi-linear form. This is similar to PAForm, but applies
  additional optimisations specific to bi-linear forms. Also has different
  default parameters.
  """
  
  def __init__(self, form, parameters = {}):
    if not extract_form_data(form).rank == 2:
      raise InvalidArgumentException("form must be a rank 2 form")
    
    PAForm.__init__(self, form, parameters = parameters, default_parameters = dolfin.parameters["timestepping"]["pre_assembly"]["bilinear_forms"])

    return

  def assemble(self, copy = False):
    """
    Return the result of assembling the Form associated with the PABilinearForm.
    If copy is False then existing data may be returned -- it is expected in
    this case that the return value will never be modified.
    """
    
    if self._non_pre_assembled_L is None:
      if self._pre_assembled_L is None:
        raise StateException("Cannot assemble empty form")
      else:
        if copy:
          L = self._pre_assembled_L.copy()
        else:
          L = self._pre_assembled_L
    else:
      if hasattr(self, "_PABilinearForm__non_pre_assembled_L_tensor"):
        L = self.__non_pre_assembled_L_tensor
        assemble(self._non_pre_assembled_L, tensor = L, reset_sparsity = False)
        if not self._pre_assembled_L is None:
          # The pre-assembled matrix and the non-pre-assembled matrix have
          # previously been configured so as to have the same sparsity pattern
          # (below)
          L.axpy(1.0, self._pre_assembled_L, True)
      else:
        L = self.__non_pre_assembled_L_tensor = assemble(self._non_pre_assembled_L)
        if not self._pre_assembled_L is None:
          self._pre_assembled_L = self._pre_assembled_L.copy()
          L.axpy(1.0, self._pre_assembled_L, False)
          # Ensure that the pre-assembled matrix and the matrix used to store
          # the non-pre-assembled matrix data have the same sparsity pattern.
          self._pre_assembled_L.axpy(0.0, L, False)

    return L
    
class PALinearForm(PAForm):
  """
  A pre-assembled linear form. This is similar to PAForm, but applies additional
  optimisations specific to linear forms. Also has different default parameters.
  """
  
  def __init__(self, form, parameters = {}):
    if not extract_form_data(form).rank == 1:
      raise InvalidArgumentException("form must be a rank 1 form")
    
    PAForm.__init__(self, form, parameters = parameters, default_parameters = dolfin.parameters["timestepping"]["pre_assembly"]["linear_forms"])

    return

  def _set_optimise(self, form):
    pre_assembled_L = []
    mult_assembled_L = OrderedDict()
    non_pre_assembled_L = []
    
    def matrix_optimisation(tform):
      args = ufl.algorithms.extract_arguments(tform)
      if not len(args) == 1:
        return None
      tcs = extract_non_static_coefficients(tform)
      if not len(tcs) == 1 or not isinstance(tcs[0], dolfin.Function) or \
        (not dolfin.MPI.num_processes() == 1 and not tcs[0].function_space().num_sub_spaces() == 0) or \
        (not dolfin.MPI.num_processes() == 1 and (is_r0_function_space(args[0].function_space()) or is_r0_function(tcs[0]))):
        return None
      fn = tcs[0]
        
      mat_form = derivative(tform, fn)
      if n_non_static_coefficients(mat_form) > 0:
        return None
      else:
        return fn, mat_form
    
    if is_static_form(form):
      pre_assembled_L.append(form)
    elif self.parameters["term_optimisation"]:
      quadrature_degree = form_quadrature_degree(form)
    
      for integral in form_integrals(form):
        integrand, iargs = preprocess_integral(form, integral)
        if self.parameters["expand_form"]:
          if isinstance(integrand, ufl.algebra.Sum):
            terms = [(term, expand_expr(term)) for term in integrand.operands()]
          else:
            terms = [(integrand, expand_expr(integrand))]
        else:
          if isinstance(integrand, ufl.algebra.Sum):
            terms = [[(term, term)] for term in integrand.operands()]
          else:
            terms = [[(term, term)]]        
        for term in terms:
          pterm = [], [], []
          for sterm in term[1]:
            stform = QForm([ufl.Integral(sterm, *iargs)], quadrature_degree = quadrature_degree)
            if is_static_form(stform):
              pterm[0].append(stform)
            elif self.parameters["matrix_optimisation"]:
              mform = matrix_optimisation(stform)
              if mform is None:
                pterm[2].append(stform)
              else:
                pterm[1].append(mform)
            else:
              pterm[2].append(stform)
          if len(pterm[0]) == 0 and len(pterm[1]) == 0:
            tform = QForm([ufl.Integral(term[0], *iargs)], quadrature_degree = quadrature_degree)
            non_pre_assembled_L.append(tform)
          else:
            pre_assembled_L += pterm[0]
            for mform in pterm[1]:
              if mform[0] in mult_assembled_L:
                mult_assembled_L[mform[0]].append(mform[1])
              else:
                mult_assembled_L[mform[0]] = [mform[1]]
            non_pre_assembled_L += pterm[2]
    elif self.parameters["matrix_optimisation"]:
      mform = matrix_optimisation(form)
      if mform is None:
        non_pre_assembled_L.append(form)
      else:
        mult_assembled_L[mform[0]] = [mform[1]]
    else:
      non_pre_assembled_L.append(form)

    self._set_pa(pre_assembled_L, mult_assembled_L, non_pre_assembled_L)
          
    return

  def _set_pa(self, pre_assembled_L, mult_assembled_L, non_pre_assembled_L):
    if len(pre_assembled_L) == 0:
      self._pre_assembled_L = None
    else:
      l_L = pre_assembled_L[0]
      for L in pre_assembled_L[1:]:
        l_L += L
      self._pre_assembled_L = assembly_cache.assemble(l_L)

    n_mult_assembled_L = 0
    if len(mult_assembled_L) == 0:
      self._mult_assembled_L = None
    else:
      self._mult_assembled_L = []
      for fn in mult_assembled_L:
        mat_forms = mult_assembled_L[fn]
        n_mult_assembled_L += len(mat_forms)
        mat_form = mat_forms[0]
        for lmat_form in mat_forms[1:]:
          mat_form += lmat_form
        self._mult_assembled_L.append([assembly_cache.assemble(mat_form), fn])

    if len(non_pre_assembled_L) == 0:
      self._non_pre_assembled_L = None
    else:
      self._non_pre_assembled_L = non_pre_assembled_L[0]
      for L in non_pre_assembled_L[1:]:
        self._non_pre_assembled_L += L

    self._n_pre_assembled = len(pre_assembled_L) + n_mult_assembled_L
    self._n_non_pre_assembled = len(non_pre_assembled_L)
    self.__static = (self._n_non_pre_assembled == 0) and (n_mult_assembled_L == 0)

    return

  def assemble(self, copy = False):
    """
    Return the result of assembling the Form associated with this PALinearForm.
    If copy is False then an existing GenericVector may be returned -- it is
    expected in this case that the return value will never be modified.
    """
    
    if self._non_pre_assembled_L is None:
      if self._mult_assembled_L is None:
        if self._pre_assembled_L is None:
          raise StateException("Cannot assemble empty form")
        else:
          if copy:
            L = self._pre_assembled_L.copy()
          else:
            L = self._pre_assembled_L
      else:
        L = self._mult_assembled_L[0][0] * self._mult_assembled_L[0][1].vector()
        for i in range(1, len(self._mult_assembled_L)):
          L += self._mult_assembled_L[i][0] * self._mult_assembled_L[i][1].vector()
        if not self._pre_assembled_L is None:
          L += self._pre_assembled_L
    else:
      if hasattr(self, "_PALinearForm__non_pre_assembled_L_tensor"):
        L = self.__non_pre_assembled_L_tensor
        assemble(self._non_pre_assembled_L, tensor = L, reset_sparsity = False)
      else:
        L = self.__non_pre_assembled_L_tensor = assemble(self._non_pre_assembled_L)
      if not self._mult_assembled_L is None:
        for i in range(len(self._mult_assembled_L)):
          L += self._mult_assembled_L[i][0] * self._mult_assembled_L[i][1].vector()
      if not self._pre_assembled_L is None:
        L += self._pre_assembled_L

    return L
  
  def is_static(self):
    """
    Return whether the PALinearForm is static.
    """
    
    return self.__static
  
  def replace(self, mapping):
    """
    Replace coefficients.
    """
    
    PAForm.replace(self, mapping)
    if not self._mult_assembled_L is None:
      for i in range(len(self._mult_assembled_L)):
        mat, fn = self._mult_assembled_L[i]
        if fn in mapping:
          self._mult_assembled_L[i] = mat, mapping[fn]
    
    return
