#!/usr/bin/env python

# Copyright (C) 2007-2013 Anders Logg and Kristian B. Oelgaard
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

# Copyright (C) 2007-2013 Anders Logg and Kristian B. Oelgaard from FFC file
# ffc/analysis.py, bzr trunk 1839
# Code first added: 2012-12-06

import dolfin
import ffc
import ufl

from exceptions import *
from fenics_utils import *
from versions import *

__all__ =\
  [
    "QForm",
    "form_quadrature_degree"
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
    return self.__add__(self, -other)
  
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

if ffc_version() < (1, 2, 0):
  def form_quadrature_degree(form):
    """
    Determine the quadrature degree with which the supplied Form is to be
    assembled. If form is a QForm, return the quadrature degree of the QForm.
    Otherwise, return the default quadrature degree if one is set, or return
    the quadrature degree that would be selected by FFC. The final case duplicates
    the internal behaviour of FFC.
    """
    
    if isinstance(form, QForm):
      return form.quadrature_degree()
    elif isinstance(form, ufl.form.Form):
      if dolfin.parameters["form_compiler"]["quadrature_degree"] > 0:
        quadrature_degree = dolfin.parameters["form_compiler"]["quadrature_degree"]
      else:
        # This is based upon code from _analyze_form and
        # _attach_integral_metadata in analysis.py, FFC bzr trunk revision 1761
        form = extract_form_data(form).preprocessed_form  # jit_form in jitcompiler.py passes the preprocessed form to
                                                          # compile_form, which eventually gets passed to _analyze_form
        element_mapping = ffc.analysis._compute_element_mapping(form, None)
        form_data = form.compute_form_data(element_mapping = element_mapping)
        quadrature_degree = -1
        for integral_data in form_data.integral_data:
          for integral in integral_data[2]:
            rep = ffc.analysis._auto_select_representation(integral, form_data.unique_sub_elements)
            quadrature_degree = max(quadrature_degree, ffc.analysis._auto_select_quadrature_degree(integral, rep, form_data.unique_sub_elements))
      return quadrature_degree
    else:
      raise InvalidArgumentException("form must be a Form")
else:
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
          rep = ffc.analysis._auto_select_representation(integral, form_data.unique_sub_elements, form_data.function_replace_map)
          quadrature_degree = max(quadrature_degree, ffc.analysis._auto_select_quadrature_degree(integral, rep, form_data.unique_sub_elements, form_data.element_replace_map))
      return quadrature_degree
    else:
      raise InvalidArgumentException("form must be a Form")