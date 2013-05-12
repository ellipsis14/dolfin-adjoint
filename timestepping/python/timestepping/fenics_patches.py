#!/usr/bin/env python

# Copyright (C) 2007 Anders Logg
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

# Copyright (C) 2007 Anders Logg from FFC file ffc/codesnippets.py, bzr branch
# 1.1.x 1771

# Copyright (C) 2008-2013 Martin Sandve Alnes from UFL file ufl/coefficient.py,
# bzr trunk 1571, ufl/form.py, bzr branch 1.1.x 1484

import ctypes
import sys

import dolfin
import ffc
import numpy
import scipy.optimize
import ufl

from embedded_cpp import *
from exceptions import *
from versions import *

__all__ = []

# Only versions 1.0.x, 1.1.x, and 1.2.x have been tested.
if dolfin_version() < (1, 0, 0) or dolfin_version() >= (1, 3, 0):
  raise VersionException("DOLFIN version %s not supported" % dolfin.__version__)
if ufl_version() < (1, 0, 0) or ufl_version() >= (1, 3, 0):
  raise VersionException("UFL version %s not supported" % ufl.__version__)
if ffc_version() < (1, 0, 0) or ffc_version() >= (1, 3, 0):
  raise VersionException("FFC version %s not supported" % ffc.__version__)

# Backwards compatibility for older versions of DOLFIN.
if dolfin_version() < (1, 1, 0):
  __all__ += \
    [
      "GenericLinearSolver",
      "GenericLUSolver",
      "FacetFunction",
      "MeshFunction",
      "UnitCubeMesh",
      "UnitIntervalMesh",
      "UnitSquareMesh",
      "RectangleMesh",
      "has_krylov_solver_method",
      "has_krylov_solver_preconditioner"
    ]

  GenericLinearSolver = dolfin.GenericLinearSolver = (dolfin.KrylovSolver, dolfin.LUSolver, dolfin.LinearSolver, dolfin.PETScLUSolver, dolfin.PETScKrylovSolver)
  GenericLUSolver = dolfin.GenericLUSolver = (dolfin.LUSolver, dolfin.PETScLUSolver)
  UnitCubeMesh = dolfin.UnitCube
  UnitIntervalMesh = dolfin.UnitInterval
  UnitSquareMesh = dolfin.UnitSquare
  RectangleMesh = dolfin.Rectangle

  def has_krylov_solver_method(method):
    return method in [k_method[0] for k_method in dolfin.krylov_solver_methods()]

  def has_krylov_solver_preconditioner(pc):
    return pc in [k_pc[0] for k_pc in dolfin.krylov_solver_preconditioners()]

  class FacetFunction(dolfin.FacetFunction):
    def __new__(cls, tp, mesh, value = 0):
      if tp == "size_t":
        tp = "uint"
      return dolfin.FacetFunction.__new__(cls, tp, mesh, value = value)
    
  class MeshFunction(dolfin.MeshFunction):
    def __new__(cls, tp, *args):
      if tp == "size_t":
        tp = "uint"
      return dolfin.MeshFunction.__new__(cls, tp, *args)

  __GenericVector_gather_orig = dolfin.GenericVector.gather
  __GenericVector_gather_code = EmbeddedCpp(
    code =
      """
      Array< double > lx(n);
      Array< unsigned int > lnodes(n);
      for(size_t i = 0;i < n;i++){
        lnodes[i] = nodes[i];
      }
      v->gather(lx, lnodes);
      for(size_t i = 0;i < n;i++){
        x[i] = lx[i];
      }
      """,
    n = int, nodes = int_arr, v = dolfin.GenericVector, x = double_arr)
  def GenericVector_gather(self, *args):
    if len(args) == 1 and isinstance(args[0], numpy.ndarray) and len(args[0].shape) == 1 and args[0].dtype == "int32":
      x = numpy.empty(args[0].shape[0])
      __GenericVector_gather_code.run(n = args[0].shape[0], nodes = args[0], v = self, x = x)
      return x
    else:
      return __GenericVector_gather_orig(self, *args)
  dolfin.GenericVector.gather = GenericVector_gather
  del(GenericVector_gather)

  sys.setdlopenflags(ctypes.RTLD_GLOBAL + sys.getdlopenflags())
if dolfin_version() < (1, 2, 0):
  __all__ += \
    [
      "grad", 
      "FacetNormal"
    ]
              
  def grad(f):
    g = dolfin.grad(f)
    if len(g.shape()) == 0:
      return dolfin.as_vector([g])
    else:
      return g
  
  def FacetNormal(mesh):
    nm = dolfin.FacetNormal(mesh)
    if len(nm.shape()) == 0:
      return dolfin.as_vector([nm])
    else:
      return nm

# Backwards compatibility for older versions of UFL.
if ufl_version() < (1, 1, 0):
  # Modified version of code from coefficient.py, UFL bzr trunk revision 1463
  def Coefficient__eq__(self, other):
    if self is other:
      return True
    if not isinstance(other, ufl.coefficient.Coefficient):
      return False
    return self._count == other._count and self._element == other._element
  ufl.coefficient.Coefficient.__eq__ = Coefficient__eq__
  del(Coefficient__eq__)
  
  # Modified version of code from form.py, UFL bzr 1.1.x branch revision 1484
  def Form__mul__(self, coefficient):
    if isinstance(coefficient, ufl.expr.Expr):
      return ufl.formoperators.action(self, coefficient)
    return NotImplemented
  ufl.Form.__mul__ = Form__mul__
  del(Form__mul__)
# UFL patches.
elif ufl_version() >= (1, 2, 0) and ufl_version() < (1, 3, 0):
  __Form_compute_form_data_orig = ufl.Form.compute_form_data
  def Form_compute_form_data(self, object_names = None, common_cell = None, element_mapping = None):
    if element_mapping is None:
      element_mapping = ffc.jitcompiler._compute_element_mapping(self, common_cell = common_cell)
    return __Form_compute_form_data_orig(self, object_names = object_names, common_cell = common_cell, element_mapping = element_mapping)
  ufl.Form.compute_form_data = Form_compute_form_data
  del(Form_compute_form_data)
  
  def Cell_is_undefined(self):
    return False
  ufl.geometry.Cell.is_undefined = Cell_is_undefined
  del(Cell_is_undefined)

# FFC patches.
if ffc_version() < (1, 2, 0):
  # Patch to code from FFC file codesnippets.py, FFC bzr 1.1.x branch revision
  # 1771
  ffc.codesnippets._circumradius_1D = """\
  // Compute circumradius, in 1D it is equal to the cell volume/2.0.
  const double circumradius%(restriction)s = std::abs(detJ%(restriction)s)/2.0;"""
  ffc.codesnippets.circumradius[1] = ffc.codesnippets._circumradius_1D