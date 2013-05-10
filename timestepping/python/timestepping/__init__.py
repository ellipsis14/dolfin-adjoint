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

from fenics_patches import *

import embedded_cpp
import exceptions
import timestepping
import versions

from embedded_cpp import *
from exceptions import *
from timestepping import *
from versions import *

__doc__ = \
"""
A timestepping abstraction and automated adjoining library. This library
utilises the FEniCS system for symbolic manipulation and automated code
generation, and supplements this system with a syntax for the description of
timestepping finite element models.
"""

__license__ = "LGPL-3"

__version__ = "1.2.0"

__all__ = \
  embedded_cpp.__all__ + \
  exceptions.__all__ + \
  timestepping.__all__ + \
  versions.__all__ + \
  [ 
    "__doc__",
    "__license__",
    "__name__",
    "__version__",
    "embedded_cpp",
    "exceptions",
    "timestepping",
    "versions"
  ]