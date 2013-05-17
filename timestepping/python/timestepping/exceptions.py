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

__all__ = \
  [
    "AbstractMethodException",
    "CheckpointException",
    "DependencyException",
    "InvalidArgumentException",
    "NotImplementedException",
    "ParameterException",
    "StateException",
    "TimeLevelException",
    "VersionException"
  ]

class AbstractMethodException(Exception):
  pass

class CheckpointException(Exception):
  pass

class DependencyException(Exception):
  pass

class InvalidArgumentException(TypeError):
  pass

class NotImplementedException(Exception):
  pass

class ParameterException(Exception):
  pass

class StateException(Exception):
  pass

class TimeLevelException(Exception):
  pass

class VersionException(ImportError):
  pass