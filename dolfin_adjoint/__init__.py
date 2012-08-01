"""

The entire dolfin-adjoint interface should be imported with a single
call:

.. code-block:: python

  from dolfin import *
  from dolfin_adjoint import *

It is essential that the importing of the :py:mod:`dolfin_adjoint` module happen *after*
importing the :py:mod:`dolfin` module. dolfin-adjoint relies on *overloading* many of
the key functions of dolfin to achieve its degree of automation.
"""

import options
import solving 
import assembly
import expressions
import utils
import assign
import matrix_free
import functional
import split_annotation
import lusolver
import gst
import function
import genericmatrix
import optimisation
import reduced_functional
from ui import *
