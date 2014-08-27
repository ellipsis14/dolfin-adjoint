"""
The optimization module is automatically imported by dolfin-adjoint 
"""

__version__ = '1.4'
__author__  = 'Patrick Farrell and Simon Funke'
__credits__ = ['Patrick Farrell', 'Simon Funke', 'David Ham', 'Marie Rognes']
__license__ = 'LGPL-3'
__maintainer__ = 'Simon Funke'
__email__ = 'simon@simula.no'

from armijo import ArmijoLineSearch
from strong_wolfe import StrongWolfeLineSearch
from fixed import FixedLineSearch