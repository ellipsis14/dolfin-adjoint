''' 
   Imports the dolfin or firedrake module, depending on parameters["adjoint_backend"] 
'''
from backend_selector.backend import *
import backend_selector

def dolfin():
   return backend_selector.backend == "dolfin"
def firedrake():
   return backend_selector.backend == "firedrake"
