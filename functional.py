import libadjoint
import ufl
import dolfin
import hashlib

import solving

class Functional(libadjoint.Functional):
  '''This class implements the libadjoint.Functional abstract base class for the Dolfin adjoint.
  It takes in a form, and implements the necessary routines such as calling the functional
  and taking its derivative.'''

  def __init__(self, form):

    self.form = form
    self.activated = False

  def __call__(self, dependencies, values):

    dolfin_dependencies=[dep for dep in ufl.algorithms.extract_coefficients(self.form) if hasattr(dep, "function_space")]

    dolfin_values=[val.data for val in values]

    return dolfin.assemble(dolfin.replace(self.form, dict(zip(dolfin_dependencies, dolfin_values))))

  def derivative(self, variable, dependencies, values):

    # Find the dolfin Function corresponding to variable.
    dolfin_variable = values[dependencies.index(variable)].data

    dolfin_dependencies = [dep for dep in ufl.algorithms.extract_coefficients(self.form) if hasattr(dep, "function_space")]

    dolfin_values = [val.data for val in values]

    current_form = dolfin.replace(self.form, dict(zip(dolfin_dependencies, dolfin_values)))
    test = dolfin.TestFunction(dolfin_variable.function_space())

    return solving.Vector(dolfin.derivative(current_form, dolfin_variable, test))

  def dependencies(self, adjointer, timestep):

    if self.activated is False:
      deps = [solving.adj_variables[coeff] for coeff in ufl.algorithms.extract_coefficients(self.form) if hasattr(coeff, "function_space")]      
      self.activated = True
    else:
      deps = []
    
    return deps

  def __str__(self):
    
    return hashlib.md5(str(self.form)).hexdigest()


