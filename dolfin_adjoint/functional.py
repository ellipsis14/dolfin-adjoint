import libadjoint
import ufl
import dolfin
import hashlib

import solving

class FinalFunctional(libadjoint.Functional):
  '''This class implements the libadjoint.Functional abstract base class for the Dolfin adjoint.
  It takes in a form that evaluates the functional at the final timestep, and implements the 
  necessary routines such as calling the functional  and taking its derivative.'''

  def __init__(self, form):

    self.form = form
    self.activated = False

  def __call__(self, adjointer, dependencies, values):

    dolfin_dependencies=[dep for dep in ufl.algorithms.extract_coefficients(self.form) if hasattr(dep, "function_space")]

    dolfin_values=[val.data for val in values]

    return dolfin.assemble(dolfin.replace(self.form, dict(zip(dolfin_dependencies, dolfin_values))))

  def derivative(self, adjointer, variable, dependencies, values):

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


class TimeFunctional(libadjoint.Functional):
  '''This class implements the libadjoint.Functional abstract base class for the Dolfin adjoint for implementing functionals of the form:
      \int_{t=0..T} form(t) + finalform(T).
    The integration of the time integral is performed using the midpoint quadrature rule (i.e. exact if the solution is piecewise linear in time) assuming that a constant timestep dt is employed.
    The two forms, form and finalform, may only use UFL coefficients of the same timelevel, except the UFL coefficients that are listed in the optional parameter staticvariables. 
    If finalform is not provided, the second term is neglected.'''

  def __init__(self, form, dt, finalform=None, staticvariables = []):

    self.form = form
    self.staticvariables = staticvariables
    self.dt = dt

    if finalform is not None:
      raise libadjoint.exceptions.LibadjointErrorNotImplemented("finalform is not supported yet.")

  def __call__(self, adjointer, timestep, dependencies, values):

    dolfin_dependencies_form = [dep for dep in ufl.algorithms.extract_coefficients(self.form) if (hasattr(dep, "function_space") and dep not in self.staticvariables)]
    dolfin_values = [val.data for val in values]

    # The quadrature weight for the midpoint rule is 1.0 for interiour points and 0.5 at the end points.
    if timestep==0 or timestep==adjointer.timestep_count-1: 
      quad_weight = 0.5
    else: 
      quad_weight = 1.0
    return dolfin.assemble(dolfin.replace(quad_weight*self.dt*self.form, dict(zip(dolfin_dependencies, dolfin_values))))

  def derivative(self, adjointer, variable, dependencies, values):

    # Find the dolfin Function corresponding to variable.
    dolfin_variable = values[dependencies.index(variable)].data

    dolfin_dependencies_form = [dep for dep in ufl.algorithms.extract_coefficients(self.form) if (hasattr(dep, "function_space") and dep not in self.staticvariables)]
    dolfin_values = [val.data for val in values]

    test = dolfin.TestFunction(dolfin_variable.function_space())
    current_form = dolfin.replace(self.form, dict(zip(dolfin_dependencies_form, dolfin_values)))

    # The quadrature weight for the midpoint rule is 1.0 for interiour points and 0.5 at the end points.
    if variable.timestep==0 or variable.timestep==adjointer.timestep_count-1: 
      quad_weight = 0.5
    else: 
      quad_weight = 1.0
    return solving.Vector(dolfin.derivative(quad_weight*self.dt*current_form, dolfin_variable, test))

  def dependencies(self, adjointer, timestep):
    deps = [solving.adj_variables[coeff] for coeff in ufl.algorithms.extract_coefficients(self.form) if (hasattr(coeff, "function_space") and coeff not in self.staticvariables)]
    # Set the time level of the dependencies:
    for i in range(len(deps)):
      deps[i].var.timestep = timestep
      deps[i].var.iteration = deps[i].iteration_count(adjointer) - 1 

    return deps

  def __str__(self):
    
    return hashlib.md5(str(self.form)).hexdigest()
