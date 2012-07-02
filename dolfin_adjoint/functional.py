import libadjoint
import ufl
import dolfin
import hashlib

import solving
import adjglobals
import adjlinalg

class FinalFunctional(libadjoint.Functional):
  '''This class implements the libadjoint.Functional abstract base class for the Dolfin adjoint.
  It takes in a form that evaluates the functional at the final timestep, and implements the 
  necessary routines such as calling the functional  and taking its derivative.'''

  def __init__(self, form, name=None):

    self.form = form
    self.activated = False
    self.name = name

  def __call__(self, adjointer, dependencies, values):

    dolfin_dependencies=[dep for dep in ufl.algorithms.extract_coefficients(self.form) if hasattr(dep, "function_space")]
    # Remove dependencies for which no equation is registered 
    for dep in dolfin_dependencies:
        if not adjointer.variable_known(adjglobals.adj_variables[dep]):
            dolfin_dependencies.remove(dep)

    dolfin_values=[val.data for val in values]

    return dolfin.assemble(dolfin.replace(self.form, dict(zip(dolfin_dependencies, dolfin_values))))

  def derivative(self, adjointer, variable, dependencies, values):

    # Find the dolfin Function corresponding to variable.
    dolfin_variable = values[dependencies.index(variable)].data

    dolfin_dependencies = [dep for dep in ufl.algorithms.extract_coefficients(self.form) if hasattr(dep, "function_space")]
    # Remove dependencies for which no equation is registered 
    for dep in dolfin_dependencies:
        if not adjointer.variable_known(adjglobals.adj_variables[dep]):
            dolfin_dependencies.remove(dep)

    dolfin_values = [val.data for val in values]

    current_form = dolfin.replace(self.form, dict(zip(dolfin_dependencies, dolfin_values)))
    test = dolfin.TestFunction(dolfin_variable.function_space())

    return adjlinalg.Vector(dolfin.derivative(current_form, dolfin_variable, test))

  def dependencies(self, adjointer, timestep):

    if self.activated is False:
      deps = [adjglobals.adj_variables[coeff] for coeff in ufl.algorithms.extract_coefficients(self.form) if hasattr(coeff, "function_space")] 
      # If there is no equation annotated for a dependency variable we remove it from the dependency list
      for dep in deps:
          if not adjointer.variable_known(dep):
              deps.remove(dep)
               
      self.activated = True
    else:
      deps = []
    
    return deps

  def __str__(self):
    if self.name is not None:
      return self.name
    else:
      return hashlib.md5(str(self.form)).hexdigest()


class TimeFunctional(libadjoint.Functional):
  '''This class implements the libadjoint.Functional abstract base class for the Dolfin adjoint for implementing functionals of the form:
      \int_{t=0..T} form(t) + final_form(T).
    The integration of the time integral is performed using the midpoint quadrature rule (i.e. exact if the solution is piecewise linear in time) assuming that a constant timestep dt is employed.
    The two forms, form and final_form, may only use UFL coefficients of the same timelevel.
    If final_form is not provided, the second term is neglected.'''

  def __init__(self, form, dt, final_form=None, verbose=False, name=None):

    self.form = form
    self.dt = dt
    self.final_form = final_form
    self.verbose = verbose
    self.name = name

  def __call__(self, adjointer, timestep, dependencies, values):

    # Select the correct value for the first timestep, as it has dependencies both at the end 
    # and, for the initial conditions, at the beginning.
    if variable.timestep == 0:
      if variable.iteration == 0:
        dolfin_values = [val.data for val in values[:len(values)/2]]
      else:
        dolfin_values = [val.data for val in values[len(values)/2:]]
    else:            
      dolfin_values = [val.data for val in values]

    # The quadrature weight for the midpoint rule is 1.0 for interiour points and 0.5 at the end points.
    if (timestep==0 and variable.iteration == 0) or timestep==adjointer.timestep_count-1: 
      quad_weight = 0.5
    else: 
      quad_weight = 1.0

    dolfin_dependencies_form = [dep for dep in ufl.algorithms.extract_coefficients(self.form) if (hasattr(dep, "function_space")) and adjointer.variable_known(adjglobals.adj_variables[dep])]
    functional_value = dolfin.replace(quad_weight*self.dt*self.form, dict(zip(dolfin_dependencies_form, dolfin_values)))

    # Add the contribution of the integral at the last timestep
    if self.final_form != None and timestep==adjointer.timestep_count-1:
      dolfin_dependencies_final_form = [dep for dep in ufl.algorithms.extract_coefficients(self.final_form) if (hasattr(dep, "function_space")) and adjointer.variable_known(adjglobals.adj_variables[dep])]
      functional_value += dolfin.replace(self.final_form, dict(zip(dolfin_dependencies_final_form, dolfin_values)))

    return dolfin.assemble(functional_value)

  def derivative(self, adjointer, variable, dependencies, values):

    # Find the dolfin Function corresponding to variable.
    dolfin_variable = values[dependencies.index(variable)].data

    dolfin_dependencies_form = [dep for dep in ufl.algorithms.extract_coefficients(self.form) if (hasattr(dep, "function_space")) and adjointer.variable_known(adjglobals.adj_variables[dep])]
    # Select the correct value for the first timestep, as it has dependencies both at the end 
    # and, for the initial conditions, at the beginning.
    if variable.timestep == 0:
      if variable.iteration == 0:
        dolfin_values = [val.data for val in values[:len(values)/2]]
      else:
        dolfin_values = [val.data for val in values[len(values)/2:]]
    else:            
      dolfin_values = [val.data for val in values]

    test = dolfin.TestFunction(dolfin_variable.function_space())
    current_form = dolfin.replace(self.form, dict(zip(dolfin_dependencies_form, dolfin_values)))

    # The quadrature weight for the midpoint rule is 1.0 for interiour points and 0.5 at the end points.
    if (variable.timestep == 0 and variable.iteration == 0) or variable.timestep==adjointer.timestep_count-1: 
      quad_weight = 0.5
    else: 
      quad_weight = 1.0
    functional_deriv_value = dolfin.derivative(quad_weight*self.dt*current_form, dolfin_variable, test)

    # Add the contribution of the integral at the last timestep
    if self.final_form != None and variable.timestep==adjointer.timestep_count-1:
      dolfin_dependencies_final_form = [dep for dep in ufl.algorithms.extract_coefficients(self.final_form) if (hasattr(dep, "function_space")) and adjointer.variable_known(adjglobals.adj_variables[dep])]
      final_form = dolfin.replace(self.final_form, dict(zip(dolfin_dependencies_final_form, dolfin_values)))
      functional_deriv_value += dolfin.derivative(final_form, dolfin_variable, test)

    if self.verbose:
      dolfin.info("Returning dJ/du term for %s" % str(variable))

    return adjlinalg.Vector(functional_deriv_value)

  def dependencies(self, adjointer, timestep):

    if adjglobals.adj_variables.libadjoint_timestep == 0:
      dolfin.info_red("Warning: instantiating a TimeFunctional without having called adj_inc_timestep. This probably won't work.")

    deps = [adjglobals.adj_variables[coeff] for coeff in ufl.algorithms.extract_coefficients(self.form) if (hasattr(coeff, "function_space")) and adjointer.variable_known(adjglobals.adj_variables[coeff])]

    # Add the dependencies for the final_form
    if self.final_form != None and timestep==adjointer.timestep_count-1:
      deps += [adjglobals.adj_variables[coeff] for coeff in ufl.algorithms.extract_coefficients(self.final_form) if (hasattr(coeff, "function_space")) and adjointer.variable_known(adjglobals.adj_variables[coeff])]
      # Remove duplicates
      deps = list(set(deps))

    # Set the time level of the dependencies:
    for i in range(len(deps)):
      deps[i].var.timestep = timestep
      deps[i].var.iteration = deps[i].iteration_count(adjointer) - 1 

    # The first timestep has dependencies both at the end and, for the initial conditions,
    # at the beginning.
    if timestep == 0:
        deps *= 2
        for i in range(len(deps)/2):
          deps[i] = deps[i].copy()
          deps[i].var.iteration = 0 

    return deps

  def __str__(self):
    if self.name is not None:
      return self.name
    else:
      return hashlib.md5(str(self.form)).hexdigest()


class Functional(libadjoint.Functional):
  '''This class implements the libadjoint.Functional abstract base class for the Dolfin adjoint for implementing functionals of the form:
      FIX THIS'''

  def __init__(self, timeform, verbose=False, name=None):

    self.timeform = timeform
    self.verbose = verbose
    self.name = name

  def __call__(self, adjointer, timestep, dependencies, values):

    deps={str(dep): val.data for dep, val in zip(dependencies, values)}

    functional_value = None

    # Get the necessary timestep information about the adjointer.
    # For integrals, we're integrating over /two/ timesteps.
    timestep_start, timestep_end = adjointer.get_times(timestep)
    point_interval = slice(timestep_start, timestep_end)

    if timestep > 0:
      prev_start, prev_end = adjointer.get_times(timestep - 1)
      integral_interval = slice(prev_start, timestep_end)
    else:
      integral_interval = slice(timestep_start, timestep_end)

    for term in self.timeform.terms:
      if isinstance(term.time, slice):
        # Integral.
        this_interval=_slice_intersect(integral_interval, term.time)
        if this_interval:
          # Get adj_variables for dependencies. Time level is not yet specified.

          # Dependency replacement dictionary.
          replace={}

          term_deps = _coeffs(adjointer, term.form)
          term_vars = _vars(adjointer, term.form)

          for term_dep, term_var in zip(term_deps,term_vars):
            start_var = self.get_vars(adjointer, timestep, term_var)[0]
            replace[term_dep] = deps[str(start_var)]

          # Trapezoidal rule over given interval.
          quad_weight = 0.5*(this_interval.stop-this_interval.start)
          # Calculate i
          if functional_value is None:
            functional_value = dolfin.replace(quad_weight*term.form, replace)
          else:
            functional_value += dolfin.replace(quad_weight*term.form, replace)

        if adjointer.finished and timestep == adjointer.timestep_count - 1: # we're at the end, and need to add the extra terms
                                                                            # associated with that
          final_interval = slice(timestep_start, timestep_end)
          this_interval = _slice_intersect(final_interval, term.time)
          if this_interval:
            # Get adj_variables for dependencies. Time level is not yet specified.

            # Dependency replacement dictionary.
            replace={}

            term_deps = _coeffs(adjointer, term.form)
            term_vars = _vars(adjointer, term.form)

            for term_dep, term_var in zip(term_deps,term_vars):
              start_var = self.get_vars(adjointer, timestep, term_var)[1]
              replace[term_dep] = deps[str(start_var)]

            # Trapezoidal rule over given interval.
            quad_weight = 0.5*(this_interval.stop-this_interval.start)
            # Calculate i
            if functional_value is None:
              functional_value = dolfin.replace(quad_weight*term.form, replace)
            else:
              functional_value += dolfin.replace(quad_weight*term.form, replace)

      else:
        # Point evaluation.

        if (term.time>=point_interval.start and term.time < point_interval.stop):
          replace = {}

          term_deps = _coeffs(adjointer, term.form)
          term_vars = _vars(adjointer, term.form)

          for term_dep, term_var in zip(term_deps, term_vars):
            (start, end) = self.get_vars(adjointer, timestep, term_var)
            theta = 1.0 - (term.time - point_interval.start)/(point_interval.stop - point_interval.start)
            replace[term_dep] = theta*deps[str(start)] + (1-theta)*deps[str(end)]

          if functional_value is None:
            functional_value = dolfin.replace(term.form, replace)
          else:
            functional_value += dolfin.replace(term.form, replace)

    if functional_value is not None:
      return dolfin.assemble(functional_value)
    else:
      return 0.0

  def get_vars(self, adjointer, timestep, model):
    # Using the adjointer, get the start and end variables associated
    # with this timestep
    start = model.copy()
    end = model.copy()

    if timestep == 0:
      start.timestep = 0
      start.iteration = 0
      end.timestep = 0
      end.iteration = end.iteration_count(adjointer) - 1
    else:
      start.timestep = timestep - 1
      start.iteration = start.iteration_count(adjointer) - 1
      end.timestep = timestep
      end.iteration = end.iteration_count(adjointer) - 1

    return (start, end)


  def derivative(self, adjointer, variable, dependencies, values):

    raise Exception("Not implemented")

  def dependencies(self, adjointer, timestep):

    if adjglobals.adj_variables.libadjoint_timestep == 0:
      dolfin.info_red("Warning: instantiating a TimeFunctional without having called adj_inc_timestep. This probably won't work.")

    point_deps=set()
    integral_deps=set()

    point_interval=slice(adjointer.time.time_levels[timestep],
                         adjointer.time.time_levels[timestep+1])

    integral_interval=slice(adjointer.time.time_levels[max(timestep-1,0)],
                         adjointer.time.time_levels[timestep+1])


    for term in self.timeform.terms:
      if isinstance(term.time, slice):
        # Integral.

        # Get adj_variables for dependencies. Time level is not yet specified.

        if _slice_intersect(integral_interval, term.time):
          integral_deps.update(_vars(adjointer, term.form))
        
      else:
        # Point evaluation.

        if (term.time>=point_interval.start and term.time < point_interval.stop):
          point_deps.update(_vars(adjointer, term.form))
        
    integral_deps=list(integral_deps)
    point_deps=list(point_deps)

    # Set the time level of the dependencies:
    
    # Point deps always need the current and previous timestep values.
    point_deps *= 2
    for i in range(len(point_deps)/2):
      point_deps[i]= point_deps[i].copy()
      if timestep !=0:
        point_deps[i].var.timestep = timestep-1
        point_deps[i].var.iteration = point_deps[i].iteration_count(adjointer) - 1 
      else:
        point_deps[i].var.timestep = timestep
        point_deps[i].var.iteration = 0
    for i in range(len(point_deps)/2, len(point_deps)):
      point_deps[i].var.timestep = timestep
      point_deps[i].var.iteration = point_deps[i].iteration_count(adjointer) - 1 

    # Integral deps depend on the previous time level.
    for i in range(len(integral_deps)):
      if timestep !=0:
        integral_deps[i].var.timestep = timestep - 1
        integral_deps[i].var.iteration = integral_deps[i].iteration_count(adjointer) - 1 
      else:
        integral_deps[i].var.timestep = timestep
        integral_deps[i].var.iteration = 0
      
    # Except at the final timestep, integrals only depend on the previous
    # value.
    if  timestep==adjointer.timestep_count-1 and adjointer.time.finished:
      integral_deps*=2
      for i in range(len(integral_deps)/2, len(integral_deps)):
        integral_deps[i]= integral_deps[i].copy()
        integral_deps[i].var.timestep = timestep
        integral_deps[i].var.iteration = integral_deps[i].iteration_count(adjointer) - 1 
    
    deps=set(point_deps).union(set(integral_deps))

    return list(deps)

  def __str__(self):
    if self.name is not None:
      return self.name
    else:
      formstr = " ".join([str(term) for term in self.timeform.terms])
      return hashlib.md5(formstr).hexdigest()

def _slice_intersect(slice1, slice2):

  if slice1.stop>=slice2.start and slice2.stop>slice1.start:
    intersect=slice(max(slice1.start, slice2.start),min(slice1.stop,slice2.stop))
    if intersect.start==intersect.stop:
      # A zero length intersect doesn't count.
      return None
    else:
      return intersect
  else:
    return None
  
def _vars(adjointer, form):
  # Return the libadjoint variables corresponding to the coeffs in form.
  return [adjglobals.adj_variables[coeff].copy() 
          for coeff in ufl.algorithms.extract_coefficients(form) 
          if (hasattr(coeff, "function_space")) and adjointer.variable_known(adjglobals.adj_variables[coeff])]

def _coeffs(adjointer, form):
  return [coeff 
          for coeff in ufl.algorithms.extract_coefficients(form) 
          if (hasattr(coeff, "function_space")) and adjointer.variable_known(adjglobals.adj_variables[coeff])]
