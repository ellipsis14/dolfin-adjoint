import libadjoint
import ufl
import dolfin
import hashlib

import solving
import adjglobals
import adjlinalg
from timeforms import NoTime, StartTimeConstant, FinishTimeConstant

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
    
    functional_value = self._substitute_form(adjointer, timestep, dependencies, values)

    if functional_value is not None:
      return dolfin.assemble(functional_value)
    else:
      return 0.0

  def derivative(self, adjointer, variable, dependencies, values):
    
    functional_value = None
    for timestep in self._derivative_timesteps(adjointer, variable):
      functional_value = _add(functional_value,
                              self._substitute_form(adjointer, timestep, dependencies, values))

    d = dolfin.derivative(functional_value, values[dependencies.index(variable)].data)
    d = ufl.algorithms.expand_derivatives(d)
    return adjlinalg.Vector(d)

  def _derivative_timesteps(self, adjointer, variable):
    
    timestep = variable.timestep
    iteration = variable.iteration
    if timestep == 0 and iteration == 0:
      return [0]
    elif timestep == adjointer.timestep_count-1:
      return [timestep]
    else:
      return [timestep, timestep+1]

  def _substitute_form(self, adjointer, timestep, dependencies, values):
    ''' Perform the substitution of the dependencies and values
    provided. This is common to __call__ and __derivative__'''

    deps={str(dep): val.data for dep, val in zip(dependencies, values)}

    functional_value = None
    final_time = _time_levels(adjointer, adjointer.timestep_count - 1)[1]

    # Get the necessary timestep information about the adjointer.
    # For integrals, we're integrating over /two/ timesteps.
    timestep_start, timestep_end = _time_levels(adjointer, timestep)
    point_interval = slice(timestep_start, timestep_end)

    if timestep > 0:
      prev_start, prev_end = _time_levels(adjointer, timestep - 1)
      integral_interval = slice(prev_start, timestep_end)
    else:
      integral_interval = slice(timestep_start, timestep_end)

    for term in self.timeform.terms:
      if isinstance(term.time, slice):
        # Integral.

        def trapezoidal(interval, iteration):
          # Evaluate the trapezoidal rule for the given interval and
          # iteration. Iteration is used to select start and end of timestep.
          this_interval=_slice_intersect(interval, term.time)
          if this_interval:
            # Get adj_variables for dependencies. Time level is not yet specified.
            
            # Dependency replacement dictionary.
            replace={}

            term_deps = _coeffs(adjointer, term.form)
            term_vars = _vars(adjointer, term.form)

            for term_dep, term_var in zip(term_deps,term_vars):
              this_var = self.get_vars(adjointer, timestep, term_var)[iteration]
              replace[term_dep] = deps[str(this_var)]

            # Trapezoidal rule over given interval.
            quad_weight = 0.5*(this_interval.stop-this_interval.start)
            
            return dolfin.replace(quad_weight*term.form, replace)

        # Calculate the integral contribution from the previous time level.
        functional_value = _add(functional_value, trapezoidal(integral_interval, 0))

        # On the final occasion, also calculate the contribution from the
        # current time leve.
        if adjointer.finished and timestep == adjointer.timestep_count - 1: # we're at the end, and need to add the extra terms
                                                                            # associated with that
          final_interval = slice(timestep_start, timestep_end)
          functional_value = _add(functional_value, trapezoidal(final_interval, 1))

      else:
        # Point evaluation.

        if point_interval.start <= term.time < point_interval.stop:
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

        # Special case for evaluation at the end of time: we can't pass over to the
        # right-hand timestep, so have to do it here.
        elif (term.time == final_time or isinstance(term.time, FinishTimeConstant)) and point_interval.stop == final_time:
          replace = {}

          term_deps = _coeffs(adjointer, term.form)
          term_vars = _vars(adjointer, term.form)

          for term_dep, term_var in zip(term_deps, term_vars):
            end = self.get_vars(adjointer, timestep, term_var)[1]
            replace[term_dep] = deps[str(end)]

          if functional_value is None:
            functional_value = dolfin.replace(term.form, replace)
          else:
            functional_value += dolfin.replace(term.form, replace)
    
    return functional_value

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

  def dependencies(self, adjointer, timestep):

    point_deps = set()
    integral_deps = set()
    final_deps = set()

    levels = _time_levels(adjointer, timestep)
    point_interval=slice(levels[0],levels[1])
    final_time = _time_levels(adjointer, adjointer.timestep_count - 1)[1]

    prev_levels = _time_levels(adjointer, max(timestep-1,0))
    integral_interval=slice(prev_levels[0],levels[1])

    for term in self.timeform.terms:
      if isinstance(term.time, slice):
        # Integral.

        # Get adj_variables for dependencies. Time level is not yet specified.

        if _slice_intersect(integral_interval, term.time):
          integral_deps.update(_vars(adjointer, term.form))
        
      else:
        # Point evaluation.

        if point_interval.start <= term.time < point_interval.stop:
          point_deps.update(_vars(adjointer, term.form))

        # Special case for evaluation at the end of time: we can't pass over to the
        # right-hand timestep, so have to do it here.
        elif (term.time == final_time or isinstance(term.time, FinishTimeConstant)) and point_interval.stop == final_time:
          final_deps.update(_vars(adjointer, term.form))
        
    integral_deps = list(integral_deps)
    point_deps = list(point_deps)
    final_deps = list(final_deps)

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

    # Final deps depend only at the very last value.
    for i in range(len(final_deps)):
      final_deps[i].timestep = timestep
      final_deps[i].iteration = final_deps[i].iteration_count(adjointer) - 1
    
    deps=set(point_deps).union(set(integral_deps)).union(set(final_deps))

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

def _add(value, increment):
  # Add increment to value correctly taking into account None.
  if increment is None:
    return value
  elif value is None:
    return increment
  else:
    return value+increment

def _time_levels(adjointer, timestep):
  
  try:
    return adjointer.get_times(timestep)
  except Exception as exc:
    start = NoTime(str(exc))
    end = NoTime(str(exc))

    if timestep == adjointer.timestep_count - 1:
      end = FinishTimeConstant()

    if timestep == 0:
      start = StartTimeConstant()

    return (start, end)
