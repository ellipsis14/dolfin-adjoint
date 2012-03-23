import libadjoint
from solving import adj_variables, Vector

class InitialConditionParameter(libadjoint.Parameter):
  '''This Parameter is used as input to the tangent linear model (TLM)
  when one wishes to compute dJ/d(initial condition) in a particular direction (perturbation).'''
  def __init__(self, coeff, perturbation):
    '''coeff: the variable whose initial condition you wish to perturb.
       perturbation: the perturbation direction in which you wish to compute the gradient. Must be a Function.'''
    self.var = adj_variables[coeff]
    self.var.c_object.timestep = 0 # we want to put in the source term only at the initial condition.
    self.var.c_object.iteration = 0 # we want to put in the source term only at the initial condition.
    self.perturbation = Vector(perturbation).duplicate()
    self.perturbation.axpy(1.0, Vector(perturbation))

  def __call__(self, dependencies, values, variable):
    # The TLM source term only kicks in at the start, for the initial condition:
    if self.var == variable:
      return self.perturbation
    else:
      return None

  def __str__(self):
    return self.var.name + ':InitialCondition'

