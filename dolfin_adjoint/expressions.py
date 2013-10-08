import firedrake
import collections
import copy

# Our equation may depend on Expressions, and those Expressions may have parameters 
# (e.g. for time-dependent boundary conditions).
# In order to successfully replay the forward solve, we need to keep those parameters around.
# Here, we overload the Expression class to record all of the parameters

expression_attrs = collections.defaultdict(set)

def update_expressions(d):
  pass

def freeze_dict():
  new_dict = {}
  for expression in expression_attrs:
    attr_list = expression_attrs[expression]
    new_dict[expression] = {}

    for attr in attr_list:
      new_dict[expression][attr] = copy.copy(getattr(expression, attr))

  return new_dict
