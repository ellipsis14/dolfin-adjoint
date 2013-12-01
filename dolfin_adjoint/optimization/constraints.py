"""This module offers a standard interface for control constraints,
that can be used with different optimisation algorithms."""

class Constraint(object):
  def function(self, m):
    """Return a vector-like object (numpy array or dolfin Vector), which must be zero for the point to be feasible."""

    raise Exception, "Constraint.function must be supplied"

  def jacobian(self, m):
    """Return a list of vector-like objects representing the gradient of the constraint function with respect to the parameter m.

       The objects returned must be of the same type as m's data."""

    raise Exception, "Constraint.jacobian must be supplied"

  def length(self):
    """Return the number of constraints (len(function(m)))."""

  def __len__(self):
    return self.length()

class EqualityConstraint(Constraint):
  """This class represents equality constraints of the form

  c_i(m) == 0

  for 0 <= i < n, where m is the parameter.
  """

class InequalityConstraint(Constraint):
  """This class represents constraints of the form

  c_i(m) >= 0

  for 0 <= i < n, where m is the parameter.
  """

class MergedConstrants(Constraint):
  def __init__(self, constraints):
    self.constraints = constraints

  def function(self, m):
    return sum([list(c.function(m)) for c in self.constraints], [])

  def jacobian(self, m):
    return sum([list(c.jacobian(m)) for c in self.constraints], [])

  def __iter__(self):
    return iter(self.constraints)

  def length(self):
    return sum(c.length() for c in self.constraints)

def canonicalise(constraints):
  if constraints is None:
    return None

  if isinstance(constraints, MergedConstrants):
    return constraints

  if not isinstance(constraints, list):
    return MergedConstrants([constraints])

  else:
    return MergedConstrants(constraints)
