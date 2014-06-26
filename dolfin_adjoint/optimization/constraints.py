"""This module offers a standard interface for control constraints,
that can be used with different optimisation algorithms."""

from ..utils import gather
from numpy import append

class Constraint(object):
  def function(self, m):
    """Return a vector-like object (numpy array or dolfin Vector), which must be zero for the point to be feasible."""

    raise NotImplementedError, "Constraint.function must be supplied"

  def jacobian(self, m):
    """Returns the full Jacobian matrix as a list of vector-like objects representing the gradient of the constraint function with respect to the parameter m.

       The objects returned must be of the same type as m's data."""

    raise NotImplementedError, "Constraint.jacobian not implemented"

  def jacobian_action(self, m, dm, result):
    """Computes the Jacobian action in direction dm and stores the result in result. """ 

    raise NotImplementedError, "Constraint.jacobian_action is not implemented"

  def jacobian_adjoint_action(self, m, dp, result):
    """Computes the Jacobian adjoint action in direction dp and stores the result in result. """ 

    raise NotImplementedError, "Constraint.jacobian_adjoint_action is not implemented"

  def hessian_action(self, m, dm, dp, result):
    """Computes the Hessian action in direction dm and dp and stores the result in result. """ 

    raise NotImplementedError, "Constraint.hessian_action is not implemented"

  def length(self):
    """Return the number of constraints (len(function(m)))."""

    raise NotImplementedError, "Constraint.length must be supplied"

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

class MergedConstraints(Constraint):
  def __init__(self, constraints):
    self.constraints = constraints

  def function(self, m):
    return reduce(append, [gather(c.function(m)) for c in self.constraints], [])

  def jacobian(self, m):
    return reduce(append, [gather(c.jacobian(m)) for c in self.constraints], [])

  def jacobian_action(self, m, dm, result):
    start = 0
    stop  = 0
    for c in self.constraints:
      stop += c.length()
      c.jacobian_action(m, dm, result[start:stop])
      start = stop

  def jacobian_adjoint_action(self, m, dp, result):
    start = 0
    stop  = 0
    for c in self.constraints:
      stop += c.length()
      c.jacobian_adjoint_action(m, dp[start:stop], result)
      start = stop

  def hessian_action(self, m, dm, dp, result):
    start = 0
    stop  = 0
    for c in self.constraints:
      stop += c.length()
      c.hessian_action(m, dm, dp[start:stop], result)
      start = stop

  def __iter__(self):
    return iter(self.constraints)

  def length(self):
    return sum(c.length() for c in self.constraints)

  def equality_constraints(self):
    ''' Filters out the equality constraints '''
    constraints = [c for c in self.constraints if isinstance(c, EqualityConstraint)]
    return MergedConstraints(constraints)

  def inequality_constraints(self):
    ''' Filters out the inequality constraints '''
    constraints = [c for c in self.constraints if isinstance(c, InequalityConstraint)]
    return MergedConstraints(constraints)

def canonicalise(constraints):
  if constraints is None:
    return None

  if isinstance(constraints, MergedConstraints):
    return constraints

  if not isinstance(constraints, list):
    return MergedConstraints([constraints])

  else:
    return MergedConstraints(constraints)
