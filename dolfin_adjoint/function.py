import dolfin
import ufl
from solving import solve, annotate as solving_annotate
import libadjoint
import assign
import adjlinalg
import adjglobals

dolfin_assign = dolfin.Function.assign
dolfin_split  = dolfin.Function.split

def dolfin_adjoint_assign(self, other, annotate=True):
  '''We also need to monkeypatch the Function.assign method, as it is often used inside 
  the main time loop, and not annotating it means you get the adjoint wrong for totally
  nonobvious reasons. If anyone objects to me monkeypatching your objects, my apologies
  in advance.'''

  if self is other:
    return

  # ignore anything not a dolfin.Function
  if not isinstance(other, dolfin.Function) or annotate is False or dolfin.parameters["adjoint"]["stop_annotating"]:
    return dolfin_assign(self, other)

  # ignore anything that is an interpolation, rather than a straight assignment
  if str(self.function_space()) != str(other.function_space()):
    return dolfin_assign(self, other)

  other_var = adjglobals.adj_variables[other]
  self_var = adjglobals.adj_variables[self]
  # ignore any functions we haven't seen before -- we DON'T want to
  # annotate the assignment of initial conditions here. That happens
  # in the main solve wrapper.
  if not adjglobals.adjointer.variable_known(other_var) and not adjglobals.adjointer.variable_known(self_var):
    adjglobals.adj_variables.forget(other)
    adjglobals.adj_variables.forget(self)
    if hasattr(other, "split"):
      if other.split is True:
        errmsg = '''Cannot use Function.split() (yet). To adjoint this, we need functionality
        not yet present in DOLFIN. See https://bugs.launchpad.net/dolfin/+bug/891127 .

        Your model may well work if you use split(func) instead of func.split().'''
        raise libadjoint.exceptions.LibadjointErrorNotImplemented(errmsg)

    return dolfin_assign(self, other)

  # OK, so we have a variable we've seen before. Beautiful.
  out = dolfin_assign(self, other)
  assign.register_assign(self, other)
  return out

def dolfin_adjoint_split(self, *args, **kwargs):
  out = dolfin_split(self, *args, **kwargs)
  for i, fn in enumerate(out):
    fn.split = True
    fn.split_fn = self
    fn.split_i  = i
    fn.split_args = args
    fn.split_kwargs = kwargs

  return out

class Function(dolfin.Function):
  '''The Function class is overloaded so that you can give :py:class:`Functions` *names*. For example,

    .. code-block:: python

      u = Function(V, name="Velocity")

    This allows you to refer to the :py:class:`Function` by name throughout dolfin-adjoint, rather than
    needing to have the specific :py:class:`Function` instance available.

    For more details, see :doc:`the dolfin-adjoint documentation </documentation/misc>`.'''

  def __init__(self, *args, **kwargs):
    if "name" in kwargs:
      self.adj_name = kwargs["name"]
      del kwargs["name"]

    annotate = False
    if 'annotate' in kwargs:
      annotate = kwargs['annotate']
      del kwargs['annotate']
    if dolfin.parameters["adjoint"]["stop_annotating"]:
      annotate = False

    dolfin.Function.__init__(self, *args, **kwargs)

    if isinstance(args[0], dolfin.Function) and annotate:
      other_var = adjglobals.adj_variables[args[0]]
      if adjglobals.adjointer.variable_known(other_var):
        assign.register_assign(self, args[0])

  def assign(self, other, annotate=True):
    '''To disable the annotation, just pass :py:data:`annotate=False` to this routine, and it acts exactly like the
    Dolfin assign call.'''

    return dolfin_adjoint_assign(self, other, annotate=annotate)

  def split(self, *args, **kwargs):
    return dolfin_adjoint_split(self, *args, **kwargs)

  def __str__(self):
    if hasattr(self, "adj_name"):
      return self.adj_name
    else:
      return dolfin.Function.__str__(self)

dolfin.Function.assign = dolfin_adjoint_assign # so that Functions produced inside Expression etc. get it too
dolfin.Function.split  = dolfin_adjoint_split

