import dolfin

constant_values = {}
constant_objects = {}

class Constant(dolfin.Constant):
  '''The Constant class is overloaded so that you can give :py:class:`Constants` *names*. For example,

    .. code-block:: python

      nu = Constant(1.0e-4, name="Diffusivity")

    This allows you to refer to the :py:class:`Constant` by name throughout dolfin-adjoint, rather than
    needing to have the specific :py:class:`Constant` instance available.

    For more details, see :doc:`the dolfin-adjoint documentation </documentation/misc>`.'''

  def __init__(self, value, cell=None, name=None):
    dolfin.Constant.__init__(self, value, cell)
    if name is not None:
      self.adj_name = name

      if name in constant_values:
        dolfin.info_red("Warning: redefing constant with name %s" % name)

      constant_values[name] = value
      constant_objects[name] = self

def get_constant(a):
  if isinstance(a, Constant):
    return a
  else:
    return constant_objects[a]
