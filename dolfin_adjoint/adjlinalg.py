import libadjoint
import dolfin
import ufl
import adjglobals
import os
import os.path
import numpy
import misc

class Vector(libadjoint.Vector):
  '''This class implements the libadjoint.Vector abstract base class for the Dolfin adjoint.
  In particular, it must implement the data callbacks for tasks such as adding two vectors
  together, duplicating vectors, taking norms, etc., that occur in the process of constructing
  the adjoint equations.'''

  def __init__(self, data, zero=False, fn_space=None):


    self.data=data
    if not (self.data is None or isinstance(self.data, dolfin.Function) or isinstance(self.data, ufl.Form)):
      dolfin.info_red("Got " + str(self.data.__class__) + " as input to the Vector() class. Don't know how to handle that.")
      raise AssertionError

    # self.zero is true if we can prove that the vector is zero.
    if data is None:
      self.zero=True
    else:
      self.zero=zero

    if fn_space is not None:
      self.fn_space = fn_space

  def duplicate(self):

    if isinstance(self.data, ufl.form.Form):
      # The data type will be determined by the first addto.
      data = None
    elif isinstance(self.data, dolfin.Function):
      try:
        fn_space = self.data.function_space().collapse()
      except:
        fn_space = self.data.function_space()
      data = dolfin.Function(fn_space)
    else:
      data = None

    fn_space = None
    if hasattr(self, "fn_space"):
      fn_space = self.fn_space

    return Vector(data, zero=True, fn_space=fn_space)

  def axpy(self, alpha, x):

    if hasattr(x, 'nonlinear_form'):
      self.nonlinear_form = x.nonlinear_form
      self.nonlinear_u = x.nonlinear_u
      self.nonlinear_bcs = x.nonlinear_bcs

    if x.zero:
      return

    if (self.data is None):
      # self is an empty form.
      if isinstance(x.data, dolfin.Function):
        self.data = dolfin.Function(x.data)
        self.data.vector()._scale(alpha)
      else:
        self.data=alpha*x.data

    elif x.data is None:
      pass
    elif isinstance(self.data, dolfin.Coefficient):
      if isinstance(x.data, dolfin.Coefficient):
        self.data.vector().axpy(alpha, x.data.vector())
      else:
        # This occurs when adding a RHS derivative to an adjoint equation
        # corresponding to the initial conditions.
        self.data.vector().axpy(alpha, dolfin.assemble(x.data))
    elif isinstance(x.data, ufl.form.Form) and isinstance(self.data, ufl.form.Form):

      # Let's do a bit of argument shuffling, shall we?
      xargs = ufl.algorithms.extract_arguments(x.data)
      sargs = ufl.algorithms.extract_arguments(self.data)

      if xargs != sargs:
        # OK, let's check that all of the function spaces are happy and so on.
        for i in range(len(xargs)):
          assert xargs[i].element() == sargs[i].element()
          assert xargs[i].function_space() == sargs[i].function_space()

        # Now that we are happy, let's replace the xargs with the sargs ones.
        x_form = dolfin.replace(x.data, dict(zip(xargs, sargs)))
      else:
        x_form = x.data

      self.data+=alpha*x_form
    elif isinstance(self.data, ufl.form.Form) and isinstance(x.data, dolfin.Function):
      x_vec = x.data.vector().copy()
      self_vec = dolfin.assemble(self.data)
      self_vec.axpy(alpha, x_vec)
      new_fn = dolfin.Function(x.data.function_space())
      new_fn.vector()[:] = self_vec
      self.data = new_fn
      self.fn_space = self.data.function_space()

    else:
      print "self.data.__class__: ", self.data.__class__
      print "x.data.__class__: ", x.data.__class__
      assert False

    self.zero = False

  def norm(self):

    if isinstance(self.data, dolfin.Function):
      return (abs(dolfin.assemble(dolfin.inner(self.data, self.data)*dolfin.dx)))**0.5
    elif isinstance(self.data, ufl.form.Form):
      return dolfin.assemble(self.data).norm("l2")

  def dot_product(self,y):

    if isinstance(self.data, ufl.form.Form):
      return dolfin.assemble(dolfin.inner(self.data, y.data)*dolfin.dx)
    elif isinstance(self.data, dolfin.Function):
      if isinstance(y.data, ufl.form.Form):
        other = dolfin.assemble(y.data)
      else:
        other = y.data.vector()
      return self.data.vector().inner(other)
    else:
      raise libadjoint.exceptions.LibadjointErrorNotImplemented("Don't know how to dot anything else.")

  def set_random(self):
    assert isinstance(self.data, dolfin.Function) or hasattr(self, "fn_space")

    if self.data is None:
      self.data = dolfin.Function(self.fn_space)

    vec = self.data.vector()
    for i in range(len(vec)):
      vec[i] = random.random()

    self.zero = False

  def size(self):
    if hasattr(self, "fn_space") and self.data is None:
      self.data = dolfin.Function(self.fn_space)

    if isinstance(self.data, dolfin.Function):
      return self.data.vector().local_size()

    if isinstance(self.data, ufl.form.Form):
      return dolfin.assemble(self.data).local_size()

    raise libadjoint.exceptions.LibadjointErrorNotImplemented("Don't know how to get the size.")

  def set_values(self, array):
    if isinstance(self.data, dolfin.Function):
      vec = self.data.vector()
      vec.set_local(array)
      vec.apply("insert")

      self.zero = False
    elif self.data is None and hasattr(self, 'fn_space'):
      self.data = dolfin.Function(self.fn_space)
      vec = self.data.vector()
      vec.set_local(array)
      self.zero = False
    else:
      raise libadjoint.exceptions.LibadjointErrorNotImplemented("Don't know how to set values.")
  
  def get_values(self, array):
    if isinstance(self.data, dolfin.Function):
      vec = self.data.vector()
      try:
        vec.get_local(array)
      except NotImplementedError:
        array[:] = vec.get_local()

    else:
      raise libadjoint.exceptions.LibadjointErrorNotImplemented("Don't know how to get values.")
  
  def write(self, var):
    filename = str(var)
    suffix = "xml"
    #if not os.path.isfile(filename+".%s" % suffix):
    #  dolfin.info_red("Warning: Overwriting checkpoint file "+filename+"."+suffix)
    file = dolfin.File(filename+".%s" % suffix)
    file << self.data

    # Save the function space into adjglobals.checkpoint_fs. It will be needed when reading the variable back in.
    adjglobals.checkpoint_fs[filename] = self.data.function_space()

  @staticmethod
  def read(var):

    filename = str(var)
    suffix = "xml"

    V = adjglobals.checkpoint_fs[filename]
    v = dolfin.Function(V, filename+".%s" % suffix)
    return Vector(v)

  @staticmethod
  def delete(var):
    try:
      filename = str(var)
      suffix = "xml"

      import os
      os.remove(filename+".%s" % suffix)
    except OSError:
      pass

class Matrix(libadjoint.Matrix):
  '''This class implements the libadjoint.Matrix abstract base class for the Dolfin adjoint.
  In particular, it must implement the data callbacks for tasks such as adding two matrices
  together, duplicating matrices, etc., that occur in the process of constructing
  the adjoint equations.'''

  def __init__(self, data, bcs=None, solver_parameters=None, adjoint=None):

    if bcs is None:
      self.bcs = []
    else:
      self.bcs=bcs

    self.data=data

    if solver_parameters is not None:
      self.solver_parameters = solver_parameters
    else:
      self.solver_parameters = {}

  def basic_solve(self, var, b):

    if isinstance(self.data, IdentityMatrix):
      x=b.duplicate()
      x.axpy(1.0, b)
    else:
      if var.type in ['ADJ_TLM', 'ADJ_ADJOINT', 'ADJ_SOA']:
        dirichlet_bcs = [dolfin.homogenize(bc) for bc in self.bcs if isinstance(bc, dolfin.DirichletBC)]
        other_bcs  = [bc for bc in self.bcs if not isinstance(bc, dolfin.DirichletBC)]
        bcs = dirichlet_bcs + other_bcs
      else:
        bcs = self.bcs

      x = Vector(dolfin.Function(self.test_function().function_space()))

      if b.data is None and not hasattr(b, 'nonlinear_form'):
        # This means we didn't get any contribution on the RHS of the adjoint system. This could be that the
        # simulation ran further ahead than when the functional was evaluated, or it could be that the
        # functional is set up incorrectly.
        dolfin.info_red("Warning: got zero RHS for the solve associated with variable %s" % var)
      elif isinstance(b.data, dolfin.Function):

        assembled_lhs = dolfin.assemble(self.data)
        [bc.apply(assembled_lhs) for bc in bcs]
        assembled_rhs = dolfin.Function(b.data).vector()
        [bc.apply(assembled_rhs) for bc in bcs]

        pc = self.solver_parameters.get("preconditioner", "default")
        ksp = self.solver_parameters.get("linear_solver", "default")
        dolfin.fem.solving.solve(assembled_lhs, x.data.vector(), assembled_rhs, ksp, pc)
      else:
        if hasattr(b, 'nonlinear_form'): # was a nonlinear solve
          x.data.vector()[:] = b.nonlinear_u.vector()
          F = dolfin.replace(b.nonlinear_form, {b.nonlinear_u: x.data})
          dolfin.fem.solving.solve(F == 0, x.data, b.nonlinear_bcs, solver_parameters=self.solver_parameters)
        else:
          assembled_lhs = dolfin.assemble(self.data)
          [bc.apply(assembled_lhs) for bc in bcs]
          assembled_rhs = dolfin.assemble(b.data)
          [bc.apply(assembled_rhs) for bc in bcs]

          pc = self.solver_parameters.get("preconditioner", "default")
          ksp = self.solver_parameters.get("linear_solver", "default")
          dolfin.fem.solving.solve(assembled_lhs, x.data.vector(), assembled_rhs, ksp, pc)

    return x

  def caching_solve(self, var, b):
    if isinstance(self.data, IdentityMatrix):
        output = b.duplicate()
        output.axpy(1.0, b)
    else:
        dirichlet_bcs = [dolfin.homogenize(bc) for bc in self.bcs if isinstance(bc, dolfin.DirichletBC)]
        other_bcs  = [bc for bc in self.bcs if not isinstance(bc, dolfin.DirichletBC)]
        bcs = dirichlet_bcs + other_bcs

        output = Vector(dolfin.Function(self.test_function().function_space()))
        if isinstance(b.data, ufl.Form):
            assembled_rhs = dolfin.assemble(b.data)
        else:
            assembled_rhs = b.data.vector()
        [bc.apply(assembled_rhs) for bc in bcs]

        if not var in adjglobals.lu_solvers:
          if dolfin.parameters["adjoint"]["debug_cache"]:
            dolfin.info_red("Got a cache miss for %s" % var)
          assembled_lhs = dolfin.assemble(self.data)
          [bc.apply(assembled_lhs) for bc in bcs]
          adjglobals.lu_solvers[var] = dolfin.LUSolver(assembled_lhs, "mumps")
          adjglobals.lu_solvers[var].parameters["reuse_factorization"] = True
        else:
          if dolfin.parameters["adjoint"]["debug_cache"]:
            dolfin.info_green("Got a cache hit for %s" % var)

        adjglobals.lu_solvers[var].solve(output.data.vector(), assembled_rhs)

    return output

  def solve(self, var, b):
    if dolfin.parameters["adjoint"]["cache_factorizations"] and var.type != "ADJ_FORWARD":
      x = self.caching_solve(var, b)
    else:
      x = self.basic_solve(var, b)

    return x

  def action(self, x, y):
    assert isinstance(x.data, dolfin.Function)
    assert isinstance(y.data, dolfin.Function)

    action_form = dolfin.action(self.data, x.data)
    action_vec  = dolfin.assemble(action_form)
    y.data.vector()[:] = action_vec

  def axpy(self, alpha, x):
    assert isinstance(x.data, ufl.Form)
    assert isinstance(self.data, ufl.Form)

    # Let's do a bit of argument shuffling, shall we?
    xargs = ufl.algorithms.extract_arguments(x.data)
    sargs = ufl.algorithms.extract_arguments(self.data)

    if xargs != sargs:
      # OK, let's check that all of the function spaces are happy and so on.
      for i in range(len(xargs)):
        assert xargs[i].element() == sargs[i].element()
        assert xargs[i].function_space() == sargs[i].function_space()

      # Now that we are happy, let's replace the xargs with the sargs ones.
      x_form = dolfin.replace(x.data, dict(zip(xargs, sargs)))
    else:
      x_form = x.data

    self.data+=alpha*x_form
    self.bcs += x.bcs # Err, I hope they are compatible ...
    self.bcs = misc.uniq(self.bcs)

  def test_function(self):
    '''test_function(self)

    Return the ufl.Argument corresponding to the trial space for the form'''

    return ufl.algorithms.extract_arguments(self.data)[-1]

class IdentityMatrix(object):
  '''Placeholder object for identity matrices'''
  pass

