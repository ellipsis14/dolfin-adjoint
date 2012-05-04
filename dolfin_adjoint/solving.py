import ufl
import ufl.classes
import ufl.algorithms
import ufl.operators

import dolfin.fem.solving
import dolfin

import libadjoint
import libadjoint.exceptions

import hashlib
import time
import copy
import os
import os.path
import random

import numpy
import scipy
import scipy.linalg

import assembly
import expressions
import coeffstore
import adjrhs
import adjglobals
import adjlinalg

def annotate(*args, **kwargs):
  '''This routine handles all of the annotation, recording the solves as they
  happen so that libadjoint can rewind them later.'''

  if 'matrix_class' in kwargs:
    matrix_class = kwargs['matrix_class']
    del kwargs['matrix_class']
  else:
    matrix_class = adjlinalg.Matrix

  if 'initial_guess' in kwargs:
    initial_guess = kwargs['initial_guess']
    del kwargs['initial_guess']
  else:
    initial_guess = False

  replace_map = False
  if 'replace_map' in kwargs:
    replace_map = kwargs['replace_map']
    del kwargs['replace_map']

  if isinstance(args[0], ufl.classes.Equation):
    # annotate !

    # Unpack the arguments, using the same routine as the real Dolfin solve call
    unpacked_args = dolfin.fem.solving._extract_args(*args, **kwargs)
    eq = unpacked_args[0]
    u  = unpacked_args[1]
    bcs = unpacked_args[2]
    J = unpacked_args[3]
    solver_parameters = copy.deepcopy(unpacked_args[7])

    if isinstance(eq.lhs, ufl.Form) and isinstance(eq.rhs, ufl.Form):
      eq_lhs = eq.lhs
      eq_rhs = eq.rhs
      eq_bcs = bcs
      linear = True
    else:
      eq_lhs, eq_rhs = define_nonlinear_equation(eq.lhs, u)
      F = eq.lhs
      eq_bcs = []
      linear = False

  elif isinstance(args[0], dolfin.cpp.Matrix):
    linear = True
    try:
      eq_lhs = args[0].form
    except KeyError:
      raise libadjoint.exceptions.LibadjointErrorInvalidInputs("dolfin_adjoint did not assemble your form, and so does not recognise your matrix. Did you from dolfin_adjoint import *?")

    try:
      eq_rhs = args[2].form
    except KeyError:
      raise libadjoint.exceptions.LibadjointErrorInvalidInputs("dolfin_adjoint did not assemble your form, and so does not recognise your right-hand side. Did you from dolfin_adjoint import *?")

    u = args[1]
    assert isinstance(u, dolfin.cpp.GenericVector)
    u = u.function

    solver_parameters = {}

    try:
      solver_parameters["linear_solver"] = args[3]
    except IndexError:
      pass

    try:
      solver_parameters["preconditioner"] = args[4]
    except IndexError:
      pass

    try:
      eq_bcs = list(set(args[0].bcs + args[2].bcs))
    except AttributeError:
      assert not hasattr(args[0], 'bcs') and not hasattr(args[2], 'bcs')
      eq_bcs = []
  else:
    raise libadjoint.exceptions.LibadjointErrorNotImplemented("Don't know how to annotate your equation, sorry!")

  # Suppose we are solving for a variable w, and that variable shows up in the
  # coefficients of eq_lhs/eq_rhs.
  # Does that mean:
  #  a) the /previous value/ of that variable, and you want to timestep?
  #  b) the /value to be solved for/ in this solve?
  # i.e. is it timelevel n-1, or n?
  # if Dolfin is doing a linear solve, we want case a);
  # if Dolfin is doing a nonlinear solve, we want case b).
  # so if we are doing a nonlinear solve, we bump the timestep number here
  # /before/ we map the coefficients -> dependencies,
  # so that libadjoint records the dependencies with the right timestep number.
  if not linear:
    var = adjglobals.adj_variables.next(u)
  else:
    var = None

  # Set up the data associated with the matrix on the left-hand side. This goes on the diagonal
  # of the 'large' system that incorporates all of the timelevels, which is why it is prefixed
  # with diag.
  diag_name = hashlib.md5(str(eq_lhs) + str(eq_rhs) + str(u) + str(time.time())).hexdigest() # we don't have a useful human-readable name, so take the md5sum of the string representation of the forms
  diag_deps = [adjglobals.adj_variables[coeff] for coeff in ufl.algorithms.extract_coefficients(eq_lhs) if hasattr(coeff, "function_space")]
  diag_coeffs = [coeff for coeff in ufl.algorithms.extract_coefficients(eq_lhs) if hasattr(coeff, "function_space")]

  if initial_guess and linear: # if the initial guess matters, we're going to have to add this in as a dependency of the system
    initial_guess_var = adjglobals.adj_variables[u]
    diag_deps.append(initial_guess_var)
    diag_coeffs.append(u)

  diag_block = libadjoint.Block(diag_name, dependencies=diag_deps, test_hermitian=dolfin.parameters["adjoint"]["test_hermitian"], test_derivative=dolfin.parameters["adjoint"]["test_derivative"])

  # Similarly, create the object associated with the right-hand side data.
  if linear:
    rhs = adjrhs.RHS(eq_rhs)
  else:
    rhs = adjrhs.NonlinearRHS(eq_rhs, F, u, bcs, mass=eq_lhs, solver_parameters=solver_parameters, J=J)


  # We need to check if this is the first equation,
  # so that we can register the appropriate initial conditions.
  # These equations are necessary so that libadjoint can assemble the
  # relevant adjoint equations for the adjoint variables associated with
  # the initial conditions.
  register_initial_conditions(zip(rhs.coefficients(),rhs.dependencies()) + zip(diag_coeffs, diag_deps), linear=linear, var=var)

  # c.f. the discussion above. In the linear case, we want to bump the
  # timestep number /after/ all of the dependencies' timesteps have been
  # computed for libadjoint.
  if linear:
    var = adjglobals.adj_variables.next(u)

  # With the initial conditions out of the way, let us now define the callbacks that
  # define the actions of the operator the user has passed in on the lhs of this equation.

  # Our equation may depend on Expressions, and those Expressions may have parameters 
  # (e.g. for time-dependent boundary conditions).
  # In order to successfully replay the forward solve, we need to keep those parameters around.
  # In expressions.py, we overloaded the Expression class to record all of the parameters
  # as they are set. We're now going to copy that dictionary as it is at the annotation time,
  # so that we can get back to this exact state:
  frozen_expressions_dict = expressions.freeze_dict()

  def diag_assembly_cb(dependencies, values, hermitian, coefficient, context):
    '''This callback must conform to the libadjoint Python block assembly
    interface. It returns either the form or its transpose, depending on
    the value of the logical hermitian.'''

    assert coefficient == 1

    value_coeffs=[v.data for v in values]
    expressions.update_expressions(frozen_expressions_dict)
    eq_l=dolfin.replace(eq_lhs, dict(zip(diag_coeffs, value_coeffs)))

    if hermitian:
      # Homogenise the adjoint boundary conditions. This creates the adjoint
      # solution associated with the lifted discrete system that is actually solved.
      adjoint_bcs = [dolfin.homogenize(bc) for bc in eq_bcs if isinstance(bc, dolfin.DirichletBC)] + [bc for bc in eq_bcs if not isinstance(bc, dolfin.DirichletBC)]
      if len(adjoint_bcs) == 0: adjoint_bcs = None

      kwargs = {}
      kwargs['bcs'] = adjoint_bcs
      kwargs['solver_parameters'] = solver_parameters
      kwargs['adjoint'] = True

      if initial_guess:
        kwargs['initial_guess'] = value_coeffs[dependencies.index(initial_guess_var)]

      if replace_map:
        kwargs['replace_map'] = dict(zip(diag_coeffs, value_coeffs))

      return (matrix_class(dolfin.adjoint(eq_l), **kwargs), adjlinalg.Vector(None, fn_space=u.function_space()))
    else:

      kwargs = {}
      kwargs['bcs'] = eq_bcs
      kwargs['solver_parameters'] = solver_parameters
      kwargs['adjoint'] = False

      if initial_guess:
        kwargs['initial_guess'] = value_coeffs[dependencies.index(initial_guess_var)]

      if replace_map:
        kwargs['replace_map'] = dict(zip(diag_coeffs, value_coeffs))

      return (matrix_class(eq_l, **kwargs), adjlinalg.Vector(None, fn_space=u.function_space()))
  diag_block.assemble = diag_assembly_cb

  def diag_action_cb(dependencies, values, hermitian, coefficient, input, context):
    value_coeffs = [v.data for v in values]
    expressions.update_expressions(frozen_expressions_dict)
    eq_l = dolfin.replace(eq_lhs, dict(zip(diag_coeffs, value_coeffs)))

    if dolfin.parameters["adjoint"]["test_derivative"] is None:
      if hermitian:
        eq_l = dolfin.adjoint(eq_l)

      output_vec = dolfin.assemble(coefficient * dolfin.action(eq_l, input.data))
      output_fn = dolfin.Function(input.data.function_space())
      vec = output_fn.vector()
      for i in range(len(vec)):
        vec[i] = output_vec[i]

      return adjlinalg.Vector(output_fn)
    else:
      # Let's do things the proper way. We're testing the order of convergence of the
      # derivative against this action, so we need to apply the boundary conditions
      # here too.

      import os
      import os.path

      G = dolfin.assemble(eq_l)
      [bc.apply(G) for bc in eq_bcs]

      if hermitian:
        fn_space = ufl.algorithms.extract_arguments(eq_l)[-1].function_space()
        output = dolfin.Function(fn_space)
        G.transpmult(input.data.vector(), output.vector())
      else:
        fn_space = ufl.algorithms.extract_arguments(eq_l)[-2].function_space()
        output = dolfin.Function(fn_space)
        G.mult(input.data.vector(), output.vector())

      return adjlinalg.Vector(output)

  diag_block.action = diag_action_cb

  if len(diag_deps) > 0:
    # If this block is nonlinear (the entries of the matrix on the LHS
    # depend on any variable previously computed), then that will induce
    # derivative terms in the adjoint equations. Here, we define the
    # callback libadjoint will need to compute such terms.
    def derivative_action(dependencies, values, variable, contraction_vector, hermitian, input, coefficient, context):
      dolfin_variable = values[dependencies.index(variable)].data
      dolfin_values = [val.data for val in values]
      expressions.update_expressions(frozen_expressions_dict)

      current_form = dolfin.replace(eq_lhs, dict(zip(diag_coeffs, dolfin_values)))

      deriv = dolfin.derivative(current_form, dolfin_variable)
      args = ufl.algorithms.extract_arguments(deriv)
      deriv = dolfin.replace(deriv, {args[1]: contraction_vector.data}) # contract over the middle index

      # Assemble the G-matrix now, so that we can apply the Dirichlet BCs to it
      if len(ufl.algorithms.extract_arguments(ufl.algorithms.expand_derivatives(coefficient*deriv))) == 0:
        return adjlinalg.Vector(None)

      G = dolfin.assemble(coefficient * deriv)
      # Zero the rows of G corresponding to Dirichlet rows of the form
      bcs = [bc for bc in eq_bcs if isinstance(bc, dolfin.cpp.DirichletBC)]
      for bc in bcs:
        bcvals = bc.get_boundary_values()
        G.zero(numpy.array(bcvals.keys(), dtype='I'))

      if hermitian:
        output = dolfin.Function(dolfin_variable.function_space()) # output lives in the function space of the differentiating variable
        G.transpmult(input.data.vector(), output.vector())
      else:
        output = dolfin.Function(ufl.algorithms.extract_arguments(eq_lhs)[-1].function_space()) # output lives in the function space of the TestFunction
        G.mult(input.data.vector(), output.vector())

      return adjlinalg.Vector(output)
    diag_block.derivative_action = derivative_action

  eqn = libadjoint.Equation(var, blocks=[diag_block], targets=[var], rhs=rhs)

  cs = adjglobals.adjointer.register_equation(eqn)
  do_checkpoint(cs, var, rhs)

  return linear

def solve(*args, **kwargs):
  '''This solve routine comes from the dolfin_adjoint package, and wraps the real Dolfin
  solve call. Its purpose is to annotate the model, recording what solves occur and what
  forms are involved, so that the adjoint model may be constructed automatically by libadjoint. 

  To disable the annotation, just pass annotate=False to this routine, and it acts exactly
  like the Dolfin solve call. This is useful in cases where the solve is known to be irrelevant
  or diagnostic for the purposes of the adjoint computation (such as projecting fields to
  other function spaces for the purposes of visualisation).'''

  # First, decide if we should annotate or not.
  to_annotate = True
  if "annotate" in kwargs:
    to_annotate = kwargs["annotate"]
    del kwargs["annotate"] # so we don't pass it on to the real solver
  if dolfin.parameters["adjoint"]["stop_annotating"]:
    to_annotate = False

  if to_annotate:
    linear = annotate(*args, **kwargs)

  ret = dolfin.fem.solving.solve(*args, **kwargs)

  if to_annotate:
    # Finally, if we want to record all of the solutions of the real forward model
    # (for comparison with a libadjoint replay later),
    # then we should record the value of the variable we just solved for.
    if dolfin.parameters["adjoint"]["record_all"]:
      if isinstance(args[0], ufl.classes.Equation):
        unpacked_args = dolfin.fem.solving._extract_args(*args, **kwargs)
        u  = unpacked_args[1]
        adjglobals.adjointer.record_variable(adjglobals.adj_variables[u], libadjoint.MemoryStorage(adjlinalg.Vector(u)))
      elif isinstance(args[0], dolfin.cpp.Matrix):
        u = args[1].function
        adjglobals.adjointer.record_variable(adjglobals.adj_variables[u], libadjoint.MemoryStorage(adjlinalg.Vector(u)))
      else:
        raise libadjoint.exceptions.LibadjointErrorInvalidInputs("Don't know how to record, sorry")

  return ret

def adj_html(*args, **kwargs):
  '''This routine dumps the current state of the adjglobals.adjointer to a HTML visualisation.
  Use it like:
    adj_html("forward.html", "forward") # for the equations recorded on the forward run
    adj_html("adjoint.html", "adjoint") # for the equations to be assembled on the adjoint run
  '''
  return adjglobals.adjointer.to_html(*args, **kwargs)

def adj_reset():
  adjglobals.adjointer.reset()
  adjglobals.adj_variables.__init__()
  
def define_nonlinear_equation(F, u):
  # Given an F := 0,
  # we write the equation for libadjoint's annotation purposes as
  # M.u = M.u + F(u)
  # as we need to have something on the diagonal in our big time system

  fn_space = u.function_space()
  test = dolfin.TestFunction(fn_space)
  trial = dolfin.TrialFunction(fn_space)

  mass = dolfin.inner(test, trial)*dolfin.dx

  return (mass, dolfin.action(mass, u) - F)

def adj_checkpointing(strategy, steps, snaps_on_disk, snaps_in_ram, verbose=False):
  dolfin.parameters["adjoint"]["record_all"] = False
  adjglobals.adjointer.set_checkpoint_strategy(strategy)
  adjglobals.adjointer.set_revolve_options(steps, snaps_on_disk, snaps_in_ram, verbose)

def register_initial_conditions(coeffdeps, linear, var=None):
  for coeff, dep in coeffdeps:
    # If coeff is not known, it must be an initial condition.
    if not adjglobals.adjointer.variable_known(dep):

      if not linear: # don't register initial conditions for the first nonlinear solve.
        if dep == var:
          continue

      if hasattr(coeff, "split"):
        if coeff.split is True:
          errmsg = '''Cannot use Function.split() (yet). To adjoint this, we need functionality
          not yet present in DOLFIN. See https://bugs.launchpad.net/dolfin/+bug/891127 .

          Your model may well work if you use split(func) instead of func.split().'''
          raise libadjoint.exceptions.LibadjointErrorNotImplemented(errmsg)

      register_initial_condition(coeff, dep)

def register_initial_condition(coeff, dep):
  fn_space = coeff.function_space()
  block_name = "Identity: %s" % str(fn_space)
  if len(block_name) > int(libadjoint.constants.adj_constants["ADJ_NAME_LEN"]):
    block_name = block_name[0:int(libadjoint.constants.adj_constants["ADJ_NAME_LEN"])-1]
  identity_block = libadjoint.Block(block_name)

  init_rhs=adjlinalg.Vector(coeff).duplicate()
  init_rhs.axpy(1.0,adjlinalg.Vector(coeff))

  def identity_assembly_cb(variables, dependencies, hermitian, coefficient, context):
    assert coefficient == 1
    return (adjlinalg.Matrix(adjlinalg.IdentityMatrix()), adjlinalg.Vector(dolfin.Function(fn_space)))

  identity_block.assemble=identity_assembly_cb

  if dolfin.parameters["adjoint"]["record_all"]:
    adjglobals.adjointer.record_variable(dep, libadjoint.MemoryStorage(adjlinalg.Vector(coeff)))

  initial_eq = libadjoint.Equation(dep, blocks=[identity_block], targets=[dep], rhs=adjrhs.RHS(init_rhs))
  adjglobals.adjointer.register_equation(initial_eq)
  assert adjglobals.adjointer.variable_known(dep)

def do_checkpoint(cs, var, rhs):
  if cs == int(libadjoint.constants.adj_constants["ADJ_CHECKPOINT_STORAGE_MEMORY"]):
    for coeff in adjglobals.adj_variables.keys(): 
      dep = adjglobals.adj_variables[coeff]

      if dep == var:
        # We may need to checkpoint another variable if rhs is a NonlinearRHS and we need
        # to store the initial condition in order to replay the solve.
        if hasattr(rhs, 'ic_var') and rhs.ic_var is not None:
          dep = rhs.ic_var
        else:
          continue

      adjglobals.adjointer.record_variable(dep, libadjoint.MemoryStorage(adjlinalg.Vector(coeff), cs=True))

  elif cs == int(libadjoint.constants.adj_constants["ADJ_CHECKPOINT_STORAGE_DISK"]):
    for coeff in adjglobals.adj_variables.keys(): 
      dep = adjglobals.adj_variables[coeff]

      if dep == var:
        # We may need to checkpoint another variable if rhs is a NonlinearRHS and we need
        # to store the initial condition in order to replay the solve.
        if hasattr(rhs, 'ic_var') and rhs.ic_var is not None:
          dep = rhs.ic_var
        else:
          continue

      adjglobals.adjointer.record_variable(dep, libadjoint.DiskStorage(adjlinalg.Vector(coeff), cs=True))

