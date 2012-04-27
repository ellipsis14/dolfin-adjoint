import libadjoint
import dolfin
import ufl
import ufl.algorithms
import adjglobals
import adjlinalg

class RHS(libadjoint.RHS):
  '''This class implements the libadjoint.RHS abstract base class for the Dolfin adjoint.
  It takes in a form, and implements the necessary routines such as calling the right-hand side
  and taking its derivative.'''
  def __init__(self, form):

    self.form=form

    if isinstance(self.form, ufl.form.Form):
      self.deps = [adjglobals.adj_variables[coeff] for coeff in ufl.algorithms.extract_coefficients(self.form) if hasattr(coeff, "function_space")]      
    else:
      self.deps = []

    if isinstance(self.form, ufl.form.Form):
      self.coeffs = [coeff for coeff in ufl.algorithms.extract_coefficients(self.form) if hasattr(coeff, "function_space")]      
    else:
      self.coeffs = []

  def __call__(self, dependencies, values):

    if isinstance(self.form, ufl.form.Form):

      dolfin_dependencies=[dep for dep in ufl.algorithms.extract_coefficients(self.form) if hasattr(dep, "function_space")]

      dolfin_values=[val.data for val in values]

      return adjlinalg.Vector(dolfin.replace(self.form, dict(zip(dolfin_dependencies, dolfin_values))))

    else:
      # RHS is a adjlinalg.Vector.
      assert isinstance(self.form, adjlinalg.Vector)
      return self.form
    

  def derivative_action(self, dependencies, values, variable, contraction_vector, hermitian):

    if isinstance(self.form, ufl.form.Form):
      # Find the dolfin Function corresponding to variable.
      dolfin_variable = values[dependencies.index(variable)].data

      dolfin_dependencies = [dep for dep in ufl.algorithms.extract_coefficients(self.form) if hasattr(dep, "function_space")]

      dolfin_values = [val.data for val in values]

      current_form = dolfin.replace(self.form, dict(zip(dolfin_dependencies, dolfin_values)))
      trial = dolfin.TrialFunction(dolfin_variable.function_space())

      d_rhs=dolfin.derivative(current_form, dolfin_variable, trial)

      if hermitian:
        action = dolfin.action(dolfin.adjoint(d_rhs),contraction_vector.data)
      else:
        action = dolfin.action(d_rhs,contraction_vector.data)

      return adjlinalg.Vector(action)
    else:
      # RHS is a adjlinalg.Vector. Its derivative is therefore zero.
      raise exceptions.LibadjointErrorNotImplemented("No derivative method for constant RHS.")


  def dependencies(self):

    return self.deps

  def coefficients(self):

    return self.coeffs

  def __str__(self):
    
    return hashlib.md5(str(self.form)).hexdigest()

class NonlinearRHS(RHS):
  '''For nonlinear problems, the source term isn't assembled in the usual way.
  If the nonlinear problem is given as
  F(u) = 0,
  we annotate it as
  M.u = M.u - F(u) .
  So in order to actually assemble the right-hand side term,
  we first need to solve F(u) = 0 to find the specific u,
  and then multiply that by the mass matrix.'''
  def __init__(self, form, F, u, bcs, mass, solver_parameters, J):
    '''form is M.u - F(u). F is the nonlinear equation, F(u) := 0.'''
    RHS.__init__(self, form)
    self.F = F
    self.u = u
    self.bcs = bcs
    self.mass = mass
    self.solver_parameters = solver_parameters
    self.J = J

    # We want to mark that the RHS term /also/ depends on
    # the previous value of u, as that's what we need to initialise
    # the nonlinear solver.
    var = adjglobals.adj_variables[self.u]
    self.ic_var = None

    if dolfin.parameters["adjoint"]["fussy_replay"]:
      can_depend = False
      if var.timestep > 0:
        prev_var = libadjoint.Variable(var.name, var.timestep-1, var.iteration)
        if adjglobals.adjointer.variable_known(prev_var):
          can_depend = True

      if can_depend:
        self.deps.append(prev_var)
        self.ic_var = prev_var
      else:
        self.ic_copy = dolfin.Function(u)
        self.ic_var = None

  def __call__(self, dependencies, values):
    assert isinstance(self.form, ufl.form.Form)

    ic = self.u.function_space() # by default, initialise with a blank function in the solution FunctionSpace

    if hasattr(self, "ic_copy"):
      ic = self.ic_copy

    replace_map = {}

    for i in range(len(self.deps)):
      if self.deps[i] in dependencies:
        j = dependencies.index(self.deps[i])
        if self.deps[i] == self.ic_var:
          ic = values[j].data # ahah, we have found an initial condition!
        else:
          replace_map[self.coeffs[i]] = values[j].data

    current_F    = dolfin.replace(self.F, replace_map)
    u = dolfin.Function(ic)
    current_F    = dolfin.replace(current_F, {self.u: u})

    try:
      if self.J is not None:
        J = dolfin.replace(self.J, replace_map)
        J = dolfin.replace(J, {self.u: u})
      else:
        J = self.J
    except ufl.log.UFLException:
      J = None

    # OK, here goes nothing:
    dolfin.solve(current_F == 0, u, self.bcs, solver_parameters=self.solver_parameters, J=J)

    return adjlinalg.Vector(dolfin.action(self.mass, u))

  def derivative_action(self, dependencies, values, variable, contraction_vector, hermitian):
    '''If variable is the variable for the initial condition, we want to ignore it,
    and set the derivative to zero. Assuming the solver converges, the sensitivity of
    the solution to the initial condition should be extremely small, and computing it
    is very difficult (one would have to do a little adjoint solve to compute it).
    Even I'm not that fussy.'''

    if variable == self.ic_var:
      deriv_value = values[dependencies.index(variable)].data
      return adjlinalg.Vector(None, fn_space=deriv_value.function_space())
    else:
      return RHS.derivative_action(self, dependencies, values, variable, contraction_vector, hermitian)

  def derivative_assembly(self, dependencies, values, variable, hermitian):
    replace_map = {}

    for i in range(len(self.deps)):
      if self.deps[i] == self.ic_var: continue
      j = dependencies.index(self.deps[i])
      replace_map[self.coeffs[i]] = values[j].data

    diff_var = values[dependencies.index(variable)].data

    current_form = dolfin.replace(self.form, replace_map)
    deriv = dolfin.derivative(current_form, diff_var)

    if hermitian:
      deriv = dolfin.adjoint(deriv)
      bcs = [dolfin.homogenize(bc) for bc in self.bcs if isinstance(bc, dolfin.DirichletBC)] + [bc for bc in self.bcs if not isinstance(bc, dolfin.DirichletBC)]
    else:
      bcs = self.bcs

    return adjlinalg.Matrix(deriv, bcs=bcs)
