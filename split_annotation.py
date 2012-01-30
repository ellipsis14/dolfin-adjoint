
import dolfin
import libadjoint

import solving
import assign

import hashlib
import time

Function_split = dolfin.Function.split
def adjoint_split(self, **kwargs):
  annotate = False
  if "annotate" in kwargs:
    annotate = kwargs["annotate"]
    del kwargs["annotate"]

  out = Function_split(self, **kwargs)

  if annotate:
    for idx, fn in enumerate(out):
      annotate_split(self, idx, fn)

  return out
dolfin.Function.split = adjoint_split

def annotate_split(bigfn, idx, smallfn):
  fn_space = smallfn.function_space().collapse()
  test = dolfin.TestFunction(fn_space)
  trial = dolfin.TrialFunction(fn_space)
  eq_lhs = dolfin.inner(test, trial)*dolfin.dx

  diag_name = "Split:%s:" % idx + hashlib.md5(str(eq_lhs) + "split" + str(smallfn) + str(bigfn) + str(idx) + str(time.time())).hexdigest()

  diag_deps = []
  diag_block = libadjoint.Block(diag_name, dependencies=diag_deps, test_hermitian=solving.debugging["test_hermitian"], test_derivative=solving.debugging["test_derivative"])

  solving.register_initial_conditions([(bigfn, solving.adj_variables[bigfn])], linear=True, var=None)

  var = solving.adj_variables.next(smallfn)

  def diag_assembly_cb(dependencies, values, hermitian, coefficient, context):
    '''This callback must conform to the libadjoint Python block assembly
    interface. It returns either the form or its transpose, depending on
    the value of the logical hermitian.'''

    assert coefficient == 1

    value_coeffs=[v.data for v in values]
    eq_l = eq_lhs

    if hermitian:
      return (solving.Matrix(dolfin.adjoint(eq_l)), solving.Vector(None, fn_space=fn_space))
    else:
      return (solving.Matrix(eq_l), solving.Vector(None, fn_space=fn_space))
  diag_block.assemble = diag_assembly_cb

  rhs = SplitRHS(test, bigfn, idx)

  eqn = libadjoint.Equation(var, blocks=[diag_block], targets=[var], rhs=rhs)

  cs = solving.adjointer.register_equation(eqn)
  solving.do_checkpoint(cs, var)

  if solving.debugging["fussy_replay"]:
    mass = eq_lhs
    smallfn_massed = dolfin.Function(fn_space)
    dolfin.solve(mass == dolfin.action(mass, smallfn), smallfn_massed)
    assert False, "No idea how to assign to a subfunction yet .. "
    #assign.dolfin_assign(bigfn, smallfn_massed)

  if solving.debugging["record_all"]:
    smallfn_record = dolfin.Function(fn_space)
    assign.dolfin_assign(smallfn_record, smallfn)
    solving.adjointer.record_variable(var, libadjoint.MemoryStorage(solving.Vector(smallfn_record)))

class SplitRHS(solving.RHS):
  def __init__(self, test, function, index):
    self.test = test
    self.function = function
    self.idx = index
    self.deps = [solving.adj_variables[function]]
    self.coeffs = [function]

  def __call__(self, dependencies, values):
    fn = Function_split(values[0].data, deepcopy=True)[self.idx]
    return solving.Vector(dolfin.inner(self.test, fn)*dolfin.dx)

  def derivative_action(self, dependencies, values, variable, contraction_vector, hermitian):
    if not hermitian:
      fn = Function_split(contraction_vector)[self.idx]
      action = dolfin.inner(self.test, fn)
    else:
      bigtest = dolfin.TestFunction(self.function.function_space())
      outfn   = dolfin.Function(self.function.function_space())

      # DOLFIN is a bit annoying when it comes to splits. Actually, it is very annoying.
      # You can't do anything like
      # outfn[idx].vector()[:] = values_I_want_to_assign_to_outfn[idx]
      # or
      # fn = outfn.split()[idx]; fn.vector()[:] = values_I_want_to_assign_to_outfn[idx]
      # for whatever reason
      assert False, "No idea how to assign to a subfunction yet .. "
      assign.dolfin_assign(outfn, contraction_vector.data)

      action = dolfin.inner(bigtest, outfn)*dolfin.dx

    return solving.Vector(action)

