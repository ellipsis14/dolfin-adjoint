import coeffstore
import libadjoint
import dolfin

# Create the adjointer, the central object that records the forward solve
# as it happens.
adjointer = libadjoint.Adjointer()

mem_checkpoints = set()
disk_checkpoints = set()

adj_variables = coeffstore.CoeffStore()

def adj_start_timestep(time=0.0):
  '''Dolfin does not supply us with information about timesteps, and so more information
  is required from the user for certain features. This function should be called at the
  start of the time loop with the initial time (defaults to 0).
  
  See also: :py:func:`dolfin_adjoint.adj_inc_timestep`
  '''

  if not dolfin.parameters["adjoint"]["stop_annotating"]:
    adjointer.time.start(time)

def adj_inc_timestep(time=None, finished=False):
  '''Dolfin does not supply us with information about timesteps, and so more information
  is required from the user for certain features. This function should be called at
  the end of the time loop with two arguments:

    - :py:data:`time` -- the time at the end of the timestep just computed
    - :py:data:`finished` -- whether this is the final timestep.

  With this information, complex functional expressions using the :py:class:`Functional` class
  can be used.

  The finished argument is necessary because the final step of a functional integration must perform
  additional calculations.

  See also: :py:func:`dolfin_adjoint.adj_start_timestep`
  '''

  if not dolfin.parameters["adjoint"]["stop_annotating"]:
    adj_variables.increment_timestep()
    if time:
      adjointer.time.next(time)

    if finished:
      adjointer.time.finish()

# A dictionary that saves the functionspaces of all checkpoint variables that have been saved to disk
checkpoint_fs = {}

function_names = set()

def adj_check_checkpoints():
  adjointer.check_checkpoints()

# For caching strategies: a dictionary that maps adj_variable to LUSolver
# Not used by default
class VariableDict(dict):
  def __getitem__(self, x):
    return dict.__getitem__(self, str(x))

  def __setitem__(self, x, y):
    return dict.__setitem__(self, str(x), y)

lu_solvers = VariableDict()

def clear_solver_cache():
  for x in lu_solvers:
    del lu_solvers[x]
