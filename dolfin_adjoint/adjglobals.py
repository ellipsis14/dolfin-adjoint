import coeffstore
import libadjoint

# Create the adjointer, the central object that records the forward solve
# as it happens.
adjointer = libadjoint.Adjointer()

adj_variables = coeffstore.CoeffStore()
def adj_inc_timestep(time=None, finished=False):
  '''Dolfin does not supply us with information about timesteps, and so more information
  is required from the user for certain features. This function should be called at
  the end of the time loop with two arguments:

    - :py:data:`time` -- the time at the end of the timestep just computed
    - :py:data:`finished` -- whether this is the final timestep.

  With this information, complex functional expressions using the :py:class:`Functional` class
  can be used.
  '''
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
