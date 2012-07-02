import coeffstore
import libadjoint

# Create the adjointer, the central object that records the forward solve
# as it happens.
adjointer = libadjoint.Adjointer()

adj_variables = coeffstore.CoeffStore()
def adj_inc_timestep(time=None, finished=False):
  adj_variables.increment_timestep()
  if time:
    adjointer.time.next(time)

  if finished:
    adjointer.time.finish()

# A dictionary that saves the functionspaces of all checkpoint variables that have been saved to disk
checkpoint_fs = {}

def adj_check_checkpoints():
  adjointer.check_checkpoints()
