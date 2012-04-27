import coeffstore
import libadjoint

adj_variables = coeffstore.CoeffStore()
def adj_inc_timestep():
  adj_variables.increment_timestep()

# Create the adjointer, the central object that records the forward solve
# as it happens.
adjointer = libadjoint.Adjointer()
adjointer.first_solve = True

# A dictionary that saves the functionspaces of all checkpoint variables that have been saved to disk
checkpoint_fs = {}

