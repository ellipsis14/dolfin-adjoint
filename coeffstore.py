import libadjoint

class CoeffStore(object):
  '''This object manages the mapping from Dolfin coefficients to libadjoint Variables.
  In the process, it also manages the incrementing of the timestep associated with each
  variable, so that the user does not have to manually manage the time information.'''
  def __init__(self):
    self.coeffs={}
    self.libadjoint_timestep = -1

  def next(self, coeff):
    '''Increment the timestep corresponding to the provided Dolfin
    coefficient and then return the corresponding libadjoint variable.'''

    try:
      (timestep, iteration) = self.coeffs[coeff]
      if timestep == max(self.libadjoint_timestep, 0):
        self.coeffs[coeff] = (timestep, iteration + 1)
      else:
        self.coeffs[coeff] = (self.libadjoint_timestep, 0)
    except KeyError:
      self.coeffs[coeff] = (max(self.libadjoint_timestep, 0), 0)

    (timestep, iteration) = self.coeffs[coeff]
    return libadjoint.Variable(str(coeff), timestep, iteration)

  def __getitem__(self, coeff):
    '''Return the libadjoint variable corresponding to coeff.'''

    if not self.coeffs.has_key(coeff):
      self.coeffs[coeff] = (max(self.libadjoint_timestep, 0), 0)

    (timestep, iteration) = self.coeffs[coeff]
    return libadjoint.Variable(str(coeff), timestep, iteration)

  def increment_timestep(self):
    self.libadjoint_timestep += 1

  def forget(self, coeff):
    del self.coeffs[coeff]
