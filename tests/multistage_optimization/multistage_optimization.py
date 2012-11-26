from dolfin import *
from dolfin_adjoint import *

mesh = UnitSquare(2, 2)
minimize_multistage(None, mesh, 3)


from IPython import embed
embed()
