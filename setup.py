from distutils.core import setup

setup (name = 'dolfin_adjoint',
       version = '1.4',
       description = 'Automatically derive the discrete adjoint of DOLFIN models',
       author = 'The dolfin_adjoint team',
       author_email = 'patrick.farrell@maths.ox.ac.uk',
       packages = ['dolfin_adjoint', 'dolfin_adjoint.optimization', 'dolfin_adjoint.optimization.line_search'],
       package_dir = {'dolfin_adjoint': 'dolfin_adjoint'})
