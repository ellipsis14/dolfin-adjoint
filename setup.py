from distutils.core import setup

setup (name = 'dolfin_adjoint',
       version = '0.6',
       description = 'Automatically derive the discrete adjoint of DOLFIN models',
       author = 'The dolfin_adjoint team',
       author_email = 'patrick.farrell06@imperial.ac.uk',
       packages = ['dolfin_adjoint', 'dolfin_adjoint.optimization'],
       package_dir = {'dolfin_adjoint': 'dolfin_adjoint'})
