''' 
   Imports the dolfin or firedrake module, depending on parameters["adjoint_backend"] 
'''

import backend_selector
print backend_selector.backend
if backend_selector.backend == "dolfin":
    from dolfin import *
elif backend_selector.backend == "firedrake":
    from firedrake import *
else:
    raise ValueError, "Unknown backend selected. Valid vales are 'dolfin' and 'firedrake'."
