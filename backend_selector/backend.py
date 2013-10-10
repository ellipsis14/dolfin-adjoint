''' 
   Imports the dolfin or firedrake module, depending on the backend chosen. 
'''

import backend_selector
if backend_selector.backend == "dolfin":
    from dolfin import *
elif backend_selector.backend == "firedrake":
    from firedrake import *
else:
    raise ValueError, "Unknown backend selected. Valid vales are 'dolfin' and 'firedrake'."
