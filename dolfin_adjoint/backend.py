''' 
   Imports the dolfin or firedrake module, depending on parameters["adjoint_backend"] 
'''

# If unset, default to dolfin 
#if not "adjoint_backend" in parameters.keys():
#    parameters.add("adjoint_backend", "dolfin")

#if parameters["adjoint_backend"] == "dolfin":
if False:
    from dolfin import *
else:
    from firedrake import *
