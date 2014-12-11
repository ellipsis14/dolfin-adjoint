import backend

if backend.__name__ == "dolfin":
    solve = backend.fem.solving.solve
    matrix_types = lambda: (backend.cpp.Matrix, backend.GenericMatrix)
    _extract_args = backend.fem.solving._extract_args
    function_type = backend.cpp.Function
    function_space_type = backend.cpp.FunctionSpace

else:
    solve = backend.solving.solve
    matrix_types = lambda: backend.op2.petsc_base.Mat
    function_type = backend.Function
    function_space_type = (backend.FunctionSpace, backend.MixedFunctionSpace)

    def _extract_args(*args, **kwargs):
        eq, u, bcs, _, _, _, _, solver_parameters, _ = backend.solving._extract_args(*args, **kwargs)
        return eq, u, bcs, None, None, None, None, solver_parameters
