import backend

if backend.__name__ == "dolfin":
    solve = backend.fem.solving.solve

else:
    solve = backend.solving.solve
