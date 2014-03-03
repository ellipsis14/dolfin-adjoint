import backend 


def uniq(seq):
  '''Remove duplicates from a list, preserving order'''
  seen = set()
  seen_add = seen.add
  return [ x for x in seq if x not in seen and not seen_add(x)]


def pause_annotation():
  flag = backend.parameters["adjoint"]["stop_annotating"]
  backend.parameters["adjoint"]["stop_annotating"] = True
  return flag

def continue_annotation(flag):
  backend.parameters["adjoint"]["stop_annotating"] = flag

def rank():
  # No idea what to do with firedrake here, so I assume one of them will fix it!
  try:
    return backend.MPI.process_number()
  except RuntimeError:
    return backend.MPI.rank(backend.mpi_comm_world())
