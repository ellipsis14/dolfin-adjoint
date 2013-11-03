import backend 


def uniq(seq):
  '''Remove duplicates from a list, preserving order'''
  seen = set()
  seen_add = seen.add
  return [ x for x in seq if x not in seen and not seen_add(x)]


annotating_flag = None 
def pause_annotation():
  global annotating_flag
  annotating_flat = backend.parameters["adjoint"]["stop_annotating"]
  backend.parameters["adjoint"]["stop_annotating"] = True

def continue_annotation():
  global annotating_flag
  backend.parameters["adjoint"]["stop_annotating"] = annotating_flag
