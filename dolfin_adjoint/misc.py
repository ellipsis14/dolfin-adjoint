
def uniq(seq):
  '''Remove duplicates from a list, preserving order'''
  seen = set()
  seen_add = seen.add
  return [ x for x in seq if x not in seen and not seen_add(x)]
