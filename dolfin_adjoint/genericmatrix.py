import dolfin
import solving

dolfin_genericmatrix_add = dolfin.GenericMatrix.__add__

def adjoint_genericmatrix_add(self, other):
  out = dolfin_genericmatrix_add(self, other)
  if hasattr(self, 'form') and hasattr(other, 'form'):
    out.form = self.form + other.form

  return out

dolfin.GenericMatrix.__add__ = adjoint_genericmatrix_add
