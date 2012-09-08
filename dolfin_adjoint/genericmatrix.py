import dolfin
import solving

dolfin_genericmatrix_add = dolfin.GenericMatrix.__add__

def adjoint_genericmatrix_add(self, other):
  out = dolfin_genericmatrix_add(self, other)
  if hasattr(self, 'form') and hasattr(other, 'form'):
    out.form = self.form + other.form

  return out

dolfin.GenericMatrix.__add__ = adjoint_genericmatrix_add

dolfin_genericmatrix_mul = dolfin.GenericMatrix.__mul__

def adjoint_genericmatrix_mul(self, other):
  out = dolfin_genericmatrix_mul(self, other)
  if hasattr(self, 'form') and isinstance(other, dolfin.GenericVector):
    if hasattr(other, 'form'):
      out.form = dolfin.action(self.form, other.form)
    elif hasattr(other, 'function'):
      out.form = dolfin.action(self.form, other.function)

  return out

dolfin.GenericMatrix.__mul__ = adjoint_genericmatrix_mul
