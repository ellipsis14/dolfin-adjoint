import dolfin

class CoefficientList(list):
    ''' This class enables easy manipulation and operations of multiple 
        dolfin.Coefficient that are common for optimisation algorithms.
    '''

    def scale(self, s):
        ''' Scales all coefficients by s. '''
        for c in self:
            # c is Function
            if hasattr(c, "vector"):
                try:
                    c.assign(s*c)
                except TypeError:
                    c.vector()[:] = s*c.vector() 
            # c is Constant
            else:
                c.assign(float(s*c))

    def inner(self, ll):
        ''' Computes the componentwise inner product with of itself and ll. ''' 
        assert len(ll) == len(self)

        r = 0
        for c1, c2 in zip(self, ll):
            # c1 is Function
            if hasattr(c1, "vector"):
                r += c1.vector().inner(c2.vector()) 
            # c1 is Constant
            else:
                r += float(c1)*float(c2)

        return r

    def normL2(self):
        ''' Computes the L2 norm. '''
        return self.inner(self)**0.5

    def deep_copy(self):
        ll = []
        for c in self:
            # c is Function
            if hasattr(c, "vector"):
                ll.append(dolfin.Function(c))
            # c is Constant
            else:
                ll.append(dolfin.Constant(float(c)))

        return CoefficientList(ll)
    
    def axpy(self, a, x):
        ''' Computes componentwise self = self + a*x. '''
        assert(len(self) == len(x))

        for yi, xi in zip(self, x):
            # yi is Function
            if hasattr(yi, "vector"):
                yi.vector().axpy(a, xi.vector())
            # yi is Constant
            else:
                yi.assign(float(yi) + a*float(xi))

    def assign(self, x):
        ''' Assigns componentwise self = x. '''
        assert(len(self) == len(x))

        for yi, xi in zip(self, x):
            yi.assign(xi)

