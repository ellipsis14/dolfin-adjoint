from dolfin import *
from dolfin_adjoint import *

mesh = UnitSquareMesh(4,4)
V = FunctionSpace(mesh, "CG", 1)

c = Constant(2)

class SourceExpression(Expression):
    def __init__(self, c):
        self.c = c

    def eval(self, value, x):
        value[0] = float(self.c)**2

    def deval(self, value, x, coeff):
        value[0] = 2*self.c 

    def dependencies(self):
        return [self.c]

    def copy(self):
        return SourceExpression(self.c)



def taylor_test_expression(exp, V):
    """ Warning: This function resets the adjoint tape! """

    adj_reset()

    # Annotate test model
    s = project(exp, V, annotate=True)

    Jform = s**2*dx
    J = Functional(Jform)
    J0 = assemble(Jform)

    controls = [Control(c) for c in exp.dependencies()]
    dJd0 = compute_gradient(J, controls, forget=False)

    def Jfunc(dep_values):

        dep_values = enlist(dep_values)

        for new_val, dep in zip(dep_values, exp.dependencies()):
            dep.assign(new_val)

        s = project(source, V, annotate=False)
        return assemble(s**2*dx)

    #HJ0 = hessian(J, controls, warn=False)

    #minconv = taylor_test(Jfunc, controls, J0, dJdic, HJm=HJ0, seed=1.0e-3)
    minconv = taylor_test(Jfunc, controls, J0, dJd0)
    assert minconv > 2.0
    print minconv

    adj_reset()

source = SourceExpression(c)
taylor_test_expression(source, V)
