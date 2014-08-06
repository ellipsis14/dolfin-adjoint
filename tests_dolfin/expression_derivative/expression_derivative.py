from dolfin import *
from dolfin_adjoint import *

mesh = UnitSquareMesh(4, 4)
V = FunctionSpace(mesh, "CG", 1)

c = Constant(2)
d = Constant(3)


class SourceExpression(Expression):
    def __init__(self, c, d):
        self.c = c
        self.d = d

    def eval(self, value, x):
        value[0] = self.c**2
        value[0] *= self.d

    def deval(self, value, x, derivative_coeff):

        if self.c == derivative_coeff:
            value[0] = 2*self.c*self.d

        elif self.d == derivative_coeff:
            value[0] = self.c**2

    def dependencies(self):
        return [self.c, self.d]

    def copy(self):
        return SourceExpression(self.c, self.d)



def taylor_test_expression(exp, V):
    """ Warning: This function resets the adjoint tape! """

    adj_reset()

    # Annotate test model
    s = project(exp, V, annotate=True)

    Jform = s**2*dx + exp*dx(domain=mesh)

    J = Functional(Jform)
    J0 = assemble(Jform)

    deps = exp.dependencies()
    controls = [Control(c) for c in deps]
    dJd0 = compute_gradient(J, controls, forget=False)

    for i in range(len(controls)):
        def Jfunc(new_val):
            dep = exp.dependencies()[i]

            # Remember the old dependency value for later
            old_val = float(dep)

            # Compute the functional value
            dep.assign(new_val)
            s = project(exp, V, annotate=False)
            out = assemble(s**2*dx + exp*dx(domain=mesh))

            # Restore the old dependency value
            dep.assign(old_val)

            return out

        #HJ = hessian(J, controls[i], warn=False)
        #minconv = taylor_test(Jfunc, controls[i], J0, dJd0[i], HJm=HJ)
        minconv = taylor_test(Jfunc, controls[i], J0, dJd0[i])
    assert minconv > 1.9


    adj_reset()

source = SourceExpression(c, d)
taylor_test_expression(source, V)
