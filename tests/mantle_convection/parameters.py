__author__ = "Lyudmyla Vynnytska"
__copyright__ = "Copyright (C) 2011 Simula Research Laboratory and %s" % __author__
__license__  = "GNU LGPL Version 3 or any later version"

# Last changed: 2011-10-17

from dolfin import *
from scipy.special import erf

class InitialTemperature(Expression):

    def __init__(self, Ra, lam):
        self.Ra = Ra
        self.lam = lam

    def eval(self, values, x):

        u0 = self.lam**(7.0/3.0)/((1 + self.lam**4)**(2.0/3.0))*(self.Ra/sqrt(pi)/2.0)**(2.0/3.0)

        Q = 2*sqrt(self.lam/pi/u0)
        v0 = u0
        if abs(x[0]) < DOLFIN_EPS and abs(x[1] - 1.0) > DOLFIN_EPS:
            Tu = 0.5
        elif abs(x[0]) < DOLFIN_EPS and abs(x[1] - 1.0) <= DOLFIN_EPS:
            Tu = 0.0
        else:
            Tu = 0.5*erf( (1.0-x[1])/2.0*sqrt(u0/(x[0])) )
        if abs(x[0] - 2.0) < DOLFIN_EPS:
            Tl = 0.5
        else:
            Tl = 1.0 - 0.5*erf( x[1]/2.0*sqrt(u0/(self.lam - x[0])) )

        Tr = 0.5 + Q/2.0/sqrt(pi)*sqrt(v0/(x[1] + 1.0))*exp(-x[0]*x[0]*v0/(4*x[1] + 4))
        td = (self.lam - x[0])*(self.lam - x[0])*v0
        Ts = 0.5 - Q/2.0/sqrt(pi)*sqrt(v0/(2.0 - x[1]))*exp( -td/(8.0 - 4.0*x[1]) )
        values[0] = Tr + Tu + Tl +  Ts - 1.5
        if values[0] < 0.0:
            values[0] = 0.0
        if values[0] > 1.0:
            values[0] = 1.0

class InitialTemperatureSimple(Expression):

    def __init__(self, h, l):
        self.h = h
        self.l = l

    def eval(self, values, x):
        values[0] = x[1] + 0.05*cos(pi*x[0]/self.l)*sin(pi*x[1]/self.h)

class Properties(Expression):

    def __init__(self, nutop, nubottom):
        self.nutop = nutop
        self.nubottom = nubottom

    def eval(self, values, x):
        interface = 0.05
        #if x[1] > interface:
        if (x[1] - interface) > 0.00001:
            values[0] = self.nutop
        else:
            values[0] = self.nubottom

# Parameter values
Ra = 1.0e+6
Rb = 1.0e+6

eta0 = 1.0

#b_val = 0.9 #ln(16384.0)
#c_val = 1.0 #ln(2.0) # ln(64.0)

deltaT = 1.0
b_val = ln(1.0)
c_val = ln(2.0)

rbottom = 1.0

rtop = 0.0
rho0 = Properties(rtop, rbottom)

g = Constant((0.0, -1.0))
