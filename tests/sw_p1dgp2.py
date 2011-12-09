import kelvin_new as kelvin
import sw
from dolfin import *

W=sw.p1dgp2(kelvin.mesh)

state=Function(W)

state.interpolate(kelvin.InitialConditions())

kelvin.params["basename"]="p1dgp2"
kelvin.params["finish_time"]=kelvin.params["dt"]*10
kelvin.params["finish_time"]=kelvin.params["dt"]*2

M,G=sw.construct_shallow_water(W,kelvin.params)

sw.timeloop_theta(M,G,state,kelvin.params)

sw.replay()