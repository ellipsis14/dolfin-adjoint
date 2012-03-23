import sys
import divett
import sw_lib
from dolfin import *
from dolfin_adjoint import *

W=sw_lib.p1dgp2(divett.mesh)
debugging["record_all"]=True

state=Function(W)

state.interpolate(divett.InitialConditions())

divett.params["basename"]="p1dgp2"
divett.params["finish_time"]=2*pi/(sqrt(divett.params["g"]*divett.params["depth"])*pi/3000)
divett.params["dt"]=divett.params["finish_time"]/5
divett.params["period"]=60*60*1.24
divett.params["dump_period"]=1

M,G,rhs_contr,ufl,ufr=sw_lib.construct_shallow_water(W,divett.ds,divett.params)

state = sw_lib.timeloop_theta(M,G,rhs_contr,ufl,ufr,state,divett.params)

adj_html("sw_forward.html", "forward")
adj_html("sw_adjoint.html", "adjoint")
replay_dolfin()

J = FinalFunctional(dot(state, state)*dx)
f_direct = assemble(dot(state, state)*dx)
for (adj_state, var) in compute_adjoint(J):
  pass

ic = Function(W)
ic.interpolate(divett.InitialConditions())
def J(ic):
  state = sw_lib.timeloop_theta(M, G, rhs_contr,ufl,ufr,ic, divett.params, annotate=False)
  return assemble(dot(state, state)*dx)

minconv = test_initial_condition_adjoint(J, ic, adj_state, seed=0.001)
if minconv < 1.9:
  exit_code = 1
else:
  exit_code = 0
sys.exit(exit_code)
