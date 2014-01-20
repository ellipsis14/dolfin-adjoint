
from dolfin import *
from numpy import *
from time import sleep
import pylab

dirname = "bin"
velocities = TimeSeries("%s/velocity" % dirname)
densities = TimeSeries("%s/density" % dirname)

# Retrieve mesh
mesh = Mesh()
densities.retrieve(mesh, 0.0)
#velocities.retrieve(mesh, 0.0)

# Define function spaces
CG2 = VectorFunctionSpace(mesh, "CG", 2)
velocity = Function(CG2)

height = 1.0
length = 2.0

times = velocities.vector_times()
urms = []

outfile = open('veloc.dat', 'w')
num = 0

for t in times:
    # Retrieve data at this time step
    velocities.retrieve(velocity.vector(), t)
    value = norm(velocity)/sqrt(length*height)
    #value = 1.0/(d_b*length)*assemble(velocity*dx(1), cell_domains=top)
    urms += [value]
    outfile.write('%s %.8f %.5f' % (num, t, value))
    outfile.write('\n')
    num +=1

outfile.close()
pylab.plot([t for t in times], urms)
pylab.xlabel("t")
pylab.ylabel("RMS velocity")
pylab.show()
#pylab.savefig("urms.pdf")
