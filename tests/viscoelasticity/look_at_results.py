from dolfin import *
from viscoelasticity import Z
import glob

def look_at_adjoints(dirname, var_name):
    adjoints = glob.glob("%s/adjoint*.xml" % dirname)
    adjoints.sort()

    filenames = [f for f in adjoints if var_name in f]
    filenames.sort(reverse=True)

    z = Function(Z)
    Z02 = Z.sub(0).sub(0).collapse()
    tau02 = Function(Z02)
    xmlfile = File("%s/viscous_stress.pvd" % dirname)
    for f in filenames:
        file = File(f)
        file >> z
        (sigma0, sigma1, v, gamma) = z.split()
        tau02.assign(project(sigma0[2], Z02))
        xmlfile << tau02
        plot(tau02)

    interactive()

def look_at_forwards(dirname):

    forwards = glob.glob("%s/forward*.xml" % dirname)

    iterations = [int(f.split("_")[-2]) for f in forwards]
    iterations.sort()
    times = [float(f.split("_")[-1][:-4]) for f in forwards]
    times.sort()
    print iterations
    print times

    z = Function(Z)
    w = Function(Z.sub(2).collapse())
    for k in iterations:
        t = times[k]

        file = File("%s/forward_%d_%g.xml" % (dirname, k, t))
        file >> z
        (sigma0, sigma1, v, gamma) = z.split()
        w.assign(v)
        plot(w)

    interactive()

dirname = "results"

look_at_adjoints(dirname, "w_3")
