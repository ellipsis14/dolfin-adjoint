from dolfin_utils.pjobs import submit

# Define name for job
name = "mantle_convection_final"

# Define job
job = "python demo.py"

# Submit job
print "Submitting %s with name %s" % (job, name)
submit(job, nodes=1, ppn=8, keep_environment=True, name=name,
       walltime=24*5)
