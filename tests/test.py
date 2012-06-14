#!/usr/bin/env python

import os, os.path
import sys
import subprocess
import sys
import multiprocessing
import multiprocessing.pool
import threading
import time

test_cmds = {'tlm_simple': 'mpirun -n 2 python tlm_simple.py',
             'navier_stokes': 'mpirun -n 2 python navier_stokes.py',
             'svd_simple': 'mpirun -n 2 python svd_simple.py',
             'optimisation': 'mpirun -n 2 python optimisation.py',
             'differentiability-dg-upwind': None,
             'differentiability-stokes': None,
             'checkpoint_online': None}

chdirlock = threading.Lock()
appendlock = threading.Lock()

num_procs = 1
if len(sys.argv) > 1:
  if sys.argv[1] == "-n":
    if len(sys.argv) > 2:
      num_procs = int(sys.argv[2])
    else:
      num_procs = None
  else:
    print "Usage: test.py [-n THREADS]"
    print "Run the dolfin-adjoint test suite."
    print "To run on N cores, use -n N; to use all"
    print "processors available, just run test.py -n."
    sys.exit(0)

pool = multiprocessing.pool.ThreadPool(num_procs)
fails = []

basedir = os.path.dirname(os.path.abspath(sys.argv[0]))
subdirs = [x for x in os.listdir(basedir) if os.path.isdir(os.path.join(basedir, x))]

# Keep path variables (for buildbot's sake for instance)
orig_pythonpath = os.getenv('PYTHONPATH', '')
pythonpath = os.pathsep.join([os.path.abspath(os.path.join(basedir,
                                                           os.path.pardir)),
                              orig_pythonpath])
os.putenv('PYTHONPATH', pythonpath)

def f(subdir):
  test_cmd = test_cmds.get(subdir, 'python %s.py' % subdir)
  if test_cmd is not None:

    chdirlock.acquire()
    os.chdir(os.path.join(basedir, subdir))
    if num_procs > 1:
      time.sleep(1)

    print "--------------------------------------------------------"
    print "Running %s " % subdir
    print "--------------------------------------------------------"

    chdirlock.release()

    exit = os.system(test_cmd)
    if exit != 0:
      print "subdir: ", subdir
      print "exit: ", exit
      appendlock.acquire()
      fails.append(subdir)
      appendlock.release()

pool.map(f, sorted(subdirs))

if len(fails) > 0:
  print "Failures: ", set(fails)
  sys.exit(1)
