#!/usr/bin/env python

import os, os.path
import sys
import subprocess
import sys
import multiprocessing
import multiprocessing.pool
import threading
import time
from optparse import OptionParser

test_cmds = {'tlm_simple': 'mpirun -n 2 python tlm_simple.py',
             'navier_stokes': 'mpirun -n 2 python navier_stokes.py',
             'svd_simple': 'mpirun -n 2 python svd_simple.py',
             'gst_mass': 'mpirun -n 2 python gst_mass.py',
             'optimization': 'mpirun -n 2 python optimization.py && python optimization_checkpointing.py',
             'optimal_control_mms': 'mpirun -n 2 python optimal_control_mms.py',
             'differentiability-dg-upwind': None,
             'differentiability-stokes': None,
             'checkpoint_online': None,
             'changing_vector': None,
             'matrix_free_burgers': None,
             'matrix_free_heat': None,
             'matrix_free_simple': None,
             'mantle_convection': None}

chdirlock = threading.Lock()
appendlock = threading.Lock()

parser = OptionParser()
parser.add_option("-n", type="int", dest="num_procs", default = 1, help = "To run on N cores, use -n N; to use all processors available, run test.py -n 0.")
parser.add_option("-t", type="string", dest="test_name", help = "To run one specific test, use -t TESTNAME. By default all test are run.")
parser.add_option("-s", dest="short_only", default = False, action="store_true", help = "To run the short tests only, use -s. By default all test are run.")
(options, args) = parser.parse_args(sys.argv)

if options.num_procs <= 0:
  options.num_procs = None

pool = multiprocessing.pool.ThreadPool(options.num_procs)
fails = []

basedir = os.path.dirname(os.path.abspath(sys.argv[0]))
subdirs = [x for x in os.listdir(basedir) if os.path.isdir(os.path.join(basedir, x))]
if options.test_name:
  if not options.test_name in subdirs:
    print "Specified test not found."
    sys.exit(1)
  else:
    subdirs = [options.test_name]

long_tests = ["viscoelasticity"] # special case the very long tests for speed
for test in long_tests:
  subdirs.remove(test)

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
    if options.num_procs > 1:
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

tests = sorted(subdirs)
if not options.short_only:
  tests = long_tests + tests
pool.map(f, tests)

if len(fails) > 0:
  print "Failures: ", set(fails)
  sys.exit(1)
