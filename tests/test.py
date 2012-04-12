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
             'differentiability': None}

chdirlock = threading.Lock()
appendlock = threading.Lock()

try:
  if sys.argv[1] == "-n":
    if len(sys.argv) > 2:
      num_procs = int(sys.argv[2])
    else:
      num_procs = None
except:
  num_procs = 1

pool = multiprocessing.pool.ThreadPool(num_procs)
fails = []

basedir = os.path.dirname(os.path.abspath(sys.argv[0]))
subdirs = [x for x in os.listdir(basedir) if os.path.isdir(os.path.join(basedir, x))]

os.putenv('PYTHONPATH', os.path.abspath(os.path.join(basedir, os.path.pardir)))

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
