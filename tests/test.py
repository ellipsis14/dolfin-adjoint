#!/usr/bin/env python

import os, os.path
import sys
import subprocess

test_cmds = {'tlm_simple': 'mpirun -n 2 python tlm_simple.py',
             'navier_stokes': 'mpirun -n 2 python navier_stokes.py',
             'viscoelasticity': None}

basedir = os.path.dirname(os.path.abspath(sys.argv[0]))
subdirs = [x for x in os.listdir(basedir) if os.path.isdir(os.path.join(basedir, x))]

os.putenv('PYTHONPATH', os.path.abspath(os.path.join(basedir, os.path.pardir, os.path.pardir)))

for subdir in sorted(subdirs):
  test_cmd = test_cmds.get(subdir, 'python %s.py' % subdir)
  if test_cmd is not None:
    print "--------------------------------------------------------"
    print "Running %s " % subdir
    print "--------------------------------------------------------"
    os.chdir(os.path.join(basedir, subdir))
    exit = os.system(test_cmd)
    assert exit == 0
