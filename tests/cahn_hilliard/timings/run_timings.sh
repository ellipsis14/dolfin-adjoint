#!/bin/bash

for i in $(seq 1 5)
do
  time mpiexec -n 8 python $1
done
