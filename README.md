How to compile:
 1) Set PETSC_DIR and PETSC_ARCH variables in your path
 2) make all

How to run (examples):
 For Q1 mesh on 1 processors:
     mpiexec -n 1 ./main -order 1  -f cube8.exo -dim 3 -num_fields 1 -num_components 1
