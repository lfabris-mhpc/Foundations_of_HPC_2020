#!/bin/bash
mpif77 -ffixed-line-length-none jacobi_fortran.F -o jacobi_fortran.x -O3 -march=native
mpirun ./jacobi_fortran.x < jacobi_fortran_stdin