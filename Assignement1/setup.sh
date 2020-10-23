#!/bin/bash

wdir=work
mkdir -p ${wdir}

gcc code/pi.c -o ${wdir}/pi.x

module purge
module load openmpi/4.0.3/gnu/9.3.0

mpicc code/mpi_pi.c -o ${wdir}/mpi_pi.x

