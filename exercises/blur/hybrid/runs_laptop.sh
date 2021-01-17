#!/bin/bash

logdir=logs_laptop
mkdir -p ${logdir}

./omp_strong_scaling.sh >> ${logdir}/omp_strong.o000
./omp_weak_scaling.sh >> ${logdir}/omp_weak.o000
./mpi_strong_scaling.sh >> ${logdir}/mpi_strong.o000
./mpi_weak_scaling.sh >> ${logdir}/mpi_weak.o000