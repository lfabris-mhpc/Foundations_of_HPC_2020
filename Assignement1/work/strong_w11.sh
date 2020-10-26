#!/bin/bash
#PBS -l nodes=1:ppn=48
#PBS -l walltime=12:00:00
#PBS -q dssc
#PBS -j oe
#PBS -N strong_w11

workdir=${PBS_O_WORKDIR}
cd ${workdir}

module purge
module load openmpi/4.0.3/gnu/9.3.0

nbase=100000000000
for p in 1 4 8 12 16 20 24 28 32 36 40 44 48
do
	/usr/bin/time -f "elapsed: %e\\nuser: %U\\nsystem: %S" mpirun --mca btl '^openib' -np ${p} ../mpi_pi.x ${nbase}
done

exit 0
