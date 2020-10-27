#!/bin/bash
#PBS -l nodes=1:ppn=24
#PBS -l walltime=00:02:00
#PBS -q thin
#PBS -j oe
#PBS -N weak_w08

workdir=${PBS_O_WORKDIR}
cd ${workdir}

module purge
module load openmpi/4.0.3/gnu/9.3.0

nbase=100000000
for p in 4 8 12 16 20 24
do
	(( n = p * nbase ))
	/usr/bin/time -f "elapsed: %e\\nuser: %U\\nsystem: %S" mpirun --mca btl '^openib' -np ${p} ../mpi_pi.x ${n}
done

exit 0
