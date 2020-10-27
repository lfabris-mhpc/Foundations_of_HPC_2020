#!/bin/bash
#PBS -l nodes=1:ppn=24
#PBS -l walltime=00:40:00
#PBS -q dssc
#PBS -j oe
#PBS -N weak_pE10

workdir=${PBS_O_WORKDIR}
cd ${workdir}

module purge
module load openmpi/4.0.3/gnu/9.3.0

nbase=10000000000
for p in 2 4 6 8 10 12 14 16 18 20 22 24
do
	(( n = p * nbase ))
	/usr/bin/time -f "elapsed: %e\\nuser: %U\\nsystem: %S" mpirun --mca btl '^openib' -np ${p} ../mpi_pi.x ${n}
done

exit 0
