#!/bin/bash
#PBS -l nodes=1:ppn=48
#PBS -l walltime=12:00:00
#PBS -q dssc
#PBS -j oe
#PBS -N weak_w11

workdir=${PBS_O_WORKDIR}
cd ${workdir}

module purge
module load openmpi/4.0.3/gnu/9.3.0

nbase=100000000000
for p in 12 24 48
do
	(( n = p * nbase ))
	/usr/bin/time -f "elapsed: %e\\nuser: %U\\nsystem: %S" mpirun --mca btl '^openib' -np ${p} ../mpi_pi.x ${n}
done

exit 0
