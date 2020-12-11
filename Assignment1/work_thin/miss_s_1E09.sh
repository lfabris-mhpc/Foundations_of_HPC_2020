#!/bin/bash
#PBS -l nodes=1:ppn=24
#PBS -l walltime=00:02:00
#PBS -q dssc
#PBS -j oe
#PBS -N miss_s_1E09

workdir=${PBS_O_WORKDIR}
cd ${workdir}

module purge
module load openmpi/4.0.3/gnu/9.3.0

nbase=1000000000
for p in 2 6 10 14 18 22
do
	/usr/bin/time -f "elapsed: %e\\nuser: %U\\nsystem: %S" mpirun --mca btl '^openib' -np ${p} ../mpi_pi.x ${nbase}
done

exit 0
