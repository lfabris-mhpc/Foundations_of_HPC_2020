#!/bin/bash
#PBS -l nodes=1:ppn=48
#PBS -l walltime=01:00:00
#PBS -q dssc
#PBS -j oe
#PBS -N serial

workdir=${PBS_O_WORKDIR}
cd ${workdir}

module purge
module load openmpi/4.0.3/gnu/9.3.0

n=100000000
for rep in {1..4}
do
	/usr/bin/time -f "elapsed: %e\\nuser: %U\\nsystem: %S"  ../pi.x ${n}
	(( n *= 10 ))
done

exit 0
