#!/bin/bash

#PBS -l nodes=1:ncpus=24
#PBS -l walltime=00:30:00
#PBS -q dssc
#PBS -j oe
#PBS -N omp_weak

p_mpi=1
out=blurred${PBS_JOBID}.pgm
cooldown=5

img=../images/test_picture_1.pgm

if [ -n "${PBS_O_WORKDIR}" ]
then
	workdir=${PBS_O_WORKDIR}
	cd ${workdir}

	module purge
	module load openmpi/4.0.3/gnu/9.3.0
fi

scaling_type="weak"
source scaling_utils.sh

hostname
date
echo

#warm up disk
../tools/img_diff.x ${img} ${img}

for kernel_size in 11 31
do
	for kernel_type in 1
	do
		kernel_params="${kernel_type} ${kernel_size}"
		if ((kernel_type == 1))
		then
			kernel_params="${kernel_params} 0.2"
		fi

		for p_omp in 1 {2..24..2}
		do
			img=../images/test_picture_${p_omp}.pgm

			run_hybrid

			sleep ${cooldown}
		done
	done
done
