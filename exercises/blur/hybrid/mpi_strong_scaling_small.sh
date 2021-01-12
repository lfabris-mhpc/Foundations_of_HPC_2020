#!/bin/bash

#PBS -l nodes=1:ppn=48
#PBS -l walltime=12:00:00
#PBS -q dssc
#PBS -j oe
#PBS -N mpi_strong

p_omp=1
out=blurred${PBS_JOBID}.pgm
cooldown=5

if [ -n "${PBS_O_WORKDIR}" ]
then
	img=../images/earth-large.pgm
else
	img=../images/test_picture.pgm
fi

if [ -n "${PBS_O_WORKDIR}" ]
then
	workdir=${PBS_O_WORKDIR}
	cd ${workdir}

	module purge
	module load openmpi/4.0.3/gnu/9.3.0
fi

scaling_type="strong"
source scaling_utils.sh

hostname
date
echo

#warm up disk
../tools/img_diff.x ${img} ${img}

for kernel_size in 11 101
do
	for kernel_type in 1
	do
		kernel_params="${kernel_type} ${kernel_size}"
		if ((kernel_type == 1))
		then
			kernel_params="${kernel_params} 0.2"
		fi

		#skip p_mpi=1, p_omp=1
		for 1 {4..48..4}
		do
			run_mpi

			sleep ${cooldown}
		done
	done
done
