#!/bin/bash

#PBS -l nodes=1:ncpus=24
#PBS -l walltime=01:00:00
#PBS -q dssc
#PBS -j oe
#PBS -N mpi_weak

p_omp=1
out=blurred${PBS_JOBID}.pgm
cooldown=5

img=../images/gradient_1.pgm

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

for kernel_size in 11 31
do
	for kernel_type in 1
	do
		kernel_params="${kernel_type} ${kernel_size}"
		if ((kernel_type == 1))
		then
			kernel_params="${kernel_params} 0.2"
		fi

		for p_mpi in 1 {2..24..2}
		do
			img=../images/gradient_${p_mpi}.pgm
			
			#warm up disk
			../tools/img_diff.x ${img} ${img}

			run_hybrid

			sleep ${cooldown}
		done
	done
done
