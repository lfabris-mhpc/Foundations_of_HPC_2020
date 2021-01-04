#!/bin/bash

#PBS -l nodes=1:ppn=48
#PBS -l walltime=10:00:00
#PBS -q dssc
#PBS -j oe
#PBS -N omp_strong

if [ -n "${PBS_O_WORKDIR}" ]
then
	workdir=${PBS_O_WORKDIR}
	cd ${workdir}
	
	module purge
	module load openmpi/4.0.3/gnu/9.3.0
	
	img=/scratch/dssc/lfabris/earth-large.pgm
else
	img=../images/test_picture.pgm
fi

if [ -n "${PBS_NUM_PPN}" ]
then
	p_max=${PBS_NUM_PPN}
else
	p_max=$(grep -c "physical id" /proc/cpuinfo )
fi

out=blurred.pgm

cooldown=1

scaling_type="strong"
source scaling_utils.sh

if [ -n "${PBS_O_WORKDIR}" ]
then
	workdir=${PBS_O_WORKDIR}
	cd ${workdir}
	
	module purge
	module load openmpi/4.0.3/gnu/9.3.0
fi

#warm up disk
../tools/img_diff.x ${img} ${img}

for kernel_size in 11 101
do
	for kernel_type in 1
	do
		kernel_params="${kernel_type} ${kernel_size}"
		if ((kernel_type == 1))
		then
			kernel_params="${kernel_params}  0.2"
		fi
		
		for ((p_omp = 1; p_omp <= ${p_max}; p_omp *= 2))
		do
			run_omp

			sleep ${cooldown}
		done
		
		if ((p_omp / 2 != p_max))
		then
			p_omp=${p_max}
			
			run_omp
			
			sleep ${cooldown}
		fi
	done
done
