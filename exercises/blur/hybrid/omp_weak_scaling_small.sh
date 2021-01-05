#!/bin/bash

#PBS -l nodes=1:ppn=48
#PBS -l walltime=10:00:00
#PBS -q dssc
#PBS -j oe
#PBS -N omp_strong

cores=$(lscpu | awk 'BEGIN {total = 0; cores = 0} /Core\(s\) per socket:/ {cores = $4} /Socket\(s\):/ {total += cores * $2; cores = 0} END { print total }')
hwthreads=$(grep -c "physical id" /proc/cpuinfo)
p_mpi=1
out=blurred.pgm
cooldown=5

if [ -n "${PBS_O_WORKDIR}" ]
then
	img=/scratch/dssc/lfabris/earth-large.pgm
	out=/scratch/dssc/lfabris/${out}
else
	img=../images/test_picture.pgm
fi

p_max=${cores}

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
		for ((p_omp = 2; p_omp <= ${p_max}; p_omp *= 2))
		do
			if [ -n "${PBS_O_WORKDIR}" ]
			then
				img=/scratch/dssc/lfabris/earth-large_${p_omp}.pgm
			else
				img=../images/test_picture_${p_omp}.pgm
			fi

			run_omp_nompirun

			if [ -n "${PBS_JOBID}" ]
			then
				printf "done mpi ${p_mpi} omp ${p_omp} kernel_params ${kernel_params}\n" > ${PBS_JOBID}.progress
			fi

			sleep ${cooldown}
		done

		if ((p_omp / 2 != p_max))
		then
			p_omp=${p_max}

			if [ -n "${PBS_O_WORKDIR}" ]
			then
				img=/scratch/dssc/lfabris/earth-large_${p_omp}.pgm
			else
				img=../images/test_picture_${p_omp}.pgm
			fi

			run_omp_nompirun

			if [ -n "${PBS_JOBID}" ]
			then
				printf "done mpi ${p_mpi} omp ${p_omp} kernel_params ${kernel_params}\n" > ${PBS_JOBID}.progress
			fi

			sleep ${cooldown}
		fi
	done
done
