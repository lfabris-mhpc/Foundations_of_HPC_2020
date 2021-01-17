#!/bin/bash

#PBS -l nodes=1:ncpus=24
#PBS -l walltime=00:10:00
#PBS -q dssc
#PBS -j oe
#PBS -N test_pics

p_mpi=1
p_omp=24
out=""
cooldown=5

img=../images/test_picture.pgm

if [ -n "${PBS_O_WORKDIR}" ]
then
    workdir=${PBS_O_WORKDIR}
    cd ${workdir}

    module purge
    module load openmpi/4.0.3/gnu/9.3.0
fi

scaling_type="test_pics"
source scaling_utils.sh

hostname
date
echo

for kernel_size in 101
do
    for kernel_type in 0 1
    do
        kernel_params="${kernel_type} ${kernel_size}"
        if ((kernel_type == 1))
        then
            kernel_params="${kernel_params} 0.2"
        fi

        #run_hybrid

	printf "run mpi ${p_mpi} omp ${p_omp} ${kernel_params} ${scaling_type}\n"

	mappings="--map-by node:PE=${p_omp} --oversubscribe --bind-to core:overload-allowed --oversubscribe"
	if ((p_mpi == 1))
	then
		mappings="--bind-to none"
	fi
	
	/usr/bin/time -f "elapsed: %e\\nuser: %U\\nsystem: %S" \
	mpirun --report-bindings \
	--np ${p_mpi} \
	${mappings} \
	-x OMP_NUM_THREADS=${p_omp} -x OMP_PLACES=cores -x OMP_PROC_BIND=close \
	./blur.mpi_omp.x ${img} ${kernel_params} ${out} < mesh0.stdin
	echo

        sleep ${cooldown}
    done
done
