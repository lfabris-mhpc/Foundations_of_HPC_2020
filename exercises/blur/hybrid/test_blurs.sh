#!/bin/bash

#PBS -l nodes=1:ppn=24:kind=thin
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

        run_hybrid
        sleep ${cooldown}
    done
done
