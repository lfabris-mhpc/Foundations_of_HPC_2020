#!/bin/bash

for exe in blur_hybrid.x blur_hybrid_wcomm.x
do
	#rm blurred.pgm
	
	printf "testing %s:\n" ${exe}
	img=images/check_me_4.pgm
	params="2 51"
	
	export OMP_NUM_THREADS=4
	export OMP_PLACES=cores
	#export OMP_PROC_BIND=spread
	perf stat --detailed ./${exe} ${img} ${params} blurred.pgm < blur_mpi_np0.stdin
	#mpirun --np 1 -x OMP_NUM_THREADS=${OMP_NUM_THREADS} -x OMP_PLACES=${OMP_PLACES} -x OMP_PROC_BIND=${OMP_PROC_BIND} ${exe} ${img} ${params} blurred.pgm < blur_mpi_np0.stdin
	
	sleep 10
	echo
done