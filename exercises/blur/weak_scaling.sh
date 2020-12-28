#!/bin/bash

max_p=8
k_diam=9
p=
#for mpi_p in 1
#for ((mpi_p = 2; mpi_p <= ${max_p}; mpi_p *= 2)) 
for ((mpi_p = 1; mpi_p <= ${max_p}; mpi_p *= 2)) 
do 
	#for ((omp_p = 1; omp_p <= ${max_p}; omp_p *= 2)) 
	for omp_p in 1
	do
		if ((mpi_p * omp_p <= max_p))
		then
			((p = 4 * mpi_p * omp_p))
			img=images/check_me_${p}.pgm
			printf "weak scaling (mpi: %d omp: %d) on %s\n" ${mpi_p} ${omp_p} ${img}
			
			#((pe = max_p / mpi_p))
			#mapping="--map-by socket:PE=${pe} --oversubscribe -x OMP_PLACES=cores -x OMP_PROC_BIND=close"
			#mapping="--map-by socket:PE=${mpi_p}"
			mapping="--map-by socket:OVERSUBSCRIBE --bind-to core:overload-allowed"
			
			/usr/bin/time -f "elapsed: %e\\nuser: %U\\nsystem: %S" \
			mpirun --mca btl '^openib' --np ${mpi_p} --report-bindings \
			${mapping} \
			-x OMP_NUM_THREADS=${omp_p} \
			blur_hybrid.x ${img} 0 11 blurred.pgm < blur_mpi_np0.stdin
			#> /dev/null
			echo
			
			rm blurred.pgm
			
			sleep 2
		fi
	done
done