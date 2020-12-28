#!/bin/bash

max_p=8
k_diam=9
img=images/check_me_4.pgm
#for mpi_p in 8
for ((mpi_p = 2; mpi_p <= ${max_p}; mpi_p *= 2)) 
#for ((mpi_p = 1; mpi_p <= ${max_p}; mpi_p *= 2)) 
do
	for ((omp_p = 1; omp_p <= ${max_p}; omp_p *= 2)) 
	do
		if ((mpi_p * omp_p <= max_p))
		then
			printf "strong scaling (mpi: %d omp: %d) on %s\n" ${mpi_p} ${omp_p} ${img}
			
			if ((mpi_p == 1))
			then
				#threads = max_p
				mapping="--bind-to none"
			elif ((mpi_p < max_p))
			then
				mapping=
				#"-x OMP_PLACES=cores -x OMP_PROC_BIND=spread"
				#"--map-by core --bind-to core -x OMP_PLACES=cores -x OMP_PROC_BIND=spread"
			elif ((mpi_p == max_p))
			then
				#threads = 1
				mapping="--oversubscribe"
			fi
			
			/usr/bin/time -f "elapsed: %e\\nuser: %U\\nsystem: %S" \
			mpirun --mca btl '^openib' --np ${mpi_p} --report-bindings \
			${mapping} \
			-x OMP_NUM_THREADS=${omp_p} \
			blur_hybrid.x ${img} 0 11 blurred.pgm < blur_mpi_np0.stdin
			#-x OMP_PLACES=cores -x OMP_PROC_BIND=spread \
			#> /dev/null
			echo
			
			rm blurred.pgm
			
			sleep 2
		fi
	done
done