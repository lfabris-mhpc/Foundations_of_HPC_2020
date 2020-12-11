#!/bin/bash

procs=4
iterations=10
for mesh_dims in 2 #1 2 3
do
	for comm_type in COMM_NOBLOCK #COMM_SENDRECV
	do
		for sort_dims in DIMS_SORT_NO #DIMS_SORT 
		do
			for upd_field in UPD_FIELD_BLOCKS #UPD_FIELD_BLOCKS #UPD_FIELD_SIMD UPD_FIELD
			do
				for simd in SIMD #SIMD_NO
				do
					for threading in OMP_NONE #OMP_LOOPS #OMP_TASKS
					do
						for block in 16 # 8 16 24 32 40 48 64
						do
							#printf "\ncomm_type: %s sort_dims: %s upd_field: %s simd: %s mesh_dims: %d\n" ${comm_type} ${sort_dims} ${upd_field} ${simd} ${mesh_dims}
							printf "\nblock: %s threading: %s mesh_dims: %d\n" ${block} ${threading} ${mesh_dims}
							mpicc jacobi.c -o jacobi.x -O3 -march=native -lm -D${comm_type} -D${sort_dims} -D${upd_field} -D${simd} -DBLOCK=${block} -D${threading} #-DDBG

							sleep 2
							perf stat --detailed mpirun --oversubscribe --np ${procs} ./jacobi.x ${mesh_dims} 100 100 100 ${iterations} 0
							# --map-by L1cache
							#perf stat --detailed mpirun --oversubscribe --np ${procs} ./jacobi.x ${mesh_dims} 600 600 600 ${iterations} 0

							#sleep 2
							#perf stat --detailed mpirun --oversubscribe --np ${procs} --map-by L1cache ./jacobi.x ${mesh_dims} 500 500 100 ${iterations} > /dev/null
							#sleep 2
							#perf stat --detailed mpirun --oversubscribe --np ${procs} --map-by L1cache ./jacobi.x ${mesh_dims} 500 100 500 ${iterations} > /dev/null
							#sleep 2
							#perf stat --detailed mpirun --oversubscribe --np ${procs} --map-by L1cache ./jacobi.x ${mesh_dims} 100 500 500 ${iterations} > /dev/null
						done
					done
				done
			done
		done
	done
done
