#!/bin/bash

iterations=100
for comm_type in COMM_NOBLOCK #COMM_SENDRECV
do
	for sort_dims in NO_SORT_DIMS #SORT_DIMS 
	do
		for upd_field in UPD_FIELD_BLOCKS UPD_FIELD
		do
			mpicc jacobi.c -o jacobi.x -O3 -march=native -lm -D${comm_type} -D${sort_dims} -D${upd_field}

			#time mpirun --oversubscribe --np 4 ./jacobi.x 3 4 4 4 10

			for mesh_dims in 1 2 3
			do
				printf "\ncomm_type: %s sort_dims: %s upd_field: %s mesh_dims: %d\n" ${comm_type} ${sort_dims} ${upd_field} ${mesh_dims}

				sleep 2
				perf stat --detailed mpirun --oversubscribe --np 4 --map-by L1cache ./jacobi.x ${mesh_dims} 500 500 100 ${iterations} > /dev/null
				sleep 2
				perf stat --detailed mpirun --oversubscribe --np 4 --map-by L1cache ./jacobi.x ${mesh_dims} 500 100 500 ${iterations} > /dev/null
				sleep 2
				perf stat --detailed mpirun --oversubscribe --np 4 --map-by L1cache ./jacobi.x ${mesh_dims} 100 500 500 ${iterations} > /dev/null
			done
		done
	done
done
