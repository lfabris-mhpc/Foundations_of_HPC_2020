#!/bin/bash

id_max=$(ls | grep -Eo "\.o[0-9]+$" | grep -Eo "[0-9]+$" | awk 'BEGIN {max = 0} {if ($1 > max) {max = $1}} END {print max}')
printf "%s\n" ${id_max}

for i in {1..3}
do
	(( id_max += 1 ))
	./omp_strong_scaling_small.sh > zzlog.o${id_max} 2>&1
	(( id_max += 1 ))
	./omp_weak_scaling_small.sh > zzlog.o${id_max} 2>&1
	
	(( id_max += 1 ))
	./mpi_strong_scaling_small.sh > zzlog.o${id_max} 2>&1
	(( id_max += 1 ))
	./mpi_weak_scaling_small.sh > zzlog.o${id_max} 2>&1
done