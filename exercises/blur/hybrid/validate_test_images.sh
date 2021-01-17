#!/bin/bash

img=../images/test_picture.pgm

source scaling_utils.sh

kernel_size=101
for kernel_type in 0 1
do
	kernel_params="${kernel_type} ${kernel_size}"
	if ((kernel_type == 1))
	then
		kernel_params="${kernel_params}  0.2"
	fi
	out=blurred_${kernel_type}.pgm

	printf "mpi_p 9 ${kernel_type} ${kernel_size}\n"
	
	p_mpi=9
	p_omp=1

	run_hybrid
	echo

	../tools/img_diff.x ${img%.pgm}.b_${kernel_type}*.pgm ${out} diff_${out} \
	| awk '/^difference/ { c[$10]++ } END {for (w in c) {print w, ":", c[w]}}'

	sleep 10
done
