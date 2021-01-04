#!/bin/bash

img=../images/test_picture.pgm

kernel_size=101
for kernel_type in 0 1
do
	kernel_params="${kernel_type} ${kernel_size}"
	if ((kernel_type == 1))
	then
		kernel_params="${kernel_params}  0.2"
	fi
	out=${img%.pgm}.b_${kernel_type}.pgm

	printf "mpi_p 9 ${kernel_type} ${kernel_size}\n"

	/usr/bin/time -f "elapsed: %e\\nuser: %U\\nsystem: %S" \
	mpirun --mca btl "^openib" --oversubscribe --np 9 \
	-x OMP_NUM_THREADS=1 \
	blur_hybrid.x ${img} ${kernel_params} ${out} < mesh0.stdin
	echo
	
	../tools/img_diff.x ${img%.pgm}.b_${kernel_type}*.pgm ${out} diff_${out} \
	| awk '/^difference/ { c[$10]++ } END {for (w in c) {print w, ":", c[w]}}'

	if [[ -f "${out}" ]]; then
		rm ${out}
	fi

	sleep 10
done
