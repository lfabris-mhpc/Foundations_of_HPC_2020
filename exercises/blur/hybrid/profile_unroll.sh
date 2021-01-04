#!/bin/bash

img=../images/test_picture.pgm
max_p=1

out=blurred.pgm

#warm up disk
../tools/img_diff.x ${img} ${img}
kernel_type=0
kernel_size=101

printf "testing ccflags ${ccflags}\n"
for unroll in 2 4 8
do
	make clean
	make CCFLAGS="-DUNROLL=${unroll}"
	
	kernel_params="${kernel_type} ${kernel_size}"

	printf "test unroll ${unroll}\n"

	/usr/bin/time -f "elapsed: %e\\nuser: %U\\nsystem: %S" \
	mpirun --mca btl "^openib" --np 1 \
	-x OMP_NUM_THREADS=1 \
	blur_hybrid.x ${img} ${kernel_params} ${out} < mesh0.stdin
	#| grep "timing_blur"
	echo
	
	../tools/img_diff.x ${img%.pgm}.b_${kernel_type}*.pgm ${out} diff_${out} \
	| awk '/^difference/ { c[$10]++ } END {for (w in c) {print w, ":", c[w]}}'

	if [[ -f "${out}" ]]; then
		rm ${out}
	fi
	if [[ -f "diff_${out}" ]]; then
		rm diff_${out}
	fi

	sleep 5
done
