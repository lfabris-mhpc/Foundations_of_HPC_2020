#!/bin/bash

#img=images/earth-large.pgm
img=../images/check_me.pgm
max_p=4

tmp_img=blurred.pgm

#warm up disk
../tools/img_diff.x ${img} ${img}

for kernel_size in 11 101 501
do
	for kernel_type in 0 1 2
	do
		kernel_params="${kernel_type} ${kernel_size}"
		if ((kernel_type == 1))
		then
			kernel_params="${kernel_params}  0.2"
		fi
		
		for ((np = 4; np <= ${max_p}; np += 4))
		do
			printf "strong scaling mpi ${np} ${kernel_type} ${kernel_size}\n"

			/usr/bin/time -f "elapsed: %e\\nuser: %U\\nsystem: %S" \
			mpirun --mca btl "^openib" --np ${np} --report-bindings \
			--map-by core --bind-to core \
			-x OMP_NUM_THREADS=1 \
			blur_hybrid.x ${img} ${kernel_params} ${tmp_img} < ../blur_mpi_np0.stdin
			echo
			
			if [[ -f "${tmp_img}" ]]; then
				rm ${tmp_img}
			fi

			rm blurred.pgm

			sleep 10
		done
	done
done
