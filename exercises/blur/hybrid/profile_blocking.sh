#!/bin/bash

img=../images/test_picture.pgm
max_p=1

tmp_img=blurred.pgm

#warm up disk
../tools/img_diff.x ${img} ${img}
kernel_type=0
ccflags="-DBLOCKING_BLUR_OFF -DBLOCKING_POS_OFF"

#comment='
make clean
make CCFLAGS="${ccflags}"

printf "testing ccflags ${ccflags}\n"
for np in 1 4
do
	for kernel_size in 101
	do
		kernel_params="${kernel_type} ${kernel_size}"

		printf "  test blocking ${kernel_type} ${kernel_size}\n"

		/usr/bin/time -f "elapsed: %e\\nuser: %U\\nsystem: %S" \
		mpirun --mca btl "^openib" --np 1 \
		--bind-to none \
		-x OMP_NUM_THREADS=${np} -x OMP_PLACES=cores -x OMP_PROC_BIND=spread \
		blur_hybrid.x ${img} ${kernel_params} ${tmp_img} < ../blur_mpi_np0.stdin | grep "timing_blur"
		echo

		if [[ -f "${tmp_img}" ]]; then
			rm ${tmp_img}
		fi

		sleep 5
	done
done
#'

#blocking_blur has no effect
for block_enabled in 1 2 3
do	
	for block_size in 32 64 #128 #256
	do
		if ((block_enabled == 1))
		then
			ccflags="-DBLOCKING_BLUR_OFF -DBLOCKING_POS_ROWS=${block_size} -DBLOCKING_POS_COLUMNS=${block_size}"
		elif ((block_enabled == 2))
		then
			ccflags="-DBLOCKING_BLUR_ROWS=${block_size} -DBLOCKING_BLUR_COLUMNS=${block_size} -DBLOCKING_POS_OFF"
		elif ((block_enabled == 3))
		then
			ccflags="-DBLOCKING_BLUR_ROWS=${block_size} -DBLOCKING_BLUR_COLUMNS=${block_size} -DBLOCKING_POS_ROWS=${block_size} -DBLOCKING_POS_COLUMNS=${block_size}"
		fi

		make clean
		make CCFLAGS="${ccflags}"

		printf "testing ccflags ${ccflags}\n"
		for np in 1 4
		do
			for kernel_size in 101
			do
				kernel_params="${kernel_type} ${kernel_size}"

				printf "  test blocking ${kernel_type} ${kernel_size}\n"

				/usr/bin/time -f "elapsed: %e\\nuser: %U\\nsystem: %S" \
				mpirun --mca btl "^openib" --np 1 \
				--bind-to none \
				-x OMP_NUM_THREADS=${np} -x OMP_PLACES=cores -x OMP_PROC_BIND=spread \
				blur_hybrid.x ${img} ${kernel_params} ${tmp_img} < ../blur_mpi_np0.stdin | grep "timing_blur"
				echo

				if [[ -f "${tmp_img}" ]]; then
					rm ${tmp_img}
				fi

				sleep 5
			done
		done
	done
done
