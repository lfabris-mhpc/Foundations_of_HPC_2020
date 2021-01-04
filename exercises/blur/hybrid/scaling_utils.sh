function run_omp {
	printf "run mpi 1 omp ${p_omp} ${kernel_params} ${scaling_type}\n"

	/usr/bin/time -f "elapsed: %e\\nuser: %U\\nsystem: %S" \
	mpirun --mca btl "^openib" --np 1 --report-bindings \
	--map-by core --bind-to none \
	-x OMP_NUM_THREADS=${p_omp} -x OMP_PLACES=cores -x OMP_PROC_BIND=spread \
	blur_hybrid.x ${img} ${kernel_params} ${out} < mesh0.stdin
	echo

	if [[ -f "${out}" ]]
	then
		rm ${out}
	fi
}

function run_mpi {
	printf "run mpi ${p_mpi} omp 1 ${kernel_params} ${scaling_type}\n"

	/usr/bin/time -f "elapsed: %e\\nuser: %U\\nsystem: %S" \
	mpirun --mca btl "^openib" --np ${p_mpi} --report-bindings \
	--map-by core --bind-to core \
	-x OMP_NUM_THREADS=1 \
	blur_hybrid.x ${img} ${kernel_params} ${out} < mesh0.stdin
	echo

	if [[ -f "${out}" ]]
	then
		rm ${out}
	fi
}

function run_hybrid {
	printf "run mpi ${p_mpi} omp ${p_omp} ${kernel_params} ${scaling_type}\n"

	/usr/bin/time -f "elapsed: %e\\nuser: %U\\nsystem: %S" \
	mpirun --mca btl "^openib" --np ${p_mpi} --report-bindings \
	--map-by core --bind-to none \
	-x OMP_NUM_THREADS=${omp_p} -x OMP_PLACES=cores -x OMP_PROC_BIND=spread \
	blur_hybrid.x ${img} ${kernel_params} ${out} < mesh0.stdin
	echo

	if [[ -f "${out}" ]]
	then
		rm ${out}
	fi
}