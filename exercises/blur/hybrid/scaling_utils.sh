
function run_hybrid {
	printf "run mpi ${p_mpi} omp ${p_omp} ${kernel_params} ${scaling_type}\n"

	mappings="--map-by node:PE=${p_omp} --oversubscribe --bind-to core:overload-allowed --oversubscribe"
	if ((p_mpi == 1))
	then
		mappings="--bind-to none"
	fi
	
	/usr/bin/time -f "elapsed: %e\\nuser: %U\\nsystem: %S" \
	mpirun --mca btl "^openib" --np ${p_mpi} --report-bindings \
	--np ${p_mpi} \
	${mappings} \
	-x OMP_NUM_THREADS=${p_omp} -x OMP_PLACES=cores -x OMP_PROC_BIND=close \
	./blur.mpi_omp.x ${img} ${kernel_params} ${out} < mesh0.stdin
	echo

	if [[ -f "${out}" ]]
	then
		rm ${out}
	fi
}
