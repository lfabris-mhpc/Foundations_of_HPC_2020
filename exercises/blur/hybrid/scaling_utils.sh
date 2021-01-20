function run_omp {
	printf "run mpi 1 omp ${p_omp} ${kernel_params} ${scaling_type}\n"

	/usr/bin/time -f "elapsed: %e\\nuser: %U\\nsystem: %S" \
	mpirun --mca btl "^openib" --np 1 --report-bindings \
	--map-by core --bind-to none \
	-x OMP_NUM_THREADS=${p_omp} -x OMP_PLACES=cores -x OMP_PROC_BIND=close \
	./blur.mpi_omp.x ${img} ${kernel_params} ${out} < mesh0.stdin
	echo

	if [[ -f "${out}" ]]
	then
		rm ${out}
	fi
}

function run_omp_nompirun {
	printf "run mpi 1 omp ${p_omp} ${kernel_params} ${scaling_type}\n"

	export OMP_NUM_THREADS=${p_omp}
	export OMP_PLACES=cores
	export OMP_PROC_BIND=close
	export OMPI_MCA_btl="^openib"
	/usr/bin/time -f "elapsed: %e\\nuser: %U\\nsystem: %S" \
	./blur.mpi_omp.x ${img} ${kernel_params} ${out} < mesh0.stdin
	echo

	if [[ -f "${out}" ]]
	then
		rm ${out}
	fi
}

function run_mpi {
	printf "run mpi ${p_mpi} omp 1 ${kernel_params} ${scaling_type}\n"

	mappings="--map-by core --bind-to core"
	if ((p_mpi > 24))
	then
		mappings="${mappings}:overload-allowed --oversubscribe"
	fi

	/usr/bin/time -f "elapsed: %e\\nuser: %U\\nsystem: %S" \
	mpirun --mca btl "^openib" --np ${p_mpi} --report-bindings \
	${mappings} \
	-x OMP_NUM_THREADS=1 \
	blur.mpi_omp.x ${img} ${kernel_params} ${out} < mesh0.stdin
	echo

	if [[ -f "${out}" ]]
	then
		rm ${out}
	fi
}

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
