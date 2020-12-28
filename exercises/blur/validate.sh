#!/bin/bash

for exe in blur_hybrid.x blur_hybrid_wcomm.x
do
	printf "testing %s:\n" ${exe}
	for img in images/gradient_small.pgm images/gradient.pgm images/check_me.pgm images/eevee.pgm
	do
		for params in "0 1" "1 9 1" "2 1"
		do
			printf "testing %s with kernel parameters %s:\n" ${img} "${params}"

			#perf stat --detailed mpirun --np 1 --report-bindings --map-by core blur_hybrid.x images/eevee.pgm 0 51 < blur_mpi_np0.stdin
			out=${img%.pgm}_id.pgm

			mpirun --oversubscribe --np 8 --map-by core ${exe} ${img} ${params} ${out} < blur_mpi_np0.stdin > /dev/null 2> /dev/null
			echo

			printf "img_diff %s %s:\n" ${img} ${out}
			tools/img_diff.x ${img} ${out} | head -n 10
			echo

			xxd ${img} > img.hex
			xxd ${out} > out.hex
			printf "diff (hex dumps) %s %s:\n" ${img} ${out}
			diff img.hex out.hex | head -n 10
			echo

			rm ${out} img.hex out.hex

			sleep 2
		done
	done
done