#!/bin/bash

#build exes
make -C hybrid/ clean
make -C hybrid/ CCFLAGS=-DNDEBUG
mv hybrid/blur_hybrid.x .

#build tools
make -C tools/

#generate gradient images
tools/make_gradient.x 16 16 65535
mv tools/gradient.pgm images/gradient_small.pgm
tools/make_gradient.x 512 512 65535
mv tools/gradient.pgm images/gradient.pgm

for img in images/gradient_small.pgm images/gradient.pgm images/check_me.pgm images/eevee.pgm
do
	tools/img_normalize.x ${img}
	echo

	for params in "0 1" "1 9 1" "2 1"
	do
		printf "testing %s with kernel parameters %s:\n" ${img} "${params}"

		#perf stat --detailed mpirun --np 1 --report-bindings --map-by core blur_hybrid.x images/eevee.pgm 0 51 < blur_mpi_np0.stdin
		out=${img%.pgm}_id.pgm

		mpirun --oversubscribe --np 8 --map-by core blur_hybrid.x ${img} ${params} ${out} < blur_mpi_np0.stdin #> /dev/null
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