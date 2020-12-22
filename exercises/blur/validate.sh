#!/bin/bash

make clean
make CCFLAGS=-DNDEBUG

gcc -o read_write_pgm_image.x read_write_pgm_image.c
./read_write_pgm_image.x > /dev/null
mv image.pgm gradient.pgm
rm check_me.back.pgm

gcc -o img_diff.x img_diff.c
gcc -o normalize.x normalize.c

for img in eevee.pgm gradient_small.pgm gradient.pgm check_me.pgm eevee.pgm
do
	./normalize.x ${img}
	echo

	for params in "0 1" "1 51 1" "2 1"
	do
		printf "testing %s with kernel parameters %s:\n" ${img} "${params}"

		#perf stat --detailed mpirun --np 1 --report-bindings --map-by core blur_hybrid.x images/eevee.pgm 0 51 < blur_mpi_np0.stdin
		out=${img%.pgm}_id.pgm

		mpirun --np 1 --map-by core blur_hybrid.x ${img} ${params} ${out} < blur_mpi_np0.stdin > /dev/null
		echo

		printf "img_diff %s %s:\n" ${img} ${out}
		python img_diff.py ${img} ${out} | head -n 10
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