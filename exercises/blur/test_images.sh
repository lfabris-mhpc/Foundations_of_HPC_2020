#!/bin/bash

export OMP_NUM_THREADS=4
./blur_hybrid.x images/test_picture.pgm 0 101 tmp0.pgm < blur_mpi_np0.stdin
tools/img_diff.x tmp0.pgm images/test_picture.b_0_101x101.pgm tmp_diff0.pgm | awk '1 { c[$10]++ } END {for (w in c) {print w, ":", c[w]}}' > diff0
cat diff0
echo

./blur_hybrid.x images/test_picture.pgm 1 101 0.2 tmp1.pgm < blur_mpi_np0.stdin
tools/img_diff.x tmp1.pgm images/test_picture.b_1_101x101_02.pgm tmp_diff1.pgm | awk '1 { c[$10]++ } END {for (w in c) {print w, ":", c[w]}}' > diff1
cat diff1