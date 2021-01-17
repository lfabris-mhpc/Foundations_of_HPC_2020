#!/bin/bash

source scaling_utils.sh

p_mpi=$1
p_omp=$2
kernel_params="2 101"  #"0 101" #"1 101 0.2" #"2 101"
scaling_type=test
img=../images/test_picture.pgm #../images/eevee.pgm #../images/test_picture.pgm
out=""

for kernel_params in "0 101" "1 101 0.2" "2 101"
do
	run_hybrid
done