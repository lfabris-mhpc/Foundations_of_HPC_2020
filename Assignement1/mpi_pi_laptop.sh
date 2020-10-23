#!/bin/bash

if [ $1 == "strong" ]
then
	workdir=strong_scaling
elif [ $1 == "weak" ]
then
	workdir=weak_scaling
else
	exit 1
fi

mkdir -p ${workdir}
cd ${workdir}

reps=3

for p in 1 2 4 #1 4 8 12 16 20 24 28 32 36 40 44 48
do
	for exp in 8 9 10 11
	do
		f=mpi_pi_p$(printf "%02d" ${p})_10to$(printf "%02d" ${exp})
		if [[ -f "${f}.sh" ]]
		then
			echo "${f}.sh"
			for (( i = 0; i < reps; ++i ))
			do
				./${f}.sh >> ${f}.o${i} 2>> ${f}.o${i}
			done
		fi
	done
done

cd ..
