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

module purge
module load openmpi/4.0.3/gnu/9.3.0

mpicc code/mpi_pi.c -o mpi_pi.x

mkdir -p ${workdir}
cd ${workdir}

nbase=100000000
ebase=8
eup=11

for p in 1 2 4 #8 16 20 24 28 32 36 40 44 48
do
	n=nbase
	if [ $1 == "weak" ]
	then
		(( n = p * nbase ))
	fi
	
	for (( exp=ebase; exp <= eup; ++exp ))
	do
		script=mpi_pi_p$(printf "%02d" ${p})_10to$(printf "%02d" ${exp})
		#n=10**8 p=48 -> 70s
		walltime=$(date -u -d @$(( (n / nbase) * 25 * 48 / p )) +"%H:%M:%S")
		
		cat << EOF > ${script}.sh
#!/bin/bash
#PBS -l nodes=1:ppn=48
#PBS -l walltime=${walltime}
#PBS -q dssc
#PBS -j oe
#PBS -N ${script}

workdir=\${PBS_O_WORKDIR}
cd \${workdir}

#module purge
#module load openmpi/4.0.3/gnu/9.3.0

/usr/bin/time -f "elapsed: %e\\nuser: %U\\nsys: %S" mpirun --mca btl '^openib' -np ${p} ../mpi_pi.x ${n}

exit 0
EOF
		
		chmod u+x ${script}.sh
		(( n *= 10 ))
	done
done

cd ..
