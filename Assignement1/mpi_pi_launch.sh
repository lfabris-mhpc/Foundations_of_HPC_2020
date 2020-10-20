#!/bin/bash

module purge
module load openmpi/4.0.3/gnu/9.3.0

mpicc code/mpi_pi.c -o mpi_pi.x

dir=weak_scaling
mkdir -p ${dir}

#cumulative=0
for p in 1 #4 8 16 20 24 28 32 36 40 44 48
do
	draws=100000000
	fac=1
	for exp in {8..9}
	do
		wtime=$(date -u -d @$((fac * 48 / p)) +"%H:%M:%S")
		#((cumulative += fac * 48 / p))
		#delay=$(date -u -d @$((cumulative / 60)) +"%H%M")
		cat << EOF > ${dir}/mpi_pi_p${p}_10to${exp}.sh
#!/bin/bash
#PBS -l nodes=1:ppn=48
#${p}
#PBS -l walltime=${wtime}
##PBS -a ${delay}
#PBS -q dssc
#PBS -j oe
#PBS -N mpi_pi_p${p}_10to${exp}

workdir=\${PBS_O_WORKDIR}
cd \${workdir}

module purge
module load openmpi/4.0.3/gnu/9.3.0

/usr/bin/time -f "elapsed: %e\\nuser: %U\\nsys: %S" mpirun --mca btl '^openib' -np ${p} ../mpi_pi.x ${draws}

exit 0
EOF
		cd ${dir}
		for d in {1..3}
		do
			qsub mpi_pi_p${p}_10to${exp}.sh
		done
		cd ..
		((draws *= 10))
		((fac *= 10))
	done
done

cd ..

