#!/bin/bash

#10to08
for n in 1 4 8 16 20 24 28 32 36 40 44 48 ; do
do
	draws=100000000
	hrmin=$(date -u -d@$(($i * 10 * 60)) +"%H:%M")
	cat <<EOF > mpi_pi_n${n}.sh
#!/bin/bash
#PBS -l nodes=1:ppn=${n}
#PBS -l walltime=1:00:00
#PBS -q dssc
#PBS -N mpi_pi_n${n}

workdir=${PBS_O_WORKDIR}
cd ${workdir}

module purge
module load openmpi/4.0.3/gnu/9.3.0

time -f "%e" mpirun --mca btl '^openib' -np ${n} mpi_pi.x ${draws} > mpi_pi_n${n}

exit 0
EOF
	for d in {1..3} do
		qsub mpi_pi_n${n}.sh
	done
done

