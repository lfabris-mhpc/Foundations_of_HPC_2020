#include <stdlib.h>
#include <unistd.h>
#include <stdio.h>
#include <math.h>
#include <string.h>
#include <mpi.h>
#define USE MPI
#define SEED 35791246

int main(int argc, char *argv[])
{
	double start_time, end_time;   
	int rankOld, rank, rankNum, proc;
	MPI_Status status;
	MPI_Request request;
	
	int master = 0;
	int tag = 123;

	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &rankNum);
	MPI_Comm_rank(MPI_COMM_WORLD, &rankOld);

	int* ringProcs = (int*) malloc(sizeof(int) * rankNum);
	int i = 0;
	for (; i <= rankNum / 2; ++i) {
		ringProcs[i] = 2 * i;
	}
	for (; i < rankNum; ++i) {
		ringProcs[i] = 2 * (i - rankNum / 2) - 1;
	}
	if (rankOld == 0) {
		for (int i = 0; i < rankNum; ++i) {
			printf("ringProcs[%d]: %d\n", i, ringProcs[i]);
		}
	}

	MPI_Group ringGroup;
	MPI_Comm_group(MPI_COMM_WORLD, &ringGroup);
	MPI_Group_incl(ringGroup, rankNum, ringProcs, &ringGroup);
	MPI_Comm newComm;
	MPI_Comm_create(MPI_COMM_WORLD, ringGroup, &newComm);

	MPI_Comm_rank(newComm, &rank);
	printf("rank in newComm: %d (old: %d)\n", rank, rankOld);

	int qperiodic = 1;
	MPI_Comm cartComm;
	MPI_Cart_create(newComm, 1, &rankNum, &qperiodic, 0, &cartComm);
	MPI_Comm_rank(cartComm, &rank);

	start_time = MPI_Wtime();

	//MPI_Scatterv(values, sendCounts, displacements, MPI_DOUBLE, localValues, localSize, MPI_DOUBLE, master, MPI_COMM_WORLD);

	printf("rank in cart: %d (old: %d)\n", rank, rankOld);

	//MPI_Reduce(&localSum, &sum, 1, MPI_DOUBLE, MPI_SUM, master, MPI_COMM_WORLD);

	if (rank == master) {
		end_time = MPI_Wtime();
		
		//printf("# of elements = %llu , parallel sum is %f check sum is %f \n", N, sum, checkSum);
		//printf("# walltime on master processor : %10.8f \n", end_time - start_time);
	} else {
		end_time = MPI_Wtime();
		//printf("\n # walltime on processor %i : %10.8f \n", rank, end_time - start_time);
	}

	MPI_Finalize();
}
