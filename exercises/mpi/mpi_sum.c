#include <stdlib.h>
#include <unistd.h>
#include <stdio.h>
#include <math.h>
#include <string.h>
#include <mpi.h>
#define USE MPI
#define SEED 35791246

int main ( int argc , char *argv[ ] )
{
	long long int M, local_M; 

	double start_time, end_time;   
	int rank, rankNum, proc;
	MPI_Status status;
	MPI_Request request;
	
	int master = 0;
	int tag = 123;

	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &rankNum);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	if (argc <= 1) {
		fprintf (stderr, " Usage : mpi_sum -np n %s number_of_elements \n", argv[0]);
		MPI_Finalize();
		exit(-1);
	}

	long long int N = atoll(argv[1]);
	int localSize = N / rankNum + (rank < (N % rankNum) ? 1 : 0);
	double* values = NULL;
	double* localValues = NULL;
	double localSum, sum, checkSum;

	int* sendCounts = NULL;
	int* displacements = NULL;

	start_time = MPI_Wtime();

	localValues = (double*) malloc(sizeof(double) * localSize);

	if (rank == master) {
		sendCounts = (int*) malloc(sizeof(int) * rankNum);
		displacements = (int*) malloc(sizeof(int) * rankNum);

		int baseSize = N / rankNum;
		for (int i = 0; i < rankNum; ++i) {
			sendCounts[i] = baseSize;
		}
		int remainder = N % rankNum;
		for (int i = 0; i < remainder; ++i) {
			sendCounts[i] += 1;
		}
		displacements[0] = 0;
		for (int i = 0; i < rankNum; ++i) {
			displacements[i] = displacements[i-1] + sendCounts[i-1];
		}
		/*
		for (int i = 0; i < rankNum; ++i) {
			printf("rank %d: [%d, %d]\n", i, displacements[i], displacements[i] + sendCounts[i] - 1);
		}
		*/
		/*
		   for (int i = 0; i < rankNum; ++i) {
			printf("sendCounts[%d]: %d\n", i, sendCounts[i]);
		}
		*/

		values = (double*) malloc(sizeof(double) * N);

		srand48(SEED);
		checkSum = 0;
		for (size_t i = 0; i < N; ++i) {
			values[i] = drand48();
			checkSum += values[i];
		}
	}

	MPI_Scatterv(values, sendCounts, displacements, MPI_DOUBLE, localValues, localSize, MPI_DOUBLE, master, MPI_COMM_WORLD);

	localSum = 0;
	for (size_t i = 0; i < localSize; ++i) {
		localSum += localValues[i];
	}

	//printf("local sum on rank %d (localSize: %d): %f\n", rank, localSize, localSum);

	MPI_Reduce(&localSum, &sum, 1, MPI_DOUBLE, MPI_SUM, master, MPI_COMM_WORLD);

	if (rank == master) {
		end_time = MPI_Wtime();
		
		printf("# of elements = %llu , parallel sum is %f check sum is %f \n", N, sum, checkSum);
		printf("# walltime on master processor : %10.8f \n", end_time - start_time);
		
		free(values);
		free(sendCounts);
		free(displacements);
	} else {
		end_time = MPI_Wtime();
		//printf("\n # walltime on processor %i : %10.8f \n", rank, end_time - start_time);
	}

	free(localValues);

	MPI_Finalize();
}
