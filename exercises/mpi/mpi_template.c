#include <stdlib.h>
#include <unistd.h>
#include <stdio.h>
#include <math.h>
#include <string.h>

#include <mpi.h>

#define USE MPI
#define ROOT 0
#define SEED 35791246

int main (int argc , char *argv[])
{
	//general MPI util vars
	int ranksNum, rank, processorNameLen;
	char processorName[MPI_MAX_PROCESSSOR_NAME];
	MPI_Status status;
	MPI_Request request;
	
	int tag = 123;

	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &rankNum);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Get_processor_name(processorName, &processorNameLen); 

	//time recording
	double startTime, endTime;   
	//startTime = MPI_Wtime();
	
	//type rootVar, localVar;
	int var, localVar;
	int totalSize, localSize;

	typedef vType int;
	MPI_DataType mpiVType = MPI_INT;

	//calculate totalSize and localSize

	//support for MPI_Scatterv
	vType* values = NULL;
	vType* localValues = NULL;
	int* sendCounts = NULL;
	int* displacements = NULL;
	
	/*
	if (rank == ROOT) {
		values = (vType*) malloc(sizeof(vType) * totalSize);
		sendCounts = (int*) malloc(sizeof(int) * ranksNum);
		displacements = (int*) malloc(sizeof(int) * ranksNum);
	}
	localValues = (vType*) malloc(sizeof(vType) * totalSize);
	*/

	if (argc <= 1) {
		fprintf (stderr, " Usage : mpi -np n %s number_of_iterations \n", argv[0]);
		MPI_Finalize();
		exit(-1);
	}

	long long int N = atoll(argv[1]) / rankNum;

	srand48(SEED * (rank + 1));
	local_M = 0;
	long long int i;
	for (i = 0; i < N; i++) {
		x = drand48(); 
		y = drand48();

		if ((x * x + y * y) < 1) {
			local_M++;
		}
	}

	MPI_Reduce(&local_M, &M, 1, MPI_LONG_LONG, MPI_SUM, master, MPI_COMM_WORLD);

	if (rank == master) {
		pi = 4.0 * M / (N * rankNum);
		end_time = MPI_Wtime();
		
		printf("\n # of trials = %llu , estimate of pi is %1.9f \n", N * rankNum, pi);
		printf("\n # walltime on master processor : %10.8f \n", end_time - start_time);
		
	} else {
		end_time = MPI_Wtime();
		printf("\n # walltime on processor %i : %10.8f \n", rank, end_time - start_time);
	}

	MPI_Finalize();

	//specialized frees
	if (rank == ROOT) {
		//master frees
		if (values) {
			free(values);
		}
		if (sendCounts) {
			free(sendCounts);
		}
		if (displacements) {
			free(displacements);
		}
	} else {
		//slaves frees
	}

	//common frees
	if (localValues) {
		free(localValues);
	}
}
