#include <stdlib.h>
#include <unistd.h>
#include <stdio.h>
#include <math.h>
#include <string.h>
#include <mpi.h>

#define USE MPI
#define SEED 35791246

/*
void calcMaster(int ranksNum, int rank, int iter, int* curMaster, int* curNum) {
	if (!iter) {
		if (rank < ranksNum / 2) {
			*curMaster = 0;
			*curNum = ranksNum / 2;
		} else {
			*curMaster = ranksNum / 2;
			*curNum = ranksNum - ranksNum / 2;
		}
	}
	for (int i = 0; i < iter; ++i) {
	
	}
}
*/

int main(int argc, char *argv[])
{
	long long int M, local_M; 

	double start_time, end_time;   
	int rank, ranksNum, proc;
	MPI_Status status;
	MPI_Request request;
	
	int master = 0;
	int tag = 123;

	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &ranksNum);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	if (argc <= 1) {
		fprintf (stderr, " Usage : mpi_sum -np n %s number_of_elements \n", argv[0]);
		MPI_Finalize();
		exit(-1);
	}

	long long int N = atoll(argv[1]);
	int localSize = N / ranksNum + (rank < (N % ranksNum) ? 1 : 0);
	double* values = NULL;
	double* localValues = NULL;
	double localSum, checkSum;

	start_time = MPI_Wtime();


	if (rank == master) {
		localValues = values = (double*) malloc(sizeof(double) * N);

		srand48(SEED);
		checkSum = 0;
		for (size_t i = 0; i < N; ++i) {
			values[i] = drand48();
			checkSum += values[i];
		}
	}

	int blockSize = N / ranksNum;
	int curMaster = 0;
	int curRanksNum = ranksNum;
	int curSize = N;
	int sendSize;
	while (curRanksNum > 1) {
		int otherRanksNum = curRanksNum - curRanksNum / 2;
		int otherMaster = curMaster + curRanksNum / 2;
		//should send blocks, not elements
		//int sendSize = curSize - curSize / 2;
		int blockSize = curSize / curRanksNum;
		sendSize = otherRanksNum * blockSize;
		//adjust for remainder
		int remainder = curSize % curRanksNum;
		if (remainder && curRanksNum / 2 < remainder) {
			sendSize += remainder - curRanksNum / 2;
		}

		if (rank == curMaster) {
			//printf("rank %d: sending size: %d (of %d with %d ranks) to rank %d\n", rank, sendSize, curSize, curRanksNum, otherMaster);
			//printf("send msg[from: %d, to: %d, extremes: %f, %f\n", rank, otherMaster, localValues[curSize-sendSize], localValues[curSize-1]);
			//send sendSize
			MPI_Send(localValues + curSize - sendSize, sendSize, MPI_DOUBLE, otherMaster, tag, MPI_COMM_WORLD);
		} else if (rank == otherMaster) {
			//printf("rank %d: receiving size: %d from rank %d\n", rank, sendSize, curMaster);
			//allocate, receive sendSize
			localValues = (double*) malloc(sizeof(double) * sendSize);
			MPI_Recv(localValues, sendSize, MPI_DOUBLE, curMaster, tag, MPI_COMM_WORLD, &status);
			//printf("receive msg[from: %d, to: %d, extremes: %f, %f\n", curMaster, rank, localValues[0], localValues[sendSize-1]);
		}
		//else, not active

		if (rank >= otherMaster) {
			curMaster = otherMaster;
			curRanksNum = otherRanksNum;
			curSize = sendSize;
		} else {
			//master remains the same
			curRanksNum = curRanksNum / 2;
			curSize -= sendSize;
		}
	}
	//printf("rank %d: localSize: %d\n", rank, curSize);

	localSum = 0;
	for (size_t i = 0; i < curSize; ++i) {
		localSum += localValues[i];
	}

	//printf("local sum on rank %d (localSize: %d): %f\n", rank, localSize, localSum);

	for (int step = 2; step <= ranksNum; step *= 2) {
		if (rank % step) {
			//send to rank - step / 2
			//printf("rank %d send localSum to %d\n", rank, rank - step / 2);
			MPI_Send(&localSum, 1, MPI_DOUBLE, rank - step / 2, tag, MPI_COMM_WORLD);
			break;
		} else if (rank + step / 2 < ranksNum) {
			//receive from rank + step / 2
			//printf("rank %d receive localSum from %d\n", rank, rank + step / 2);
			double tmp;
			MPI_Recv(&tmp, 1, MPI_DOUBLE, rank + step / 2, tag, MPI_COMM_WORLD, &status);
			localSum += tmp;
			//printf("rank %d localSum now %f after partial from %d\n", rank, localSum, rank + step / 2);
		}
	}

	if (rank == master) {
		end_time = MPI_Wtime();
		
		printf("# of elements = %llu , parallel sum is %f check sum is %f \n", N, localSum, checkSum);
		printf("# walltime on master processor : %10.8f \n", end_time - start_time);
		
		//free(values);
	} else {
		end_time = MPI_Wtime();
		//printf("\n # walltime on processor %i : %10.8f \n", rank, end_time - start_time);
	}

	free(localValues);

	MPI_Finalize();
}
