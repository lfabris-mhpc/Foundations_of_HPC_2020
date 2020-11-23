/*
 * Copyright (C) 2016 Master in High Performance Computing
 *
 * Adapted from the net by Alberto Sartori. 
 */

// pi.c: Montecarlo integration to determine pi

// We have a circle inscribed inside a square of unit lenght. The
// ratio between the area of the square (1) and the circle (pi/4) is
// pi/4. Therefore, if we randomly choose N points inside the square,
// on average, only M=N*pi/4 points will belong to the circle. From
// the last relation we can estimate pi.

#include <stdio.h>
#include <time.h>
#include <stdlib.h>

#include <omp.h>

// if you don ' t have drand48 uncomment the following two lines 10
// #define drand48 1.0/RANDMAXrand
// #define srand48 srand
//#define seed 68111 // seed for number generator
#define seed 35791246

union seed48_ {
	//short size: 2
	unsigned short int sshort[sizeof(long long int)/sizeof(unsigned short int)];
	long long int sllong;
};

int main (int argc, char ** argv) {
	if (argc < 2)
	{
		printf(" Usage: %s iterations (nthreads)\n",argv[0]);
		return 1;
	}
	
	long long int N = atoll(argv[1]);
	
	if (argc >= 3) {
		int nthreads = atoi(argv[2]);
		omp_set_num_threads(nthreads);
	}

	long long int M = 0;
	//the reduction implies firstprivate
	#pragma omp parallel reduction (+: M)
	{
		union seed48_ s;
		int threadId = omp_get_thread_num();
		s.sllong = seed * (1 + threadId);
		//printf("thread %d uses long seed %lld\n", threadId, s.sllong);
		//printf("thread %d uses short seeds: %hu %hu %hu\n", threadId, s.sshort[2], s.sshort[1], s.sshort[0]);

		struct drand48_data randState;
		seed48_r((unsigned short int*) &s.sshort, &randState);

		double x, y;

		#pragma omp for //schedule(dynamic)
		for (long long int i = 0; i < N; ++i)
		{
			drand48_r(&randState, &x);
			drand48_r(&randState, &y);
			
			if ((x*x + y*y) < 1) { ++M; }
		}

		//printf("thread %d has local M: %lld\n", threadId, M);
		//printf("thread %d did %d iterations\n", threadId, iters);
	}

	double pi = 4.0 * M / N;
	printf("\n # of trials = %llu , estimate of pi is %1.9f \n", N, pi);
	
	return 0;
}
