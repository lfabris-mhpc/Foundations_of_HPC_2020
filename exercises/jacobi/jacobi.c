#include <stdlib.h>
#include <unistd.h>
#include <stdio.h>

#include <math.h>
//#include <string.h>

#include <immintrin.h>

#include <mpi.h>

#define USE MPI

#include <omp.h>

#include "utils_print.h"

#define boundary 1

#ifndef BLOCK
	#define block0 16
	#define block1 16
	#define block2 16
#else
	#define block0 BLOCK
	#define block1 BLOCK
	#define block2 BLOCK
#endif

void print_rank_prefix(const int rank, const int* block_coords) {
	printf("rank %d(%d, %d, %d)", rank, block_coords[0], block_coords[1], block_coords[2]);
}

void print_field(const double* field, const int* block_sizes) {
	const int adv0 = 2 * boundary + block_sizes[1];
	const int adv1 = 2 * boundary + block_sizes[2];

	printf("[");
	for (int i = 0; i < 2 * boundary + block_sizes[0]; ++i) {
		const int jpos = i * adv0;
		printf("[");
		for (int j = 0; j < adv0; ++j) {
			const int kpos = (jpos + j) * adv1;
			printf("[");
			for (int k = 0; k < adv1; ++k) {
				printf("%f ", field[kpos + k]);
			}
			printf("]\n");
		}
		printf("]\n");
	}
	printf("]\n");
}

inline double update_field_slice(double* restrict field, const double* restrict field_prev
	, const int adv0, const int adv1
	, const int idx_lower0, const int idx_upper0
	, const int idx_lower1, const int idx_upper1
	, const int idx_lower2, const int idx_upper2) {
	#if DBG
	//printf("update_field_slice field[%d:%d, %d:%d, %d:%d]\n", idx_lower0, idx_upper0, idx_lower1, idx_upper1, idx_lower2, idx_upper2);
	#endif
	const double factor = 0.1666666666;//1.0 / (6.0 * boundary);
	double res = 0.0;

	for (int i = idx_lower0; i < idx_upper0; ++i) {
		const int jpos = i * adv0;

		for (int j = idx_lower1; j < idx_upper1; ++j) {
			const int kpos = (jpos + j) * adv1;

			for (int k = idx_lower2; k < idx_upper2; ++k) {
				const int pos = kpos + k;

				field[pos] = 0.0;
				//use accumulators? and a smarter visiting pattern?
				for (int b = 1; b <= boundary; ++b) {
					//dim 2
					field[pos] += field_prev[pos - b];
					field[pos] += field_prev[pos + b];
					//dim 1
					field[pos] += field_prev[pos - b * adv1];
					field[pos] += field_prev[pos + b * adv1];
					//dim 0
					field[pos] += field_prev[pos - b * adv0 * adv1];
					field[pos] += field_prev[pos + b * adv0 * adv1];
				}

				field[pos] *= factor;

				double t = field[pos] - field_prev[pos];
				res += t * t;
			}
		}
	}

	return res;
}

inline double update_field_slice_simd(double* restrict field, const double* restrict field_prev
	, const int adv0, const int adv1
	, const int idx_lower0, const int idx_upper0
	, const int idx_lower1, const int idx_upper1
	, const int idx_lower2, const int idx_upper2) {
	#if DBG
	//printf("update_field_slice_simd field[%d:%d, %d:%d, %d:%d]\n", idx_lower0, idx_upper0, idx_lower1, idx_upper1, idx_lower2, idx_upper2);
	#endif
	const double factor = 0.1666666666;//1.0 / (6.0 * boundary);
	double res = 0.0;

	const int block = 4;
	const int nblocks = (idx_upper2 - idx_lower2) / block;

	__m256d factors = _mm256_broadcast_sd(&factor);

	for (int i = idx_lower0; i < idx_upper0; ++i) {
		const int jpos = i * adv0;

		for (int j = idx_lower1; j < idx_upper1; ++j) {
			const int kpos = (jpos + j) * adv1;

			for (int kk = 0; kk < nblocks; ++kk) {
				const int pos = kpos + idx_lower2 + kk * block;

				for (int b = 1; b <= boundary; ++b) {
					__m256d f2 = _mm256_add_pd(_mm256_loadu_pd(field_prev + pos - 1), _mm256_loadu_pd(field_prev + pos + 1));
					__m256d f1 = _mm256_add_pd(_mm256_loadu_pd(field_prev + pos - adv1), _mm256_loadu_pd(field_prev + pos + adv1));
					f2 = _mm256_add_pd(f2, f1);

					__m256d f0 = _mm256_add_pd(_mm256_loadu_pd(field_prev + pos - adv0 * adv1), _mm256_loadu_pd(field_prev + pos + adv0 * adv1));
					f0 = _mm256_add_pd(f0, f2);

					f0 = _mm256_mul_pd(f0, factors);

					/*
					__m256d u = _mm256_loadu_pd(field_prev + pos - b);
					__m256d l = _mm256_loadu_pd(field_prev + pos + b);
					__m256d f0 = _mm256_add_pd(u, l);

					u = _mm256_loadu_pd(field_prev + pos - b * adv1);
					f0 = _mm256_add_pd(f0, u);
					l = _mm256_loadu_pd(field_prev + pos + b * adv1);
					f0 = _mm256_add_pd(f0, l);

					u = _mm256_loadu_pd(field_prev + pos - b * adv0 * adv1);
					f0 = _mm256_add_pd(f0, u);
					l = _mm256_loadu_pd(field_prev + pos + b * adv0 * adv1);
					f0 = _mm256_add_pd(f0, u);

					f0 = _mm256_mul_pd(f0, factors);
					*/
					_mm256_storeu_pd(field + pos, f0);

					__m256d p = _mm256_loadu_pd(field_prev + pos);

					f0 = _mm256_sub_pd(f0, p);
					f0 = _mm256_mul_pd(f0, f0);

					res += f0[0] + f0[1] + f0[2] + f0[3];
				}
			}

			for (int k = idx_lower2 + nblocks * block; k < idx_upper2; ++k) {
				const int pos = kpos + k;

				field[pos] = 0.0;
				//use accumulators? and a smarter visiting pattern?
				for (int b = 1; b <= boundary; ++b) {
					//dim 2
					field[pos] += field_prev[pos - b];
					field[pos] += field_prev[pos + b];
					//dim 1
					field[pos] += field_prev[pos - b * adv1];
					field[pos] += field_prev[pos + b * adv1];
					//dim 0
					field[pos] += field_prev[pos - b * adv0 * adv1];
					field[pos] += field_prev[pos + b * adv0 * adv1];
				}

				field[pos] *= factor;

				double t = field[pos] - field_prev[pos];
				res += t * t;
			}
		}
	}

	return res;
}

double update_field_slice_by_blocks(double* restrict field, const double* restrict field_prev, const int* block_sizes, const int* idx_lower, const int* idx_upper) {
	#if DBG
	//printf("update_field_slice_by_blocks field[%d:%d, %d:%d, %d:%d]\n", idx_lower[0], idx_upper[0], idx_lower[1], idx_upper[1], idx_lower[2], idx_upper[2]);
	#endif
	const int adv0 = 2 * boundary + block_sizes[1];
	const int adv1 = 2 * boundary + block_sizes[2];
	double res = 0.0;

	//const int block0 = 16;
	//const int block1 = 16;
	//const int block2 = 16;

	const int nblocks0 = (idx_upper[0] - idx_lower[0]) / block0;
	const int nblocks1 = (idx_upper[1] - idx_lower[1]) / block1;
	const int nblocks2 = (idx_upper[2] - idx_lower[2]) / block2;

	#pragma omp parallel for reduction(+: res)
	for (int i = 0; i < nblocks0; ++i) {
		for (int j = 0; j < nblocks1; ++j) {
			for (int k = 0; k < nblocks2; ++k) {
				#ifdef SIMD
				res += update_field_slice_simd(field, field_prev
					, adv0, adv1
					, idx_lower[0] + i * block0, idx_lower[0] + (i+1) * block0
					, idx_lower[1] + j * block1, idx_lower[1] + (j+1) * block1
					, idx_lower[2] + k * block2, idx_lower[2] + (k+1) * block2);
				#else
				res += update_field_slice(field, field_prev
					, adv0, adv1
					, idx_lower[0] + i * block0, idx_lower[0] + (i+1) * block0
					, idx_lower[1] + j * block1, idx_lower[1] + (j+1) * block1
					, idx_lower[2] + k * block2, idx_lower[2] + (k+1) * block2);
				#endif
			}
		}
	}

	#ifdef SIMD
	res += update_field_slice_simd(field, field_prev
		, adv0, adv1
		, idx_lower[0] + nblocks0 * block0, idx_upper[0]
		, idx_lower[1] + nblocks1 * block1, idx_upper[1]
		, idx_lower[2] + nblocks2 * block2, idx_upper[2]);
	#else
	res += update_field_slice(field, field_prev
		, adv0, adv1
		, idx_lower[0] + nblocks0 * block0, idx_upper[0]
		, idx_lower[1] + nblocks1 * block1, idx_upper[1]
		, idx_lower[2] + nblocks2 * block2, idx_upper[2]);
	#endif

	return res;
}

double update_field(double* restrict field, const double* restrict field_prev, const int* block_sizes) {
	#ifdef UPD_FIELD_BLOCKS
	const int idx_lower[3] = {boundary, boundary, boundary};
	const int idx_upper[3] = {boundary + block_sizes[0], boundary + block_sizes[1], boundary + block_sizes[2]};
	return update_field_slice_by_blocks(field, field_prev, block_sizes, idx_lower, idx_upper);
	#elif defined(UPD_FIELD_SIMD)
	return update_field_slice_simd(field, field_prev
		, 2 * boundary + block_sizes[1], 2 * boundary + block_sizes[2]
		, boundary, boundary + block_sizes[0]
		, boundary, boundary + block_sizes[1]
		, boundary, boundary + block_sizes[2]);
	#else
	return update_field_slice(field, field_prev
		, 2 * boundary + block_sizes[1], 2 * boundary + block_sizes[2]
		, boundary, boundary + block_sizes[0]
		, boundary, boundary + block_sizes[1]
		, boundary, boundary + block_sizes[2]);
	#endif
}

double get_field_slice_residual(double* restrict field, const double* restrict field_prev
	, const int adv0, const int adv1
	, const int idx_lower0, const int idx_upper0
	, const int idx_lower1, const int idx_upper1
	, const int idx_lower2, const int idx_upper2) {
	#if DBG
	//printf("get_field_slice_residual field[%d:%d, %d:%d, %d:%d]\n", idx_lower0, idx_upper0, idx_lower1, idx_upper1, idx_lower2, idx_upper2);
	#endif
	double res = 0.0;

	for (int i = idx_lower0; i < idx_upper0; ++i) {
		const int jpos = i * adv0;

		for (int j = idx_lower1; j < idx_upper1; ++j) {
			const int kpos = (jpos + j) * adv1;

			for (int k = idx_lower2; k < idx_upper2; ++k) {
				const int pos = kpos + k;

				double t = field[pos] - field_prev[pos];
				res += t * t;
			}
		}
	}

	return res;
}

//field_slice to buffer and opposite
inline void field_slice2buffer(const double* restrict field, double* restrict buffer
	, const int adv0, const int adv1
	, const int idx_lower0, const int idx_upper0
	, const int idx_lower1, const int idx_upper1
	, const int idx_lower2, const int idx_upper2) {
	#if DBG
	//printf("field_slice2buffer field[%d:%d, %d:%d, %d:%d]\n", idx_lower0, idx_upper0, idx_lower1, idx_upper1, idx_lower2, idx_upper2);
	#endif
	//const int adv2 = idx_upper2 - idx_lower2;
	int buf = 0;
	for (int i = idx_lower0; i < idx_upper0; ++i) {
		const int jpos = i * adv0;

		for (int j = idx_lower1; j < idx_upper1; ++j) {
			//memcpy(buffer + buf, field + (jpos + j) * adv1 + idx_lower2, sizeof(double) * adv2);
			//buf += adv2;
			const int kpos = (jpos + j) * adv1;

			for (int k = idx_lower2; k < idx_upper2; ++k) {
				buffer[buf++] = field[kpos + k];
			}
		}
	}
}

inline void buffer2field_slice(double* restrict field, double* restrict buffer
	, const int adv0, const int adv1
	, const int idx_lower0, const int idx_upper0
	, const int idx_lower1, const int idx_upper1
	, const int idx_lower2, const int idx_upper2) {
	#if DBG
	//printf("buffer2field_slice field[%d:%d, %d:%d, %d:%d]\n", idx_lower0, idx_upper0, idx_lower1, idx_upper1, idx_lower2, idx_upper2);
	#endif
	//const int adv2 = idx_upper2 - idx_lower2;

	int buf = 0;
	for (int i = idx_lower0; i < idx_upper0; ++i) {
		const int jpos = i * adv0;

		for (int j = idx_lower1; j < idx_upper1; ++j) {
			//memcpy(field + (jpos + j) * adv1 + idx_lower2, buffer + buf, sizeof(double) * adv2);
			//buf += adv2;
			const int kpos = (jpos + j) * adv1;

			for (int k = idx_lower2; k < idx_upper2; ++k) {
				field[kpos + k] = buffer[buf++];
			}
		}
	}
}


//from inside block, to send buffers
void boundaries2buffers(const double* restrict field, double* restrict* buffer, const int* block_sizes) {
	#pragma omp parallel
	{
		#pragma omp first
		{
			//dim 0
			if (buffer[0]) {
				#pragma omp task
				field_slice2buffer(field, buffer[0]
					, 2 * boundary + block_sizes[1], 2 * boundary + block_sizes[2]
					, boundary, 2 * boundary
					, boundary, boundary + block_sizes[1]
					, boundary, boundary + block_sizes[2]);
			}
			if (buffer[1]) {
				#pragma omp task
				field_slice2buffer(field, buffer[1]
					, 2 * boundary + block_sizes[1], 2 * boundary + block_sizes[2]
					, block_sizes[0], boundary + block_sizes[0]
					, boundary, boundary + block_sizes[1]
					, boundary, boundary + block_sizes[2]);
			}
			//dim 1
			if (buffer[2]) {
				#pragma omp task
				field_slice2buffer(field, buffer[2]
					, 2 * boundary + block_sizes[1], 2 * boundary + block_sizes[2]
					, boundary, boundary + block_sizes[0]
					, boundary, 2 * boundary
					, boundary, boundary + block_sizes[2]);
			}
			if (buffer[3]) {
				#pragma omp task
				field_slice2buffer(field, buffer[3]
					, 2 * boundary + block_sizes[1], 2 * boundary + block_sizes[2]
					, boundary, boundary + block_sizes[0]
					, block_sizes[1], boundary + block_sizes[1]
					, boundary, boundary + block_sizes[2]);
			}
			//dim 2
			if (buffer[4]) {
				#pragma omp task
				field_slice2buffer(field, buffer[4]
					, 2 * boundary + block_sizes[1], 2 * boundary + block_sizes[2]
					, boundary, boundary + block_sizes[0]
					, boundary, boundary + block_sizes[1]
					, boundary, 2 * boundary);
			}
			if (buffer[5]) {
				#pragma omp task
				field_slice2buffer(field, buffer[5]
					, 2 * boundary + block_sizes[1], 2 * boundary + block_sizes[2]
					, boundary, boundary + block_sizes[0]
					, boundary, boundary + block_sizes[1]
					, block_sizes[2], boundary + block_sizes[2]);
			}
		}
	}
}

//from recv buffers, to boundaryes
void buffers2boundaries(double* restrict field, double* restrict* buffer, const int* block_sizes) {
	#pragma omp parallel
	{
		#pragma omp first
		{
			//dim 0
			if (buffer[0]) {
				#pragma omp task
				buffer2field_slice(field, buffer[0]
					, 2 * boundary + block_sizes[1], 2 * boundary + block_sizes[2]
					, 0, boundary
					, boundary, boundary + block_sizes[1]
					, boundary, boundary + block_sizes[2]);
			}
			if (buffer[1]) {
				#pragma omp task
				buffer2field_slice(field, buffer[1]
					, 2 * boundary + block_sizes[1], 2 * boundary + block_sizes[2]
					, boundary + block_sizes[0], 2 * boundary + block_sizes[0]
					, boundary, boundary + block_sizes[1]
					, boundary, boundary + block_sizes[2]);
			}
			//dim 1
			if (buffer[2]) {
				#pragma omp task
				buffer2field_slice(field, buffer[2]
					, 2 * boundary + block_sizes[1], 2 * boundary + block_sizes[2]
					, boundary, boundary + block_sizes[0]
					, 0, boundary
					, boundary, boundary + block_sizes[2]);
			}
			if (buffer[3]) {
				#pragma omp task
				buffer2field_slice(field, buffer[3]
					, 2 * boundary + block_sizes[1], 2 * boundary + block_sizes[2]
					, boundary, boundary + block_sizes[0]
					, boundary + block_sizes[1], 2 * boundary + block_sizes[1]
					, boundary, boundary + block_sizes[2]);
			}
			//dim 2
			if (buffer[4]) {
				#pragma omp task
				buffer2field_slice(field, buffer[4]
					, 2 * boundary + block_sizes[1], 2 * boundary + block_sizes[2]
					, boundary, boundary + block_sizes[0]
					, boundary, boundary + block_sizes[1]
					, 0, boundary);
			}
			if (buffer[5]) {
				#pragma omp task
				buffer2field_slice(field, buffer[5]
					, 2 * boundary + block_sizes[1], 2 * boundary + block_sizes[2]
					, boundary, boundary + block_sizes[0]
					, boundary, boundary + block_sizes[1]
					, boundary + block_sizes[2], 2 * boundary + block_sizes[2]);
			}
		}
	}
}

inline void init_field_slice(double* restrict field, const double val
	, const int adv0, const int adv1
	, const int idx_lower0, const int idx_upper0
	, const int idx_lower1, const int idx_upper1
	, const int idx_lower2, const int idx_upper2) {
	#if DBG
	//printf("init_field_slice field[%d:%d, %d:%d, %d:%d]\n", idx_lower0, idx_upper0, idx_lower1, idx_upper1, idx_lower2, idx_upper2);
	#endif
	#pragma omp parallel for
	for (int i = idx_lower0; i < idx_upper0; ++i) {
		const int jpos = i * adv0;

		for (int j = idx_lower1; j < idx_upper1; ++j) {
			const int kpos = (jpos + j) * adv1;

			for (int k = idx_lower2; k < idx_upper2; ++k) {
				field[kpos + k] = val;
			}
		}
	}
}

void init_boundaries(double* restrict field, const double val, const int* block_sizes, const int* boundary_active) {
	//"left" boundaries for coordinates 0 take val
	//"right" boundaries for coordinates n-1 take 0
	//dim 0
	if (!boundary_active[0]) {
		init_field_slice(field, val
			, 2 * boundary + block_sizes[1], 2 * boundary + block_sizes[2]
			, 0, boundary
			, boundary, boundary + block_sizes[1]
			, boundary, boundary + block_sizes[2]);
	}
	if (!boundary_active[1]) {
		init_field_slice(field, 0
			, 2 * boundary + block_sizes[1], 2 * boundary + block_sizes[2]
			, boundary + block_sizes[0], 2 * boundary + block_sizes[0]
			, boundary, boundary + block_sizes[1]
			, boundary, boundary + block_sizes[2]);
	}
	//dim 1
	if (!boundary_active[2]) {
		init_field_slice(field, val
			, 2 * boundary + block_sizes[1], 2 * boundary + block_sizes[2]
			, boundary, boundary + block_sizes[0]
			, 0, boundary
			, boundary, boundary + block_sizes[2]);
	}
	if (!boundary_active[3]) {
		init_field_slice(field, 0
			, 2 * boundary + block_sizes[1], 2 * boundary + block_sizes[2]
			, boundary, boundary + block_sizes[0]
			, boundary + block_sizes[1], 2 * boundary + block_sizes[1]
			, boundary, boundary + block_sizes[2]);
	}
	//dim 2
	if (!boundary_active[4]) {
		init_field_slice(field, val
			, 2 * boundary + block_sizes[1], 2 * boundary + block_sizes[2]
			, boundary, boundary + block_sizes[0]
			, boundary, boundary + block_sizes[1]
			, 0, boundary);
	}
	if (!boundary_active[5]) {
		init_field_slice(field, 0
			, 2 * boundary + block_sizes[1], 2 * boundary + block_sizes[2]
			, boundary, boundary + block_sizes[0]
			, boundary, boundary + block_sizes[1]
			, boundary + block_sizes[2], 2 * boundary + block_sizes[2]);
	}
}

#ifdef COMM_SENDRECV
void communicate_sendrecv(MPI_Comm* mesh_comm, double* restrict* boundary_send, double* restrict* boundary_recv, const int* block_coords, const int* boundary_sizes, const int* boundary_ranks) {
	int rank;
	MPI_Comm_rank(*mesh_comm, &rank);

	for (int i = 0; i < 3; ++i) {
		const int i2 = 2 * i;
		int other = block_coords[i] % 2 ? boundary_ranks[i2] : boundary_ranks[i2 + 1];
		int buf = i2 + (block_coords[i] % 2 ? 0 : 1);

		MPI_Status status;
		if (other != MPI_PROC_NULL) {
			//print_rank_prefix(rank, block_coords);
			//printf(": sendrecv with %d size %d\n", other, boundary_sizes[i]);
			MPI_Sendrecv(boundary_send[buf], boundary_sizes[i], MPI_DOUBLE, other, i//int sendtag
				, boundary_recv[buf], boundary_sizes[i], MPI_DOUBLE, other, i//int recvtag
				, *mesh_comm, &status);
		}

		other = block_coords[i] % 2 ? boundary_ranks[i2 + 1] : boundary_ranks[i2];
		buf = i2 + (block_coords[i] % 2 ? 1 : 0);

		if (other != MPI_PROC_NULL) {
			//print_rank_prefix(rank, block_coords);
			//printf(": sendrecv with %d size %d\n", other, boundary_sizes[i]);
			MPI_Sendrecv(boundary_send[buf], boundary_sizes[i], MPI_DOUBLE, other, i//int sendtag
				, boundary_recv[buf], boundary_sizes[i], MPI_DOUBLE, other, i//int recvtag
				, *mesh_comm, &status);
		}
	}
}
#else
void communicate_nonblocking(MPI_Comm* mesh_comm, double* restrict* boundary_send, double* restrict* boundary_recv, const int* block_coords, const int* boundary_sizes, const int* boundary_ranks) {
	int rank;
	MPI_Comm_rank(*mesh_comm, &rank);
	int ncomms = 0;
	MPI_Request requests[12];
	MPI_Status statuses[12];

	for (int i = 0; i < 6; ++i) {
		if (boundary_ranks[i] != MPI_PROC_NULL) {
			//print_rank_prefix(rank, block_coords);
			//printf(": sendrecv with %d size %d\n", other, boundary_sizes[i]);
			MPI_Isend(boundary_send[i], boundary_sizes[i / 2], MPI_DOUBLE, boundary_ranks[i], i / 2, *mesh_comm, requests + ncomms++);
			//print_rank_prefix(rank, block_coords);
			//printf(": sendrecv with %d size %d\n", other, boundary_sizes[i]);
			MPI_Irecv(boundary_recv[i], boundary_sizes[i / 2], MPI_DOUBLE, boundary_ranks[i], i / 2, *mesh_comm, requests + ncomms++);
		}
	}

	MPI_Waitall(ncomms, requests, statuses);
}
#endif

/*
//fine grained tasks
double task_boundary(MPI_Comm* mesh_comm
	, double* restrict field, double* restrict field_prev
	, double* restrict boundary_send, double* restrict boundary_recv
	, const int boundary_rank
	, const int adv0, const int adv1
	, const int idx_lower0, const int idx_upper0
	, const int idx_lower1, const int idx_upper1
	, const int idx_lower2, const int idx_upper2) {
	//copy own boundaries from prev field to send buffers
	field_slice2buffer(field, boundary_send
		, adv0, adv1
		, idx_lower0, idx_upper0
		, idx_lower1, idx_upper1
		, idx_lower2, idx_upper2);

	int rank;
	MPI_Comm_rank(*mesh_comm, &rank);
	MPI_Request requests[2];
	MPI_Status statuses[2];

	//communications
	MPI_Isend(boundary_send, boundary_size, MPI_DOUBLE, boundary_rank, 0, *mesh_comm, requests);
	MPI_Irecv(boundary_recv, boundary_size, MPI_DOUBLE, boundary_rank, 0, *mesh_comm, requests + 1);
	MPI_Waitall(2, requests, statuses);

	//copy external boundaries from recv buffers to prev field
	buffer2field_slice(field, boundary_recv
		, 2 * boundary + block_sizes[1], 2 * boundary + block_sizes[2]
		, idx_lower0 - boundary, boundary
		, boundary, boundary + block_sizes[1]
		, boundary, boundary + block_sizes[2]);

	//compute update
	return update_field_slice(field, field_prev
		, 2 * boundary + block_sizes[1], 2 * boundary + block_sizes[2]
		, boundary, 2 * boundary
		, boundary, boundary + block_sizes[1]
		, boundary, boundary + block_sizes[2]);
}
*/

int compare_ints(const void* a, const void* b) {
    const int arg1 = *(const int*) a;
    const int arg2 = *(const int*) b;

    if (arg1 < arg2) return -1;
    if (arg1 > arg2) return 1;
    return 0;

    // return (arg1 > arg2) - (arg1 < arg2); // possible shortcut
    // return arg1 - arg2; // erroneous shortcut (fails if INT_MIN is present)
}

int main (int argc , char *argv[])
{
	//general MPI util vars
	int nranks, rank, processor_name_len;
	char processor_name[MPI_MAX_PROCESSOR_NAME];

	#ifdef _OPENMP
	int mpi_thread_provided;
	MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &mpi_thread_provided);
	#else
	MPI_Init(&argc, &argv);
	#endif

	MPI_Comm_size(MPI_COMM_WORLD, &nranks);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Get_processor_name(processor_name, &processor_name_len);

	#ifdef _OPENMP
	if (rank == 0) {
		char* t = "none";
		switch (mpi_thread_provided) {
			case MPI_THREAD_SINGLE:
				t = "MPI_THREAD_SINGLE";
				break;
			case MPI_THREAD_FUNNELED:
				t = "MPI_THREAD_FUNNELED";
				break;
			case MPI_THREAD_SERIALIZED:
				t = "MPI_THREAD_SERIALIZED";
				break;
			case MPI_THREAD_MULTIPLE:
				t = "MPI_THREAD_MULTIPLE";
				break;
		}
		printf("mpi_thread_provided: %d\n", mpi_thread_provided);
		printf("omp_get_max_threads: %d\n", omp_get_max_threads());
	}
	#endif

	const int spatial_dims = 3;
	int dim_elems[3] = {1, 1, 1};

	int proc_dims;
	int dim_blocks[3] = {1, 1, 1};

	//const int boundary = 1;
	int iterations = 100;
	const double tolerance = 1e-10;

	if (argc < 6) {
		fprintf(stderr, "usage: %s [mesh dimensionality] [dimension 1 elements] [dimension 2 elements] [dimension 3 elements]\n", argv[0]);
		MPI_Abort(MPI_COMM_WORLD, 1);
		exit(1);
	}

	int rank_dbg;

	if (rank == 0) {
		int res = sscanf(argv[1], "%d", &proc_dims);
		if (res != 1 || proc_dims < 1 || proc_dims > 3) {
			fprintf(stderr, "unusable mesh dimensionality\n");
			MPI_Abort(MPI_COMM_WORLD, 1);
			exit(1);
		}
		for (int i = 0; i < proc_dims; ++i) {
			dim_blocks[i] = 0;
		}

		for (int i = 0; i < 3; ++i) {
			res = sscanf(argv[2+i], "%d", dim_elems+i);
			if (res != 1 || dim_elems[i] < 1) {
				fprintf(stderr, "unusable dimension %d elements\n", i+1);
				MPI_Abort(MPI_COMM_WORLD, 1);
				exit(1);
			}
		}

		res = sscanf(argv[5], "%d", &iterations);
		if (res != 1 || !iterations) {
			fprintf(stderr, "unusable max iterations\n");
			MPI_Abort(MPI_COMM_WORLD, 1);
			exit(1);
		}

		if (argc >= 7) {
			res = sscanf(argv[6], "%d", &rank_dbg);
			if (res != 1) {
				fprintf(stderr, "unusable rank_dbg\n");
				MPI_Abort(MPI_COMM_WORLD, 1);
				exit(1);
			}
		} else {
			rank_dbg = MPI_PROC_NULL;
		}

		//get balanced divisors list
		MPI_Dims_create(nranks, 3, dim_blocks);
		//optimize dim_blocks vs dim_elems to minimize boundaries buffer sizes
		//handle dimension depth = 1 && dim_blocks > 1 -> move those procs to another dim
		//TODO
		#ifdef DIMS_SORT
		qsort((void*) dim_blocks, 3, sizeof(int), compare_ints);
		#endif

		printf("dim_blocks: (%d, %d, %d)\n", dim_blocks[0], dim_blocks[1], dim_blocks[2]);
	}

	MPI_Bcast(dim_blocks, 3, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Bcast(dim_elems, 3, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Bcast(&iterations, 1, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Bcast(&rank_dbg, 1, MPI_INT, 0, MPI_COMM_WORLD);

	int dim_periodic[3] = {0, 0, 0};
	MPI_Comm mesh_comm;
	MPI_Cart_create(MPI_COMM_WORLD, 3, dim_blocks, dim_periodic, 1, &mesh_comm);

	if (mesh_comm != MPI_COMM_NULL) {
		MPI_Comm_rank(mesh_comm, &rank);
		MPI_Comm_size(mesh_comm, &nranks);

		int block_coords[3];
		MPI_Cart_coords(mesh_comm, rank, 3, block_coords);

		int block_sizes[3];
		int field_sizes[3];
		for (int i = 0; i < 3; ++i) {
			block_sizes[i] = dim_elems[i] / dim_blocks[i];
			if (block_coords[i] < dim_elems[i] % dim_blocks[i]) {
				++block_sizes[i];
			}
			field_sizes[i] = 2 * boundary + block_sizes[i];
		}

		if (rank == rank_dbg) {
			print_rank_prefix(rank, block_coords);
			printf(": block_sizes: (%d, %d, %d)\n", block_sizes[0], block_sizes[1], block_sizes[2]);
		}

		//allocate 2 * extended matrix wasting the edge elements
		const int field_elems = block_sizes[0] * block_sizes[1] * block_sizes[2];
		const int field_extended_elems = (2 * boundary + block_sizes[0]) * (2 * boundary + block_sizes[1]) * (2 * boundary + block_sizes[2]);

		//boundaries metadata
		int boundary_sizes[3];
		boundary_sizes[0] = boundary * block_sizes[1] * block_sizes[2];
		boundary_sizes[1] = boundary * block_sizes[0] * block_sizes[2];
		boundary_sizes[2] = boundary * block_sizes[0] * block_sizes[1];

		int boundary_active[3 * 2];
		int boundary_ranks[3 * 2];
		int boundary_total_elems = 0;
		for (int i = 0; i < 3; ++i) {
			int i2 = 2 * i;
			MPI_Cart_shift(mesh_comm, i, 1, boundary_ranks + i2, boundary_ranks + i2 + 1);

			boundary_active[i2] = boundary_ranks[i2] != MPI_PROC_NULL ? 1 : 0;
			boundary_active[i2 + 1] = boundary_ranks[i2 + 1] != MPI_PROC_NULL ? 1 : 0;

			boundary_total_elems += boundary_sizes[i] * (boundary_active[i2] + boundary_active[i2 + 1]);
		}

		if (sizeof(double) * (boundary_total_elems + field_extended_elems) > 8L << 30) {
			fprintf(stderr, "memory allocation size exceeds 8GB\n");
			MPI_Abort(mesh_comm, 1);
			exit(1);
		}

		//double* blob = (double*) calloc(sizeof(double), 2 * field_extended_elems);
		/*
		double* blob = (double*) aligned_alloc(sizeof(double) * 8, sizeof(double) * 2 * field_extended_elems);
		for (int i = 0; i < 2 * field_extended_elems; ++i) {
			blob[i] = 0.0;
		}
		*/
		double* fields[2];// = {blob, blob + field_extended_elems};

		fields[0] = (double*) aligned_alloc(sizeof(double), sizeof(double) * field_extended_elems);
		for (int i = 0; i < field_extended_elems; ++i) {
			fields[0][i] = 0.0;
		}
		fields[1] = (double*) aligned_alloc(sizeof(double), sizeof(double) * field_extended_elems);
		for (int i = 0; i < field_extended_elems; ++i) {
			fields[1][i] = 0.0;
		}

		//allocate one buffers for sending and receiving boundaries
		double* boundary_send[3 * 2];
		double* boundary_recv[3 * 2];

		//boundary_send[0] = (double*) malloc(sizeof(double) * 2 * boundary_total_elems);
		boundary_send[0] = (double*) malloc(sizeof(double) * boundary_total_elems);
		for (int i = 0; i < 3 * 2 - 1; ++i) {
			boundary_send[i+1] = boundary_send[i] + boundary_active[i] * boundary_sizes[i / 2];
		}
		//boundary_recv[0] = boundary_send[5] + boundary_active[5] * boundary_sizes[2];
		boundary_recv[0] = (double*) malloc(sizeof(double) * boundary_total_elems);
		for (int i = 0; i < 3 * 2 - 1; ++i) {
			boundary_recv[i+1] = boundary_recv[i] + boundary_active[i] * boundary_sizes[i / 2];
		}
		for (int i = 0; i < 3 * 2; ++i) {
			boundary_send[i] = boundary_active[i] ? boundary_send[i] : NULL;
			boundary_recv[i] = boundary_active[i] ? boundary_recv[i] : NULL;
		}

		int field_previous = 0;
		int field_current = 1;
		const double val = 10;
		//init boundaries where missing adjacent processes
		init_boundaries(fields[field_previous], val, block_sizes, boundary_active);
		init_boundaries(fields[field_current], val, block_sizes, boundary_active);

		//time recording
		double t_comm = 0, t_update = 0;

		if (rank == 0) {
			printf("# StartResidual %f\n", 0.0);
		}

		for (int i = 0; i < iterations; ++i) {
			//MPI_Barrier(mesh_comm);

			if (rank == rank_dbg) {
				#ifdef DBG
				print_field(fields[field_previous], block_sizes);
				#endif
			}

			t_comm -= MPI_Wtime();
			//copy own boundaries from prev field to send buffers
			boundaries2buffers(fields[field_previous], boundary_send, block_sizes);

			//communications
			#ifdef COMM_SENDRECV
			communicate_sendrecv(&mesh_comm, boundary_send, boundary_recv, block_coords, boundary_sizes, boundary_ranks);
			#else
			communicate_nonblocking(&mesh_comm, boundary_send, boundary_recv, block_coords, boundary_sizes, boundary_ranks);
			#endif

			//copy external boundaries from recv buffers to prev field
			buffers2boundaries(fields[field_previous], boundary_recv, block_sizes);
			t_comm += MPI_Wtime();

			t_update -= MPI_Wtime();
			//compute update
			const double res = update_field(fields[field_current], fields[field_previous], block_sizes);
			t_update += MPI_Wtime();

			double res_total;
			MPI_Allreduce(&res, &res_total, 1, MPI_DOUBLE, MPI_SUM, mesh_comm);

			if (rank == rank_dbg) {
				//print_rank_prefix(rank, block_coords);
				//print_matrix_dbl("current matrix", fields[field_current], block_sizes[0] + 2 * boundary, block_sizes[1] + 2 * boundary);
				printf("rank %d(%d, %d, %d): residual %g\n", rank, block_coords[0], block_coords[1], block_coords[2], res_total);
			}

			if (res_total < tolerance) {
				break;
			}

			const int tmp = field_current;
			field_current = field_previous;
			field_previous = tmp;
		}

		double t_comm_min, t_comm_max, t_update_min, t_update_max;
		MPI_Allreduce(&t_comm, &t_comm_min, 1, MPI_DOUBLE, MPI_MIN, mesh_comm);
		MPI_Allreduce(&t_comm, &t_comm_max, 1, MPI_DOUBLE, MPI_MAX, mesh_comm);
		MPI_Allreduce(&t_update, &t_update_min, 1, MPI_DOUBLE, MPI_MIN, mesh_comm);
		MPI_Allreduce(&t_update, &t_update_max, 1, MPI_DOUBLE, MPI_MAX, mesh_comm);

		if (rank == rank_dbg) {
			printf("rank %d(%d, %d, %d): t_comm_min %g t_comm_max %f t_update_min %g t_update_max %f\n"
				   , rank, block_coords[0], block_coords[1], block_coords[2]
				   , t_comm_min, t_comm_max
				   , t_update_min, t_update_max);
		}

		free(fields[0]);
		free(fields[1]);

		for (int i = 0; i < 6; ++i) {
			if (boundary_send[i]) {
				free(boundary_send[i]);
				break;
			}
		}
		for (int i = 0; i < 6; ++i) {
			if (boundary_recv[i]) {
				free(boundary_recv[i]);
				break;
			}
		}
	}

	MPI_Finalize();
}
