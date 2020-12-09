#include <stdlib.h>
#include <unistd.h>
#include <stdio.h>
#include <math.h>
#include <string.h>
#include <math.h>

#include <mpi.h>

#define USE MPI

#include "utils_print.h"

void print_rank_prefix(const int rank, const int* block_coords) {
	printf("rank %d(%d, %d, %d)", rank, block_coords[0], block_coords[1], block_coords[2]); 
}

void print_field(const double* field, const int* block_sizes, const int boundary) {
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

void update_field_slice(const int rank, const int* block_coords, double* restrict field, const double* restrict field_prev, const int* block_sizes, const int boundary, const int* idx_lower, const int* idx_upper) {
	//print_rank_prefix(rank, block_coords);
	//printf(": update_field_slice\n");
	const int adv0 = 2 * boundary + block_sizes[1];
	const int adv1 = 2 * boundary + block_sizes[2];
	const double factor = 1.0 / (6 * boundary);

	for (int i = idx_lower[0]; i < idx_upper[0]; ++i) {
		const int jpos = i * adv0;

		for (int j = idx_lower[1]; j < idx_upper[1]; ++j) {
			const int kpos = (jpos + j) * adv1;

			for (int k = idx_lower[2]; k < idx_upper[2]; ++k) {
				const int pos = kpos + k;

				field[pos] = 0;
				//use accumulators? and a smarter visiting pattern?
				for (int b = 1; b <= boundary; ++b) {
					//printf("rank %d: update field[%d, %d, %d]\n", rank, i, j, k);
					//dim 0
					field[pos] += field_prev[pos - b * adv0 * adv1];
					field[pos] += field_prev[pos + b * adv0 * adv1];
					//dim 1
					field[pos] += field_prev[pos - b * adv1];
					field[pos] += field_prev[pos + b * adv1];
					//dim 2
					field[pos] += field_prev[pos - b];
					field[pos] += field_prev[pos + b];
				}

				field[pos] *= factor;
			}
		}
	}
}

void update_field_slice2(double* restrict field, const double* restrict field_prev
	, const int adv0, const int adv1, const int boundary
	, const int idx_lower0, const int idx_upper0
	, const int idx_lower1, const int idx_upper1
	, const int idx_lower2, const int idx_upper2) {
	const double factor = 1.0 / (6 * boundary);

	for (int i = idx_lower0; i < idx_upper0; ++i) {
		const int jpos = i * adv0;

		for (int j = idx_lower1; j < idx_upper1; ++j) {
			const int kpos = (jpos + j) * adv1;

			for (int k = idx_lower2; k < idx_upper2; ++k) {
				const int pos = kpos + k;

				field[pos] = 0;
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
			}
		}
	}
}

void update_field_slice_by_blocks(const int rank, const int* block_coords, double* restrict field, const double* restrict field_prev, const int* block_sizes, const int boundary, const int* idx_lower, const int* idx_upper) {
	//print_rank_prefix(rank, block_coords);
	//printf(": update_field_slice_by_blocks\n");
	const int adv0 = 2 * boundary + block_sizes[1];
	const int adv1 = 2 * boundary + block_sizes[2];
	const double factor = 1.0 / (6 * boundary);

	const int block0 = 8;
	const int block1 = 8;
	const int block2 = 8;
	
	const int nblocks0 = (idx_upper[0] - idx_lower[0]) / block0;
	const int nblocks1 = (idx_upper[1] - idx_lower[1]) / block1;
	const int nblocks2 = (idx_upper[2] - idx_lower[2]) / block2;
	for (int i = 0; i < nblocks0; ++i) {
		for (int j = 0; j < nblocks1; ++j) {
			for (int k = 0; k < nblocks2; ++k) {
				update_field_slice2(field, field_prev
					, adv0, adv1, boundary
					, idx_lower[0] + i * block0, idx_lower[0] + (i+1) * block0
					, idx_lower[1] + j * block1, idx_lower[1] + (j+1) * block1
					, idx_lower[2] + k * block2, idx_lower[2] + (k+1) * block2);
			}
		}
	}
	
	update_field_slice2(field, field_prev
		, adv0, adv1, boundary
		, idx_lower[0] + nblocks0 * block0, idx_upper[0]
		, idx_lower[1] + nblocks1 * block1, idx_upper[1]
		, idx_lower[2] + nblocks2 * block2, idx_upper[2]);
}

double get_field_slice_residual_max(const int rank, const int* block_coords, const double* restrict field, const double* restrict field_prev, const int* block_sizes, const int boundary, const int* idx_lower, const int* idx_upper) {
	//print_rank_prefix(rank, block_coords);
	//printf(": get_field_slice_residual_max\n");
	const int adv0 = 2 * boundary + block_sizes[1];
	const int adv1 = 2 * boundary + block_sizes[2];
	double res = 0;

	for (int i = idx_lower[0]; i < idx_upper[0]; ++i) {
		const int jpos = i * adv0;

		for (int j = idx_lower[1]; j < idx_upper[1]; ++j) {
			const int kpos = (jpos + j) * adv1;

			for (int k = idx_lower[2]; k < idx_upper[2]; ++k) {
				const int pos = kpos + k;

				res = fmax(res, (field[pos] - field_prev[pos]) * (field[pos] - field_prev[pos]));
			}
		}
	}
	
	return res;
	//return res / (block_sizes[0] * block_sizes[1] * block_sizes[2]);
}

void update_field(const int rank, const int* block_coords, double* restrict field, const double* restrict field_prev, const int* block_sizes, const int boundary) {
	//print_rank_prefix(rank, block_coords);
	//printf(": update_field\n");
	
	const int idx_lower[3] = {boundary, boundary, boundary};
	const int idx_upper[3] = {boundary + block_sizes[0], boundary + block_sizes[1], boundary + block_sizes[2]};
	#ifdef UPD_FIELD_BLOCKS
	update_field_slice_by_blocks(rank, block_coords, field, field_prev, block_sizes, boundary, idx_lower, idx_upper);
	#else
	update_field_slice(rank, block_coords, field, field_prev, block_sizes, boundary, idx_lower, idx_upper);
	#endif
}

double get_field_residual_max(const int rank, const int* block_coords, const double* restrict field, const double* restrict field_prev, const int* block_sizes, const int boundary) {
	//print_rank_prefix(rank, block_coords);
	//printf(": get_field_residual_max\n");
	
	const int idx_lower[3] = {boundary, boundary, boundary};
	const int idx_upper[3] = {boundary + block_sizes[0], boundary + block_sizes[1], boundary + block_sizes[2]};
	return get_field_slice_residual_max(rank, block_coords, field, field_prev, block_sizes, boundary, idx_lower, idx_upper);
}

void field_slice2buffer(const int rank, const int* block_coords, const double* restrict field, const int* block_sizes, const int boundary, const int* idx_lower, const int* idx_upper, double* restrict buffer) {
	const int adv0 = 2 * boundary + block_sizes[1];
	const int adv1 = 2 * boundary + block_sizes[2];
	//const int adv2 = idx_upper[2] - idx_lower[2];
	
	int buf = 0;
	for (int i = idx_lower[0]; i < idx_upper[0]; ++i) {
		const int jpos = i * adv0;

		for (int j = idx_lower[1]; j < idx_upper[1]; ++j) {
			//memcpy(buffer + buf, field + (jpos + j) * adv1 + idx_lower[2], sizeof(double) * adv2);
			//buf += adv2;
			const int kpos = (jpos + j) * adv1;
			for (int k = idx_lower[2]; k < idx_upper[2]; ++k) {
				buffer[buf++] = field[kpos + k];
			}
		}
	}
}

void buffer2field_slice(const int rank, const int* block_coords, double* restrict field, const int* block_sizes, const int boundary, const int* idx_lower, const int* idx_upper, double* restrict buffer) {
	const int adv0 = 2 * boundary + block_sizes[1];
	const int adv1 = 2 * boundary + block_sizes[2];
	//const int adv2 = idx_upper[2] - idx_lower[2];
	
	int buf = 0;
	for (int i = idx_lower[0]; i < idx_upper[0]; ++i) {
		const int jpos = i * adv0;

		for (int j = idx_lower[1]; j < idx_upper[1]; ++j) {
			//memcpy(field + (jpos + j) * adv1 + idx_lower[2], buffer + buf, sizeof(double) * adv2);
			//buf += adv2;
			const int kpos = (jpos + j) * adv1;
			for (int k = idx_lower[2]; k < idx_upper[2]; ++k) {
				field[kpos + k] = buffer[buf++];
			}
		}
	}
}

void init_field_slice(const int rank, const int* block_coords, double* restrict field, const int* block_sizes, const int boundary, const int* idx_lower, const int* idx_upper, double val) {
	const int adv0 = 2 * boundary + block_sizes[1];
	const int adv1 = 2 * boundary + block_sizes[2];
	
	int buf = 0;
	for (int i = idx_lower[0]; i < idx_upper[0]; ++i) {
		const int jpos = i * adv0;

		for (int j = idx_lower[1]; j < idx_upper[1]; ++j) {
			const int kpos = (jpos + j) * adv1;

			for (int k = idx_lower[2]; k < idx_upper[2]; ++k) {
				field[kpos + k] = val;
			}
		}
	}
}

//from inside block, to send buffers
void boundaries2buffers(const int rank, const int* block_coords, const double* restrict field, const int* block_sizes, const int boundary, double* restrict* buffer) {
	//print_rank_prefix(rank, block_coords);
	//printf(": boundaries2buffers\n");

	//dim 0
	if (buffer[0]) {
		const int idx_lower[3] = {boundary, boundary, boundary};
		const int idx_upper[3] = {2 * boundary, boundary + block_sizes[1], boundary + block_sizes[2]};
		field_slice2buffer(rank, block_coords, field, block_sizes, boundary, idx_lower, idx_upper, buffer[0]);
	}
	if (buffer[1]) {
		const int idx_lower[3] = {block_sizes[0], boundary, boundary};
		const int idx_upper[3] = {boundary + block_sizes[0], boundary + block_sizes[1], boundary + block_sizes[2]};
		field_slice2buffer(rank, block_coords, field, block_sizes, boundary, idx_lower, idx_upper, buffer[1]);
	}
	//dim 1
	if (buffer[2]) {
		const int idx_lower[3] = {boundary, boundary, boundary};
		const int idx_upper[3] = {boundary + block_sizes[0], 2 * boundary, boundary + block_sizes[2]};
		field_slice2buffer(rank, block_coords, field, block_sizes, boundary, idx_lower, idx_upper, buffer[2]);
	}
	if (buffer[3]) {
		const int idx_lower[3] = {boundary, block_sizes[1], boundary};
		const int idx_upper[3] = {boundary + block_sizes[0], boundary + block_sizes[1], boundary + block_sizes[2]};
		field_slice2buffer(rank, block_coords, field, block_sizes, boundary, idx_lower, idx_upper, buffer[3]);
	}
	//dim 2
	if (buffer[4]) {
		const int idx_lower[3] = {boundary, boundary, boundary};
		const int idx_upper[3] = {boundary + block_sizes[0], boundary + block_sizes[1], 2 * boundary};
		field_slice2buffer(rank, block_coords, field, block_sizes, boundary, idx_lower, idx_upper, buffer[4]);
	}
	if (buffer[5]) {
		const int idx_lower[3] = {boundary, boundary, block_sizes[2]};
		const int idx_upper[3] = {boundary + block_sizes[0], boundary + block_sizes[1], boundary + block_sizes[2]};
		field_slice2buffer(rank, block_coords, field, block_sizes, boundary, idx_lower, idx_upper, buffer[5]);
	}
}

//from recv buffers, to boundaryes
void buffers2boundaries(const int rank, const int* block_coords, double* restrict field, const int* block_sizes, const int boundary, double* restrict* buffer) {
	//print_rank_prefix(rank, block_coords);
	//printf(": buffers2boundaries\n");
	
	//dim 0
	if (buffer[0]) {
		const int idx_lower[3] = {0, boundary, boundary};
		const int idx_upper[3] = {boundary, boundary + block_sizes[1], boundary + block_sizes[2]};
		buffer2field_slice(rank, block_coords, field, block_sizes, boundary, idx_lower, idx_upper, buffer[0]);
	}
	if (buffer[1]) {
		const int idx_lower[3] = {boundary + block_sizes[0], boundary, boundary};
		const int idx_upper[3] = {2 * boundary + block_sizes[0], boundary + block_sizes[1], boundary + block_sizes[2]};
		buffer2field_slice(rank, block_coords, field, block_sizes, boundary, idx_lower, idx_upper, buffer[1]);
	}
	//dim 1
	if (buffer[2]) {
		const int idx_lower[3] = {boundary, 0, boundary};
		const int idx_upper[3] = {boundary + block_sizes[0], boundary, boundary + block_sizes[2]};
		buffer2field_slice(rank, block_coords, field, block_sizes, boundary, idx_lower, idx_upper, buffer[2]);
	}
	if (buffer[3]) {
		const int idx_lower[3] = {boundary, boundary + block_sizes[1], boundary};
		const int idx_upper[3] = {boundary + block_sizes[0], 2 * boundary + block_sizes[1], boundary + block_sizes[2]};
		buffer2field_slice(rank, block_coords, field, block_sizes, boundary, idx_lower, idx_upper, buffer[3]);
	}
	//dim 2
	if (buffer[4]) {
		const int idx_lower[3] = {boundary, boundary, 0};
		const int idx_upper[3] = {boundary + block_sizes[0], boundary + block_sizes[1], boundary};
		buffer2field_slice(rank, block_coords, field, block_sizes, boundary, idx_lower, idx_upper, buffer[4]);
	}
	if (buffer[5]) {
		const int idx_lower[3] = {boundary, boundary, boundary + block_sizes[2]};
		const int idx_upper[3] = {boundary + block_sizes[0], boundary + block_sizes[1], 2 * boundary + block_sizes[2]};
		buffer2field_slice(rank, block_coords, field, block_sizes, boundary, idx_lower, idx_upper, buffer[5]);
	}
}

void init_boundaries(const int rank, const int* block_coords, double* restrict field, const int* block_sizes, int boundary, const int* boundary_active, const double val) {
	//print_rank_prefix(rank, block_coords);
	//printf(": init_boundaries\n");
	
	//dim 0
	if (!boundary_active[0]) {
		const int idx_lower[3] = {0, boundary, boundary};
		const int idx_upper[3] = {boundary, boundary + block_sizes[1], boundary + block_sizes[2]};
		init_field_slice(rank, block_coords, field, block_sizes, boundary, idx_lower, idx_upper, val);
	}
	if (!boundary_active[1]) {
		const int idx_lower[3] = {boundary + block_sizes[0], boundary, boundary};
		const int idx_upper[3] = {2 * boundary + block_sizes[0], boundary + block_sizes[1], boundary + block_sizes[2]};
		init_field_slice(rank, block_coords, field, block_sizes, boundary, idx_lower, idx_upper, val);
	}
	//dim 1
	if (!boundary_active[2]) {
		const int idx_lower[3] = {boundary, 0, boundary};
		const int idx_upper[3] = {boundary + block_sizes[0], boundary, boundary + block_sizes[2]};
		init_field_slice(rank, block_coords, field, block_sizes, boundary, idx_lower, idx_upper, val);
	}
	if (!boundary_active[3]) {
		const int idx_lower[3] = {boundary, boundary + block_sizes[1], boundary};
		const int idx_upper[3] = {boundary + block_sizes[0], 2 * boundary + block_sizes[1], boundary + block_sizes[2]};
		init_field_slice(rank, block_coords, field, block_sizes, boundary, idx_lower, idx_upper, val);
	}
	//dim 2
	if (!boundary_active[4]) {
		const int idx_lower[3] = {boundary, boundary, 0};
		const int idx_upper[3] = {boundary + block_sizes[0], boundary + block_sizes[1], boundary};
		init_field_slice(rank, block_coords, field, block_sizes, boundary, idx_lower, idx_upper, val);
	}
	if (!boundary_active[5]) {
		const int idx_lower[3] = {boundary, boundary, boundary + block_sizes[2]};
		const int idx_upper[3] = {boundary + block_sizes[0], boundary + block_sizes[1], 2 * boundary + block_sizes[2]};
		init_field_slice(rank, block_coords, field, block_sizes, boundary, idx_lower, idx_upper, val);
	}
}

#ifdef COMM_SENDRECV
void communicate_sendrecv(MPI_Comm* mesh_comm, int* block_coords, int* boundary_sizes, int* boundary_rank, int boundary, double* restrict* boundary_send, double* restrict* boundary_recv) {
	int rank;
	MPI_Comm_rank(*mesh_comm, &rank);

	for (int i = 0; i < 3; ++i) {
		const int i2 = 2 * i;
		int other = block_coords[i] % 2 ? boundary_rank[i2] : boundary_rank[i2 + 1];
		int buf = i2 + (block_coords[i] % 2 ? 0 : 1);

		MPI_Status status;
		if (other != MPI_PROC_NULL) {
			//print_rank_prefix(rank, block_coords);
			//printf(": sendrecv with %d size %d\n", other, boundary_sizes[i]);
			MPI_Sendrecv(boundary_send[buf], boundary_sizes[i], MPI_DOUBLE, other, i//int sendtag
				, boundary_recv[buf], boundary_sizes[i], MPI_DOUBLE, other, i//int recvtag
				, *mesh_comm, &status);
		}

		other = block_coords[i] % 2 ? boundary_rank[i2 + 1] : boundary_rank[i2];
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
void communicate_nonblocking(MPI_Comm* mesh_comm, int* block_coords, int* boundary_sizes, int* boundary_rank, int boundary, double* restrict* boundary_send, double* restrict* boundary_recv) {
	int rank;
	MPI_Comm_rank(*mesh_comm, &rank);
	int ncomms = 0;
	MPI_Request requests[12];
	MPI_Status statuses[12];

	for (int i = 0; i < 6; ++i) {
		if (boundary_rank[i] != MPI_PROC_NULL) {
			//print_rank_prefix(rank, block_coords);
			//printf(": sendrecv with %d size %d\n", other, boundary_sizes[i]);
			MPI_Isend(boundary_send[i], boundary_sizes[i / 2], MPI_DOUBLE, boundary_rank[i], i / 2, *mesh_comm, requests + ncomms++);
			//print_rank_prefix(rank, block_coords);
			//printf(": sendrecv with %d size %d\n", other, boundary_sizes[i]);
			MPI_Irecv(boundary_recv[i], boundary_sizes[i / 2], MPI_DOUBLE, boundary_rank[i], i / 2, *mesh_comm, requests + ncomms++);
		}
	}
	
	MPI_Waitall(ncomms, requests, statuses);
}
#endif

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

	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &nranks);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Get_processor_name(processor_name, &processor_name_len); 

	//time recording
	double start_time, end_time;   
	//start_time = MPI_Wtime();

	const int spatial_dims = 3;
	int dim_elems[3] = {1, 1, 1};
	
	int proc_dims;
	int dim_blocks[3] = {1, 1, 1};
	
	const int boundary = 1;
	int iterations = 100;
	const double tolerance = 1e-10;
	
	if (argc < 6) {
		fprintf(stderr, "usage: %s [mesh dimensionality] [dimension 1 elements] [dimension 2 elements] [dimension 3 elements]\n", argv[0]);
		exit(1);
	}
	
	int rank_dbg;
	
	if (rank == 0) {
		int res = sscanf(argv[1], "%d", &proc_dims);
		if (res != 1 || proc_dims < 1 || proc_dims > 3) {
			fprintf(stderr, "unusable mesh dimensionality\n");
			exit(1);
		}
		for (int i = 0; i < proc_dims; ++i) {
			dim_blocks[i] = 0;
		}

		for (int i = 0; i < 3; ++i) {
			res = sscanf(argv[2+i], "%d", dim_elems+i);
			if (res != 1) {
				fprintf(stderr, "unusable dimension %d elements\n", i+1);
				exit(1);
			}
		}

		res = sscanf(argv[5], "%d", &iterations);
		if (res != 1 || !iterations) {
			fprintf(stderr, "unusable max iterations\n");
			exit(1);
		}
		
		if (argc >= 7) {
			res = sscanf(argv[6], "%d", &rank_dbg);
			if (res != 1) {
				fprintf(stderr, "unusable rank_dbg\n");
				exit(1);
			}
		} else {
			rank_dbg = MPI_PROC_NULL;
		}
		
		MPI_Dims_create(nranks, proc_dims, dim_blocks);
		//optimize dim_blocks vs dim_elems to minimize boundaries buffer sizes
		//TODO
		#ifdef SORT_DIMS
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
		
		//double* blob = (double*) malloc(sizeof(double) * 2 * field_extended_elems);
		double* blob = (double*) calloc(sizeof(double), 2 * field_extended_elems);
		//init prev field to 0
		//for (int i = 0; i < 2 * field_extended_elems; ++i) {
		//	blob[i] = 0.0;
		//}

		double* const fields[2] = {blob, blob + field_extended_elems};
		
		//boundaries metadata
		int boundary_sizes[3];
		boundary_sizes[0] = boundary * block_sizes[1] * block_sizes[2];
		boundary_sizes[1] = boundary * block_sizes[0] * block_sizes[2];
		boundary_sizes[2] = boundary * block_sizes[0] * block_sizes[1];
		
		int boundary_active[3 * 2];
		int boundary_rank[3 * 2];
		int boundary_total_elems = 0;
		for (int i = 0; i < 3; ++i) {
			int i2 = 2 * i;
			MPI_Cart_shift(mesh_comm, i, 1, boundary_rank + i2, boundary_rank + i2 + 1);
			
			boundary_active[i2] = boundary_rank[i2] != MPI_PROC_NULL ? 1 : 0;
			boundary_active[i2 + 1] = boundary_rank[i2 + 1] != MPI_PROC_NULL ? 1 : 0;
			
			boundary_total_elems += boundary_sizes[i] * (boundary_active[i2] + boundary_active[i2 + 1]);
		}
		
		//allocate one buffers for sending and receiving boundaries
		double* boundary_send[3 * 2];
		double* boundary_recv[3 * 2];
		
		boundary_send[0] = (double*) malloc(sizeof(double) * 2 * boundary_total_elems);
		for (int i = 0; i < 3 * 2 - 1; ++i) {
			boundary_send[i+1] = boundary_send[i] + boundary_active[i] * boundary_sizes[i / 2];
		}
		boundary_recv[0] = boundary_send[5] + boundary_active[5] * boundary_sizes[2];
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
		init_boundaries(rank, block_coords, fields[field_previous], block_sizes, boundary, boundary_active, val);
		init_boundaries(rank, block_coords, fields[field_current], block_sizes, boundary, boundary_active, val);

		for (int i = 0; i < iterations; ++i) {
			if (rank == rank_dbg) {
				print_field(fields[field_previous], block_sizes, boundary);
			}
			
			//copy own boundaries from prev field to send buffers
			boundaries2buffers(rank, block_coords, fields[field_previous], block_sizes, boundary, boundary_send);

			//communications
			#ifdef COMM_SENDRECV
			communicate_sendrecv(&mesh_comm, block_coords, boundary_sizes, boundary_rank, boundary, boundary_send, boundary_recv);
			#else
			communicate_nonblocking(&mesh_comm, block_coords, boundary_sizes, boundary_rank, boundary, boundary_send, boundary_recv);
			#endif

			//copy external boundaries from recv buffers to prev field 
			buffers2boundaries(rank, block_coords, fields[field_previous], block_sizes, boundary, boundary_recv);
			
			//compute update
			update_field(rank, block_coords, fields[field_current], fields[field_previous], block_sizes, boundary);
			
			//check residuals
			const double res = get_field_residual_max(rank, block_coords, fields[field_current], fields[field_previous], block_sizes, boundary);
			
			double res_max;
			MPI_Allreduce(&res, &res_max, 1, MPI_DOUBLE, MPI_MAX, mesh_comm);
			if (rank == rank_dbg) {
				//print_rank_prefix(rank, block_coords);
				//print_matrix_dbl("current matrix", fields[field_current], block_sizes[0] + 2 * boundary, block_sizes[1] + 2 * boundary);
				printf("rank %d(%d, %d, %d) residual max %g\n", rank, block_coords[0], block_coords[1], block_coords[2], res_max);
			}
			printf("rank %d(%d, %d, %d) residual max %g\n", rank, block_coords[0], block_coords[1], block_coords[2], res_max);

			const int tmp = field_current;
			field_current = field_previous;
			field_previous = tmp;
		}
	}

	MPI_Finalize();
}
