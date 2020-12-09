#include <stdlib.h>
#include <unistd.h>
#include <stdio.h>
#include <math.h>
#include <string.h>
#include <math.h>

#include <mpi.h>

#define USE MPI

void print_array_int(const char* name, const int* arr, const int size) {
	printf("%s: [", name);
	for (int i = 0; i < size; ++i) {
		printf("%d ", arr[i]);
	}
	printf("]\n");
}

void print_array_dbl(const char* name, const double* arr, const int size) {
	printf("%s: [", name);
	for (int i = 0; i < size; ++i) {
		printf("%f ", arr[i]);
	}
	printf("]\n");
}

void print_array_dbl_ptr(const char* name, const double** arr, const int size) {
	printf("%s: [", name);
	for (int i = 0; i < size; ++i) {
		printf("%p ", arr[i]);
	}
	printf("]\n");
}

void print_matrix_dbl(const char* name, const double* arr, const int rows, const int columns) {
	printf("%s:\n", name);
	for (int i = 0; i < rows; ++i) {
		printf("[");
		for (int j = 0; j < columns; ++j) {
			printf("%f ", arr[i * columns + j]);
		}
		printf("]\n");
	}
	printf("]\n");
}

void allocate_halo_buffer(const int rank, const int* block_coords, const int dims, const int* halo_sizes, const int* halo_ok, double** halo_buffer) {
	//printf("rank %d(%d, %d): allocate_halo_buffer\n", rank, block_coords[0], block_coords[1]);
	int halo_buffer_elems = 0;

	for (int i = 0; i < dims; ++i) {
		halo_buffer_elems += halo_sizes[i] * (halo_ok[2 * i] + halo_ok[2 * i + 1]);
	}

	halo_buffer[0] = (double*) malloc(sizeof(double) * halo_buffer_elems);
	for (int i = 1; i < 2 * dims; ++i) {
		halo_buffer[i] = halo_buffer[i - 1] + halo_ok[i - 1] * halo_sizes[i / 2];
	}

	for (int i = 0; i < 2 * dims; ++i) {
		halo_buffer[i] = halo_ok[i] ? halo_buffer[i] : NULL;
	}
}

void update_matrix(const int rank, const int* block_coords, double* matrix, const double* matrix_prev, const int dims, const int* block_sizes, const int halo) {
	//printf("rank %d(%d, %d): update_matrix\n", rank, block_coords[0], block_coords[1]);
	int rows = block_sizes[0];
	int columns = block_sizes[1];
	int disp_down = 2 * halo + columns;
	double factor = 1.0 / (4 * halo);

	for (int i = 0; i < rows; ++i) {
		int disp = (halo + i) * disp_down + halo;

		for (int j = 0; j < columns; ++j) {
			int pos = disp + j;

			matrix[pos] = 0;
			//use accumulators?
			for (int k = 1; k <= halo; ++k) {
				matrix[pos] += matrix_prev[pos - k];
				matrix[pos] += matrix_prev[pos + k];
				matrix[pos] += matrix_prev[pos - k * disp_down];
				matrix[pos] += matrix_prev[pos + k * disp_down];
			}

			matrix[pos] *= factor;
		}
	}
}

double get_residual(const double* matrix, const double* prev_matrix, const int dims, const int* block_sizes, const int halo) {
	//printf("rank %d(%d, %d): get_diffs\n", rank, block_coords[0], block_coords[1]);
	int rows = block_sizes[0];
	int columns = block_sizes[1];
	int disp_down = 2 * halo + columns;
	double res = 0;

	for (int i = 0; i < rows - 1; ++i) {
		int disp = (halo + i) * disp_down + halo;
		
		for (int j = 0; j < columns - 1; ++j) {
			int pos = disp + j;
			
			res += (matrix[pos] - prev_matrix[pos]) * (matrix[pos] - prev_matrix[pos]);
		}
	}

	return res / (rows * columns);
}

//from inside block, to send buffers
void haloes2buffers(int rank, int* block_coords, double* matrix, int dims, int* block_sizes, int halo, double** halo_buffer) {
	//printf("rank %d(%d, %d): haloes2buffers\n", rank, block_coords[0], block_coords[1]);
	int rows = block_sizes[0];
	int columns = block_sizes[1];
	int disp_down = 2 * halo + columns;

	if (halo_buffer[0]) {
		for (int i = 0; i < halo; ++i) {
			for (int j = 0; j < columns; ++j) {
				halo_buffer[0][i * columns + j] = matrix[(halo + i) * disp_down + halo + j];
			}
		}
	}
	if (halo_buffer[1]) {
		for (int i = 0; i < halo; ++i) {
			for (int j = 0; j < columns; ++j) {
				halo_buffer[1][i * columns + j] = matrix[(rows + i) * disp_down + halo + j];
			}
		}
	}
	if (halo_buffer[2]) {
		for (int i = 0; i < rows; ++i) {
			for (int j = 0; j < halo; ++j) {
				halo_buffer[2][i * halo + j] = matrix[(halo + i) * disp_down + halo + j];
			}
		}
	}
	if (halo_buffer[3]) {
		for (int i = 0; i < rows; ++i) {
			for (int j = 0; j < halo; ++j) {
				halo_buffer[3][i * halo + j] = matrix[(halo + i) * disp_down + columns + j];
			}
		}
	}
}

//from recv buffers, to haloes
void buffers2haloes(int rank, int* block_coords, double* matrix, int dims, int* block_sizes, int halo, double** halo_buffer) {
	//printf("rank %d(%d, %d): buffers2haloes\n", rank, block_coords[0], block_coords[1]);
	int rows = block_sizes[0];
	int columns = block_sizes[1];
	int disp_down = 2 * halo + columns;
	
	if (halo_buffer[0]) {
		for (int i = 0; i < halo; ++i) {
			for (int j = 0; j < columns; ++j) {
				matrix[i * disp_down + halo + j] = halo_buffer[0][i * columns + j];
			}
		}
	}
	if (halo_buffer[1]) {
		for (int i = 0; i < halo; ++i) {
			for (int j = 0; j < columns; ++j) {
				matrix[(rows + halo + i) * disp_down + halo + j] = halo_buffer[1][i * columns + j];
			}
		}
	}
	if (halo_buffer[2]) {
		for (int i = 0; i < rows; ++i) {
			for (int j = 0; j < halo; ++j) {
				matrix[(halo + i) * disp_down + j] = halo_buffer[2][i * halo + j];
			}
		}
	}
	if (halo_buffer[3]) {
		for (int i = 0; i < rows; ++i) {
			for (int j = 0; j < halo; ++j) {
				matrix[(halo + i) * disp_down + columns + halo + j] = halo_buffer[3][i * halo + j];
			}
		}
	}
}

void init_boundary(int rank, int* block_coords, double* matrix, int dims, int* block_sizes, int halo, int* halo_ok, double val) {
	//printf("rank %d(%d, %d): init_boundary\n", rank, block_coords[0], block_coords[1]);
	int rows = block_sizes[0];
	int columns = block_sizes[1];
	int disp_down = 2 * halo + columns;
	
	if (!halo_ok[0]) {
		for (int i = 0; i < halo; ++i) {
			for (int j = 0; j < columns; ++j) {
				matrix[i * disp_down + halo + j] = val;
			}
		}
	}
	if (!halo_ok[1]) {
		for (int i = 0; i < halo; ++i) {
			for (int j = 0; j < columns; ++j) {
				matrix[(rows + halo + i) * disp_down + halo + j] = val;
			}
		}
	}
	if (!halo_ok[2]) {
		for (int i = 0; i < rows; ++i) {
			for (int j = 0; j < halo; ++j) {
				matrix[(halo + i) * disp_down + j] = val;
			}
		}
	}
	if (!halo_ok[3]) {
		for (int i = 0; i < rows; ++i) {
			for (int j = 0; j < halo; ++j) {
				matrix[(halo + i) * disp_down + columns + halo + j] = val;
			}
		}
	}
}

void communicate_sendrecv(MPI_Comm* cart_comm, int dims, int* block_coords, int* halo_sizes, int halo, double** halo_send, double** halo_recv) {
	int rank;
	MPI_Comm_rank(*cart_comm, &rank);
	//print_array_dbl_ptr("halo_send", halo_send, dims*2);
	//print_array_dbl_ptr("halo_recv", halo_recv, dims*2);

	for (int d = 0; d < dims; ++d) {
		int prev, next;
		MPI_Cart_shift(*cart_comm, d, 1, &prev, &next);
		int other = block_coords[d] % 2 ? next : prev;
		int buf = 2 * d + (block_coords[d] % 2 ? 1 : 0);

		MPI_Status status;
		if (other != MPI_PROC_NULL) {
			//printf("rank %d(%d, %d) sendrecv with %d\n", rank, block_coords[0], block_coords[1], other); 
			MPI_Sendrecv(halo_send[buf], halo_sizes[d], MPI_DOUBLE, other, d//int sendtag
				, halo_recv[buf], halo_sizes[d], MPI_DOUBLE, other, d//int recvtag
				, *cart_comm, &status);
		}

		other = block_coords[d] % 2 ? prev : next;
		buf = 2 * d + (block_coords[d] % 2 ? 0 : 1);
		
		if (other != MPI_PROC_NULL) {
			//printf("rank %d(%d, %d) sendrecv with %d\n", rank, block_coords[0], block_coords[1], other); 
			MPI_Sendrecv(halo_send[buf], halo_sizes[d], MPI_DOUBLE, other, d//int sendtag
				, halo_recv[buf], halo_sizes[d], MPI_DOUBLE, other, d//int recvtag
				, *cart_comm, &status);
		}
	}
}

int main (int argc , char *argv[])
{
	//general MPI util vars
	int ranks_num, rank, processor_name_len;
	char processor_name[MPI_MAX_PROCESSOR_NAME];
	MPI_Status status;
	MPI_Request request;
	
	int tag = 123;

	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &ranks_num);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Get_processor_name(processor_name, &processor_name_len); 

	//time recording
	double start_time, end_time;   
	//start_time = MPI_Wtime();

	const int dims = 2;
	int dim_elems[2] = {2432, 3345};
	int halo = 1;
	int iterations = 100;
	double tolerance = 1e-17;

	int dim_blocks[2] = {0, 0};
	MPI_Dims_create(ranks_num, 2, dim_blocks);

	int dim_periodic[2] = {0, 0};
	MPI_Comm cart_comm;
	MPI_Cart_create(MPI_COMM_WORLD, dims, dim_blocks, dim_periodic, 1, &cart_comm);

	if (cart_comm != MPI_COMM_NULL) {
		MPI_Comm_rank(cart_comm, &rank);
		MPI_Comm_size(cart_comm, &ranks_num);

		int block_coords[2];
		MPI_Cart_coords(cart_comm, rank, 2, block_coords);

		int block_sizes[2];
		for (int i = 0; i < dims; ++i) {
			block_sizes[i] = dim_elems[i] / dim_blocks[i];
			if (block_coords[i] < dim_elems[i] % dim_blocks[i]) {
				++block_sizes[i];
			}
		}

		//allocate 2 * haloed matrix (wastes 2 * (2 * halo)^dims elements)
		int matrix_elems = block_sizes[0] * block_sizes[1];
		int matrix_haloed_elems = (2 * halo + block_sizes[0]) * (2 * halo + block_sizes[1]);
		
		int halo_sizes[2];
		halo_sizes[0] = halo * block_sizes[1];
		halo_sizes[1] = halo * block_sizes[0];

		int halo_ok[4];
		//left
		halo_ok[0] = block_coords[0] != 0;
		//right
		halo_ok[1] = block_coords[0] != (dim_blocks[0] - 1);
		//up
		halo_ok[2] = block_coords[1] != 0;
		//down
		halo_ok[3] = block_coords[1] != (dim_blocks[1] - 1);
		
		int blob_elems = 2 * matrix_haloed_elems;
		double* blob = (double*) malloc(sizeof(double) * blob_elems);
		for (int i = 0; i < blob_elems; ++i) {
			blob[i] = 0.0;
		}

		double* elems[2] = {blob, blob + matrix_haloed_elems};
		
		//allocate one buffer for send, one for receive
		double* halo_send[4];
		allocate_halo_buffer(rank, block_coords, dims, halo_sizes, halo_ok, halo_send);
		
		double* halo_recv[4];
		allocate_halo_buffer(rank, block_coords, dims, halo_sizes, halo_ok, halo_recv);

		int cur_matrix = 0;
		int prev_matrix = 1;
		//init boundaries where no halo
		double val = 10;
		init_boundary(rank, block_coords, elems[prev_matrix], dims, block_sizes, halo, halo_ok, val);
		init_boundary(rank, block_coords, elems[cur_matrix], dims, block_sizes, halo, halo_ok, val);

		/*
		if (rank == 0) {
			print_matrix_dbl("initial", elems[prev_matrix], block_sizes[0] + 2 * halo, block_sizes[1] + 2 * halo);
		}
		*/

		for (int i = 0; i < iterations; ++i) {
			//copy own halos from prev matrix to send buffers
			haloes2buffers(rank, block_coords, elems[prev_matrix], dims, block_sizes, halo, halo_send);

			//communications
			communicate_sendrecv(&cart_comm, dims, block_coords, halo_sizes, halo, halo_send, halo_recv);

			//copy external halos from recv buffers to prev matrix 
			buffers2haloes(rank, block_coords, elems[prev_matrix], dims, block_sizes, halo, halo_recv);
			
			//compute
			update_matrix(rank, block_coords, elems[cur_matrix], elems[prev_matrix], dims, block_sizes, halo);
			
			//check residuals
			double res = get_residual(elems[cur_matrix], elems[prev_matrix], dims, block_sizes, halo);
			double res_max;
			MPI_Allreduce(&res, &res_max, 1, MPI_DOUBLE, MPI_MAX, cart_comm);
			if (rank == 0) {
				//print_matrix_dbl("current matrix", elems[cur_matrix], block_sizes[0] + 2 * halo, block_sizes[1] + 2 * halo);
				printf("rank %d(%d, %d) residual max %g\n", rank, block_coords[0], block_coords[1], res_max);
			}

			int tmp = cur_matrix;
			cur_matrix = prev_matrix;
			prev_matrix = tmp;
		}
	}

	MPI_Finalize();
}
