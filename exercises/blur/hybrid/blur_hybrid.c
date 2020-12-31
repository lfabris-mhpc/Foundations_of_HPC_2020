#include <stdlib.h>
#include <stdio.h>

#include <math.h>
#include <string.h>

#include <assert.h>
#include <errno.h>

#include <mpi.h>
#ifdef _OPENMP
#include <omp.h>
#endif

#include <utils.h>

int main (int argc , char *argv[])
{
	int ret;
	
	#ifdef _OPENMP
	int mpi_thread_provided;
	ret = MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &mpi_thread_provided);
	#else
	ret = MPI_Init(&argc, &argv);
	#endif
	assert(ret == MPI_SUCCESS);
	
	#ifdef TIMING
	double timing_wall = - MPI_Wtime();
	#endif

	#ifndef NDEBUG
	int rank_dbg = 0;
	#endif
	int nranks, rank;
	ret = MPI_Comm_size(MPI_COMM_WORLD, &nranks);
	assert(ret == MPI_SUCCESS);
	ret = MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	assert(ret == MPI_SUCCESS);

	metadata meta;
	if (rank == 0) {
		ret = params_from_stdin(2, meta.mesh_sizes);
		if (ret) {
			MPI_Abort(MPI_COMM_WORLD, 1);
			exit(1);
		}
	}

	char* img_path;
	int kernel_type;
	int kernel_radiuses[2];
	double kernel_params0;
	char* img_save_path;

	ret = params_from_args(argc, argv
		, 2
		, &img_path
		, &kernel_type
		, kernel_radiuses
		, &kernel_params0
		, &img_save_path);
	if (rank == 0 && ret) {
		MPI_Abort(MPI_COMM_WORLD, 1);
	}

	#ifndef NDEBUG
	if (rank == 0) {
		printf("image_path: %s\n", img_path);
		printf("kernel_type: %d\n", kernel_type);
		printf("kernel_radiuses: %d %d\n", kernel_radiuses[0], kernel_radiuses[1]);
		if (kernel_type == KERNEL_TYPE_WEIGHTED) {
			printf("weighted_kernel_f: %lf\n", kernel_params0);
		}
		if (img_save_path) {
			printf("img_save_path: %s\n", img_save_path);
		}
	}
	#endif

	if (rank == 0) {
		#ifdef TIMING
		double timing_header_read = - MPI_Wtime();
		#endif
		
		ret = pgm_get_metadata(img_path, &meta);
		if (ret) {
			fprintf(stderr, "could not open file %s\n", img_path);
			MPI_Abort(MPI_COMM_WORLD, 1);
		}
		
		#ifdef TIMING
		timing_header_read += MPI_Wtime();
		printf("timing_header_read: %lf\n", timing_header_read);
		#endif
		
		#ifndef NDEBUG
		printf("img_sizes (%d, %d)\n", meta.img_sizes[0], meta.img_sizes[1]);
		#endif

		int reorg = !meta.mesh_sizes[0] && !meta.mesh_sizes[1];
		ret = MPI_Dims_create(nranks, 2, meta.mesh_sizes);
		assert(ret == MPI_SUCCESS);
		
		if (reorg && (meta.mesh_sizes[0] > meta.mesh_sizes[1]) != (meta.img_sizes[0] > meta.img_sizes[1])) {
			int tmp = meta.mesh_sizes[1];
			meta.mesh_sizes[1] = meta.mesh_sizes[0];
			meta.mesh_sizes[0] = tmp;
		}
		
		#ifndef NDEBUG
		printf("mesh_sizes: (%d, %d)\n", meta.mesh_sizes[0], meta.mesh_sizes[1]);
		#endif
	}
	
	MPI_Datatype meta_type;
	ret = metadata_type_commit(&meta_type);
	assert(ret == MPI_SUCCESS);
	
	ret = MPI_Bcast(&meta, 1, meta_type, 0, MPI_COMM_WORLD);
	assert(ret == MPI_SUCCESS);
	
	ret = MPI_Type_free(&meta_type);
	assert(ret == MPI_SUCCESS);
	
	int dim_periodic[2] = {0, 0};
	MPI_Comm mesh_comm;
	
	ret = MPI_Cart_create(MPI_COMM_WORLD, 2, meta.mesh_sizes, dim_periodic, 1, &mesh_comm);
	assert(ret == MPI_SUCCESS);

	if (mesh_comm != MPI_COMM_NULL) {
		ret = MPI_Comm_rank(mesh_comm, &rank);
		assert(ret == MPI_SUCCESS);
		ret = MPI_Comm_size(mesh_comm, &nranks);
		assert(ret == MPI_SUCCESS);

		int block_coords[2];
		ret = MPI_Cart_coords(mesh_comm, rank, 2, block_coords);
		assert(ret == MPI_SUCCESS);

		#ifdef _OPENMP
		if (rank == 0) {
			print_rank_prefix(stdout, rank, block_coords);
			printf(": ");
			print_thread_provided(mpi_thread_provided, omp_get_max_threads());
		}
		#endif

		int pixel_size = 1 + (meta.intensity_max > 255);
		MPI_Datatype pixel_type = meta.intensity_max > 255 ? MPI_UNSIGNED_SHORT : MPI_BYTE;
		//int pixel_channels = 1;
		
		#ifndef NDEBUG
		if (rank == rank_dbg) {
			printf("pixel_size: %d (intensity_max: %d)\n", pixel_size, meta.intensity_max);
		}
		#endif

		//{up, down, left, right}
		int neighbor_ranks[4];

		//relative to the img; no halos
		int block_sizes[2];
		int block_lower[2];
		//relative to the image; with halos
		int block_haloed_lower[2];

		//haloed block_extents
		int field_sizes[2];
		//non-haloed area to blur
		int field_lower[2];
		int field_upper[2];

		int field_elems = 1;
		int field_dst_elems = 1;

		for (int i = 0; i < 2; ++i) {
			block_sizes[i] = meta.img_sizes[i] / meta.mesh_sizes[i];
			block_lower[i] = block_sizes[i] * block_coords[i];

			int rem = meta.img_sizes[i] % meta.mesh_sizes[i];
			if (block_coords[i] < rem) {
				++block_sizes[i];
			}
			if (rem) {
				block_lower[i] += imin(block_coords[i], rem);
			}
			
			if (block_sizes[i] < kernel_radiuses[i]) {
				print_rank_prefix(stderr, rank, block_coords);
				fprintf(stderr, "domain decomposition failed, block_size[%d] %d < %d kernel_radiuses[%d]\n"
					, i, block_sizes[i], kernel_radiuses[i], i);
				MPI_Abort(mesh_comm, 1);
			}

			ret = MPI_Cart_shift(mesh_comm, i, 1, neighbor_ranks + 2 * i, neighbor_ranks + 2 * i + 1);
			assert(ret == MPI_SUCCESS);

			//the following are valid under the assumption k << img_size
			field_lower[i] = neighbor_ranks[2 * i] != MPI_PROC_NULL ? kernel_radiuses[i] : 0;
			field_upper[i] = field_lower[i] + block_sizes[i];

			field_sizes[i] = field_upper[i] + (neighbor_ranks[2 * i + 1] != MPI_PROC_NULL ? kernel_radiuses[i] : 0);

			block_haloed_lower[i] = block_lower[i] + (field_lower[i] ? - kernel_radiuses[i] : 0);

			field_elems *= field_sizes[i];
			field_dst_elems *= block_sizes[i];
		}

		//haloed buffer (input)
		unsigned short int* field = (unsigned short int*) malloc(sizeof(unsigned short int) * field_elems);
		if (!field) {
			MPI_Abort(mesh_comm, 1);
			exit(1);
		}

		#ifndef NDEBUG
		print_rank_prefix(stdout, rank, block_coords);
		printf(": img_view_input defined as img[%d:%d, %d:%d] (img shape: (%d, %d))\n"
			, block_haloed_lower[0], block_haloed_lower[0] + field_sizes[0]
			, block_haloed_lower[1], block_haloed_lower[1] + field_sizes[1]
			, meta.img_sizes[0], meta.img_sizes[1]);
		#endif
		
		//read view: haloed block
		MPI_Datatype img_view_input;
		ret = MPI_Type_create_subarray(2, meta.img_sizes
			, field_sizes, block_haloed_lower
			, MPI_ORDER_C, pixel_type
			, &img_view_input);
		assert(ret == MPI_SUCCESS);

		ret = MPI_Type_commit(&img_view_input);
		assert(ret == MPI_SUCCESS);

		#ifdef TIMING
		double timing_file_read = - MPI_Wtime();
		#endif
		
		MPI_File fin;
		ret = MPI_File_open(mesh_comm, img_path, MPI_MODE_RDONLY, MPI_INFO_NULL, &fin);
		assert(ret == MPI_SUCCESS);

		ret = MPI_File_set_view(fin, meta.header_length_input, pixel_type, img_view_input, "native", MPI_INFO_NULL);
		assert(ret == MPI_SUCCESS);

		MPI_Status status;
		int count;
		ret = MPI_File_read(fin, field, field_elems, pixel_type, &status);
		assert(ret == MPI_SUCCESS);
		ret = MPI_Get_count(&status, pixel_type, &count);
		assert(ret == MPI_SUCCESS && count == field_elems);

		ret = MPI_File_close(&fin);
		assert(ret == MPI_SUCCESS);
		
		#ifdef TIMING
		timing_file_read += MPI_Wtime();
		print_rank_prefix(stdout, rank, block_coords);
		printf(": timing_file_read: %lf (bandwidth: %lfMB/s)\n", timing_file_read, (field_elems * pixel_size) / (1024 * 1024 * timing_file_read));
		#endif

		ret = MPI_Type_free(&img_view_input);
		assert(ret == MPI_SUCCESS);
		
		preprocess_buffer(field, field_elems, pixel_size);

		//buffer without halos (output)
		unsigned short int* field_dst = (unsigned short int*) malloc(sizeof(unsigned short int) * field_dst_elems);
		if (!field) {
			MPI_Abort(mesh_comm, 1);
		}
		
		const int kernel_diameters[2] = {2 * kernel_radiuses[0] + 1, 2 * kernel_radiuses[1] + 1};
		double* kernel = (double*) malloc(sizeof(double) * kernel_diameters[0] * kernel_diameters[1]);
		if (!kernel) {
			MPI_Abort(mesh_comm, 1);
		}

		ret = kernel_init(kernel_type
			, kernel_radiuses
			, kernel_params0
			, kernel);
		assert(!ret);

		#ifndef NDEBUG
		if (rank == rank_dbg) {
			printf("kernel:\n");
			for (int i = 0; i < kernel_diameters[0]; ++i) {
				printf("[");
				for (int j = 0; j < kernel_diameters[1]; ++j) {
					printf("%lf ", kernel[i * kernel_diameters[1] + j]);
				}
				printf("]\n");
			}
		}
		#endif

		//blur
		#ifdef TIMING
		double timing_blur = - MPI_Wtime();
		#endif
		
		#if BLOCKING_ON
		#ifndef BLOCKING_ROWS
		#define BLOCKING_ROWS 256
		#endif
		#ifndef BLOCKING_COLUMNS
		#define BLOCKING_COLUMNS 256
		#endif
		const int blocking[2] = {BLOCKING_ROWS, BLOCKING_COLUMNS};
		kernel_process_byblocks(kernel
			, kernel_radiuses
			, field
			, field_sizes
			, field_lower
			, field_upper
			, field_dst
			, block_sizes
			, blocking);
		#else
		const int field_dst_lower[2] = {0, 0};
		kernel_process_block(kernel
			, kernel_radiuses
			, field
			, field_sizes
			, field_lower
			, field_upper
			, field_dst
			, block_sizes
			, field_dst_lower);
		#endif
		
		#ifdef TIMING
		timing_blur += MPI_Wtime();
		print_rank_prefix(stdout, rank, block_coords);
		printf(": timing_blur: %lf\n", timing_blur);
		#endif

		free(kernel);
		
		postprocess_buffer(field_dst, field_dst_elems, pixel_size);

		#ifndef NDEBUG
		print_rank_prefix(stdout, rank, block_coords);
		printf(": img_view_output defined as img[%d:%d, %d:%d]\n"
			, block_lower[0], block_lower[0] + block_sizes[0]
			, block_lower[1], block_lower[1] + block_sizes[1]);
		#endif
		
		//write view: block without halos
		MPI_Datatype img_view_output;
		ret = MPI_Type_create_subarray(2, meta.img_sizes
			, block_sizes, block_lower
			, MPI_ORDER_C, pixel_type
			, &img_view_output);
		assert(ret == MPI_SUCCESS);

		ret = MPI_Type_commit(&img_view_output);
		assert(ret == MPI_SUCCESS);

		if (!img_save_path) {
			//create img_save_path
			//TODO
			img_save_path = "blurred.pgm";
		}
		#ifndef NDEBUG
		if (rank == rank_dbg) {
			printf("output image: %s\n", img_save_path);
		}
		#endif

		#ifdef TIMING
		double timing_file_write = - MPI_Wtime();
		#endif

		MPI_File fout;
		ret = MPI_File_open(mesh_comm, img_save_path, MPI_MODE_WRONLY | MPI_MODE_CREATE, MPI_INFO_NULL, &fout);
		assert(ret == MPI_SUCCESS);

		MPI_Offset header_offset_output;
		if (rank == 0) {
			char header[200];
			ret = snprintf(header, sizeof(header), "P5\n# generated by\n# Lorenzo Fabris\n%d %d\n%d\n", meta.img_sizes[1], meta.img_sizes[0], meta.intensity_max);
			assert(ret >= 0);

			ret = MPI_File_write(fout, header, strlen(header), MPI_BYTE, &status);
			assert(ret == MPI_SUCCESS);

			ret = MPI_File_get_position(fout, &header_offset_output);
			assert(ret == MPI_SUCCESS);
		}
		
		ret = MPI_Bcast(&header_offset_output, 1, MPI_OFFSET, 0, mesh_comm);
		assert(ret == MPI_SUCCESS);

		ret = MPI_File_set_view(fout, header_offset_output, pixel_type, img_view_output, "native", MPI_INFO_NULL);
		assert(ret == MPI_SUCCESS);

		ret = MPI_File_write(fout, field_dst, field_dst_elems, pixel_type, &status);
		assert(ret == MPI_SUCCESS);
		ret = MPI_Get_count(&status, pixel_type, &count);
		assert(ret == MPI_SUCCESS && count == field_dst_elems);

		ret = MPI_File_close(&fout);
		assert(ret == MPI_SUCCESS);

		#ifdef TIMING
		timing_file_write += MPI_Wtime();
		print_rank_prefix(stdout, rank, block_coords);
		printf(": timing_file_write: %lf (bandwidth: %lfMB/s)\n", timing_file_write, (field_elems * pixel_size) / (1024 * 1024 * timing_file_write));
		#endif

		ret = MPI_Type_free(&img_view_output);
		assert(ret == MPI_SUCCESS);

		free(field_dst);
		free(field);
		
		#ifdef TIMING
		timing_wall += MPI_Wtime();
		print_rank_prefix(stdout, rank, block_coords);
		printf(": timing_wall: %lf\n", timing_wall);
		#endif
	}

	MPI_Finalize();
}
