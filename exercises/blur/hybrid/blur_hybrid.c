#ifndef FLOAT_T
#define FLOAT_T double
#endif

#define VERBOSITY_OFF 0
#define VERBOSITY_INFO 1
#define VERBOSITY_KERNEL 2
#define VERBOSITY_BLUR 3
#define VERBOSITY_BLUR_POS 4

#if defined(BLOCKING_BLUR_ON) || defined(BLOCKING_BLUR_ROWS) || defined(BLOCKING_BLUR_COLUMNS)
#define BLOCKING_BLUR_ON

#ifndef BLOCKING_BLUR_ROWS
#define BLOCKING_BLUR_ROWS 128
#endif
#ifndef BLOCKING_BLUR_COLUMNS
#define BLOCKING_BLUR_COLUMNS 128
#endif
#endif

#if defined(BLOCKING_POS_ON) || defined(BLOCKING_POS_ROWS) || defined(BLOCKING_POS_COLUMNS)
#define BLOCKING_POS_ON

#ifndef BLOCKING_POS_ROWS
#define BLOCKING_POS_ROWS 64
#endif
#ifndef BLOCKING_POS_COLUMNS
#define BLOCKING_POS_COLUMNS 64
#endif
#endif

#include <mpi.h>

#ifdef _OPENMP
#include <omp.h>
#endif

#include <utils.h>

int main(int argc , char** argv)
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

	#ifdef VERBOSITY
	int rank_dbg = 0;
	#endif
	int nranks, rank;
	ret = MPI_Comm_size(MPI_COMM_WORLD, &nranks);
	assert(ret == MPI_SUCCESS);
	ret = MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	assert(ret == MPI_SUCCESS);

	metadata meta;
	if (rank == 0) {
		ret = params_from_stdin(meta.mesh_sizes);
		if (ret) {
			MPI_Abort(MPI_COMM_WORLD, 1);
		}
	}

	char* img_path;
	int kernel_type;
	int kernel_sizes[2];
	FLOAT_T kernel_params0;
	char* img_save_path;

	ret = params_from_args(argc, argv
		, &img_path
		, &kernel_type
		, kernel_sizes
		, &kernel_params0
		, &img_save_path);
	if (rank == 0 && ret) {
		MPI_Abort(MPI_COMM_WORLD, 1);
	}

	#if VERBOSITY >= VERBOSITY_INFO
	if (rank == 0) {
		printf("image_path: %s\n", img_path);
		printf("kernel_type: %d\n", kernel_type);
		printf("kernel_sizes: %d %d\n", kernel_sizes[0], kernel_sizes[1]);
		if (kernel_type == KERNEL_TYPE_WEIGHTED) {
			printf("weighted_kernel_f: %f\n", kernel_params0);
		}
		if (img_save_path) {
			printf("img_save_path: %s\n", img_save_path);
		}
	}
	#endif

	int img_save_path_auto = 0;
	if (!img_save_path) {
		ret = img_save_path_generate(img_path
			, kernel_type
			, kernel_sizes
			, kernel_type == 1 ? argv[4] : NULL
			, &img_save_path);
		if (ret) {
			MPI_Abort(MPI_COMM_WORLD, 1);
		}

		img_save_path_auto = 1;

		#if VERBOSITY >= VERBOSITY_INFO
		if (rank == 0) {
			printf("img_save_path (auto generated): %s\n", img_save_path);
		}
		#endif
	}

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

		#if VERBOSITY >= VERBOSITY_INFO
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

		#if VERBOSITY >= VERBOSITY_INFO
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

		#if VERBOSITY >= VERBOSITY_INFO
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

			ret = MPI_Cart_shift(mesh_comm, i, 1, neighbor_ranks + 2 * i, neighbor_ranks + 2 * i + 1);
			assert(ret == MPI_SUCCESS);

			field_lower[i] = neighbor_ranks[2 * i] != MPI_PROC_NULL ? (kernel_sizes[i] / 2) : 0;
			field_sizes[i] = field_lower[i] + block_sizes[i] + (neighbor_ranks[2 * i + 1] != MPI_PROC_NULL ? (kernel_sizes[i] / 2) : 0);

			block_haloed_lower[i] = block_lower[i] + (field_lower[i] ? - (kernel_sizes[i] / 2) : 0);

			field_elems *= field_sizes[i];
			field_dst_elems *= block_sizes[i];
		}

		//haloed buffer (input)
		uint16_t* field = (uint16_t*) malloc(sizeof(uint16_t) * field_elems);
		if (!field) {
			MPI_Abort(mesh_comm, 1);
			exit(1);
		}

		#if VERBOSITY >= VERBOSITY_INFO
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
		printf(": timing_file_read: %lf (bandwidth: %lf GB/s)\n", timing_file_read, (field_elems * pixel_size) / (1000 * 1000 * 1000 * timing_file_read));
		#endif

		ret = MPI_Type_free(&img_view_input);
		assert(ret == MPI_SUCCESS);

		#ifdef TIMING
		double timing_preprocess = - MPI_Wtime();
		#endif

		preprocess_buffer(field, field_elems, pixel_size);

		#ifdef TIMING
		timing_preprocess += MPI_Wtime();
		print_rank_prefix(stdout, rank, block_coords);
		printf(": timing_preprocess: %lf\n", timing_preprocess);
		#endif

		//buffer without halos (output)
		uint16_t* field_dst = (uint16_t*) malloc(sizeof(uint16_t) * field_dst_elems);
		if (!field) {
			MPI_Abort(mesh_comm, 1);
		}

		FLOAT_T* kernel = (FLOAT_T*) malloc(sizeof(FLOAT_T) * kernel_sizes[0] * kernel_sizes[1]);
		if (!kernel) {
			MPI_Abort(mesh_comm, 1);
		}

		#ifdef TIMING
		double timing_kernel_init = - MPI_Wtime();
		#endif

		ret = kernel_init(kernel_type
			, kernel_sizes
			, kernel_params0
			, kernel
			, 1);
		assert(!ret);

		#ifdef TIMING
		timing_kernel_init += MPI_Wtime();
		print_rank_prefix(stdout, rank, block_coords);
		printf(": timing_kernel_init: %lf\n", timing_kernel_init);
		#endif

		#if VERBOSITY >= VERBOSITY_KERNEL
		if (rank == rank_dbg) {
			printf("kernel:\n");
			for (int i = 0; i < kernel_sizes[0]; ++i) {
				printf("[");
				for (int j = 0; j < kernel_sizes[1]; ++j) {
					printf("%f ", kernel[i * kernel_sizes[1] + j]);
				}
				printf("]\n");
			}
		}
		#endif

		//blur
		#ifdef TIMING
		double timing_blur = - MPI_Wtime();
		#endif

		const int field_dst_lower[2] = {0, 0};
		#ifdef BLOCKING_BLUR_ON
		const int blocking[2] = {BLOCKING_BLUR_ROWS, BLOCKING_BLUR_COLUMNS};

		blur_byblocks(kernel, kernel_sizes
			, field, field_sizes, field_lower
			, field_dst, block_sizes, field_dst_lower
			, block_sizes
			, blocking
			, meta.intensity_max);
		#else
		blur(kernel, kernel_sizes
			, field, field_sizes, field_lower
			, field_dst, block_sizes, field_dst_lower
			, block_sizes
			, meta.intensity_max);
		#endif

		#ifdef TIMING
		timing_blur += MPI_Wtime();
		print_rank_prefix(stdout, rank, block_coords);
		printf(": timing_blur: %lf\n", timing_blur);
		#endif

		free(kernel);

		#ifdef TIMING
		double timing_postprocess = - MPI_Wtime();
		#endif
		
		postprocess_buffer(field_dst, field_dst_elems, pixel_size);
		
		#ifdef TIMING
		timing_postprocess += MPI_Wtime();
		print_rank_prefix(stdout, rank, block_coords);
		printf(": timing_postprocess: %lf\n", timing_postprocess);
		#endif

		#if VERBOSITY >= VERBOSITY_INFO
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

		ret = MPI_File_write_all(fout, field_dst, field_dst_elems, pixel_type, &status);
		assert(ret == MPI_SUCCESS);
		ret = MPI_Get_count(&status, pixel_type, &count);
		assert(ret == MPI_SUCCESS && count == field_dst_elems);

		ret = MPI_File_close(&fout);
		assert(ret == MPI_SUCCESS);

		#ifdef TIMING
		timing_file_write += MPI_Wtime();
		print_rank_prefix(stdout, rank, block_coords);
		printf(": timing_file_write: %lf (bandwidth: %lf GB/s)\n", timing_file_write, (field_elems * pixel_size) / (1000 * 1000 * 1000 * timing_file_write));
		#endif

		ret = MPI_Type_free(&img_view_output);
		assert(ret == MPI_SUCCESS);

		if (img_save_path_auto) {
			free(img_save_path);
		}
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
