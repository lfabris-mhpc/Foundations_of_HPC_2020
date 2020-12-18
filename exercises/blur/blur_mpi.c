#include <stdlib.h>

#include <string.h>

#include <assert.h>

#include <mpi.h>

#ifdef _OPENMP
#include <omp.h>
#endif

#include <blur_utils.h>
#include <pgm_utils.h>

void print_rank_prefix(const int rank, const int* block_coords) {
	printf("rank %d(%d, %d)", rank, block_coords[0], block_coords[1]);
}

int main (int argc , char *argv[])
{
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
		printf("mpi_thread_provided: %d (%s)\n", mpi_thread_provided, t);
		printf("omp_get_max_threads: %d\n", omp_get_max_threads());
	}
	#endif

	int dim_elems[2] = {1, 1};
	unsigned short int intensity_max;
	int dim_blocks[2] = {0, 0};

	int kernel_type;
	int kernel_radius;
	double kernel_params0;

	#ifndef NDEBUG
	int rank_dbg = 0;
	#endif

	//work out cmd params
	if (argc < 4) {
		MPI_Abort(MPI_COMM_WORLD, 1);
		exit(1);
	}
	
	int param_idx = 1;

	//image path
	const char* img_path = argv[param_idx++];

	//kernel type
	int res = sscanf(argv[param_idx++], "%d", &kernel_type);
	if (res != 1 || kernel_type < 0 || kernel_type >= KERNEL_TYPE_UNRECOGNIZED) {
		if (rank == 0) {
			fprintf(stderr, "unusable kernel_type\n");
		}
		MPI_Abort(MPI_COMM_WORLD, 1);
		exit(1);
	}

	//kernel radius
	res = sscanf(argv[param_idx++], "%d", &kernel_radius);
	if (res != 1) {
		if (rank == 0) {
			fprintf(stderr, "unusable kernel_radius\n");
		}
		MPI_Abort(MPI_COMM_WORLD, 1);
		exit(1);
	}
	//for now, keep kernel squares
	//kernel_radius1 = kernel_radius0;

	if (param_idx < argc) {
		//kernel param(s)
		res = sscanf(argv[param_idx++], "%lf", &kernel_params0);
		if (res != 1) {
			if (rank == 0) {
				fprintf(stderr, "unusable kernel_params0\n");
			}
			MPI_Abort(MPI_COMM_WORLD, 1);
			exit(1);
		}
	}

	int ret = kernel_validate_parameters(kernel_type
		, kernel_radius
		, kernel_radius
		, kernel_params0);
	if (ret) {
		MPI_Abort(MPI_COMM_WORLD, 1);
		exit(1);
	}

	if (rank == 0) {
		//stdin params
		for (int i = 0; i < 2; ++i) {
			printf("dimension %i: how many subdivisions? (0 to let MPI choose)\n", i + 1);
			int check = scanf("%d", dim_blocks + i);
			if (check != 1 || dim_blocks[i] < 0) {
				fprintf(stderr, "subdivisions must be and integer >= 0\n");
				MPI_Abort(MPI_COMM_WORLD, 1);
				exit(1);
			}
		}
	}

	MPI_Bcast(dim_blocks, 2, MPI_INT, 0, MPI_COMM_WORLD);
	#ifndef NDEBUG
	if (rank == rank_dbg) {
		printf("dim_blocks: (%d, %d)\n", dim_blocks[0], dim_blocks[1]);
	}
	#endif

	//get balanced divisors list
	MPI_Dims_create(nranks, 2, dim_blocks);
	//handle dimension depth = 1 && dim_blocks > 1 -> move those procs to another dim
	//TODO

	//unused, no need to broadcast as if it were a real parameter
	//MPI_Bcast(&rank_dbg, 1, MPI_INT, 0, MPI_COMM_WORLD);

	int dim_periodic[2] = {0, 0};
	MPI_Comm mesh_comm;
	MPI_Cart_create(MPI_COMM_WORLD, 2, dim_blocks, dim_periodic, 1, &mesh_comm);

	if (mesh_comm != MPI_COMM_NULL) {
		MPI_Comm_rank(mesh_comm, &rank);
		MPI_Comm_size(mesh_comm, &nranks);

		int block_coords[2];
		MPI_Cart_coords(mesh_comm, rank, 2, block_coords);
		//print_rank_prefix(rank, block_coords);

		FILE* fp;
		ret = pgm_open(img_path, dim_elems, dim_elems + 1, &intensity_max, &fp);
		assert(!ret);

		fclose(fp);

		#ifndef NDEBUG
		if (rank == rank_dbg) {
			print_rank_prefix(rank, block_coords);
			printf(": image %s has %d rows, %d columns, %hu max intensity\n", img_path, dim_elems[0], dim_elems[1], intensity_max);
		}
		#endif

		int block_sizes[2];
		int block_offsets[2];
		int field_sizes[2];
		int neighbor_ranks[4];
		int field_lower[2];
		int field_upper[2];
		for (int i = 0; i < 2; ++i) {
			block_sizes[i] = dim_elems[i] / dim_blocks[i];
			block_offsets[i] = block_sizes[i] * block_coords[i];
			if (block_coords[i] < dim_elems[i] % dim_blocks[i]) {
				++block_sizes[i];
				block_offsets[i] += i % dim_blocks[i];
			}

			MPI_Cart_shift(mesh_comm, i, 1, neighbor_ranks + 2 * i, neighbor_ranks + 2 * i + 1);
			
			//note that kernel_radius *could* be different for each dim, here it is assumed square
			field_lower[i] = neighbor_ranks[2 * i] != MPI_PROC_NULL ? kernel_radius : 0;
			field_upper[i] = field_lower[i] + block_sizes[i];
			
			//if kernel isn't square, handle this too
			field_sizes[i] = field_upper[i] + (neighbor_ranks[2 * i + 1] != MPI_PROC_NULL ? kernel_radius : 0);
		}
		int corner_idx = 0;
		int corner_ranks[4];
		//top left
		if (neighbor_ranks[0] != MPI_PROC_NULL
			&& neighbor_ranks[2] != MPI_PROC_NULL) {
			int corner_coords[2] = {block_coords[0] - 1, block_coords[1] - 1};
			MPI_Cart_rank(mesh_comm, corner_coords, corner_ranks + corner_idx++);
		} else {
			corner_ranks[corner_idx++] = MPI_PROC_NULL;
		}
		//top right
		if (neighbor_ranks[0] != MPI_PROC_NULL
			&& neighbor_ranks[3] != MPI_PROC_NULL) {
			int corner_coords[2] = {block_coords[0] - 1, block_coords[1] + 1};
			MPI_Cart_rank(mesh_comm, corner_coords, corner_ranks + corner_idx++);
		} else {
			corner_ranks[corner_idx++] = MPI_PROC_NULL;
		}
		//bottom left
		if (neighbor_ranks[1] != MPI_PROC_NULL
			&& neighbor_ranks[2] != MPI_PROC_NULL) {
			int corner_coords[2] = {block_coords[0] + 1, block_coords[1] - 1};
			MPI_Cart_rank(mesh_comm, corner_coords, corner_ranks + corner_idx++);
		} else {
			corner_ranks[corner_idx++] = MPI_PROC_NULL;
		}
		//bottom right
		if (neighbor_ranks[1] != MPI_PROC_NULL
			&& neighbor_ranks[3] != MPI_PROC_NULL) {
			int corner_coords[2] = {block_coords[0] + 1, block_coords[1] + 1};
			MPI_Cart_rank(mesh_comm, corner_coords, corner_ranks + corner_idx++);
		} else {
			corner_ranks[corner_idx++] = MPI_PROC_NULL;
		}

		#ifndef NDEBUG
		print_rank_prefix(rank, block_coords);
		printf(": block_offsets: %d, %d\n", block_offsets[0], block_offsets[1]);
		print_rank_prefix(rank, block_coords);
		printf(": block_sizes: (%d, %d)\n", block_sizes[0], block_sizes[1]);
		print_rank_prefix(rank, block_coords);
		printf(": field_sizes: (%d, %d)\n", field_sizes[0], field_sizes[1]);
		print_rank_prefix(rank, block_coords);
		printf(": working field slice: (%d:%d, %d:%d)\n", field_lower[0], field_upper[0], field_lower[1], field_upper[1]);
		print_rank_prefix(rank, block_coords);
		printf(": neighbor_ranks: (%d, %d, %d, %d)\n", neighbor_ranks[0], neighbor_ranks[1], neighbor_ranks[2], neighbor_ranks[3]);
		print_rank_prefix(rank, block_coords);
		printf(": corner_ranks: (%d, %d, %d, %d)\n", corner_ranks[0], corner_ranks[1], corner_ranks[2], corner_ranks[3]);
		#endif

		const int field_elems = block_sizes[0] * block_sizes[1];
		const int field_extended_elems = field_sizes[0] * field_sizes[1];

		unsigned short int* field = (unsigned short int*) malloc(sizeof(unsigned short int) * field_extended_elems);
		if (!field) {
			MPI_Abort(MPI_COMM_WORLD, 1);
			exit(1);
		}

		#ifdef TIMING
		double timing_img_read = - MPI_Wtime();
		#endif
		ret = pgm_load_image_slice_into_field_slice(img_path
			, block_offsets[0], block_offsets[0] + block_sizes[0]
			, block_offsets[1], block_offsets[1] + block_sizes[1]
			, field
			, field_sizes[1]
			, field_lower[0], field_upper[0]
			, field_lower[1], field_upper[1]);
		assert(!ret);
		#ifdef TIMING
		timing_img_read += MPI_Wtime();
		print_rank_prefix(rank, block_coords);
		printf(": timing_img_read: %lf\n", timing_img_read);
		#endif
		
		#ifndef NDEBUG
		//save original unblurred image slice
		char* tmp_path = (char*) malloc(strlen(img_path) + 4 + KERNEL_SUFFIX_MAX_LEN);
		if (!tmp_path) {
			MPI_Abort(MPI_COMM_WORLD, 1);
			exit(1);
		}
		//strip ".pgm"
		memcpy(tmp_path, img_path, strlen(img_path));
		tmp_path[strlen(img_path) - 4] = 0;
		
		strcat(tmp_path, "_dbg.pgm");
		
		char* img_save_dbg_path = (char*) malloc(strlen(tmp_path) + KERNEL_SUFFIX_MAX_LEN);
		if (!img_save_dbg_path) {
			MPI_Abort(MPI_COMM_WORLD, 1);
			exit(1);
		}
		
		ret = img_save_path_init(tmp_path
			, kernel_type
			, block_coords
			, img_save_dbg_path);
		assert(!ret);
		
		ret = pgm_save_field_slice(img_save_dbg_path
			, field
			, intensity_max
			, field_sizes[1]
			, field_lower[0], field_upper[0]
			, field_lower[1], field_upper[1]);
		assert(!ret);
		
		free(img_save_dbg_path);
		free(tmp_path);
		#endif

		//init boundaries to 0 -> will give vignette effect where there's no neighbor
		//TODO: change to buffer exchanges when implementing proper decomposition
		//up
		if (neighbor_ranks[0] != MPI_PROC_NULL) {
			ret = field_slice_init(0, field
				, field_sizes[1]
				, 0, field_lower[0]
				, 0, field_sizes[1]);//both corners
				//, field_lower[1], field_upper[1]);//no corners
			assert(!ret);
		}
		//left
		if (neighbor_ranks[2] != MPI_PROC_NULL) {
			ret = field_slice_init(0, field
				, field_sizes[1]
				, field_lower[0], field_upper[0]//no corners
				, 0, field_lower[1]);
			assert(!ret);
		}
		//right
		if (neighbor_ranks[3] != MPI_PROC_NULL) {
			ret = field_slice_init(0, field
				, field_sizes[1]
				, field_lower[0], field_upper[0]//no corners
				, field_upper[1], field_sizes[1]);
			assert(!ret);
		}
		//bottom
		if (neighbor_ranks[1] != MPI_PROC_NULL) {
			ret = field_slice_init(0, field
				, field_sizes[1]
				, field_upper[0], field_sizes[0]
				, 0, field_sizes[1]);//both corners
				//, field_lower[1], field_upper[1]);//no corners
			assert(!ret);
		}

		//if kernel isn't square, adapt the malloc size
		const int k = 2 * kernel_radius + 1;
		double* kernel = (double*) malloc(sizeof(double) * k * k);
		if (!kernel) {
			MPI_Abort(MPI_COMM_WORLD, 1);
			exit(1);
		}

		ret = kernel_init(kernel_type
			, kernel_radius
			, kernel_radius
			, kernel_params0
			, kernel);
		assert(!ret);

		#ifndef NDEBUG
		if (rank == rank_dbg) {
			print_rank_prefix(rank, block_coords);
			printf(": kernel:\n");
			for (int i = 0; i < k; ++i) {
				printf("[");
				for (int j = 0; j < k; ++j) {
					printf("%lf ", kernel[i * k + j]);
				}
				printf("]\n");
			}
		}
		#endif

		unsigned short int* field_dst = (unsigned short int*) malloc(sizeof(unsigned short int) * field_elems);
		if (!field_dst) {
			MPI_Abort(MPI_COMM_WORLD, 1);
			exit(1);
		}

		#ifdef TIMING
		double timing_blur = - MPI_Wtime();
		#endif
		intensity_max = 0;
		ret = kernel_apply_to_slice(kernel
			, kernel_radius
			, kernel_radius
			, field
			, field_sizes[0], field_sizes[1]
			, field_lower[0], field_upper[0]
			, field_lower[1], field_upper[1]
			, field_dst
			, block_sizes[1]
			, 0, block_sizes[0]
			, 0, block_sizes[1]
			, &intensity_max);
		assert(!ret);
		#ifdef TIMING
		timing_blur += MPI_Wtime();
		print_rank_prefix(rank, block_coords);
		printf(": timing_blur: %lf\n", timing_blur);
		#endif
		
		char* img_save_path = (char*) malloc(strlen(img_path) + KERNEL_SUFFIX_MAX_LEN);
		if (!img_save_path) {
			MPI_Abort(MPI_COMM_WORLD, 1);
			exit(1);
		}

		ret = img_save_path_init(img_path
			, kernel_type
			, block_coords
			, img_save_path);
		assert(!ret);

		#ifndef NDEBUG
		if (rank == rank_dbg) {
			print_rank_prefix(rank, block_coords);
			printf(": output img: %s\n", img_save_path);
		}
		#endif

		#ifdef TIMING
		double timing_img_save = - MPI_Wtime();
		#endif
		//blurred image slice
		ret = pgm_save_field_slice(img_save_path
			, field_dst
			, intensity_max
			, block_sizes[1]
			, 0, block_sizes[0]
			, 0, block_sizes[1]);
		assert(!ret);
		#ifdef TIMING
		timing_img_save += MPI_Wtime();
		print_rank_prefix(rank, block_coords);
		printf(": timing_img_save: %lf\n", timing_img_save);
		#endif
		
		free(img_save_path);
		free(field_dst);
		free(kernel);
		free(field);
	}

	MPI_Finalize();
}
