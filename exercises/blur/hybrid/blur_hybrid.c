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

//changing the map-by results in different numbers of available openMP threads
// perf stat --detailed mpirun --np 4 --report-bindings --map-by core blur_hybrid.x images/eevee.pgm 0 51 < blur_mpi_np0.stdin
int main (int argc , char *argv[])
{
	int nranks, rank, processor_name_len;
	char processor_name[MPI_MAX_PROCESSOR_NAME];

	int ret;
	#ifdef _OPENMP
	int mpi_thread_provided;
	ret = MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &mpi_thread_provided);
	#else
	ret = MPI_Init(&argc, &argv);
	#endif
	assert(ret == MPI_SUCCESS);

	ret = MPI_Comm_size(MPI_COMM_WORLD, &nranks);
	assert(ret == MPI_SUCCESS);
	ret = MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	assert(ret == MPI_SUCCESS);
	ret = MPI_Get_processor_name(processor_name, &processor_name_len);
	assert(ret == MPI_SUCCESS);

	int dim_blocks[2] = {0, 0};
	int pgm_code;
	int dim_elems[2] = {0, 0};
	int intensity_max = 0;

	#ifndef NDEBUG
	int rank_dbg = 0;
	#endif

	if (rank == 0) {
		ret = params_from_stdin(2, dim_blocks);
		if (ret) {
			MPI_Abort(MPI_COMM_WORLD, 1);
			exit(1);
		}
	}

	char* img_path;
	int kernel_type;
	int kernel_diameters[2];
	int kernel_radiuses[2];
	double kernel_params0;
	char* img_save_path;

	ret = params_from_args(argc, argv
		, 2
		, &img_path
		, &kernel_type
		, kernel_diameters
		, kernel_radiuses
		, &kernel_params0
		, &img_save_path);
	if (rank == 0 && ret) {
		MPI_Abort(MPI_COMM_WORLD, 1);
		exit(1);
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

	ret = MPI_Bcast(dim_blocks, 2, MPI_INT, 0, MPI_COMM_WORLD);
	assert(ret == MPI_SUCCESS);

	//get balanced divisors list
	ret = MPI_Dims_create(nranks, 2, dim_blocks);
	assert(ret == MPI_SUCCESS);
	//handle dimension depth = 1 && dim_blocks > 1 -> move those procs to another dim
	//TODO

	#ifndef NDEBUG
	if (rank == rank_dbg) {
		printf("dim_blocks: (%d, %d)\n", dim_blocks[0], dim_blocks[1]);
	}
	#endif

	int dim_periodic[2] = {0, 0};
	MPI_Comm mesh_comm;
	ret = MPI_Cart_create(MPI_COMM_WORLD, 2, dim_blocks, dim_periodic, 1, &mesh_comm);
	assert(ret == MPI_SUCCESS);

	if (mesh_comm != MPI_COMM_NULL) {
		//get (possibly) updated & reordered info
		ret = MPI_Comm_rank(mesh_comm, &rank);
		assert(ret == MPI_SUCCESS);
		ret = MPI_Comm_size(mesh_comm, &nranks);
		assert(ret == MPI_SUCCESS);

		int block_coords[2];
		ret = MPI_Cart_coords(mesh_comm, rank, 2, block_coords);
		assert(ret == MPI_SUCCESS);

		#ifdef _OPENMP
		#ifndef NDEBUG
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
		print_rank_prefix(rank, block_coords);
		printf(": mpi_thread_provided: %d (%s)\n", mpi_thread_provided, t);
		print_rank_prefix(rank, block_coords);
		printf(": omp_get_max_threads: %d\n", omp_get_max_threads());
		#endif
		#endif

		long int offset_header_orig;
		if (rank == 0) {
			#ifdef TIMING
			double timing_header_read = - MPI_Wtime();
			#endif
			//read header; dim_elems, intensity_max, offset_header
			FILE* fp;
			ret = pgm_open(img_path
				, &pgm_code
				, dim_elems
				, dim_elems + 1
				, &intensity_max
				, &fp);
			if (ret) {
				fprintf(stderr, "could not open file %s\n", img_path);
				MPI_Abort(MPI_COMM_WORLD, 1);
			}

			offset_header_orig = ftell(fp);
			fclose(fp);
			#ifdef TIMING
			timing_header_read += MPI_Wtime();
			print_rank_prefix(rank, block_coords);
			printf(": input image: shape (%d, %d)\n", dim_elems[0], dim_elems[1]);
			print_rank_prefix(rank, block_coords);
			printf(": timing_header_read: %lf\n", timing_header_read);
			#endif
		}

		//broadcast header info
		//pack, unpack to save a couple latencies?
		ret = MPI_Bcast(dim_elems, 2, MPI_INT, 0, MPI_COMM_WORLD);
		assert(ret == MPI_SUCCESS);
		ret = MPI_Bcast(&intensity_max, 1, MPI_INT, 0, MPI_COMM_WORLD);
		assert(ret == MPI_SUCCESS);
		ret = MPI_Bcast(&offset_header_orig, 1, MPI_LONG, 0, MPI_COMM_WORLD);
		assert(ret == MPI_SUCCESS);

		int pixel_size = 1 + (intensity_max > 255);
		#ifndef NDEBUG
		if (rank == rank_dbg) {
			printf("pixel_size: %d (intensity_max: %d)\n", pixel_size, intensity_max);
		}
		#endif
		MPI_Datatype pixel_type = intensity_max > 255 ? MPI_UNSIGNED_SHORT : MPI_BYTE;
		//int pixel_colors = 1;

		//for haloes
		int neighbor_ranks[4];

		//for field_dst
		int block_sizes[2];
		int block_offsets[2];

		//haloed field
		int field_sizes[2];
		int field_lower[2];
		int field_upper[2];
		//relative to original image
		int haloed_offsets[2];

		int field_elems = 1;
		int field_dst_elems = 1;

		for (int i = 0; i < 2; ++i) {
			block_sizes[i] = dim_elems[i] / dim_blocks[i];
			block_offsets[i] = block_sizes[i] * block_coords[i];

			int rem = dim_elems[i] % dim_blocks[i];
			if (block_coords[i] < rem) {
				++block_sizes[i];
			}
			if (rem) {
				block_offsets[i] += block_coords[i];
			}
			
			if (block_sizes[i] < kernel_radiuses[i]) {
				print_rank_prefix(rank, block_coords);
				fprintf(stderr, "domain decomposition failed, block_size[%d] %d < %d kernel_radiuses[%d]\n"
					, i, block_sizes[i], kernel_radiuses[i], i);
				MPI_Abort(mesh_comm, 1);
				exit(1);
			}

			ret = MPI_Cart_shift(mesh_comm, i, 1, neighbor_ranks + 2 * i, neighbor_ranks + 2 * i + 1);
			assert(ret == MPI_SUCCESS);

			//this would be valid under the assumption k << img_size
			field_lower[i] = neighbor_ranks[2 * i] != MPI_PROC_NULL ? kernel_radiuses[i] : 0;
			//field_lower[i] = iclamp((neighbor_ranks[2 * i] != MPI_PROC_NULL) * kernel_radiuses[i] - block_offsets[i], 0, kernel_radiuses[i]);
			field_upper[i] = field_lower[i] + block_sizes[i];

			//this would be valid under the assumption k << img_size
			field_sizes[i] = field_upper[i] + (neighbor_ranks[2 * i + 1] != MPI_PROC_NULL ? kernel_radiuses[i] : 0);
			//field_sizes[i] = field_upper[i] + iclamp((neighbor_ranks[2 * i + 1] != MPI_PROC_NULL) * kernel_radiuses[i], 0, dim_elems[i] - (block_offsets[i] + block_sizes[i]));

			//this would be valid under the assumption k << img_size
			haloed_offsets[i] = block_offsets[i] + (field_lower[i] ? - kernel_radiuses[i] : 0);
			//haloed_offsets[i] = iclamp(block_offsets[i] - (field_lower[i] > 0) * kernel_radiuses[i], 0, block_offsets[i]);

			field_elems *= field_sizes[i];
			field_dst_elems *= block_sizes[i];
		}

		/*
		//calc corners - {ul, ur, bl, br}
		int corner_ranks[4];
		for (int i = 0; i < 2; ++i) {
			for (int j = 2; j < 4; ++j) {
				if (neighbor_ranks[i] != MPI_PROC_NULL
					&& neighbor_ranks[j] != MPI_PROC_NULL) {
					//TODO
				}
			}
		}
		*/

		/*
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
		#endif
		*/

		/*
		#ifdef _OPENMP
		if (mpi_thread_provided != MPI_THREAD_MULTIPLE) {
			//no concurrent MPI calls
		}
		#endif
		*/

		//allocate haloed buffer
		unsigned short int* field = (unsigned short int*) malloc(sizeof(unsigned short int) * field_elems);
		if (!field) {
			MPI_Abort(mesh_comm, 1);
			exit(1);
		}

		#ifndef NDEBUG
		print_rank_prefix(rank, block_coords);
		printf(": input subarray defined as img[%d:%d, %d:%d]\n"
			, haloed_offsets[0], haloed_offsets[0] + field_sizes[0]
			, haloed_offsets[1], haloed_offsets[1] + field_sizes[1]);
		/*
		print_rank_prefix(rank, block_coords);
		printf(": MPI_Type_create_subarray(%d" \
			", %d, %d\n" \
			", %d, %d\n" \
			", %d, %d)\n" \
			, 2
			, dim_elems[0], dim_elems[1]
			, field_sizes[0], field_sizes[1]
			, haloed_offsets[0], haloed_offsets[1]);
		*/
		#endif
		//create input subarray, with halos
		MPI_Datatype subarray;
		ret = MPI_Type_create_subarray(2
			, dim_elems
			, field_sizes
			, haloed_offsets
			, MPI_ORDER_C
			, pixel_type
			, &subarray);
		assert(ret == MPI_SUCCESS);

		ret = MPI_Type_commit(&subarray);
		assert(ret == MPI_SUCCESS);

		#ifdef TIMING
		double timing_file_read = - MPI_Wtime();
		#endif
		MPI_File fin;
		ret = MPI_File_open(mesh_comm, img_path, MPI_MODE_RDONLY, MPI_INFO_NULL, &fin);
		assert(ret == MPI_SUCCESS);

		MPI_Offset offset_header = offset_header_orig;
		ret = MPI_File_set_view(fin, offset_header, pixel_type, subarray, "native", MPI_INFO_NULL);
		assert(ret == MPI_SUCCESS);

		MPI_Status status;
		ret = MPI_File_read(fin, field, field_elems, pixel_type, &status);
		assert(ret == MPI_SUCCESS && status.MPI_ERROR == MPI_SUCCESS);

		ret = MPI_File_close(&fin);
		assert(ret == MPI_SUCCESS);

		ret = MPI_Type_free(&subarray);
		assert(ret == MPI_SUCCESS);

		if (pixel_size == 1) {
			//"widen" the char pixels
			unsigned char* field_reinterpreted = (unsigned char*) field;

			//cannot parallelize through omp due to aliasing
			for (int i = field_elems - 1; i > 0; --i) {
				field[i] = (unsigned short int) field_reinterpreted[i];
			}
			unsigned short int tmp = (unsigned short int) field_reinterpreted[0];
			field[0] = tmp;
		} else if (pixel_size == 2) {
			//check endianness and flip if needed
			//if ((0x100 & 0xf) == 0x0) {
			if (I_M_LITTLE_ENDIAN) {
				//cpu is little endian; file will be saved as big endian?
				#ifndef NDEBUG
				if (rank == rank_dbg) {
					printf("swapping bytes to handle endianness\n");
				}
				#endif

				#pragma omp parallel for shared(field)
				for (int i = 0; i < field_elems; ++i) {
					//swap first and secon byte of feach element
					field[i] = swap(field[i]);
				}
			}
		} else {
			assert(0);
		}
		#ifdef TIMING
		timing_file_read += MPI_Wtime();
		print_rank_prefix(rank, block_coords);
		printf(": timing_file_read: %lf (bandwidth: %lfMB/s)\n", timing_file_read, (field_elems * pixel_size) / (1024 * 1024 * timing_file_read));
		#endif

		//allocate buffer without halos
		unsigned short int* field_dst = (unsigned short int*) malloc(sizeof(unsigned short int) * field_dst_elems);
		if (!field) {
			MPI_Abort(mesh_comm, 1);
			exit(1);
		}

		//blur
		#ifdef TIMING
		double timing_blur = - MPI_Wtime();
		#endif
		//#pragma omp parallel reduction(max: intensity_max)
		//{
		const int kernel_diameters[2] = {2 * kernel_radiuses[0] + 1, 2 * kernel_radiuses[1] + 1};
		double* kernel = (double*) malloc(sizeof(double) * kernel_diameters[0] * kernel_diameters[1]);
		if (!kernel) {
			MPI_Abort(mesh_comm, 1);
			exit(1);
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

		#pragma omp parallel for collapse(2) schedule(dynamic) shared(kernel, kernel_radiuses, field, field_sizes, field_lower, field_dst, block_sizes)
		//#pragma omp for collapse(2) schedule(dynamic)
		for (int i = 0; i < block_sizes[0]; ++i) {
			for (int j = 0; j < block_sizes[1]; ++j) {
				const int field_i = field_lower[0] + i;
				const int field_j = field_lower[1] + j;

				int kernel_lower[2];
				int kernel_upper[2];
				//more elegant, less performant
				kernel_lower[0] = iclamp(kernel_radiuses[0] - field_i, 0, kernel_radiuses[0]);
				kernel_upper[0] = kernel_radiuses[0] + iclamp(field_sizes[0] - field_i, 1, kernel_radiuses[0] + 1);
				kernel_lower[1] = iclamp(kernel_radiuses[1] - field_j, 0, kernel_radiuses[1]);
				kernel_upper[1] = kernel_radiuses[1] + iclamp(field_sizes[1] - field_j, 1, kernel_radiuses[1] + 1);
				/*
				kernel_lower[0] = field_i >= kernel_radiuses[0] ? 0 : kernel_radiuses[0] - field_i;
				kernel_upper[0] = (field_i + kernel_radiuses[0]) < field_sizes[0] ? kernel_diameters[0] : (field_sizes[0] + kernel_radiuses[0] - field_i);
				kernel_lower[1] = field_j >= kernel_radiuses[1] ? 0 : kernel_radiuses[1] - field_j;
				kernel_upper[1] = (field_j + kernel_radiuses[1]) < field_sizes[1] ? kernel_diameters[1] : (field_sizes[1] + kernel_radiuses[1] - field_j);
				*/
				#ifndef NDEBUG
				printf("apply kernel[%d:%d, %d:%d] for field[%d, %d]\n", kernel_lower[0], kernel_upper[0], kernel_lower[1], kernel_upper[1], field_i, field_j);
				#endif

				double intensity_raw;
				#ifdef SIMD_ON
				kernel_oneshot_simd(kernel
					, kernel_radiuses
					, kernel_lower
					, kernel_upper
					, field
					, field_sizes
					, field_i, field_j
					, &intensity_raw);
				#else
				kernel_oneshot(kernel
					, kernel_radiuses
					, kernel_lower
					, kernel_upper
					, field
					, field_sizes
					, field_i, field_j
					, &intensity_raw);
				#endif
				#ifndef NDEBUG
				//comparison with basic algo
				double intensity_raw2;
				kernel_oneshot_basic(kernel
					, kernel_radiuses
					, kernel_lower
					, kernel_upper
					, field
					, field_sizes
					, field_i, field_j
					, &intensity_raw2);
				assert(fabs(intensity_raw - intensity_raw2) < 0.00001);
				#endif

				unsigned short int intensity = (unsigned short int) round(fmax(0.0, intensity_raw));
				field_dst[i * block_sizes[1] + j] = intensity;

				#ifndef NDEBUG
				printf("intensity at pos %d, %d; (%hu) %hu <- %lf\n", field_i, field_j, field[field_i * field_sizes[1] + field_j], intensity, intensity_raw);
				if (intensity > intensity_max) {
					printf("broken intensity at pos %d, %d; %hu <- %lf\n"
						, i, j
						, intensity, intensity_raw);
				}
				#endif
			}
		}

		free(kernel);
		//}
		#ifdef TIMING
		timing_blur += MPI_Wtime();
		print_rank_prefix(rank, block_coords);
		printf(": timing_blur: %lf\n", timing_blur);
		#endif

		#ifdef TIMING
		double timing_file_write = - MPI_Wtime();
		#endif

		if (pixel_size == 1) {
			//"shrink" the shorts to chars
			unsigned char* field_reinterpreted = (unsigned char*) field_dst;

			unsigned char tmp = (unsigned char) field_dst[0];
			field_reinterpreted[0] = tmp;
			//cannot parallelize through omp due to aliasing
			for (int i = 1; i < field_dst_elems; ++i) {
				field_reinterpreted[i] = (unsigned char) field_dst[i];
			}
		} else if (pixel_size == 2) {
			//check endianness and flip if needed
			//if ((0x100 & 0xf) == 0x0) {
			if (I_M_LITTLE_ENDIAN) {
				//cpu is little endian; file will be saved as big endian?
				#ifndef NDEBUG
				if (rank == rank_dbg) {
					printf("swapping bytes to handle endianness\n");
				}
				#endif

				#pragma omp parallel for shared(field)
				for (int i = 0; i < field_dst_elems; ++i) {
					//swap first and secon byte of each pixel
					//field_dst[i] = ((field_dst[i] & (short int) 0xff00) >> 8) + ((field_dst[i+1] & (short int) 0x00ff) << 8);
					field_dst[i] = swap(field_dst[i]);
				}
			}
		}

		#ifndef NDEBUG
		print_rank_prefix(rank, block_coords);
		printf(": output subarray defined as img[%d:%d, %d:%d]\n"
			, block_offsets[0], block_offsets[0] + block_sizes[0]
			, block_offsets[1], block_offsets[1] + block_sizes[1]);
		#endif
		//create output subarray, without halos
		ret = MPI_Type_create_subarray(2
			, dim_elems
			, block_sizes
			, block_offsets
			, MPI_ORDER_C
			, pixel_type
			, &subarray);
		assert(ret == MPI_SUCCESS);

		ret = MPI_Type_commit(&subarray);
		assert(ret == MPI_SUCCESS);

		if (!img_save_path) {
			//create img_save_path
			//TODO
			img_save_path = "output.pgm";
		}
		#ifndef NDEBUG
		if (rank == rank_dbg) {
			printf("output image: %s\n", img_save_path);
		}
		#endif

		MPI_File fout;
		ret = MPI_File_open(mesh_comm, img_save_path, MPI_MODE_WRONLY | MPI_MODE_CREATE, MPI_INFO_NULL, &fout);
		assert(ret == MPI_SUCCESS);

		if (rank == 0) {
			char header[200];
			ret = snprintf(header, sizeof(header), "P5\n# generated by\n# Lorenzo Fabris\n%d %d\n%d\n", dim_elems[1], dim_elems[0], intensity_max);
			assert(ret >= 0);

			//write header, get offset_header
			ret = MPI_File_write(fout, header, strlen(header), MPI_BYTE, &status);
			assert(ret == MPI_SUCCESS);

			ret = MPI_File_get_position(fout, &offset_header);
			assert(ret == MPI_SUCCESS);
		}
		//broadcast offset_header
		ret = MPI_Bcast(&offset_header, 1, MPI_OFFSET, 0, mesh_comm);
		assert(ret == MPI_SUCCESS);

		ret = MPI_File_set_view(fout, offset_header, pixel_type, subarray, "native", MPI_INFO_NULL);
		assert(ret == MPI_SUCCESS);

		#ifndef NDEBUG
		print_rank_prefix(rank, block_coords);
		printf(": writing %d elements through subarray\n"
			, field_dst_elems);
		#endif
		ret = MPI_File_write(fout, field_dst, field_dst_elems, pixel_type, &status);
		assert(ret == MPI_SUCCESS && status.MPI_ERROR == MPI_SUCCESS);

		ret = MPI_File_close(&fout);
		assert(ret == MPI_SUCCESS);

		ret = MPI_Type_free(&subarray);
		assert(ret == MPI_SUCCESS);

		/*
		//print last newline?
		ret = MPI_Barrier(mesh_comm);
		assert(ret == MPI_SUCCESS);

		if (rank == 0) {
			FILE* fp = fopen(img_save_path, "a+");
			assert(fp);
			fseek(fp, 0, SEEK_END);
			fprintf(fp, "\n");
			fclose(fp);
		}
		*/
		#ifdef TIMING
		timing_file_write += MPI_Wtime();
		print_rank_prefix(rank, block_coords);
		printf(": timing_file_write: %lf (bandwidth: %lfMB/s)\n", timing_file_write, (field_elems * pixel_size) / (1024 * 1024 * timing_file_write));
		#endif

		free(field_dst);
		free(field);
	}

	MPI_Finalize();
}