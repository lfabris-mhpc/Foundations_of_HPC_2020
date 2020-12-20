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

#if ((0x100 & 0xf) == 0x0)
#define I_M_LITTLE_ENDIAN 1
#define swap(mem) (( (mem) & (short int)0xff00) >> 8) +	\
  ( ((mem) & (short int)0x00ff) << 8)
#else
#define I_M_LITTLE_ENDIAN 0
#define swap(mem) (mem)
#endif

#include "utils.h"

//changing the map-by results in different numbers of available openMP threads
// time mpirun --np 4 --report-bindings --map-by hwthread blur_hybrid.x images/eevee.pgm 0 51 < blur_mpi_np0.stdin
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
	unsigned short int intensity_max = 0;

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

	//unused, no need to broadcast as if it were a real parameter
	//MPI_Bcast(&rank_dbg, 1, MPI_INT, 0, MPI_COMM_WORLD);

	int dim_periodic[2] = {0, 0};
	MPI_Comm mesh_comm;
	ret = MPI_Cart_create(MPI_COMM_WORLD, 2, dim_blocks, dim_periodic, 1, &mesh_comm);
	assert(ret == MPI_SUCCESS);

	if (mesh_comm != MPI_COMM_NULL) {
		//get (possibly) updated info
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
			printf(": timing_header_read: %lf\n", timing_header_read);
			#endif
		}
		
		//broadcast header info
		//pack, unpack to save a couple latencies?
		ret = MPI_Bcast(dim_elems, 2, MPI_INT, 0, MPI_COMM_WORLD);
		assert(ret == MPI_SUCCESS);
		ret = MPI_Bcast(&intensity_max, 1, MPI_UNSIGNED_SHORT, 0, MPI_COMM_WORLD);
		assert(ret == MPI_SUCCESS);
		ret = MPI_Bcast(&offset_header_orig, 1, MPI_LONG, 0, MPI_COMM_WORLD);
		assert(ret == MPI_SUCCESS);
		
		int pixel_size = 1 + intensity_max > 255;
		//int pixel_colors = 1;
		
		int block_sizes[2];
		int block_offsets[2];
		int field_sizes[2];
		int neighbor_ranks[4];
		int corner_ranks[4];
		
		int field_lower[2];
		int field_upper[2];
		
		int haloed_offsets[2];
		
		int field_dst_elems = 1;
		int field_elems = 1;
		
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

			ret = MPI_Cart_shift(mesh_comm, i, 1, neighbor_ranks + 2 * i, neighbor_ranks + 2 * i + 1);
			assert(ret == MPI_SUCCESS);
			
			field_lower[i] = neighbor_ranks[2 * i] != MPI_PROC_NULL ? kernel_radiuses[i] : 0;
			field_upper[i] = field_lower[i] + block_sizes[i];
			
			field_sizes[i] = field_upper[i] + (neighbor_ranks[2 * i + 1] != MPI_PROC_NULL ? kernel_radiuses[i] : 0);
			
			haloed_offsets[i] = block_offsets[i] + (field_lower[i] ? - kernel_radiuses[i] : 0);
			
			field_dst_elems *= block_sizes[i];
			field_elems *= field_sizes[i];
		}
		
		//calc corners
		for (int i = 0; i < 2; ++i) {
			//TODO
		}

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

		#ifdef _OPENMP
		if (mpi_thread_provided != MPI_THREAD_MULTIPLE) {
			//no concurrent MPI calls
		}
		#endif
		
		//allocate haloed buffer
		unsigned short int* field = (unsigned short int*) malloc(sizeof(unsigned short int) * field_elems);
		if (!field) {
			MPI_Abort(MPI_COMM_WORLD, 1);
			exit(1);
		}
		
		MPI_Datatype pixel_type = intensity_max > 255 ? MPI_UNSIGNED_SHORT : MPI_BYTE;
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
		#ifndef NDEBUG
		print_rank_prefix(rank, block_coords);
		printf(": subarray defined as img[%d:%d, %d:%d]\n"
			, haloed_offsets[0], haloed_offsets[0] + field_sizes[0]
			, haloed_offsets[1], haloed_offsets[1] + field_sizes[1]);
		#endif
		
		ret = MPI_Type_commit(&subarray);
		assert(!ret);
		
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

			#pragma omp parallel for shared(field, field_reinterpreted)
			for (int i = field_elems - 1; i > 0; --i) {
				field[i] = (unsigned short int) field_reinterpreted[i];
			}
			unsigned short int tmp = (unsigned short int) field_reinterpreted[0];
			field[0] = tmp;
		} else if (pixel_size == 2) {
			//check endianness and flip if needed
			if ((0x100 & 0xf) == 0x0) {
				//cpu is little endian; file will be saved as bigendian?
				#ifndef NDEBUG
				if (rank == rank_dbg) {
					printf("swapping bytes to handle endianness\n");
				}
				#endif
				
				#pragma omp parallel for shared(field)
				for (int i = 0; i < field_elems; ++i) {
					//swap first and secon byte of field's elements
					field[i] = ((field[i] & (short int) 0xff00) >> 8) + ((field[i] & (short int) 0x00ff) << 8);
				}
			}
		} else {
			assert(0);
		}
		#ifdef TIMING
		timing_file_read += MPI_Wtime();
		print_rank_prefix(rank, block_coords);
		printf(": timing_file_read: %lf\n", timing_file_read);
		#endif
		
		//allocate buffer without halos
		unsigned short int* field_dst = (unsigned short int*) malloc(sizeof(unsigned short int) * field_dst_elems);
		if (!field) {
			MPI_Abort(MPI_COMM_WORLD, 1);
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
			MPI_Abort(MPI_COMM_WORLD, 1);
			exit(1);
		}
		
		for (int i = 0; i < kernel_diameters[0]; ++i) {
			for (int j = 0; j < kernel_diameters[1]; ++j) {
				kernel[i * kernel_diameters[1] + j] = 0.0;
			}
		}

		ret = kernel_init(kernel_type
			, kernel_radiuses
			, kernel_params0
			, kernel);
		assert(!ret);
		/*
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
		*/
		
		unsigned short intensity_max_blur = intensity_max;

		#pragma omp parallel for collapse(2) schedule(dynamic) reduction(max: intensity_max_blur) shared(kernel, kernel_radiuses, field, field_sizes, field_lower, field_dst, block_sizes)
		//#pragma omp for collapse(2) schedule(dynamic) reduction(max: intensity_max_blur)
		for (int i = 0; i < block_sizes[0]; ++i) {
			for (int j = 0; j < block_sizes[1]; ++j) {
				const int src_i = field_lower[0] + i;
				const int src_j = field_lower[1] + j;

				int kernel_lower[2];
				int kernel_upper[2];
				kernel_lower[0] = src_i >= kernel_radiuses[0] ? 0 : kernel_radiuses[0] - src_i;
				kernel_upper[0] = (src_i + kernel_radiuses[0]) < field_sizes[0] ? kernel_diameters[0] : (kernel_radiuses[0] + field_sizes[0] - src_i);
				kernel_lower[1] = src_j >= kernel_radiuses[1] ? 0 : kernel_radiuses[1] - src_j;
				kernel_upper[1] = (src_j + kernel_radiuses[1]) < field_sizes[1] ? kernel_diameters[1] : (kernel_radiuses[1] + field_sizes[1] - src_j);

				double intensity_raw;
				kernel_oneshot(kernel
					, kernel_radiuses
					, kernel_lower
					, kernel_upper
					, field
					, field_sizes
					, src_i, src_j
					, &intensity_raw);

				unsigned short int intensity = (unsigned short int) fmax(0, intensity_raw);
				field_dst[i * block_sizes[1] + j] = intensity;

				if (intensity > intensity_max) {
					printf("broken intensity at pos %d, %d; %hu <- %lf\n"
						, i, j
						, intensity, intensity_raw);
				}

				//intensity_max_blur = intensity_max_blur < intensity ? intensity : intensity_max_blur;
				//intensity_max = intensity_max < intensity ? intensity : intensity_max;
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
			#pragma omp parallel for shared(field_dst, field_reinterpreted)
			for (int i = 1; i < field_dst_elems; ++i) {
				field_reinterpreted[i] = (unsigned char) field_dst[i];
			}
		} else if (pixel_size == 2) {
			//check endianness and flip if needed
			if ((0x100 & 0xf) == 0x0) {
				//cpu is little endian; file will be saved as bigendian?
				#ifndef NDEBUG
				if (rank == rank_dbg) {
					printf("swapping bytes to handle endianness\n");
				}
				#endif
				
				#pragma omp parallel for shared(field)
				for (int i = 0; i < field_dst_elems; ++i) {
					//swap first and secon byte of field's elements
					field_dst[i] = ((field_dst[i] & (short int) 0xff00) >> 8) + ((field_dst[i] & (short int) 0x00ff) << 8);
				}
			}
		}
		
		//create output subarray, without halos
		ret = MPI_Type_create_subarray(2
			, dim_elems
			, block_sizes
			, block_offsets
			, MPI_ORDER_C
			, pixel_type
			, &subarray);
		assert(ret == MPI_SUCCESS);
		#ifndef NDEBUG
		print_rank_prefix(rank, block_coords);
		printf(": subarray defined as img[%d:%d, %d:%d]\n"
			, block_offsets[0], block_offsets[0] + block_sizes[0]
			, block_offsets[1], block_offsets[1] + block_sizes[1]);
		#endif
		
		ret = MPI_Type_commit(&subarray);
		assert(!ret);
		
		if (!img_save_path) {
			//create img_save_path
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
			snprintf(header, sizeof(header), "P5\n%d %d\n#created by Lorenzo Fabris\n%hu\n", dim_elems[1], dim_elems[0], intensity_max);
			
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
		
		ret = MPI_File_write(fout, field_dst, field_dst_elems, pixel_type, &status);
		assert(ret == MPI_SUCCESS && status.MPI_ERROR == MPI_SUCCESS);
		
		ret = MPI_File_close(&fout);
		assert(ret == MPI_SUCCESS);
		
		ret = MPI_Type_free(&subarray);
		assert(ret == MPI_SUCCESS);
		#ifdef TIMING
		timing_file_write += MPI_Wtime();
		print_rank_prefix(rank, block_coords);
		printf(": timing_file_write: %lf\n", timing_file_write);
		#endif
		
		/*
		if (img_save_path != argv[argc-1]) {
			free(img_save_path);
		}
		*/
		free(field_dst);
		free(field);
	}

	MPI_Finalize();
}
