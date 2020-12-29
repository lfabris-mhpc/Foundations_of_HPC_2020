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

	int nranks, rank;
	ret = MPI_Comm_size(MPI_COMM_WORLD, &nranks);
	assert(ret == MPI_SUCCESS);
	ret = MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	assert(ret == MPI_SUCCESS);

	metadata meta;
	//meta.header_length_input = 0;
	//meta.header_length_output = 0;
	//meta.mesh_sizes = {0, 0};
	//meta.pgm_code = 0;
	//meta.img_sizes = {0, 0};
	//meta.intensity_max = 0;

	#ifndef NDEBUG
	int rank_dbg = 0;
	#endif

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

	if (rank == 0) {
		#ifdef TIMING
		double timing_header_read = - MPI_Wtime();
		#endif
		
		FILE* fp;
		ret = pgm_open(img_path
			, &meta.pgm_code
			, meta.img_sizes
			, meta.img_sizes + 1
			, &meta.intensity_max
			, &fp);
		if (ret) {
			fprintf(stderr, "could not open file %s\n", img_path);
			MPI_Abort(MPI_COMM_WORLD, 1);
		}

		meta.header_length_input = ftell(fp);
		fclose(fp);
		
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
		
		if (reorg) {
			if ((meta.mesh_sizes[0] > meta.mesh_sizes[1]) != (meta.img_sizes[0] > meta.img_sizes[1])) {
				int tmp = meta.mesh_sizes[1];
				meta.mesh_sizes[1] = meta.mesh_sizes[0];
				meta.mesh_sizes[0] = tmp;
			}
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

		//haloed block_extents
		int field_sizes[2];
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
				block_lower[i] += block_coords[i];
			}
			
			if (block_sizes[i] < kernel_radiuses[i]) {
				print_rank_prefix(stderr, rank, block_coords);
				fprintf(stderr, "domain decomposition failed, block_size[%d] %d < %d kernel_radiuses[%d]\n"
					, i, block_sizes[i], kernel_radiuses[i], i);
				MPI_Abort(mesh_comm, 1);
				exit(1);
			}

			ret = MPI_Cart_shift(mesh_comm, i, 1, neighbor_ranks + 2 * i, neighbor_ranks + 2 * i + 1);
			assert(ret == MPI_SUCCESS);

			//the following are valid under the assumption k << img_size
			field_lower[i] = neighbor_ranks[2 * i] != MPI_PROC_NULL ? kernel_radiuses[i] : 0;
			field_upper[i] = field_lower[i] + block_sizes[i];

			field_sizes[i] = field_upper[i] + (neighbor_ranks[2 * i + 1] != MPI_PROC_NULL ? kernel_radiuses[i] : 0);

			field_elems *= field_sizes[i];
			field_dst_elems *= block_sizes[i];
		}

		/*
		#ifdef _OPENMP
		if (mpi_thread_provided != MPI_THREAD_MULTIPLE) {
			//no concurrent MPI calls
		}
		#endif
		*/

		//haloed buffer (input)
		unsigned short int* field = (unsigned short int*) malloc(sizeof(unsigned short int) * field_elems);
		if (!field) {
			MPI_Abort(mesh_comm, 1);
			exit(1);
		}

		#ifndef NDEBUG
		print_rank_prefix(stdout, rank, block_coords);
		printf(": img_view_input defined as img[%d:%d, %d:%d] (img shape: (%d, %d))\n"
			, block_lower[0], block_lower[0] + block_sizes[0]
			, block_lower[1], block_lower[1] + block_sizes[1]
			, meta.img_sizes[0], meta.img_sizes[1]);
		#endif

		//read view: block, without halos
		MPI_Datatype img_view_input;
		ret = MPI_Type_create_subarray(2
			, meta.img_sizes
			, block_sizes
			, block_lower
			, MPI_ORDER_C
			, pixel_type
			, &img_view_input);
		assert(ret == MPI_SUCCESS);

		ret = MPI_Type_commit(&img_view_input);
		assert(ret == MPI_SUCCESS);
		
		#ifndef NDEBUG
		print_rank_prefix(stdout, rank, block_coords);
		printf(": img_view_input defined as field[%d:%d, %d:%d] (field shape: (%d, %d))\n"
			, field_lower[0], field_lower[0] + block_sizes[0]
			, field_lower[1], field_lower[1] + block_sizes[1]
			, field_sizes[0], field_sizes[1]);
		#endif
		//buffer slice without halos
		MPI_Datatype buffer_slice_view;
		ret = MPI_Type_create_subarray(2
			, field_sizes
			, block_sizes
			, field_lower
			, MPI_ORDER_C
			, pixel_type
			, &buffer_slice_view);
		assert(ret == MPI_SUCCESS);

		ret = MPI_Type_commit(&buffer_slice_view);
		assert(ret == MPI_SUCCESS);
		#ifndef NDEBUG
		print_rank_prefix(stdout, rank, block_coords);
		printf(": input field_subarray committed\n");
		#endif

		#ifdef TIMING
		double timing_file_read = - MPI_Wtime();
		#endif
		MPI_File fin;
		ret = MPI_File_open(mesh_comm, img_path, MPI_MODE_RDONLY, MPI_INFO_NULL, &fin);
		assert(ret == MPI_SUCCESS);

		//MPI_Offset disp = meta.header_length_input;
		ret = MPI_File_set_view(fin, meta.header_length_input, pixel_type, img_view_input, "native", MPI_INFO_NULL);
		assert(ret == MPI_SUCCESS);

		MPI_Status status;
		int count;
		ret = MPI_File_read(fin, field, 1, buffer_slice_view, &status);
		assert(ret == MPI_SUCCESS);
		ret = MPI_Get_count(&status, pixel_type, &count);
		assert(ret == MPI_SUCCESS && count == field_dst_elems);

		ret = MPI_File_close(&fin);
		assert(ret == MPI_SUCCESS);

		ret = MPI_Type_free(&img_view_input);
		assert(ret == MPI_SUCCESS);
		
		ret = MPI_Type_free(&buffer_slice_view);
		assert(ret == MPI_SUCCESS);
		
		#ifdef TIMING
		timing_file_read += MPI_Wtime();
		print_rank_prefix(stdout, rank, block_coords);
		printf(": timing_file_read: %lf (bandwidth: %lfMB/s)\n", timing_file_read, (field_elems * pixel_size) / (1024 * 1024 * timing_file_read));
		#endif
		
		#ifdef TIMING
		double timing_halo_exchange = - MPI_Wtime();
		#endif
		
		//exchange halos
		//basic: first exchange "normal" halos along a dimension...
		//custom subarrays for halos with horizontal neighbors
		if (kernel_radiuses[1]) {
			//int halo_elems = block_sizes[0] * kernel_radiuses[1];
			int halo_sizes[2] = {block_sizes[0], kernel_radiuses[1]};
			int halo_lower[2];
			
			MPI_Datatype send_left_halo;
			MPI_Datatype recv_left_halo;
			if (neighbor_ranks[2] != MPI_PROC_NULL) {
				//left halo send: field[field_lower[0]:field_lower[0] + block_sizes[0], field_lower[1]:field_lower[1] + kernel_radiuses[1]]
				halo_lower[0] = field_lower[0];
				halo_lower[1] = field_lower[1];
				
				ret = MPI_Type_create_subarray(2, field_sizes
					, halo_sizes, halo_lower
					, MPI_ORDER_C, pixel_type
					, &send_left_halo);
				assert(ret == MPI_SUCCESS);

				ret = MPI_Type_commit(&send_left_halo);
				assert(ret == MPI_SUCCESS);

				//left halo receive: field[field_lower[0]:field_lower[0] + block_sizes[0], 0:kernel_radiuses[1]]
				halo_lower[0] = field_lower[0];
				halo_lower[1] = 0;
				
				ret = MPI_Type_create_subarray(2, field_sizes
					, halo_sizes, halo_lower
					, MPI_ORDER_C, pixel_type
					, &recv_left_halo);
				assert(ret == MPI_SUCCESS);

				ret = MPI_Type_commit(&recv_left_halo);
				assert(ret == MPI_SUCCESS);
			}

			MPI_Datatype send_right_halo;
			MPI_Datatype recv_right_halo;
			if (neighbor_ranks[3] != MPI_PROC_NULL) {
				//right halo send: field[field_lower[0]:field_lower[0] + block_sizes[0], field_upper[1] - kernel_radiuses[1]:field_upper[1]]
				halo_lower[0] = field_lower[0];
				halo_lower[1] = field_upper[1] - kernel_radiuses[1];
				
				ret = MPI_Type_create_subarray(2, field_sizes
					, halo_sizes, halo_lower
					, MPI_ORDER_C, pixel_type
					, &send_right_halo);
				assert(ret == MPI_SUCCESS);

				ret = MPI_Type_commit(&send_right_halo);
				assert(ret == MPI_SUCCESS);
				
				//right halo receive: field[field_lower[0]:field_lower[0] + block_sizes[0], field_upper[1]:field_sizes[1]]
				halo_lower[0] = field_lower[0];
				halo_lower[1] = field_upper[1];
				
				ret = MPI_Type_create_subarray(2, field_sizes
					, halo_sizes, halo_lower
					, MPI_ORDER_C, pixel_type
					, &recv_right_halo);
				assert(ret == MPI_SUCCESS);

				ret = MPI_Type_commit(&recv_right_halo);
				assert(ret == MPI_SUCCESS);
			}

			int nreqs = 0;
			MPI_Request requests[4];
			MPI_Status statuses[4];

			//nonblocking exchange with left neighbor
			int other = neighbor_ranks[2];
			if (other != MPI_PROC_NULL) {
				ret = MPI_Isend(field, 1, send_left_halo, other, 1, mesh_comm, requests + nreqs);
				assert(ret == MPI_SUCCESS);
				++nreqs;
				ret = MPI_Irecv(field, 1, recv_left_halo, other, 1, mesh_comm, requests + nreqs);
				assert(ret == MPI_SUCCESS);
				++nreqs;
			}
			
			//nonblocking exchange with right neighbor
			other = neighbor_ranks[3];
			if (other != MPI_PROC_NULL) {
				ret = MPI_Isend(field, 1, send_right_halo, other, 1, mesh_comm, requests + nreqs);
				assert(ret == MPI_SUCCESS);
				++nreqs;
				ret = MPI_Irecv(field, 1, recv_right_halo, other, 1, mesh_comm, requests + nreqs);
				assert(ret == MPI_SUCCESS);
				++nreqs;
			}
			
			ret = MPI_Waitall(nreqs, requests, statuses);
			assert(ret == MPI_SUCCESS);
			for (int i = 0; i < nreqs; ++i) {
				assert(statuses[i].MPI_ERROR == MPI_SUCCESS);
			}
			
			if (neighbor_ranks[2] != MPI_PROC_NULL) {
				ret = MPI_Type_free(&send_left_halo);
				assert(ret == MPI_SUCCESS);
				
				ret = MPI_Type_free(&recv_left_halo);
				assert(ret == MPI_SUCCESS);
			}
			
			if (neighbor_ranks[3] != MPI_PROC_NULL) {
				ret = MPI_Type_free(&send_right_halo);
				assert(ret == MPI_SUCCESS);
				
				ret = MPI_Type_free(&recv_right_halo);
				assert(ret == MPI_SUCCESS);
			}
		}
		
		//then, exchange halos for the other including the receiver's corners, gotten from the previous dimension
		//contiguous buffers, no need for subarrays
		if (kernel_radiuses[0]) {
			//not using datatypes requires handling type casting for buffer pointers
			unsigned char* field_reinterpreted = (unsigned char*) field;
			
			int halo_elems = kernel_radiuses[0] * field_sizes[1];

			int nreqs = 0;
			MPI_Request requests[4];
			MPI_Status statuses[4];
			
			//nonblocking exchange with top neighbor
			int other = neighbor_ranks[0];
			if (other != MPI_PROC_NULL) {
				//top halo (+ corners) send: field[field_lower[0]:field_lower[0] + kernel_radiuses[0], :]
				int offset = field_lower[0] * field_sizes[1];
				ret = MPI_Isend(field_reinterpreted + offset * pixel_size
					, halo_elems, pixel_type, other, 2, mesh_comm, requests + nreqs);
				assert(ret == MPI_SUCCESS);
				++nreqs;
				
				//top halo (+ corners) recv: field[:kernel_radiuses[0], :]
				offset = 0;
				ret = MPI_Irecv(field_reinterpreted + offset * pixel_size
					, halo_elems, pixel_type, other, 2, mesh_comm, requests + nreqs);
				assert(ret == MPI_SUCCESS);
				++nreqs;
			}
			
			//nonblocking exchange with bottom neighbor
			other = neighbor_ranks[1];
			if (other != MPI_PROC_NULL) {
				//bottom halo (+ corners) send: field[field_upper[0] - kernel_radiuses[0]:field_upper[0], :]
				int offset = (field_upper[0] - kernel_radiuses[0]) * field_sizes[1];
				ret = MPI_Isend(field_reinterpreted + offset * pixel_size
					, halo_elems, pixel_type, other, 2, mesh_comm, requests + nreqs);
				assert(ret == MPI_SUCCESS);
				++nreqs;
				
				//bottom halo (+ corners) recv: field[field_upper[0]:field_sizes[0], :]
				offset = field_upper[0] * field_sizes[1];
				ret = MPI_Irecv(field_reinterpreted + offset * pixel_size
					, halo_elems, pixel_type, other, 2, mesh_comm, requests + nreqs);
				assert(ret == MPI_SUCCESS);
				++nreqs;
			}
			
			ret = MPI_Waitall(nreqs, requests, statuses);
			assert(ret == MPI_SUCCESS);
			for (int i = 0; i < nreqs; ++i) {
				assert(statuses[i].MPI_ERROR == MPI_SUCCESS);
			}
		}
		
		#ifdef TIMING
		timing_halo_exchange += MPI_Wtime();
		print_rank_prefix(stdout, rank, block_coords);
		printf(": timing_halo_exchange: %lf (bandwidth: %lfMB/s)\n", timing_halo_exchange, ((field_elems - field_dst_elems) * pixel_size) / (1024 * 1024 * timing_file_read));
		#endif

		preprocess_buffer(field, field_elems, pixel_size);

		//buffer without halos (output)
		unsigned short int* field_dst = (unsigned short int*) malloc(sizeof(unsigned short int) * field_dst_elems);
		if (!field) {
			MPI_Abort(mesh_comm, 1);
			exit(1);
		}
		
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

		//blur
		#ifdef TIMING
		double timing_blur = - MPI_Wtime();
		#endif

		#ifdef _OPENMP
		#pragma omp parallel for schedule(dynamic) shared(kernel, kernel_radiuses, field, field_sizes, field_lower, field_dst, block_sizes)
		#endif
		//collapse and schedule(dynamic) do not go together
		for (int i = 0; i < block_sizes[0]; ++i) {
			for (int j = 0; j < block_sizes[1]; ++j) {
				const int field_i = field_lower[0] + i;
				const int field_j = field_lower[1] + j;

				int kernel_lower[2];
				int kernel_upper[2];
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
				//printf("apply kernel[%d:%d, %d:%d] for field[%d, %d]\n", kernel_lower[0], kernel_upper[0], kernel_lower[1], kernel_upper[1], field_i, field_j);
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

				unsigned short int intensity = (unsigned short int) round(fmax(0.0, intensity_raw));
				field_dst[i * block_sizes[1] + j] = intensity;

				#ifndef NDEBUG
				//printf("intensity at pos %d, %d; (%hu) %hu <- %lf\n", field_i, field_j, field[field_i * field_sizes[1] + field_j], intensity, intensity_raw);
				if (intensity > meta.intensity_max) {
					printf("broken intensity at pos %d, %d; %hu <- %lf\n"
						, i, j
						, intensity, intensity_raw);
				}
				#endif
			}
		}

		free(kernel);
		
		#ifdef TIMING
		timing_blur += MPI_Wtime();
		print_rank_prefix(stdout, rank, block_coords);
		printf(": timing_blur: %lf\n", timing_blur);
		#endif
		
		postprocess_buffer(field_dst, field_dst_elems, pixel_size);

		#ifndef NDEBUG
		print_rank_prefix(stdout, rank, block_coords);
		printf(": img_view_output defined as img[%d:%d, %d:%d]\n"
			, block_lower[0], block_lower[0] + block_sizes[0]
			, block_lower[1], block_lower[1] + block_sizes[1]);
		#endif
		
		//write view: block without halos
		MPI_Datatype img_view_output;
		ret = MPI_Type_create_subarray(2
			, meta.img_sizes
			, block_sizes
			, block_lower
			, MPI_ORDER_C
			, pixel_type
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
