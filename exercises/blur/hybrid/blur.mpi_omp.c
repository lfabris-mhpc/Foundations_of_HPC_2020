#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>

#include <string.h>

#include <assert.h>

#include <mpi.h>

#ifdef _OPENMP
#include <omp.h>
#endif

#ifndef FLOAT_T
#define FLOAT_T double
#endif

#define VERBOSITY_OFF 0
#define VERBOSITY_INFO 1
#define VERBOSITY_KERNEL 2
#define VERBOSITY_BLUR 3
#define VERBOSITY_BLUR_POS 4

//#define VERBOSITY 4

#define UNUSED(x) ((void)(x))

#define imax(lhs, rhs) (((lhs) > (rhs)) ? (lhs) : (rhs))
#define imin(lhs, rhs) (((lhs) < (rhs)) ? (lhs) : (rhs))
#define swap(v) (((v & (uint16_t) 0xff00) >> 8) + ((v & (uint16_t) 0x00ff) << 8))

void print_rank_prefix(FILE* fp, const int rank, const int* block_coords) {
	fprintf(fp, "rank %d mesh[%d, %d]", rank, block_coords[0], block_coords[1]);
}

void print_usage(const char* program_name) {
	fprintf(stderr, "usage: %s [img_path] [kernel_type] [kernel_diameter] {weighted_kernel_f} {img_output}\n", program_name);
}

typedef struct metadata_t {
	long int header_length_input;
	int mesh_sizes[2];
	int img_sizes[2];
	int pgm_code;
	int intensity_max;
} metadata;

int metadata_type_commit(MPI_Datatype* type) {
	int lengths[5] = {1, 2, 2, 1, 1};
	const MPI_Aint displacements[5] = {0
		, sizeof(long int)
		, sizeof(long int) + sizeof(int) * 2
		, sizeof(long int) + sizeof(int) * 4
		, sizeof(long int) + sizeof(int) * 5};
    MPI_Datatype types[6] = {MPI_LONG, MPI_INT, MPI_INT, MPI_INT, MPI_INT};

	#ifndef NDEBUG
    int ret =
	#endif
	MPI_Type_create_struct(5, lengths, displacements, types, type);
	assert(ret == MPI_SUCCESS);

    return MPI_Type_commit(type);
}

void print_thread_provided(const int mpi_thread_provided, const int omp_max_threads) {
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

	printf("mpi_thread_provided: %d (%s) omp_get_max_threads: %d\n", mpi_thread_provided, t, omp_max_threads);
}

enum {KERNEL_TYPE_IDENTITY, KERNEL_TYPE_WEIGHTED, KERNEL_TYPE_GAUSSIAN, KERNEL_TYPE_UNRECOGNIZED};

int params_from_stdin(int* restrict dim_blocks) {
	assert(dim_blocks);

	for (int i = 0; i < 2; ++i) {
		printf("dimension %i: how many subdivisions? (0 to let MPI choose)\n", i + 1);
		int check = scanf("%d", dim_blocks + i);
		if (check != 1 || dim_blocks[i] < 0) {
			fprintf(stderr, "subdivisions must be and integer >= 0\n");
			return -1;
		}
	}

	return 0;
}

int params_from_args(const int argc, char** argv
	, const char** img_path
	, int* kernel_type
	, int* kernel_sizes
	, FLOAT_T* kernel_params0
	, char** img_save_path) {
	assert(img_path);
	assert(kernel_type);
	assert(kernel_sizes);
	assert(kernel_params0);
	assert(img_save_path);

	if (argc < 3) {
		print_usage(argv[0]);
		return -1;
	}

	int param_idx = 0;
	*img_path = argv[++param_idx];

	*kernel_type = (int) strtol(argv[++param_idx], NULL, 10);
	if (*kernel_type < 0 || *kernel_type >= KERNEL_TYPE_UNRECOGNIZED) {
		fprintf(stderr, "could not parse kernel_type\n");
		return -1;
	}

	//automated testing requires a single readius be passed, so read that and propagate
	for (int i = 0; i < 1; ++i) {
		const int kernel_diameter = atoi(argv[++param_idx]);

		if (kernel_diameter < 0 || kernel_diameter % 2 != 1) {
			print_usage(argv[0]);
			fprintf(stderr, "kernel_diameter must be an odd positive integer\n");
			return -1;
		}

		kernel_sizes[i] = kernel_diameter;
	}
	kernel_sizes[1] = kernel_sizes[0];

	*kernel_params0 = 0.0;
	if (*kernel_type == KERNEL_TYPE_WEIGHTED) {
		if (argc < 5) {
			print_usage(argv[0]);
			fprintf(stderr, "kernel_type %d requires weighted_kernel_f parameter\n", *kernel_type);
			return -1;
		}

		*kernel_params0 = atof(argv[++param_idx]);
		if (*kernel_params0 < 0.0 || *kernel_params0 > 1.0) {
			fprintf(stderr, "weighted_kernel_f must be in [0, 1]\n");
			return -1;
		}
	}

	if (argc > param_idx) {
		*img_save_path = argv[++param_idx];
	} else {
		*img_save_path = NULL;
	}

	return 0;
}

int pgm_get_metadata(const char* img_path, metadata* meta) {
	FILE* fp = fopen(img_path, "rb");
	if (!fp) {
		return -1;
	}

	char line[256];
	//first line must be P5\n
	char* checkc = fgets(line, sizeof(line), fp);
	if (!checkc || strncmp(line, "P5", 2)) {
		return -1;
	}

	meta->pgm_code = 5;

	//skip comments
	checkc = fgets(line, sizeof(line), fp);
	while (!checkc || !strncmp(line, "#", 1)) {
		checkc = fgets(line, sizeof(line), fp);
	}
	if (!checkc) {
		return -1;
	}
	int nread = sscanf(line, "%d %d\n", meta->img_sizes + 1, meta->img_sizes);
	if (nread != 2) {
		return -1;
	}

	//skip comments
	checkc = fgets(line, sizeof(line), fp);
	while (!checkc || !strncmp(line, "#", 1)) {
		checkc = fgets(line, sizeof(line), fp);
	}

	nread = sscanf(line, "%d\n", &meta->intensity_max);
	if (nread != 1) {
		return -1;
	}

	meta->header_length_input = ftell(fp);
	fclose(fp);

	return 0;
}

int img_save_path_generate(const char* img_path
	, const int kernel_type
	, const int* kernel_sizes
	, const char* kernel_params0_str
	, char** img_save_path) {
	assert(img_path);
	assert(kernel_sizes);
	assert(kernel_type != 1 || kernel_params0_str);

	//strip .pgm
	int extension_pos = strlen(img_path) - 4;
	if (extension_pos < 0) {
		return -1;
	}

	while (extension_pos && strncmp(img_path + extension_pos, ".pgm", 4)) {
		--extension_pos;
	}

	//strip folders
	int basename_pos = extension_pos;
	while (basename_pos && img_path[basename_pos] != '/') {
		--basename_pos;
	}
	++basename_pos;

	assert(basename_pos > 0);
	assert(extension_pos > 0);
	extension_pos -= basename_pos;

	*img_save_path = (char*) malloc(extension_pos + 5 + 128);
	if (!*img_save_path) {
		free(*img_save_path);
		*img_save_path = NULL;
		return -1;
	}

	memcpy(*img_save_path, img_path + basename_pos, extension_pos);
	(*img_save_path)[extension_pos] = 0;

	snprintf(*img_save_path + extension_pos, 5 + 128
		, ".bb_%d_%dx%d", kernel_type, kernel_sizes[0], kernel_sizes[1]);

	if (kernel_type == 1) {
		assert(kernel_params0_str);
		int start = strlen(*img_save_path);
		int new_len;
		snprintf(*img_save_path + start, sizeof(*img_save_path) - start
			, "_%s%n", kernel_params0_str, &new_len);

		char* dot = strstr(*img_save_path + start, ".");
		if (dot) {
			for (int i = 0; i < new_len; ++i) {
				dot[i] = dot[i+1];
			}
		}
	}
	strcat(*img_save_path, ".mpi_omp.pgm");

	return 0;
}

int read_pgm_slice(const char* img_path
	, const metadata* meta
	, const int* block_haloed_lower
	, const int* field_sizes
	, uint16_t* field) {
	const int pixel_size = 1 + (meta->intensity_max > 255);
	
	uint8_t* field_reinterpreted = (uint8_t*) field;
	
	#if VERBOSITY >= VERBOSITY_INFO
	printf("read_pgm_slice: img[%d:%d, %d:%d]\n"
		, block_haloed_lower[0], block_haloed_lower[0] + field_sizes[0]
		, block_haloed_lower[1], block_haloed_lower[1] + field_sizes[1]
		);
	#endif
	
	if (field_sizes[1] == meta->img_sizes[1]) {
		//read entire slice in one go
		FILE* fp = fopen(img_path, "rb");
		#ifndef NDEBUG
		int ret =
		#endif
		fseek(fp, meta->header_length_input, SEEK_SET);
		assert(ret == 0);
		
		size_t read = fread(field_reinterpreted, 1, pixel_size * field_sizes[0] * field_sizes[1], fp);
		assert(read == pixel_size * field_sizes[0] * field_sizes[1]);
		UNUSED(read);
		
		return 0;
	}
	
	#ifdef _OPENMP
	#pragma omp parallel shared(img_path, meta, block_haloed_lower, field_sizes, field, field_reinterpreted)
	#endif
	{
		FILE* fp = fopen(img_path, "rb");
		if (fp) {
			int thread_rows = field_sizes[0] / omp_get_num_threads();
			int thread_rows_lower = omp_get_thread_num() * thread_rows;
			int rem = field_sizes[0] % omp_get_num_threads();
			if (omp_get_thread_num() < rem) {
				++thread_rows;
				thread_rows_lower += omp_get_thread_num();
			} else {
				thread_rows_lower += rem;
			}
	
			#if VERBOSITY >= VERBOSITY_INFO
			printf("read_pgm_slice: thread %d will read img[%d:%d, %d:%d]\n"
				, omp_get_thread_num()
				, block_haloed_lower[0] + thread_rows_lower, block_haloed_lower[0] + thread_rows_lower + thread_rows
				, block_haloed_lower[1], block_haloed_lower[1] + field_sizes[1]
				);
			#endif
			
			#ifndef NDEBUG
			int ret =
			#endif
			fseek(fp, meta->header_length_input + pixel_size * ((block_haloed_lower[0] + thread_rows_lower) * meta->img_sizes[1] + block_haloed_lower[1]), SEEK_SET);
			assert(ret == 0);

			for (int i = 0; i < thread_rows; ++i) {
				const int pos = pixel_size * (thread_rows_lower + i) * field_sizes[1];
				
				#ifndef NDEBUG
				size_t read =
				#endif
				fread(field_reinterpreted + pos, 1, pixel_size * field_sizes[1], fp);
				assert(read == pixel_size * field_sizes[1]);

				if (i < (thread_rows - 1)) {
					#ifndef NDEBUG
					ret =
					#endif
					fseek(fp, pixel_size * (meta->img_sizes[1] - field_sizes[1]), SEEK_CUR);
					assert(ret == 0);
				}
			}

			fclose(fp);
		}
	}
	
	return 0;
}

void preprocess_buffer(uint16_t* field, const int field_elems, const int pixel_size) {
	if (pixel_size == 1) {
		//widen the char to shorts
		uint8_t* field_reinterpreted = (uint8_t*) field;

		for (int i = field_elems - 1; i > -1; --i) {
			field[i] = (uint16_t) field_reinterpreted[i];
		}
	} else if ((pixel_size == 2) && ((0x100 & 0xf) == 0x0)) {
		#ifdef _OPENMP
		#pragma omp parallel for shared(field, field_elems)
		#endif
		for (int i = 0; i < field_elems; ++i) {
			field[i] = swap(field[i]);
		}
	} else {
		assert(0);
	}
}

void postprocess_buffer(uint16_t* field, const int field_elems, const int pixel_size) {
	if (pixel_size == 1) {
		//shrink the shorts to chars
		uint8_t* field_reinterpreted = (uint8_t*) field;

		for (int i = 0; i < field_elems; ++i) {
			field_reinterpreted[i] = (uint8_t) field[i];
		}
	} else if ((pixel_size == 2) && ((0x100 & 0xf) == 0x0)) {
		#ifdef _OPENMP
		#pragma omp parallel for shared(field, field_elems)
		#endif
		for (int i = 0; i < field_elems; ++i) {
			field[i] = swap(field[i]);
		}
	} else {
		assert(0);
	}
}

void binomial_coefficients_init(FLOAT_T* coeffs0, FLOAT_T* coeffs1, const int* kernel_sizes) {
	assert(coeffs0);
	assert(coeffs1);
	assert(kernel_sizes[0] > 0);
	assert(kernel_sizes[1] > 0);

	int i = 0;
	for (; i < imin(kernel_sizes[0], kernel_sizes[1]); ++i) {
		coeffs0[i] = 1.0;
		coeffs1[i] = 1.0;
	}
	for (; i < kernel_sizes[0]; ++i) {
		coeffs0[i] = 1.0;
	}
	for (; i < kernel_sizes[1]; ++i) {
		coeffs1[i] = 1.0;
	}

	if (kernel_sizes[0] < 3 && kernel_sizes[1] < 3) {
		return;
	}

	for (i = 2; i < imin(kernel_sizes[0], kernel_sizes[1]); ++i) {
		for (int j = i - 1; j; --j) {
			coeffs0[j] += coeffs0[j - 1];
			coeffs1[j] = coeffs0[j];
		}
	}
	for (; i < kernel_sizes[0]; ++i) {
		for (int j = i - 1; j; --j) {
			coeffs0[j] += coeffs0[j - 1];
		}
	}
	for (; i < kernel_sizes[1]; ++i) {
		for (int j = i - 1; j; --j) {
			coeffs1[j] += coeffs1[j - 1];
		}
	}
}

int kernel_init(const int kernel_type
	, const int* restrict kernel_sizes
	, const FLOAT_T kernel_params0
	, FLOAT_T* restrict kernel
	, const int normalize) {
	assert(kernel);
	assert(kernel_sizes);

	const int elems = kernel_sizes[0] * kernel_sizes[1];
	assert(elems > 0);

	switch (kernel_type) {
		case KERNEL_TYPE_IDENTITY:
			{
				const FLOAT_T w = normalize ? (1.0 / elems) : 1.0;
				#ifdef _OPENMP
				#pragma omp parallel for shared(kernel, elems, w)
				#endif
				for (int i = 0; i < elems; ++i) {
					kernel[i] = w;
				}
			}
			break;
		case KERNEL_TYPE_WEIGHTED:
			{
				//kernel_params0 used as weight of original pixel
				assert(kernel_params0 >= 0.0 && kernel_params0 <= 1.0);

				const FLOAT_T w = (1.0 - kernel_params0) / (elems - 1);
				#ifdef _OPENMP
				#pragma omp parallel for shared(kernel, elems, w)
				#endif
				for (int i = 0; i < elems; ++i) {
					kernel[i] = w;
				}
				kernel[elems / 2] = kernel_params0;
			}
			break;
		case KERNEL_TYPE_GAUSSIAN:
			{
				FLOAT_T coeffs0[kernel_sizes[0]];
				FLOAT_T coeffs1[kernel_sizes[1]];
				binomial_coefficients_init(coeffs0, coeffs1, kernel_sizes);

				FLOAT_T coeffs_sum[2] = {0, 0};
				for (int i = 0; i < kernel_sizes[0]; ++i) {
					coeffs_sum[0] += coeffs0[i];
				}
				for (int i = 0; i < kernel_sizes[1]; ++i) {
					coeffs_sum[1] += coeffs1[i];
				}

				FLOAT_T norm = coeffs_sum[0] * coeffs_sum[1];
				#ifdef _OPENMP
				#pragma omp parallel for collapse(2) shared(kernel_sizes, kernel, coeffs0, coeffs1, norm)
				#endif
				for (int i = 0; i < kernel_sizes[0]; ++i) {
					for (int j = 0; j < kernel_sizes[1]; ++j) {
						kernel[i * kernel_sizes[1] + j] = coeffs0[i] * coeffs1[j];
						if (normalize) {
							kernel[i * kernel_sizes[1] + j] /= norm;
						}
					}
				}
			}
			break;
		default:
			return -1;
	}

	#ifndef NDEBUG
	if (normalize) {
		FLOAT_T norm = 0.0;
		#ifdef _OPENMP
		#pragma omp parallel for reduction(+: norm) shared(elemes, kernel)
		#endif
		for (int i = 0; i < elems; ++i) {
			norm += kernel[i];
		}
		#ifdef _OPENMP
		#pragma omp parallel for shared(elemes, kernel, norm)
		#endif
		for (int i = 0; i < elems; ++i) {
			kernel[i] /= norm;
		}
		assert(fabs(norm - 1) < 0.00000001);
	}
	#endif

	return 0;
}

void convolve_slices(const FLOAT_T* restrict kernel
	, const int* restrict kernel_sizes
	, const int* restrict kernel_lower
	, const uint16_t* restrict field
	, const int* restrict field_sizes
	, const int* restrict field_lower
	, const int* restrict extents
	, FLOAT_T* restrict output
	, FLOAT_T* restrict output_norm) {
	assert(kernel);
	assert(field);
	assert(output);
	assert(output_norm);

	for (int i = 0; i < 2; ++i) {
		assert(kernel_sizes[i] >= 1);
		assert(kernel_lower[i] >= 0);
		assert((kernel_lower[i] + extents[i]) <= kernel_sizes[i]);

		assert(field_sizes[i] >= 1);
		assert(field_lower[i] >= 0);
		assert((field_lower[i] + extents[i]) <= field_sizes[i]);
	}

	FLOAT_T tmp = 0.0, norm = 0.0;

	#ifndef NDEBUG
	int iters = 0;
	#endif

	#if !defined(UNROLL) || UNROLL < 0
	#define UNROLL 8
	#elif UNROLL > 16
	#define UNROLL 16
	#endif

	#if UNROLL > 1
	#define tmps_plus(pos) tmps[(pos)] += kernel[kernel_jpos + j + (pos)] * field[field_jpos + j + (pos)];
	#define norms_plus(pos) norms[(pos)] += kernel[kernel_jpos + j + (pos)];
	#define unrolled_op(pos) tmps_plus((pos)); \
		norms_plus((pos));

	#define unrolled_4block(start) unrolled_op((start) + 0); \
		unrolled_op((start) + 1); \
		unrolled_op((start) + 2); \
		unrolled_op((start) + 3);
	#endif

	{
		#if UNROLL > 1
		FLOAT_T tmps[UNROLL] = {0.0};
		FLOAT_T norms[UNROLL] = {0.0};
		#endif

		for (int i = 0; i < extents[0]; ++i) {
			register const int kernel_jpos = (kernel_lower[0] + i) * kernel_sizes[1] + kernel_lower[1];
			register const int field_jpos = (field_lower[0] + i) * field_sizes[1] + field_lower[1];

			int j = 0;
			#if UNROLL > 1
			for (; j < extents[1] - UNROLL; j += UNROLL) {
				unrolled_op(0);
				#if UNROLL > 1
				unrolled_op(1);
				#endif
				#if UNROLL > 2
				unrolled_op(2);
				#endif
				#if UNROLL > 3
				unrolled_op(3);
				#endif
				#if UNROLL > 4
				unrolled_op(4);
				#endif
				#if UNROLL > 5
				unrolled_op(5);
				#endif
				#if UNROLL > 6
				unrolled_op(6);
				#endif
				#if UNROLL > 7
				unrolled_op(7);
				#endif
				
				#if UNROLL > 8
				unrolled_op(8);
				#endif
				#if UNROLL > 9
				unrolled_op(9);
				#endif
				#if UNROLL > 10
				unrolled_op(10);
				#endif
				#if UNROLL > 11
				unrolled_op(11);
				#endif
				#if UNROLL > 12
				unrolled_op(12);
				#endif
				#if UNROLL > 13
				unrolled_op(13);
				#endif
				#if UNROLL > 14
				unrolled_op(14);
				#endif
				#if UNROLL > 15
				unrolled_op(15);
				#endif

				#ifndef NDEBUG
				iters += UNROLL;
				#endif
			}
			#endif

			for (; j < extents[1]; ++j) {
				tmp += kernel[kernel_jpos + j] * field[field_jpos + j];
				norm += kernel[kernel_jpos + j];

				#ifndef NDEBUG
				++iters;
				#endif
			}
		}

		#if UNROLL > 1
		for (int i = 0; i < UNROLL; ++i) {
			tmp += tmps[i];
			norm += norms[i];
		}
		#endif
	}

	#ifndef NDEBUG
	assert(iters == extents[0] * extents[1]);
	#endif

	*output = tmp;
	*output_norm = norm;
}

void init_pos_slice(const int* restrict kernel_sizes
	, const int* restrict field_sizes
	, const int field_pos0
	, const int field_pos1
	, int* restrict kernel_lower
	, int* restrict field_lower
	, int* restrict extents) {
	assert(kernel_lower);
	assert(field_lower);
	assert(extents);

	assert(kernel_sizes[0] > 0);
	assert(kernel_sizes[1] > 0);

	assert(field_pos0 >= 0);
	assert(field_pos0 < field_sizes[0]);
	assert(field_pos1 >= 0);
	assert(field_pos1 < field_sizes[1]);

	const int radius0 = kernel_sizes[0] / 2;
	kernel_lower[0] = imax(radius0 - field_pos0, 0);
	field_lower[0] = field_pos0 - (radius0 - kernel_lower[0]);
	extents[0] = imin(field_pos0 + radius0 + 1, field_sizes[0]) - field_lower[0];

	const int radius1 = kernel_sizes[1] / 2;
	kernel_lower[1] = imax(radius1 - field_pos1, 0);
	field_lower[1] = field_pos1 - (radius1 - kernel_lower[1]);
	extents[1] = imin(field_pos1 + radius1 + 1, field_sizes[1]) - field_lower[1];
}

void convolve_pos(const FLOAT_T* restrict kernel
	, const int* restrict kernel_sizes
	, const uint16_t* restrict field
	, const int* restrict field_sizes
	, const int field_pos0
	, const int field_pos1
	, FLOAT_T* restrict output) {
	assert(kernel);
	assert(field);
	assert(output);

	assert(kernel_sizes[0] > 0);
	assert(kernel_sizes[1] > 0);

	assert(field_pos0 >= 0);
	assert(field_pos0 < field_sizes[0]);
	assert(field_pos1 >= 0);
	assert(field_pos1 < field_sizes[1]);

	int kernel_lower[2];
	int field_lower[2];
	int extents[2];
	init_pos_slice(kernel_sizes, field_sizes, field_pos0, field_pos1
		, kernel_lower, field_lower, extents);

	#if VERBOSITY >= VERBOSITY_BLUR_POS
	printf("convolve kernel[%d:%d, %d:%d] and field[%d:%d, %d:%d] -> pos (%d, %d); extents: (%d, %d)\n"
		, kernel_lower[0], kernel_lower[0] + extents[0]
		, kernel_lower[1], kernel_lower[1] + extents[1]
		, field_lower[0], field_lower[0] + extents[0]
		, field_lower[1], field_lower[1] + extents[1]
		, field_pos0, field_pos1
		, extents[0], extents[1]);
	#endif

	FLOAT_T tmp = 0.0, norm = 0.0;

	convolve_slices(kernel, kernel_sizes, kernel_lower
		, field, field_sizes, field_lower
		, extents, &tmp, &norm);

	if (extents[0] != kernel_sizes[0] || extents[1] != kernel_sizes[1]) {
		tmp /= norm;
	}

	*output = tmp;
}

void blur(const FLOAT_T* restrict kernel
	, const int* restrict kernel_sizes
	, uint16_t* restrict field
	, const int* restrict field_sizes
	, const int* restrict field_lower
	, uint16_t* restrict field_dst
	, const int* restrict field_dst_sizes
	, const int* restrict field_dst_lower
	, const int* restrict extents
	, const int intensity_max) {
	assert(kernel);
	assert(field);
	assert(field_dst);
	assert(intensity_max > 0);

	for (int i = 0; i < 2; ++i) {
		assert(kernel_sizes[i] > 0);
		assert(extents[i] > 0);

		assert(field_lower[i] >= 0);
		assert(field_lower[i] + extents[i] <= field_sizes[i]);

		assert(field_dst_lower[i] >= 0);
		assert(field_dst_lower[i] + extents[i] <= field_dst_sizes[i]);
	}

	#if VERBOSITY >= VERBOSITY_BLUR
	printf("blur kernel and field[%d:%d, %d:%d]; extents: (%d, %d)\n"
		, field_lower[0], field_lower[0] + extents[0]
		, field_lower[1], field_lower[1] + extents[1]
		, extents[0], extents[1]);
	#endif

	#ifdef _OPENMP
	#pragma omp parallel for collapse(2) shared(kernel, kernel_sizes, field, field_sizes, field_lower, field_dst, field_dst_sizes, field_dst_lower, extents, intensity_max)
	#endif
	for (int i = 0; i < extents[0]; ++i) {
		for (int j = 0; j < extents[1]; ++j) {
			FLOAT_T intensity;
			convolve_pos(kernel, kernel_sizes
				, field, field_sizes, field_lower[0] + i, field_lower[1] + j
				, &intensity);

			field_dst[(field_dst_lower[0] + i) * field_dst_sizes[1] + field_dst_lower[1] + j] = (uint16_t) (intensity > intensity_max ? intensity_max : intensity);
			
			#if VERBOSITY >= VERBOSITY_BLUR
			printf("blur field[%d, %d] %hu -> %hu\n"
				, field_dst_lower[0] + i, field_dst_lower[1] + j
				, field[(field_lower[0] + i) * field_sizes[1] + field_lower[1] + j]
				, field_dst[(field_dst_lower[0] + i) * field_dst_sizes[1] + field_dst_lower[1] + j]);
			#endif
		}
	}
}

int main(int argc , char** argv)
{
	int ret;

	#ifdef _OPENMP
	int mpi_thread_provided;
	ret = MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &mpi_thread_provided);
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

	const char* img_path;
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

		ret = MPI_Dims_create(nranks, 2, meta.mesh_sizes);
		assert(ret == MPI_SUCCESS);

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
		//master's touch
		img_save_path[0] = 0;

		#if VERBOSITY >= VERBOSITY_INFO
		print_rank_prefix(stdout, rank, block_coords);
		printf(": img_view_input defined as img[%d:%d, %d:%d] (img shape: (%d, %d))\n"
			, block_haloed_lower[0], block_haloed_lower[0] + field_sizes[0]
			, block_haloed_lower[1], block_haloed_lower[1] + field_sizes[1]
			, meta.img_sizes[0], meta.img_sizes[1]);
		#endif
		
		#ifdef TIMING
		double timing_file_read = - MPI_Wtime();
		#endif

		//read haloed block, independently from other ranks
		#ifdef READ_MPI
		read_pgm_slice_mpi(img_path
			, &meta
			, block_haloed_lower
			, field_sizes
			, field
			, &mesh_comm);
		#else
		read_pgm_slice(img_path
			, &meta
			, block_haloed_lower
			, field_sizes
			, field);
		#endif

		#ifdef TIMING
		timing_file_read += MPI_Wtime();
		print_rank_prefix(stdout, rank, block_coords);
		printf(": timing_file_read: %lf (bandwidth: %lf GB/s)\n", timing_file_read, (field_elems * pixel_size) / (1000 * 1000 * 1000 * timing_file_read));
		#endif

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
		//master's touch
		field_dst[0] = 0;

		FLOAT_T* kernel = (FLOAT_T*) malloc(sizeof(FLOAT_T) * kernel_sizes[0] * kernel_sizes[1]);
		if (!kernel) {
			MPI_Abort(mesh_comm, 1);
		}
		//master's touch
		kernel[0] = 0.0;

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
		blur(kernel, kernel_sizes
			, field, field_sizes, field_lower
			, field_dst, block_sizes, field_dst_lower
			, block_sizes
			, meta.intensity_max);

		#ifdef TIMING
		timing_blur += MPI_Wtime();
		print_rank_prefix(stdout, rank, block_coords);
		printf(": timing_blur: %lf\n", timing_blur);
		#endif

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

		//concerted file write
		MPI_Status status;
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

		int count;
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
		free(kernel);
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
