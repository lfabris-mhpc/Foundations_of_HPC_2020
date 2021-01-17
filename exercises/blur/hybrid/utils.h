#ifndef __UTILS_H__
#define __UTILS_H__

#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>

#include <string.h>
#include <math.h>

#include <assert.h>

#define VERBOSITY_OFF 0
#define VERBOSITY_INFO 1
#define VERBOSITY_KERNEL 2
#define VERBOSITY_BLUR 3
#define VERBOSITY_BLUR_POS 4

#ifndef FLOAT_T
#define FLOAT_T float
#endif

void print_rank_prefix(FILE* fp, const int rank, const int* restrict block_coords) {
	fprintf(fp, "rank %d mesh[%d, %d]", rank, block_coords[0], block_coords[1]);
}

void print_usage(const char* restrict program_name) {
	fprintf(stderr, "usage: %s [img_path] [kernel_type] [kernel_diameter] {weighted_kernel_f} {img_output}\n", program_name);
}

#define imax(lhs, rhs) (((lhs) > (rhs)) ? (lhs) : (rhs))
#define imin(lhs, rhs) (((lhs) < (rhs)) ? (lhs) : (rhs))
#define swap(v) (((v & (uint16_t) 0xff00) >> 8) + ((v & (uint16_t) 0x00ff) << 8))

typedef struct metadata_t {
	long int header_length_input;
	long int header_length_output;
	int mesh_sizes[2];
	int img_sizes[2];
	int pgm_code;
	int intensity_max;
} metadata;

int metadata_type_commit(MPI_Datatype* type) {
	int lengths[6] = {1, 1, 2, 2, 1, 1};
	const MPI_Aint displacements[6] = {0
		, sizeof(long int)
		, sizeof(long int) * 2
		, sizeof(long int) * 2 + sizeof(int) * 2
		, sizeof(long int) * 2 + sizeof(int) * 4
		, sizeof(long int) * 2 + sizeof(int) * 5};
    MPI_Datatype types[6] = {MPI_LONG, MPI_LONG, MPI_INT, MPI_INT, MPI_INT, MPI_INT};

	#ifndef NDEBUG
    int ret =
	#endif
	MPI_Type_create_struct(6, lengths, displacements, types, type);
	assert(ret == MPI_SUCCESS);

    return MPI_Type_commit(type);
}

void preprocess_buffer(uint16_t* field, int field_elems, int pixel_size) {
	if (pixel_size == 1) {
		//widen the char to shorts
		uint8_t* field_reinterpreted = (uint8_t*) field;

		for (int i = field_elems - 1; i > -1; --i) {
			field[i] = (uint16_t) field_reinterpreted[i];
		}
	} else if ((pixel_size == 2) && ((0x100 & 0xf) == 0x0)) {
		#ifdef _OPENMP
		#pragma omp parallel for shared(field)
		#endif
		for (int i = 0; i < field_elems; ++i) {
			field[i] = swap(field[i]);
		}
	} else {
		assert(0);
	}
}

void postprocess_buffer(uint16_t* field, int field_elems, int pixel_size) {
	if (pixel_size == 1) {
		//shrink the shorts to chars
		uint8_t* field_reinterpreted = (uint8_t*) field;

		for (int i = 0; i < field_elems; ++i) {
			field_reinterpreted[i] = (uint8_t) field[i];
		}
	} else if ((pixel_size == 2) && ((0x100 & 0xf) == 0x0)) {
		#ifdef _OPENMP
		#pragma omp parallel for shared(field)
		#endif
		for (int i = 0; i < field_elems; ++i) {
			field[i] = swap(field[i]);
		}
	} else {
		assert(0);
	}
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

#define KERNEL_SUFFIX_MAX_LEN 20

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
	, char** img_path
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
		/*
		if (strlen(line) < 256) {
			checkc = fgets(line, sizeof(line), *fp);
		}
		*/
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
	, const int* restrict kernel_sizes
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
		, ".b_%d_%dx%d", kernel_type, kernel_sizes[0], kernel_sizes[1]);

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
	strcat(*img_save_path, ".pgm");

	return 0;
}

void binomial_coefficients_init(FLOAT_T* coeffs0, int size0, FLOAT_T* coeffs1, int size1) {
	assert(coeffs0);
	assert(coeffs1);
	assert(size0 > 0);
	assert(size1 > 0);

	int i = 0;
	for (; i < imin(size0, size1); ++i) {
		coeffs0[i] = 1.0;
		coeffs1[i] = 1.0;
	}
	for (; i < size0; ++i) {
		coeffs0[i] = 1.0;
	}
	for (; i < size1; ++i) {
		coeffs1[i] = 1.0;
	}

	if (size0 < 3 && size1 < 3) {
		return;
	}

	for (i = 2; i < imin(size0, size1); ++i) {
		for (int j = i - 1; j; --j) {
			coeffs0[j] += coeffs0[j - 1];
			coeffs1[j] = coeffs0[j];
		}
	}
	for (; i < size0; ++i) {
		for (int j = i - 1; j; --j) {
			coeffs0[j] += coeffs0[j - 1];
		}
	}
	for (; i < size1; ++i) {
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
				#pragma omp parallel for
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
				#pragma omp parallel for
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
				binomial_coefficients_init(coeffs0, kernel_sizes[0], coeffs1, kernel_sizes[1]);

				FLOAT_T coeffs_sum[2] = {0, 0};
				for (int i = 0; i < kernel_sizes[0]; ++i) {
					coeffs_sum[0] += coeffs0[i];
				}
				for (int i = 0; i < kernel_sizes[1]; ++i) {
					coeffs_sum[1] += coeffs1[i];
				}

				FLOAT_T norm = coeffs_sum[0] * coeffs_sum[1];
				#ifdef _OPENMP
				#pragma omp parallel for collapse(2)
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
		for (int i = 0; i < elems; ++i) {
			norm += kernel[i];
		}
		#ifdef _OPENMP
		#pragma omp parallel for
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

	#if !defined(UNROLL)
	#define UNROLL 4
	#elif UNROLL > 8
	#define UNROLL 8
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

	#ifdef PARALLEL_POS_ON
	#pragma omp parallel reduction(+:tmp, norm) shared(kernel, kernel_sizes, kernel_lower, field, field_sizes, field_lower, extents)
	#endif
	{
		#if UNROLL > 1
		FLOAT_T tmps[UNROLL] = {0.0};
		FLOAT_T norms[UNROLL] = {0.0};
		#endif


		#ifdef PARALLEL_POS_ON
		#pragma omp for
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

#ifdef BLOCKING_POS_ON
void convolve_pos_byblocks(const FLOAT_T* restrict kernel
	, const int* restrict kernel_sizes
	, const uint16_t* restrict field
	, const int* restrict field_sizes
	, const int field_pos0
	, const int field_pos1
	, const int* restrict blocking
	, FLOAT_T* restrict output) {
	assert(kernel);
	assert(field);
	assert(output);

	assert(kernel_sizes[0] > 0);
	assert(kernel_sizes[1] > 0);
	assert(blocking[0] > 0);
	assert(blocking[1] > 0);

	assert(field_pos0 >= 0);
	assert(field_pos0 < field_sizes[0]);
	assert(field_pos1 >= 0);
	assert(field_pos1 < field_sizes[1]);

	int kernel_lower[2];
	int field_lower[2];
	int extents[2];
	init_pos_slice(kernel_sizes, field_sizes, field_pos0, field_pos1
		, kernel_lower, field_lower, extents);

	FLOAT_T tmp = 0.0, norm = 0.0, ttmp = 0.0, tnorm = 0.0;
	int kernel_block_lower[2];
	int field_block_lower[2];

	int i = 0;
	for (; i < extents[0] - blocking[0]; i += blocking[0]) {
		kernel_block_lower[0] = kernel_lower[0] + i;
		field_block_lower[0] = field_lower[0] + i;

		int j = 0;
		for (; j < extents[1] - blocking[1]; j += blocking[1]) {
			kernel_block_lower[1] = kernel_lower[1] + j;
			field_block_lower[1] = field_lower[1] + j;

			convolve_slices(kernel, kernel_sizes, kernel_block_lower
				, field, field_sizes, field_block_lower
				, blocking, &ttmp, &tnorm);

			tmp += ttmp;
			norm += tnorm;
		}

		if (j < extents[1]) {
			const int block_extents[2] = {blocking[0], extents[1] - j};

			kernel_block_lower[1] = kernel_lower[1] + j;
			field_block_lower[1] = field_lower[1] + j;

			convolve_slices(kernel, kernel_sizes, kernel_block_lower
				, field, field_sizes, field_block_lower
				, block_extents, &ttmp, &tnorm);

			tmp += ttmp;
			norm += tnorm;
		}
	}

	if (i < extents[0]) {
		int block_extents[2] = {extents[0] - i, blocking[1]};
		kernel_block_lower[0] = kernel_lower[0] + i;
		field_block_lower[0] = field_lower[0] + i;

		int j = 0;
		for (; j < extents[1] - blocking[1]; j += blocking[1]) {
			kernel_block_lower[1] = kernel_lower[1] + j;
			field_block_lower[1] = field_lower[1] + j;

			convolve_slices(kernel, kernel_sizes, kernel_block_lower
				, field, field_sizes, field_block_lower
				, block_extents, &ttmp, &tnorm);

			tmp += ttmp;
			norm += tnorm;
		}

		if (j < extents[1]) {
			block_extents[1] = extents[1] - j;

			kernel_block_lower[1] = kernel_lower[1] + j;
			field_block_lower[1] = field_lower[1] + j;

			convolve_slices(kernel, kernel_sizes, kernel_block_lower
				, field, field_sizes, field_block_lower
				, block_extents, &ttmp, &tnorm);

			tmp += ttmp;
			norm += tnorm;
		}
	}

	#if VERBOSITY >= VERBOSITY_BLUR_POS
	printf("convolve kernel and field[%d:%d, %d:%d]; extents: (%d, %d)\n"
		, field_lower[0], field_lower[0] + extents[0]
		, field_lower[1], field_lower[1] + extents[1]
		, extents[0], extents[1]);
	#endif

	if (extents[0] != kernel_sizes[0] || extents[1] != kernel_sizes[1]) {
		tmp /= norm;
	}

	*output = tmp;
}
#endif

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

	#ifdef BLOCKING_POS_ON
	const int blocking[2] = {BLOCKING_POS_ROWS, BLOCKING_POS_COLUMNS};
	#endif

	#if VERBOSITY >= VERBOSITY_BLUR
	printf("blur kernel and field[%d:%d, %d:%d]; extents: (%d, %d)\n"
		, field_lower[0], field_lower[0] + extents[0]
		, field_lower[1], field_lower[1] + extents[1]
		, extents[0], extents[1]);
	#endif

	#ifndef PARALLEL_POS_ON
	#ifdef _OPENMP
	#pragma omp parallel for collapse(2) shared(kernel, kernel_sizes, field, field_sizes, field_lower, field_dst, field_dst_sizes, field_dst_lower, extents, intensity_max)
	#endif
	#endif
	for (int i = 0; i < extents[0]; ++i) {
		for (int j = 0; j < extents[1]; ++j) {
			FLOAT_T intensity;
			#ifdef BLOCKING_POS_ON
			convolve_pos_byblocks(kernel, kernel_sizes
				, field, field_sizes, field_lower[0] + i, field_lower[1] + j
				, blocking
				, &intensity);
			#else
			convolve_pos(kernel, kernel_sizes
				, field, field_sizes, field_lower[0] + i, field_lower[1] + j
				, &intensity);
			#endif

			field_dst[(field_dst_lower[0] + i) * field_dst_sizes[1] + field_dst_lower[1] + j] = (uint16_t) fmin(intensity, intensity_max);//fmin(round(intensity), intensity_max);
		}
	}
}

#ifdef BLOCKING_BLUR_ON
void blur_byblocks(const FLOAT_T* restrict kernel
	, const int* restrict kernel_sizes
	, uint16_t* restrict field
	, const int* restrict field_sizes
	, const int* restrict field_lower
	, uint16_t* restrict field_dst
	, const int* restrict field_dst_sizes
	, const int* restrict field_dst_lower
	, const int* restrict extents
	, const int* restrict blocking
	, const int intensity_max) {
	assert(kernel);
	assert(field);
	assert(field_dst);
	assert(intensity_max > 0);

	for (int i = 0; i < 2; ++i) {
		assert(kernel_sizes[i] > 0);
		assert(blocking[i] > 0);
		assert(extents[i] > 0);

		assert(field_lower[i] >= 0);
		assert(field_lower[i] + extents[i] <= field_sizes[i]);

		assert(field_dst_lower[i] >= 0);
		assert(field_dst_lower[i] + extents[i] <= field_dst_sizes[i]);
	}

	#if VERBOSITY >= VERBOSITY_BLUR
	printf("blur (by blocks) kernel on field[%d:%d, %d:%d]; extents: (%d, %d)\n"
		, field_lower[0], field_lower[0] + extents[0]
		, field_lower[1], field_lower[1] + extents[1]
		, extents[0], extents[1]);
	#endif

	#ifndef _OPENMP
	#define TASK_PRINT_FMT "task blur field[%d:%d, %d:%d]; extents (%d, %d)\n"
	#endif

	#ifdef _OPENMP
	#define TASK_PRINT_FMT "thread %d task blur field[%d:%d, %d:%d]; extents (%d, %d)\n"

	#pragma omp parallel
	#endif
	{
		#ifdef _OPENMP
		#pragma omp single nowait
		#endif
		{
			int i = 0;
			for (; i < extents[0] - blocking[0]; i += blocking[0]) {
				int j = 0;
				for (; j < extents[1] - blocking[1]; j += blocking[1]) {
					#ifdef _OPENMP
					#pragma omp task firstprivate(i, j) shared(kernel, kernel_sizes, field, field_sizes, field_lower, field_dst, field_dst_sizes, field_dst_lower, blocking, intensity_max)
					#endif
					{
						const int field_block_lower[2] = {field_lower[0] + i, field_lower[1] + j};
						const int field_dst_block_lower[2] = {field_dst_lower[0] + i, field_dst_lower[0] + j};

						#if VERBOSITY >= VERBOSITY_BLUR
						printf(TASK_PRINT_FMT
							#ifdef _OPENMP
							, omp_get_thread_num()
							#endif
							, field_block_lower[0], field_block_lower[0] + blocking[0]
							, field_block_lower[1], field_block_lower[1] + blocking[1]
							, blocking[0], blocking[1]);
						#endif

						blur(kernel, kernel_sizes
							, field, field_sizes, field_block_lower
							, field_dst, field_dst_sizes, field_dst_block_lower
							, blocking
							, intensity_max);
					}
				}

				if (j < extents[1]) {
					#ifdef _OPENMP
					#pragma omp task firstprivate(i, j) shared(kernel, kernel_sizes, field, field_sizes, field_lower, field_dst, field_dst_sizes, field_dst_lower, extents, blocking, intensity_max)
					#endif
					{
						const int field_block_lower[2] = {field_lower[0] + i, field_lower[1] + j};
						const int field_dst_block_lower[2] = {field_dst_lower[0] + i, field_dst_lower[1] + j};
						const int block_extents[2] = {blocking[0], extents[1] - j};

						#if VERBOSITY >= VERBOSITY_BLUR
						printf(TASK_PRINT_FMT
							#ifdef _OPENMP
							, omp_get_thread_num()
							#endif
							, field_block_lower[0], field_block_lower[0] + block_extents[0]
							, field_block_lower[1], field_block_lower[1] + block_extents[1]
							, block_extents[0], block_extents[1]);
						#endif

						blur(kernel, kernel_sizes
							, field, field_sizes, field_block_lower
							, field_dst, field_dst_sizes, field_dst_block_lower
							, block_extents
							, intensity_max);
					}
				}
			}

			if (i < extents[0]) {
				int j = 0;
				for (; j < extents[1] - blocking[1]; j += blocking[1]) {
					#ifdef _OPENMP
					#pragma omp task firstprivate(i, j) shared(kernel, kernel_sizes, field, field_sizes, field_lower, field_dst, field_dst_sizes, field_dst_lower, extents, blocking, intensity_max)
					#endif
					{
						const int field_block_lower[2] = {field_lower[0] + i, field_lower[1] + j};
						const int field_dst_block_lower[2] = {field_dst_lower[0] + i, field_dst_lower[1] + j};
						const int block_extents[2] = {extents[0] - i, blocking[1]};

						#if VERBOSITY >= VERBOSITY_BLUR
						printf(TASK_PRINT_FMT
							#ifdef _OPENMP
							, omp_get_thread_num()
							#endif
							, field_block_lower[0], field_block_lower[0] + block_extents[0]
							, field_block_lower[1], field_block_lower[1] + block_extents[1]
							, block_extents[0], block_extents[1]);
						#endif

						blur(kernel, kernel_sizes
							, field, field_sizes, field_block_lower
							, field_dst, field_dst_sizes, field_dst_block_lower
							, block_extents
							, intensity_max);
					}
				}

				if (j < extents[1]) {
					#ifdef _OPENMP
					#pragma omp task firstprivate(i, j) shared(kernel, kernel_sizes, field, field_sizes, field_lower, field_dst, field_dst_sizes, field_dst_lower, extents, blocking, intensity_max)
					#endif
					{
						const int field_block_lower[2] = {field_lower[0] + i, field_lower[1] + j};
						const int field_dst_block_lower[2] = {field_dst_lower[0] + i, field_dst_lower[1] + j};
						const int block_extents[2] = {extents[0] - i, extents[1] - j};

						#if VERBOSITY >= VERBOSITY_BLUR
						printf(TASK_PRINT_FMT
							#ifdef _OPENMP
							, omp_get_thread_num()
							#endif
							, field_block_lower[0], field_block_lower[0] + block_extents[0]
							, field_block_lower[1], field_block_lower[1] + block_extents[1]
							, block_extents[0], block_extents[1]);
						#endif

						blur(kernel, kernel_sizes
							, field, field_sizes, field_block_lower
							, field_dst, field_dst_sizes, field_dst_block_lower
							, block_extents
							, intensity_max);
					}
				}
			}
		}
	}
}
#endif

#endif