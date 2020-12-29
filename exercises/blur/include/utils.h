#ifndef __UTILS_H__
#define __UTILS_H__

void print_rank_prefix(FILE* fp, const int rank, const int* restrict block_coords) {
	fprintf(fp, "rank %d mesh[%d, %d]", rank, block_coords[0], block_coords[1]);
}

void print_usage(const char* restrict program_name) {
	fprintf(stderr, "usage: %s [img_path] [kernel_type] [kernel_diameter] {weighted_kernel_f} {img_output}\n", program_name);
}

inline int imax(const int lhs, const int rhs) {
    return (lhs > rhs ) ? lhs : rhs;
}

inline int imin(const int lhs, const int rhs) {
    return (lhs > rhs ) ? rhs : lhs;
}

inline int iclamp(const int v, const int lower, const int upper) {
    return imin(upper, imax(v, lower));
}

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

inline unsigned short int swap(unsigned short int val) {
	return ((val & (short int) 0xff00) >> 8) + ((val & (short int)0x00ff) << 8);
}

void preprocess_buffer(unsigned short int* field, int field_elems, int pixel_size) {
	if (pixel_size == 1) {
		//widen the char to shorts
		unsigned char* field_reinterpreted = (unsigned char*) field;
		for (int i = field_elems - 1; i > -1; --i) {
			field[i] = (unsigned short int) field_reinterpreted[i];
		}
		//unsigned short int tmp = (unsigned short int) field_reinterpreted[0];
		//field[0] = tmp;
	} else if (pixel_size == 2) {
		if ((0x100 & 0xf) == 0x0) {
			#ifdef _OPENMP
			#pragma omp parallel for shared(field)
			#endif
			for (int i = 0; i < field_elems; ++i) {
				field[i] = swap(field[i]);
			}
		}
	} else {
		assert(0);
	}
}

void postprocess_buffer(unsigned short int* field, int field_elems, int pixel_size) {
	if (pixel_size == 1) {
		//shrink the shorts to chars
		unsigned char* field_reinterpreted = (unsigned char*) field;

		//unsigned char tmp = (unsigned char) field[0];
		//field_reinterpreted[0] = tmp;
		for (int i = 0; i < field_elems; ++i) {
			field_reinterpreted[i] = (unsigned char) field[i];
		}
	} else if (pixel_size == 2) {
		if ((0x100 & 0xf) == 0x0) {
			#ifdef _OPENMP
			#pragma omp parallel for shared(field)
			#endif
			for (int i = 0; i < field_elems; ++i) {
				field[i] = swap(field[i]);
			}
		}
	} else {
		assert(0);
	}
}

void print_thread_provided(const int  mpi_thread_provided, const int omp_max_threads) {
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

int params_from_stdin(const int ndims, int* restrict dim_blocks) {
	assert(dim_blocks);

	for (int i = 0; i < ndims; ++i) {
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
	, const int ndims
	, char** img_path
	, int* kernel_type
	, int* kernel_radiuses
	, double* kernel_params0
	, char** img_save_path) {
	assert(img_path);
	assert(kernel_type);
	assert(kernel_radiuses);
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

		kernel_radiuses[i] = kernel_diameter / 2;
	}
	for (int i = 1; i < ndims; ++i) {
		kernel_radiuses[i] = kernel_radiuses[0];
	}

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

int pgm_open(const char* img_path
	, int* pgm_code
	, int* rows, int* columns
	, int* intensity_max
	, FILE** fp) {
	*fp = fopen(img_path, "rb");
	if (!*fp) {
		return -1;
	}

	char line[256];
	//first line must be P5\n
	char* checkc = fgets(line, sizeof(line), *fp);
	if (!checkc || strncmp(line, "P5", 2)) {
		return -1;
	}
	
	*pgm_code = 5;

	//skip comments
	checkc = fgets(line, sizeof(line), *fp);
	while (!checkc || !strncmp(line, "#", 1)) {
		checkc = fgets(line, sizeof(line), *fp);
	}
	if (!checkc) {
		return -1;
	}
	int nread = sscanf(line, "%d %d\n", columns, rows);
	if (nread != 2) {
		return -1;
	}

	//skip comments
	checkc = fgets(line, sizeof(line), *fp);
	while (!checkc || !strncmp(line, "#", 1)) {
		/*
		if (strlen(line) < 256) {
			checkc = fgets(line, sizeof(line), *fp);
		}
		*/
		checkc = fgets(line, sizeof(line), *fp);
	}
	//TODO check why %hu was not working
	nread = sscanf(line, "%d\n", intensity_max);
	if (nread != 1) {
		return -1;
	}

	return 0;
}

void binomial_coefficients_init(double* coeffs, int size) {
	for (int i = 0; i < size; ++i) {
		coeffs[i] = 1.0;
	}
	if (size < 3) {
		return;
	}
	for (int i = 2; i < size; ++i) {
		for (int j = i - 1; j; --j) {
			coeffs[j] = coeffs[j] + coeffs[j - 1];
		}
	}
}

int kernel_init(const int kernel_type
	, const int* restrict kernel_radiuses
	, const double kernel_params0
	, double* restrict kernel) {
	assert(kernel);

	const int kernel_diameters0 = 2 * kernel_radiuses[0] + 1;
	const int kernel_diameters1 = 2 * kernel_radiuses[1] + 1;
	const int elems = kernel_diameters0 * kernel_diameters1;

	switch (kernel_type) {
		case KERNEL_TYPE_IDENTITY:
			{
				const double w = 1.0 / elems;
				for (int i = 0; i < elems; ++i) {
					kernel[i] = w;
				}
			}
			break;
		case KERNEL_TYPE_WEIGHTED:
			{
				//kernel_params0 used as weight of original pixel
				assert(kernel_params0 >= 0.0 && kernel_params0 <= 1.0);

				const double w = elems > 1 ? ((1.0 - kernel_params0) / (elems - 1)) : 0.0;
				for (int i = 0; i < elems; ++i) {
					kernel[i] = w;
				}
				kernel[elems / 2] = kernel_params0;
			}
			break;
		case KERNEL_TYPE_GAUSSIAN:
			{
				double coeffs0[kernel_diameters0];
				binomial_coefficients_init(coeffs0, kernel_diameters0);
				double coeffs1[kernel_diameters1];
				binomial_coefficients_init(coeffs1, kernel_diameters1);
				
				double coeff_sum[2] = {0, 0};
				for (int i = 0; i < kernel_diameters0; ++i) {
					coeff_sum[0] += coeffs0[i];
				}
				for (int i = 0; i < kernel_diameters1; ++i) {
					coeff_sum[1] += coeffs1[i];
				}

				double norm = coeff_sum[0] * coeff_sum[1];
				for (int i = 0; i < kernel_diameters0; ++i) {
					for (int j = 0; j < kernel_diameters1; ++j) {
						kernel[i * kernel_diameters1 + j] = coeffs0[i] * coeffs1[j];
						//norm += coeffs0[i] * coeffs1[j];
					}
				}

				for (int i = 0; i < elems; ++i) {
					kernel[i] /= norm;
				}
			}
			break;
		default:
			return -1;
	}

	#ifndef NDEBUG
	double norm = 0.0;
	for (int i = 0; i < elems; ++i) {
		norm += kernel[i];
	}
	assert(fabs(norm - 1) < 0.000001);
	#endif

	return 0;
}

int kernel_oneshot(const double* restrict kernel
	, const int* restrict kernel_radiuses
	, const int* restrict kernel_lower
	, const int* restrict kernel_upper
	, const unsigned short int* restrict field
	, const int* restrict field_sizes
	, const int field_i, const int field_j
	, double* restrict output) {
	assert(kernel);
	assert(field);
	assert(output);

	//const int kernel_diameters0 = 2 * kernel_radiuses[0] + 1;
	const int kernel_diameters1 = 2 * kernel_radiuses[1] + 1;
	double tmp = 0.0, norm = 0.0;

	//unrolling
	#define unroll 4
	double tmps0 = 0.0, tmps1 = 0.0, tmps2 = 0.0, tmps3 = 0.0;
	double norms0 = 0.0, norms1 = 0.0, norms2 = 0.0, norms3 = 0.0;

	const int kernel_upper1_unroll = kernel_lower[1] + unroll * ((kernel_upper[1] - kernel_lower[1]) / unroll);

	#ifndef NDEBUG
	int iters = 0;
	#endif

	for (int i = kernel_lower[0]; i < kernel_upper[0]; ++i) {
		register const int kernel_jpos = i * kernel_diameters1;
		register const int field_jpos = (field_i + i - kernel_radiuses[0]) * field_sizes[1] + field_j - kernel_radiuses[1];

		int j;
		for (j = kernel_lower[1]; j < kernel_upper1_unroll; j += unroll) {
			tmps0 += kernel[kernel_jpos + j] * field[field_jpos + j];
			norms0 += kernel[kernel_jpos + j];
			tmps1 += kernel[kernel_jpos + j + 1] * field[field_jpos + j + 1];
			norms1 += kernel[kernel_jpos + j + 1];
			tmps2 += kernel[kernel_jpos + j + 2] * field[field_jpos + j + 2];
			norms2 += kernel[kernel_jpos + j + 2];
			tmps3 += kernel[kernel_jpos + j + 3] * field[field_jpos + j + 3];
			norms3 += kernel[kernel_jpos + j + 3];

			#ifndef NDEBUG
			iters += unroll;
			#endif
		}

		for (; j < kernel_upper[1]; ++j) {
			tmp += kernel[kernel_jpos + j] * field[field_jpos + j];
			norm += kernel[kernel_jpos + j];

			#ifndef NDEBUG
			++iters;
			#endif
		}
	}
	assert(iters == (kernel_upper[0] - kernel_lower[0]) * (kernel_upper[1] - kernel_lower[1]));

	tmp += (tmps0 + tmps1) + (tmps2 + tmps3);
	norm += (norms0 + norms1) + (norms2 + norms3);

	tmp /= norm;
	#ifndef NDEBUG
	/*
	printf("field[%d, %d] = %hu <- %lf / %lf = %lf\n"
		, field_i, field_j, field[field_i * field_sizes[1] + field_j]
		, tmp * norm, norm, tmp);
	*/
	++iters;
	#endif

	*output = tmp;

	return 0;
}

#ifdef SIMD_ON
//SIMD version
typedef double v4df __attribute__ ((vector_size (4 * sizeof(double))));
typedef union {
    v4df V;
    double v[4];
} v4df_u;

int kernel_oneshot_simd(const double* restrict kernel
	, const int* restrict kernel_radiuses
	, const int* restrict kernel_lower
	, const int* restrict kernel_upper
	, const unsigned short int* restrict field
	, const int* restrict field_sizes
	, const int field_i, const int field_j
	, double* restrict output) {
	assert(kernel);
	assert(field);
	assert(output);

	//const int kernel_diameters0 = 2 * kernel_radiuses[0] + 1;
	const int kernel_diameters1 = 2 * kernel_radiuses[1] + 1;
	double tmp = 0.0, norm = 0.0;

	#ifdef _GNU_SOURCE
	v4df tmps = {0, 0, 0, 0};
	v4df norms = {0, 0, 0, 0};
	#else
	v4df tmps = {0};
	v4df norms = {0};
	#endif

	const int kernel_upper1_unroll = kernel_lower[1] + 4 * ((kernel_upper[1] - kernel_lower[1]) / 4);

	#ifndef NDEBUG
	int iters = 0;
	#endif

	for (int i = kernel_lower[0]; i < kernel_upper[0]; ++i) {
		register const int kernel_jpos = i * kernel_diameters1;
		register const int field_jpos = (field_i + i - kernel_radiuses[0]) * field_sizes[1] + field_j - kernel_radiuses[1];

		int j;
		for (j = kernel_lower[1]; j < kernel_upper1_unroll; j += 4) {
			//cannot expect aligned values, needs to explicitly load both kernel elements and field (which are cast from shorts)
			register v4df_u field_cast, kernel_cast;
			field_cast.v[0] = field[field_jpos + j];
			field_cast.v[1] = field[field_jpos + j + 1];
			field_cast.v[2] = field[field_jpos + j + 2];
			field_cast.v[3] = field[field_jpos + j + 3];

			kernel_cast.v[0] = kernel[kernel_jpos + j];
			kernel_cast.v[1] = kernel[kernel_jpos + j + 1];
			kernel_cast.v[2] = kernel[kernel_jpos + j + 2];
			kernel_cast.v[3] = kernel[kernel_jpos + j + 3];

  			tmps += kernel_cast.V * field_cast.V;
			norms += kernel_cast.V;

			#ifndef NDEBUG
			iters += 4;
			#endif
		}

		for (; j < kernel_upper[1]; ++j) {
			tmp += kernel[kernel_jpos + j] * field[field_jpos + j];
			norm += kernel[kernel_jpos + j];

			#ifndef NDEBUG
			++iters;
			#endif
		}
	}
	assert(iters == (kernel_upper[0] - kernel_lower[0]) * (kernel_upper[1] - kernel_lower[1]));

	v4df_u tmps_collect, norms_collect;
	tmps_collect.V = tmps;
	norms_collect.V = norms;

	tmps_collect.v[0] = tmps_collect.v[0] + tmps_collect.v[1];
	tmps_collect.v[2] = tmps_collect.v[2] + tmps_collect.v[3];
	norms_collect.v[0] = norms_collect.v[0] + norms_collect.v[1];
	norms_collect.v[2] = norms_collect.v[2] + norms_collect.v[3];

	tmp += tmps_collect.v[0] + tmps_collect.v[2];
	norm += norms_collect.v[0] + norms_collect.v[2];

	tmp /= norm;

	*output = tmp;

	return 0;
}
#endif

#endif