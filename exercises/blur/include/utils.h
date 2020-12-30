#ifndef __UTILS_H__
#define __UTILS_H__

void print_rank_prefix(FILE* fp, const int rank, const int* restrict block_coords) {
	fprintf(fp, "rank %d mesh[%d, %d]", rank, block_coords[0], block_coords[1]);
}

void print_usage(const char* restrict program_name) {
	fprintf(stderr, "usage: %s [img_path] [kernel_type] [kernel_diameter] {weighted_kernel_f} {img_output}\n", program_name);
}

#define imax(lhs, rhs) (((lhs) > (rhs)) ? (lhs) : (rhs))
#define imin(lhs, rhs) (((lhs) < (rhs)) ? (lhs) : (rhs))
#define iclamp(v, lower, upper) (imin((upper), imax((v), (lower))))
#define swap(v) (((v & (short int) 0xff00) >> 8) + ((v & (short int) 0x00ff) << 8))
/*
int imax(const int lhs, const int rhs) {
	return lhs > rhs ? lhs : rhs;
}
int imin(const int lhs, const int rhs) {
	return lhs < rhs ? lhs : rhs;
}
int iclamp(const int v, const int lower, const int upper) {
	return imin(upper, imax(v, lower));
}
int swap(const unsigned int v) {
	return ((v & (short int) 0xff00) >> 8) + ((v & (short int) 0x00ff) << 8);
}
*/
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
	//TODO check why %hu was not working
	nread = sscanf(line, "%d\n", &meta->intensity_max);
	if (nread != 1) {
		return -1;
	}
	
	meta->header_length_input = ftell(fp);
	fclose(fp);

	return 0;
}

void binomial_coefficients_init(double* coeffs0, int size0, double* coeffs1, int size1) {
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
				double coeffs1[kernel_diameters1];
				binomial_coefficients_init(coeffs0, kernel_diameters0, coeffs1, kernel_diameters1);
				
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
	double tmps[unroll];
	double norms[unroll];
	for (int i = 0; i < unroll; ++i) {
		tmps[i] = 0.0;
		norms[i] = 0.0;
	}

	#define tmps_plus(pos) tmps[(pos)] += kernel[kernel_jpos + j + (pos)] * field[field_jpos + j + (pos)];
	#define norms_plus(pos) norms[(pos)] += kernel[kernel_jpos + j + (pos)];
	#define unrolled_op(pos) tmps_plus((pos)); \
		norms_plus((pos));

	#define unrolled_4block(start) unrolled_op((start) + 0); \
		unrolled_op((start) + 1); \
		unrolled_op((start) + 2); \
		unrolled_op((start) + 3);

	const int kernel_upper1_unroll = kernel_lower[1] + unroll * ((kernel_upper[1] - kernel_lower[1]) / unroll);

	#ifndef NDEBUG
	int iters = 0;
	#endif

	for (int i = kernel_lower[0]; i < kernel_upper[0]; ++i) {
		register const int kernel_jpos = i * kernel_diameters1;
		register const int field_jpos = (field_i + i - kernel_radiuses[0]) * field_sizes[1] + field_j - kernel_radiuses[1];

		int j;
		for (j = kernel_lower[1]; j < kernel_upper1_unroll; j += unroll) {
			unrolled_4block(0);
			//unrolled_op(4);
			//unrolled_op(5);
			//unrolled_4block(4);

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

	for (int i = 0; i < unroll; i += 2) {
		tmp += (tmps[i] + tmps[i + 1]);
		norm += (norms[i] + norms[i + 1]);
	}
	if (unroll % 2) {
		tmp += (tmps[unroll - 1] + tmps[unroll - 1]);
		norm += (norms[unroll - 1] + norms[unroll - 1]);
	}
	
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

void kernel_block(const double* restrict kernel
	, const int* restrict kernel_radiuses
	, const unsigned short int* restrict field
	, const int* restrict field_sizes
	, const int* restrict field_lower
	, const int* restrict field_upper
	, unsigned short int* restrict field_dst
	, const int* restrict field_dst_sizes
	, const int* restrict field_dst_lower) {
	assert(kernel);
	assert(kernel_radiuses);
	assert(field);
	assert(field_lower);
	assert(field_upper);
	assert(field_dst);
	assert(field_dst_sizes);
	assert(field_dst_lower);
	
	int kernel_lower[2];
	int kernel_upper[2];
	for (int i = 0; i < field_upper[0] - field_lower[0]; ++i) {
		for (int j = 0; j < field_upper[1] - field_lower[1]; ++j) {
			const int field_i = field_lower[0] + i;
			const int field_j = field_lower[1] + j;
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
			kernel_oneshot(kernel
				, kernel_radiuses
				, kernel_lower
				, kernel_upper
				, field
				, field_sizes
				, field_i, field_j
				, &intensity_raw);

			unsigned short int intensity = (unsigned short int) round(intensity_raw);
			field_dst[(field_dst_lower[0] + i) * field_dst_sizes[1] + field_dst_lower[1] + j] = intensity;
		}
	}
}

#endif