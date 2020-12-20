#define KERNEL_SUFFIX_MAX_LEN 20

enum {KERNEL_TYPE_IDENTITY, KERNEL_TYPE_WEIGHTED, KERNEL_TYPE_GAUSSIAN, KERNEL_TYPE_UNRECOGNIZED};

void print_usage(const char* program_name) {
	fprintf(stderr, "usage: %s [img_path] [kernel_type] [kernel_diameter] {weighted_kernel_f} {img_output}\n", program_name);
}

int params_from_stdin(const int dims, int* restrict dim_blocks) {
	assert(dim_blocks);
	
	//stdin params
	for (int i = 0; i < dims; ++i) {
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
	, const int dims
	, char** img_path
	, int* kernel_type
	, int* kernel_diameters
	, int* kernel_radiuses
	, double* kernel_params0
	, char** img_save_path) {
	assert(img_path);
	assert(kernel_type);
	assert(kernel_diameters);
	assert(kernel_radiuses);
	assert(kernel_params0);
	assert(img_save_path);
	
	if (argc < 3) {
		print_usage(argv[0]);
		return -1;
	}
	
	int param_idx = 0;
	*img_path = argv[++param_idx];
	
	//char* end;
	*kernel_type = (int) strtol(argv[++param_idx], NULL, 10);
	if (*kernel_type < 0 || *kernel_type >= KERNEL_TYPE_UNRECOGNIZED) {
		fprintf(stderr, "could not parse kernel_type\n");
		return -1;
	}
	
	//keep it 1d
	for (int i = 0; i < 1; ++i) {
		kernel_diameters[i] = atoi(argv[++param_idx]);
		/*
		if (errno) {
			fprintf(stderr, "could not parse kernel_diameter\n");
			return -1;
		} else 
		*/
		if (kernel_diameters[i] < 0 || kernel_diameters[i] % 2 != 1) {
			print_usage(argv[0]);
			fprintf(stderr, "kernel_diameter must be an odd positive integer\n");
			return -1;
		}
		
		kernel_radiuses[i] = kernel_diameters[i] / 2;
	}
	//propagate first value to the other
	for (int i = 1; i < dims; ++i) {
		kernel_diameters[i] = kernel_diameters[0];
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
		/*
		if (errno) {
			fprintf(stderr, "could not parse weighted_kernel_f\n");
			return -1;
		}
		*/
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
	, unsigned short int* intensity_max
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
	//next line can be a comment starting with #
	//next line must be columns rows
	checkc = fgets(line, sizeof(line), *fp);
	if (!checkc || !strncmp(line, "#", 1)) {
		if (strlen(line) < 256) {
			checkc = fgets(line, sizeof(line), *fp);
		}
	}
	if (!checkc) {
		return -1;
	}
	int nread = sscanf(line, "%d %d\n", columns, rows);
	if (nread != 2) {
		return -1;
	}

	//next line must be intensity_max
	checkc = fgets(line, sizeof(line), *fp);
	if (!checkc) {
		return -1;
	}
	nread = sscanf(line, "%hu\n", intensity_max);
	if (nread != 1) {
		return -1;
	}

	return 0;
}

int kernel_init(const int kernel_type
	, const int* restrict kernel_radiuses
	, const double kernel_params0
	, double* restrict kernel) {
	assert(kernel);
	
	const int k0 = 2 * kernel_radiuses[0] + 1;
	const int k1 = 2 * kernel_radiuses[1] + 1;
	const int elems = k0 * k1;

	switch (kernel_type) {
		case KERNEL_TYPE_IDENTITY:
			{
				const double w = 1.0 / elems;
				#ifndef NDEBUG
				printf("creating identity kernel; w: %lf\n", w);
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
				
				const double w = elems > 1 ? (1.0 - kernel_params0) / (elems - 1) : 0.0;
				#ifndef NDEBUG
				printf("creating weighted kernel; center = %lf, other = %lf\n", kernel_params0, w);
				#endif
				for (int i = 0; i < elems; ++i) {
					kernel[i] = w;
				}
				kernel[elems / 2] = kernel_params0;
			}
			break;
		case KERNEL_TYPE_GAUSSIAN:
			{
				const double sigma2 = kernel_radiuses[0] * kernel_radiuses[1];
				#ifndef NDEBUG
				printf("creating gaussian kernel; sigma2 %lf\n", sigma2);
				#endif
				
				double norm = 0.0;
				for (int i = 0; i < kernel_radiuses[0] + 1; ++i) {
					const int jpos = i * k1;
					const double x = i - kernel_radiuses[0] - 1;

					//init upper left quarter, copy in the others
					for (int j = 0; j < kernel_radiuses[1] + 1; ++j) {
						const double y = j - kernel_radiuses[1] - 1;

						kernel[jpos + j] = exp(-(x * x + y * y) / (2 * sigma2)) / (2 * M_PI * sigma2);
						norm += kernel[jpos + j];
					}
					//upper right
					for (int j = 0; j < kernel_radiuses[1]; ++j) {
						kernel[jpos + kernel_radiuses[0] + 1 + j] = kernel[jpos + kernel_radiuses[0] - 1 - j];
						assert(jpos + kernel_radiuses[0] + 1 + j < elems);
						norm += kernel[jpos + kernel_radiuses[0] + 1 + j];
					}
				}
				
				for (int i = 0; i < kernel_radiuses[0]; ++i) {
					const int jpos_read = (kernel_radiuses[0] - 1 - i) * k1;
					const int jpos_write = (kernel_radiuses[0] + 1 + i) * k1;

					for (int j = 0; j < k1; ++j) {
						//lower half
						kernel[jpos_write + j] = kernel[jpos_read + j];
						assert(jpos_write + j < elems);
						norm += kernel[jpos_write + j];
					}
				}
				
				/*
				#ifndef NDEBUG
				printf("raw gaussian kernel has norm %lf\n", norm);
				#endif
				*/
				
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

	const int k0 = 2 * kernel_radiuses[0] + 1;
	const int k1 = 2 * kernel_radiuses[1] + 1;	
	double tmp = 0.0, norm = 0.0;

	for (int i = kernel_lower[0]; i < kernel_upper[0]; ++i) {
		const int field_ii = field_i + i - kernel_radiuses[0];
		
		for (int j = kernel_lower[1]; j < kernel_upper[1]; ++j) {
			tmp += kernel[i * k1 + j] * field[field_ii * field_sizes[1] + field_j + j - kernel_radiuses[1]];
			norm += kernel[i * k1 + j];
		}
	}

	if ((kernel_upper[0] - kernel_lower[0]) < k0
		|| (kernel_upper[1] - kernel_lower[1]) < k1) {
		#ifndef NDEBUG
		if (tmp / norm > 32000) {
			printf("renormalizing pos %d, %d: %lf -> %lf\n"
				, field_i, field_j
				, tmp, tmp / norm);
		}
		#endif
		tmp /= norm;
	} else {
		assert(fabs(norm - 1) < 0.001);
	}
	
	*output = tmp;

	return 0;
}

void print_rank_prefix(const int rank, const int* block_coords) {
	printf("rank %d(%d, %d)", rank, block_coords[0], block_coords[1]);
}