#include <stdio.h>

#include <math.h>
#include <string.h>

#include <errno.h>
#include <assert.h>

#include <blur_utils.h>

int kernel_validate_parameters(const int kernel_type
	, const int kernel_radius0
	, const int kernel_radius1
	, const double kernel_params0) {
	if (kernel_radius0 < 0) {
		fprintf(stderr, "kernel radius must be >= 0; found %d\n", kernel_radius0);
		errno = -1;
		return -1;
	}
	if (kernel_radius1 < 0) {
		fprintf(stderr, "kernel radius must be >= 0; found %d\n", kernel_radius1);
		errno = -1;
		return -1;
	}
	
	switch (kernel_type) {
		case KERNEL_TYPE_IDENTITY:
			break;
		case KERNEL_TYPE_WEIGHTED:
			//kernel_params0 used as weight of original pixel
			if (kernel_params0 < 0.0 || kernel_params0 > 1.0) {
				fprintf(stderr, "weighted kernel parameter f must be in [0, 1]; found %lf\n", kernel_params0);
				errno = -1;
				return -1;
			}
			break;
		case KERNEL_TYPE_GAUSSIAN:
			//kernel_params0 is used as sigma2
			if (kernel_params0 <= 0.0) {
				fprintf(stderr, "gaussian kernel parameter sigma2 must be >= 0; found %lf\n", kernel_params0);
				errno = -1;
				return -1;
			}
			break;
		default:
			errno = -1;
			return -1;
	}
	
	return 0;
}

int kernel_init(const int kernel_type
	, const int kernel_radius0
	, const int kernel_radius1
	, const double kernel_params0
	, double* restrict kernel) {
	if (!kernel) {
		errno = -1;
		return -1;
	}
	
	const int k0 = 2 * kernel_radius0 + 1;
	const int k1 = 2 * kernel_radius1 + 1;
	const int elems = k0 * k1;

	switch (kernel_type) {
		case KERNEL_TYPE_IDENTITY:
			{
				const double w = 1.0 / elems;
				#ifndef NDEBUG
				printf("creating identity kernel; elements = %lf\n", w);
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
				//kernel_params0 is used as sigma2
				assert(kernel_params0 > 0.0);
				#ifndef NDEBUG
				printf("creating gaussian kernel; sigma2 = %lf\n", kernel_params0);
				#endif
				
				for (int i = 0; i < kernel_radius0 + 1; ++i) {
					const int jpos = i * kernel_radius1;
					const double x = i - kernel_radius0 - 1;

					//init upper left quarter, copy in the others
					for (int j = 0; j < kernel_radius1 + 1; ++j) {
						const double y = j - kernel_radius1 - 1;

						kernel[jpos + j] = exp(-(x * x + y * y) / (2 * kernel_params0)) / (2 * M_PI * kernel_params0);
					}
					//upper right
					for (int j = 0; j < kernel_radius1; ++j) {
						kernel[jpos + kernel_radius1 + 1 + j] = kernel[jpos + kernel_radius1 - 1 - j];
						assert(jpos + kernel_radius1 + 1 + j < k * k);
					}
				}

				for (int i = 0; i < kernel_radius0; ++i) {
					const int jpos_read = (kernel_radius0 - 1 - i) * kernel_radius1;
					const int jpos_write = (kernel_radius0 + 1 + i) * kernel_radius1;

					for (int j = 0; j < kernel_radius1; ++j) {
						//lower half
						kernel[jpos_write + j] = kernel[jpos_read + j];
						assert(jpos_write + j < elems);
					}
				}
			}
			break;
		default:
			errno = -1;
			return -1;
	}

	double norm = 0.0;
	for (int i = 0; i < elems; ++i) {
		norm += kernel[i];
	}
	for (int i = 0; i < elems; ++i) {
		kernel[i] /= norm;
	}

	return 0;
}

int kernel_oneshot(const double* kernel
	, const int kernel_radius0
	, const int kernel_radius1
	, const int kernel_lower0, const int kernel_upper0
	, const int kernel_lower1, const int kernel_upper1
	, const unsigned short int* restrict field
	, const int field_adv0
	, const int field_i, const int field_j
	, double* restrict output) {
	assert(kernel);
	assert(field);
	assert(output);

	//validate something?
	//assert((field_src_upper0 - field_src_lower0) == (field_dst_upper0 - field_dst_lower0));
	//assert((field_src_upper1 - field_src_lower1) == (field_dst_upper1 - field_dst_lower1));

	const int k0 = 2 * kernel_radius0 + 1;
	const int k1 = 2 * kernel_radius1 + 1;
	//const int elems = k0 * k1;
	
	double tmp = 0.0, norm = 0.0;

	for (int i = kernel_lower0; i < kernel_upper0; ++i) {
		const int field_ii = field_i + i - kernel_radius0;
		
		for (int j = kernel_lower1; j < kernel_upper1; ++j) {
			tmp += kernel[i * k1 + j] * field[field_ii * field_adv0 + field_j + j - kernel_radius1];
			/*
			printf("kernel[%d, %d] * field[%d + %d, %d + %d]\n"
				, i, j
				, field_i, i - kernel_radius
				, field_j, j - kernel_radius);
			*/
			norm += kernel[i * k1 + j];
		}
	}

	if ((kernel_upper0 - kernel_lower0) < k0
		|| (kernel_upper1 - kernel_lower1) < k1) {
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

int kernel_apply_to_slice(const double* kernel
	, const int kernel_radius0
	, const int kernel_radius1
	, const unsigned short int* restrict field_src
	, const int field_src_rows, const int field_src_columns
	, const int field_src_lower0, const int field_src_upper0
	, const int field_src_lower1, const int field_src_upper1
	, unsigned short int* restrict field_dst
	, const int field_dst_adv0
	, const int field_dst_lower0, const int field_dst_upper0
	, const int field_dst_lower1, const int field_dst_upper1
	, unsigned short int* restrict max_intensity) {
	assert(kernel);
	assert(field_src);
	assert(field_dst);

	assert((field_src_upper0 - field_src_lower0) == (field_dst_upper0 - field_dst_lower0));
	assert((field_src_upper1 - field_src_lower1) == (field_dst_upper1 - field_dst_lower1));
	
	const int k0 = 2 * kernel_radius0 + 1;
	const int k1 = 2 * kernel_radius1 + 1;
	//const int elems = k0 * k1;
	
	#ifndef NDEBUG
	printf("kernel_apply_to_slice: field[%d:%d, %d:%d] <- kernel(img[%d:%d, %d:%d])\n"
		, field_dst_lower0, field_dst_upper0
		, field_dst_lower1, field_dst_upper1
		, field_src_lower0, field_src_upper0
		, field_src_lower1, field_src_upper1);
	#endif

	for (int i = 0; i < field_dst_upper0 - field_dst_lower0; ++i) {
		const int src_i = field_src_lower0 + i;
		const int dst_i = field_dst_lower0 + i;
		
		for (int j = 0; j < field_dst_upper1 - field_dst_lower1; ++j) {
			const int src_j = field_src_lower1 + j;
			const int dst_j = field_dst_lower1 + j;
			
			int kernel_lower0 = src_i >= kernel_radius0 ? 0 : kernel_radius0 - src_i;
			int kernel_upper0 = (src_i + kernel_radius0) < field_src_rows ? k0 : (kernel_radius0 + field_src_rows - src_i);
			int kernel_lower1 = src_j >= kernel_radius1 ? 0 : kernel_radius1 - src_j;
			int kernel_upper1 = (src_j + kernel_radius1 < field_src_columns) ? k1 : (kernel_radius1 + field_src_columns - src_j);
			
			/*
			if (src_i + kernel_radius0 > field_src_rows
				|| src_j + kernel_radius1 > field_src_columns) {
				printf("field[%d, %d] requires slicing the kernel with [%d:%d, %d:%d]\n"
					, src_i, src_j
					, kernel_lower0, kernel_upper0
					, kernel_lower1, kernel_upper1);
			}
			*/

			double intensity_raw;
			kernel_oneshot(kernel
				, kernel_radius0
				, kernel_radius1
				, kernel_lower0, kernel_upper0
				, kernel_lower1, kernel_upper1
				, field_src
				, field_src_columns
				, src_i, src_j
				, &intensity_raw);
			
			unsigned short int intensity = (unsigned short int) fmax(0, intensity_raw);
			field_dst[dst_i * field_dst_adv0 + dst_j] = intensity;
			
			if (intensity > 32000) {
				printf("broken intensity at pos %d, %d; %hu <- %lf\n"
					, dst_i, dst_j
					, intensity, intensity_raw);
				//exit(0);
			}
						
			*max_intensity = *max_intensity < intensity ? intensity : *max_intensity;
		}
	}

	return 0;
}

int field_slice_init(const unsigned short value
	, unsigned short int* restrict field
	, const int field_adv0
	, const int field_lower0, const int field_upper0
	, const int field_lower1, const int field_upper1) {
	assert(field);

	for (int i = field_lower0; i < field_upper0; ++i) {
		for (int j = field_lower1; j < field_upper1; ++j) {
			field[i * field_adv0 + j] = value;
		}
	}

	return 0;
}

int img_save_path_init(const char* restrict img_path
	, const int kernel_type
	, const int* restrict block_coords
	, char* restrict img_save_path) {
	const int img_path_len = strlen(img_path);
	char* suffix;
	switch (kernel_type) {
		case KERNEL_TYPE_IDENTITY:
			suffix = block_coords ? "_identity_%03d_%03d" : "_identity";
			break;
		case KERNEL_TYPE_WEIGHTED:
			suffix = block_coords ? "_weighted_%03d_%03d" : "_weighted";
			break;
		case KERNEL_TYPE_GAUSSIAN:
			suffix = block_coords ? "_gaussian_%03d_%03d" : "_gaussian";
			break;
		default:
			errno = -1;
			return -1;
	}
	const int suffix_len = strlen(suffix);

	//find idx of extension
	int img_path_ext = img_path_len - 1;
	while (img_path[img_path_ext] != '.') {
		--img_path_ext;
	}

	if (!img_save_path) {
		errno = -1;
		return -1;
	}

	memcpy(img_save_path, img_path, img_path_ext);

	int res;
	if (block_coords) {
		res = snprintf(img_save_path + img_path_ext, suffix_len + 1, suffix, block_coords[0], block_coords[1]);
	} else {
		//res = snprintf(*img_save_path + img_path_ext, suffix_len + 1, "%s", suffix);
		memcpy(img_save_path + img_path_ext, suffix, suffix_len + 1);
	}

	if (res < 0) {
		errno = -1;
		return -1;
	}

	strcat(img_save_path, img_path + img_path_ext);

	return 0;
}