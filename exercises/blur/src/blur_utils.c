#include <stdlib.h>
#include <unistd.h>
#include <stdio.h>

#include <math.h>
#include <string.h>

#include <errno.h>
#include <assert.h>

#include <blur_utils.h>

int kernel_validate_parameters(const int kernel_type
	, const int kernel_radius
	, const double kernel_params0) {
	if (kernel_radius < 0) {
		fprintf(stderr, "kernel radius must be >= 0; found %d\n", kernel_radius);
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

int kernel_init(const int kernel_type, const int kernel_radius
	, const double kernel_params0
	, double* kernel) {
	const int k = 2 * kernel_radius + 1;
	if (!kernel) {
		errno = -1;
		return -1;
	}

	switch (kernel_type) {
		case KERNEL_TYPE_IDENTITY:
			{
				const double w = 1.0 / (k * k);
				#ifndef NDEBUG
				printf("creating identity kernel; elements = %lf\n", w);
				#endif
				for (int i = 0; i < k * k; ++i) {
					kernel[i] = w;
				}
			}
			break;
		case KERNEL_TYPE_WEIGHTED:
			{
				//kernel_params0 used as weight of original pixel
				assert(kernel_params0 >= 0.0 && kernel_params0 <= 1.0);
				
				const double w = k > 1 ? (1.0 - kernel_params0) / (k * k - 1) : 0.0;
				#ifndef NDEBUG
				printf("creating weighted kernel; center = %lf, other = %lf\n", kernel_params0, w);
				#endif
				for (int i = 0; i < k * k; ++i) {
					kernel[i] = w;
				}
				kernel[(k * k) / 2] = kernel_params0;
			}
			break;
		case KERNEL_TYPE_GAUSSIAN:
			{
				//kernel_params0 is used as sigma2
				assert(kernel_params0 > 0.0);
				#ifndef NDEBUG
				printf("creating gaussian kernel; sigma2 = %lf\n", kernel_params0);
				#endif
				
				for (int i = 0; i < kernel_radius + 1; ++i) {
					const int jpos = i * k;
					const double x = i - kernel_radius - 1;

					//init upper left quarter, copy in the others
					for (int j = 0; j < kernel_radius + 1; ++j) {
						const double y = j - kernel_radius - 1;

						kernel[jpos + j] = exp(-(x * x + y * y) / (2 * kernel_params0)) / (2 * M_PI * kernel_params0);
					}
					//upper right
					for (int j = 0; j < kernel_radius; ++j) {
						kernel[jpos + kernel_radius + 1 + j] = kernel[jpos + kernel_radius - 1 - j];
						assert(jpos + kernel_radius + 1 + j < k * k);
					}
				}

				for (int i = 0; i < kernel_radius; ++i) {
					const int jpos_read = (kernel_radius - 1 - i) * k;
					const int jpos_write = (kernel_radius + 1 + i) * k;

					for (int j = 0; j < k; ++j) {
						//lower half
						kernel[jpos_write + j] = kernel[jpos_read + j];
						assert(jpos_write + j < k * k);
					}
				}
			}
			break;
		default:
			errno = -1;
			return -1;
	}

	double norm = 0.0;
	for (int i = 0; i < k * k; ++i) {
		norm += kernel[i];
	}
	for (int i = 0; i < k * k; ++i) {
		kernel[i] /= norm;
	}

	return 0;
}

int kernel_calc(const double* kernel
	, const int kernel_radius
	, const int kernel_lower0, const int kernel_upper0
	, const int kernel_lower1, const int kernel_upper1
	, const unsigned short int* field
	, const int field_adv0
	, const int field_i, const int field_j
	, double* output) {
	assert(kernel);
	assert(field);
	assert(output);

	//validate something?
	//assert((field_src_upper0 - field_src_lower0) == (field_dst_upper0 - field_dst_lower0));
	//assert((field_src_upper1 - field_src_lower1) == (field_dst_upper1 - field_dst_lower1));

	const int k = 2 * kernel_radius + 1;
	double tmp = 0.0, norm = 0.0;

	for (int i = kernel_lower0; i < kernel_upper0; ++i) {
		for (int j = kernel_lower1; j < kernel_upper1; ++j) {
			tmp += kernel[i * k + j] * field[(field_i + i - kernel_radius) * field_adv0 + field_j + j - kernel_radius];
			/*
			printf("kernel[%d, %d] * field[%d + %d, %d + %d]\n"
				, i, j
				, field_i, i - kernel_radius
				, field_j, j - kernel_radius);
			*/
			norm += kernel[i * k + j];
		}
	}

	if ((kernel_upper0 - kernel_lower0) < k
		|| (kernel_upper1 - kernel_lower1) < k) {
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
	, const int kernel_radius
	, const unsigned short int* field_src
	, const int field_src_rows, const int field_src_columns
	, const int field_src_lower0, const int field_src_upper0
	, const int field_src_lower1, const int field_src_upper1
	, unsigned short int* field_dst
	, const int field_dst_adv0
	, const int field_dst_lower0, const int field_dst_upper0
	, const int field_dst_lower1, const int field_dst_upper1
	, unsigned short int* max_intensity) {
	assert(kernel);
	assert(field_src);
	assert(field_dst);

	assert((field_src_upper0 - field_src_lower0) == (field_dst_upper0 - field_dst_lower0));
	assert((field_src_upper1 - field_src_lower1) == (field_dst_upper1 - field_dst_lower1));
	
	const int k = 2 * kernel_radius + 1;
	
	#ifndef NDEBUG
	printf("kernel_apply_to_slice: field[%d:%d, %d:%d] <- kernel(img[%d:%d, %d:%d])\n"
		, field_dst_lower0, field_dst_upper0
		, field_dst_lower1, field_dst_upper1
		, field_src_lower0, field_src_upper0
		, field_src_lower1, field_src_upper1);
	#endif

	for (int i = 0; i < field_dst_upper0 - field_dst_lower0; ++i) {
		const int src_i = field_src_lower0 + i;
		
		for (int j = 0; j < field_dst_upper1 - field_dst_lower1; ++j) {
			const int src_j = field_src_lower1 + j;
			
			int kernel_lower0 = src_i >= kernel_radius ? 0 : kernel_radius - src_i;
			int kernel_upper0 = (src_i + kernel_radius) < field_src_rows ? k : (kernel_radius + field_src_rows - src_i);
			int kernel_lower1 = src_j >= kernel_radius ? 0 : kernel_radius - src_j;
			int kernel_upper1 = (src_j + kernel_radius < field_src_columns) ? k : (kernel_radius + field_src_columns - src_j);
			
			/*
			if (src_i + kernel_radius > field_src_rows
				|| src_j + kernel_radius > field_src_columns) {
				printf("field[%d, %d] requires slicing the kernel with [%d:%d, %d:%d]\n"
					, src_i, src_j
					, kernel_lower0, kernel_upper0
					, kernel_lower1, kernel_upper1);
			}
			*/

			double intensity_raw;
			kernel_calc(kernel
				, kernel_radius
				, kernel_lower0, kernel_upper0
				, kernel_lower1, kernel_upper1
				//, 0, k, 0, k
				, field_src
				, field_src_columns
				, src_i, src_j
				, &intensity_raw);
			
			unsigned short int intensity = (unsigned short int) fmax(0, intensity_raw);
			field_dst[(field_dst_lower0 + i) * field_dst_adv0 + field_dst_lower1 + j] = intensity;
			
			if (intensity > 32000) {
				printf("broken intensity at pos %d, %d; %hu <- %lf\n"
					, field_dst_lower0 + i, field_dst_lower1 + j
					, intensity, intensity_raw);
				//exit(0);
			}
						
			*max_intensity = *max_intensity < intensity ? intensity : *max_intensity;
		}
	}

	return 0;
}

int field_slice_init(const unsigned short value
	, unsigned short int* field
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

int img_save_path_init(const char* img_path, const int kernel_type
	, const int* block_coords
	, char* img_save_path) {
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