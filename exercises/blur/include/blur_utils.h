#ifndef __BLUR_UTILS_H__
#define __BLUR_UTILS_H__

#define KERNEL_SUFFIX_MAX_LEN 20

enum {KERNEL_TYPE_IDENTITY, KERNEL_TYPE_WEIGHTED, KERNEL_TYPE_GAUSSIAN, KERNEL_TYPE_UNRECOGNIZED};

int kernel_validate_parameters(const int kernel_type
	, const int kernel_radius
	, const double kernel_params0);

int kernel_init(const int kernel_type, const int kernel_radius
	, const double kernel_params0
	, double* kernel);

int kernel_calc(const double* kernel
	, const int kernel_radius
	, const int kernel_lower0, const int kernel_upper0
	, const int kernel_lower1, const int kernel_upper1
	, const unsigned short int* field
	, const int field_adv0
	, const int field_i, const int field_j
	, double* output);

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
	, unsigned short int* max_intensity);

int field_slice_init(const unsigned short value
	, unsigned short int* field
	, const int field_adv0
	, const int field_lower0, const int field_upper0
	, const int field_lower1, const int field_upper1);

int img_save_path_init(const char* img_path, const int kernel_type
	, const int* block_coords
	, char* img_save_path);

#endif