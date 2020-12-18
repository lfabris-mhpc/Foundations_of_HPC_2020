#ifndef __PGM_UTILS_H__
#define __PGM_UTILS_H__

#include <stdio.h>

int pgm_get_header_info(const char* restrict img_path
	, int* restrict rows, int* restrict columns
	, unsigned short int* restrict intensity_max
	, int* restrict pixel_size
	, long int* offset_header_orig);

int pgm_open(const char* restrict img_path
	, int* restrict rows, int* restrict columns
	, unsigned short int* restrict intensity_max
	, FILE** fp);

//field[field_lower0:field_upper0, field_lower1:field_upper1] <- img[img_lower0:img_upper0, img_lower1:img_upper1]
int pgm_load_image_slice_into_field_slice(const char* restrict img_path
	, const int img_lower0, const int img_upper0
	, const int img_lower1, const int img_upper1
	, unsigned short int* restrict field
	, const int field_adv0
	, const int field_lower0, const int field_upper0
	, const int field_lower1, const int field_upper1);

int pgm_save_field_slice(const char* restrict img_path
	, const unsigned short int* restrict field
	, const short int intensity_max
	, const int field_adv0
	, const int field_lower0, const int field_upper0
	, const int field_lower1, const int field_upper1);

#endif