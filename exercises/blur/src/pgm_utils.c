#include <stdlib.h>

#include <math.h>
#include <string.h>

#include <errno.h>

#include <pgm_utils.h>

int pgm_get_header_info(const char* restrict img_path
	, int* restrict rows, int* restrict columns
	, unsigned short int* restrict intensity_max
	, int* restrict pixel_size
	, long int* restrict offset_header_orig) {
	FILE* fp = fopen(img_path, "rb");
	if (!fp) {
		errno = -1;
		return -1;
	}
	
	*rows = *columns = *intensity_max = *offset_header_orig = -1;

	//char MagicN[2];
	char* line = NULL;
	size_t k = 0;//, n = 0;
	
	if (k > 0) {
		k = sscanf(line, "%d%*c%d%*c%hu%*c", columns, rows, intensity_max);
		if (k < 3) {
			fscanf(fp, "%hu%*c", intensity_max);
		}
	} else {
		*intensity_max = 0;         // this is the signal that there was an I/O error
				// while reading the image header
		free(line);
		return -1;
	}
	
	free(line);
	
	*offset_header_orig = ftell(fp);
	*pixel_size = 1 + (*intensity_max > 255);
	//int color_depth = 1 + (*intensity_max > 255);
	//unsigned int size = *columns * *rows * intensity_max;
	
	fclose(fp);

	return 0;
}

int pgm_open(const char* restrict img_path
	, int* rows, int* columns
	, unsigned short int* restrict intensity_max
	, FILE** fp) {
	*fp = fopen(img_path, "rb");
	if (!*fp) {
		errno = -1;
		return -1;
	}

	char line[256];
	//first line must be P5\n
	char* checkc = fgets(line, sizeof(line), *fp);
	if (!checkc || strncmp(line, "P5", 2)) {
		errno = -1;
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
		errno = -1;
		return -1;
	}
	int nread = sscanf(line, "%d %d\n", columns, rows);
	if (nread != 2) {
		errno = -1;
		return -1;
	}

	//next line must be intensity_max
	checkc = fgets(line, sizeof(line), *fp);
	if (!checkc) {
		errno = -1;
		return -1;
	}
	nread = sscanf(line, "%hu\n", intensity_max);
	if (nread != 1) {
		errno = -1;
		return -1;
	}

	return 0;
}

//field[field_lower0:field_upper0, field_lower1:field_upper1] = img[img_lower0:img_upper0, img_lower1:img_upper1]
int pgm_load_image_slice_into_field_slice(const char* restrict img_path
	, const int img_lower0, const int img_upper0
	, const int img_lower1, const int img_upper1
	, unsigned short int* restrict field
	, const int field_adv0
	, const int field_lower0, const int field_upper0
	, const int field_lower1, const int field_upper1) {
	if ((img_upper0 - img_lower0) != (field_upper0 - field_lower0)
		|| (img_upper1 - img_lower1) != (field_upper1 - field_lower1)
		|| field_adv0 < (field_upper1 - field_lower1)) {
		errno = -1;
		return -1;
	}

	#ifndef NDEBUG
	printf("field[%d:%d, %d:%d] <- %s[%d:%d, %d:%d]\n"
		, field_lower0, field_upper0
		, field_lower1, field_upper1
		, img_path
		, img_lower0, img_upper0
		, img_lower1, img_upper1);
	#endif

	int columns, rows;
	unsigned short int intensity_max;
	FILE* fp;
	int check = pgm_open(img_path, &rows, &columns, &intensity_max, &fp);
	if (check) {
		return check;
	}

	if (!fp) {
		errno = -1;
		return -1;
	}

	int px_size = intensity_max > 255 ? sizeof(unsigned short int) : sizeof(unsigned char);
	//skip img_lower0 rows
	fseek(fp, img_lower0 * columns * px_size, SEEK_CUR);

	for (int i = field_lower0; i < field_upper0; ++i) {
		const int jpos = i * field_adv0;
		//skip first img_lower1 columns
		fseek(fp, img_lower1 * px_size, SEEK_CUR);

		size_t nread = 0;
		if (intensity_max > 255) {
			nread = fread(field + jpos + field_lower1, px_size, field_upper1 - field_lower1, fp);
			if (nread != (size_t) (field_upper1 - field_lower1)) {
				errno = -1;
				return -1;
			}
		} else {
			for (int j = field_lower1; j < field_upper1; ++j) {
				field[jpos + j] = (unsigned short int) getc(fp);
			}
		}

		//skip last columns - img_upper1 columns
		fseek(fp, fmax(0, columns - img_upper1) * px_size, SEEK_CUR);
	}

	fclose(fp);

	return 0;
}

int pgm_save_field_slice(const char* restrict img_path
	, const unsigned short int* restrict field
	, const short int intensity_max
	, const int field_adv0
	, const int field_lower0, const int field_upper0
	, const int field_lower1, const int field_upper1) {
	#ifndef NDEBUG
	printf("%s[0:%d, 0:%d] <- field[%d:%d, %d:%d] intensity_max %hu\n"
		, img_path
		, field_upper0 - field_lower0
		, field_upper1 - field_lower1
		, field_lower0, field_upper0
		, field_lower1, field_upper1
		, intensity_max);
	#endif

	FILE* fp = fopen(img_path, "wb");
	if (!fp) {
		errno = -1;
		return -1;
	}

	int nprint = fprintf(fp, "P5\n%d %d\n%hu\n", field_upper1 - field_lower1, field_upper0 - field_lower0, intensity_max);
	if (nprint < 0) {
		errno = -1;
		return -1;
	}

	if (intensity_max > 256) {
		for (int i = field_lower0; i < field_upper0; ++i) {
			size_t nwrite = fwrite(field + i * field_adv0 + field_lower1, sizeof(short int), field_upper1 - field_lower1, fp);
			if (nwrite != (size_t) (field_upper1 - field_lower1)) {
				errno = -1;
				return -1;
			}
		}
	} else {
		for (int i = field_lower0; i < field_upper0; ++i) {
			const int jpos = i * field_adv0;
			for (int j = field_lower1; j < field_upper1; ++j) {
				int checkc = putc(field[jpos + j], fp);
				if (checkc != (unsigned char) field[jpos + j]) {
					errno = -1;
					return -1;
				}
			}
		}
	}

	fclose(fp);

	return 0;
}