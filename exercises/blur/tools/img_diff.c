#include <string.h>

#include <pgm_utils.h>

int main(int argc, char **argv) {
	if (argc < 3) {
		printf("usage: %s image1 image2\n", argv[0]);
		exit(1);
	}

    int dims1[2] = {0, 0};
    int dims2[2] = {0, 0};
    int maxval1, maxval2 = 0;
	void *ptr1, *ptr2;

    read_pgm_image(&ptr1, &maxval1, dims1 + 1, dims1, argv[1]);
	if (maxval1 < 0) {
		fprintf(stderr, "error while reading %s\n", argv[1]);
		exit(1);
	}
    read_pgm_image(&ptr2, &maxval2, dims2 + 1, dims2, argv[2]);
	if (maxval2 < 0) {
		fprintf(stderr, "error while reading %s\n", argv[2]);
		exit(1);
	}
	
	if (dims1[0] != dims2[0]) {
		printf("images have different numbers of rows: %d vs %d\n", dims1[0], dims2[0]);
		exit(0);
	}
	if (dims1[1] != dims2[1]) {
		printf("images have different numbers of columns: %d vs %d\n", dims1[1], dims2[1]);
		exit(0);
	}
	if (maxval1 != maxval2) {
		printf("images have different maxval: %d vs %d\n", maxval1, maxval2);
		exit(0);
	}
	
	int pixel_size = 1 + (maxval1 > 255);
	void* diff = NULL;
	if (argc > 3) {
		diff = malloc(pixel_size * dims1[0] * dims1[1]);
	}
	
	swap_image(ptr1, dims1[1], dims1[0], maxval1);
	swap_image(ptr2, dims2[1], dims2[0], maxval2);

	int diff_max = 0, diff_count = 0;
	for (int i = 0; i < dims1[0]; ++i) {
		for (int j = 0; j < dims1[1]; ++j) {
			if (pixel_size == 1) {
				unsigned char a = ((unsigned char*) ptr1)[i * dims1[1] + j];
				unsigned char b = ((unsigned char*) ptr2)[i * dims1[1] + j];
				if (a != b) {
					unsigned char d = (a > b ? a - b : b - a);
					printf("difference at pixel (%d %d): %c vs %c diff %c\n", i, j, a, b, d);
					((unsigned short int*) diff)[i * dims1[1] + j] = d;
					diff_max = ((int) d > diff_max) ? d : diff_max;
					++diff_count;
				} else if (diff) {
					((unsigned short int*) diff)[i * dims1[1] + j] = 0;
				}
			} else {
				unsigned short int a = ((unsigned short int*) ptr1)[i * dims1[1] + j];
				unsigned short int b = ((unsigned short int*) ptr2)[i * dims1[1] + j];
				if (a != b) {
					unsigned short int d = (a > b ? a - b : b - a);
					printf("difference at pixel (%d %d): %hu vs %hu diff %hu\n", i, j, a, b, d);
					((unsigned short int*) diff)[i * dims1[1] + j] = d;
					diff_max = ((int) d > diff_max) ? d : diff_max;
					++diff_count;
				} else if (diff) {
					((unsigned short int*) diff)[i * dims1[1] + j] = 0;
				}
			}
		}
	}
	
	printf("different pixels: %d out of %d (%lf) diff_max: %d\n", diff_count, dims1[0] * dims1[1], diff_count / (double) (dims1[0] * dims1[1]), diff_max);
	
	if (diff) {
		if ((diff_max > 255) != (maxval1 > 255)) {
			//shrink
			unsigned char* diff2 = (unsigned char*) malloc(dims1[0] * dims1[1]);
			for (int i = 0; i < dims1[0] * dims1[1]; ++i) {
				diff2[i] = (unsigned char) ((unsigned short int*) diff)[i];
			}
			
    		write_pgm_image(diff2, diff_max, dims1[1], dims1[0], argv[3]);
			free(diff2);
		} else {
    		write_pgm_image(diff, diff_max, dims1[1], dims1[0], argv[3]);
		}
		free(diff);
	}
	free(ptr2);
	free(ptr1);

    return 0;
} 



