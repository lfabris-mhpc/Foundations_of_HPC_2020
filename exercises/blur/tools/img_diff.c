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
    read_pgm_image(&ptr2, &maxval2, dims2 + 1, dims2, argv[2]);
	
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

	for (int i = 0; i < dims1[0]; ++i) {
		for (int j = 0; j < dims1[1]; ++j) {
			if (pixel_size == 1) {
				unsigned char a = ((unsigned char*) ptr1)[i * dims1[1] + j];
				unsigned char b = ((unsigned char*) ptr2)[i * dims1[1] + j];
				if (a != b) {
					printf("difference at pixel %d %d: %c vs %c\n", i, j, a, b);
				}
			} else {
				unsigned short int a = ((unsigned short int*) ptr1)[i * dims1[1] + j];
				unsigned short int b = ((unsigned short int*) ptr2)[i * dims1[1] + j];
				if (a != b) {
					printf("difference at pixel %d %d: %hu vs %hu\n", i, j, a, b);
				}
				
			}
		}
	}

	free(ptr2);
	free(ptr1);

    return 0;
} 



