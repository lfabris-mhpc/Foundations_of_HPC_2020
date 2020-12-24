#include <string.h>

#include <pgm_utils.h>

int main(int argc, char **argv) {
	if (argc < 2) {
		printf("usage: %s image\n", argv[0]);
		exit(1);
	}

    int dims[2] = {0, 0};
    int maxval = 0;
	void* ptr;

    read_pgm_image(&ptr, &maxval, dims + 1, dims, argv[1]);
    write_pgm_image(ptr, maxval, dims[1], dims[0], argv[1]);
    printf("%s has been normalized\n", argv[1]);

	free(ptr);

    return 0;
}