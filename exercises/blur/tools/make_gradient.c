#include <string.h>

#include <pgm_utils.h>

int main(int argc, char **argv) {
	if (argc < 4) {
		printf("usage: %s rows columns maxval\n", argv[0]);
		exit(1);
	}

    int dims[2] = {atoi(argv[1]), atoi(argv[2])};
    int maxval = atoi(argv[3]) % 65536;
									   
	void* ptr = generate_gradient(maxval, dims[1], dims[0]);
    write_pgm_image(ptr, maxval, dims[1], dims[0], "gradient.pgm");
    printf("gradient has been generated\n");

	free(ptr);

    return 0;
}