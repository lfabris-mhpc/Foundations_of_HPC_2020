#include <string.h>

#include <pgm_utils.h>

int main(int argc, char** argv) {
	if (argc < 5) {
		printf("usage: %s image image_output vertical-repeats horizontal-repeats\n", argv[0]);
		exit(1);
	}

	int rep_vert = atoi(argv[3]);
	int rep_hori = atoi(argv[4]);
	if (!rep_vert || !rep_hori) {
		printf("invalid tiling argument\n");
		exit(1);
	}

    int dims[2] = {0, 0};
    int maxval = 0;
	void* ptr;

    read_pgm_image(&ptr, &maxval, dims + 1, dims, argv[1]);
	int pixel_size = 1 + (maxval > 255);
	size_t rowsize = pixel_size * dims[1];

	printf("base image %s has %d rows and %d columns\n", argv[1], dims[0], dims[1]);

	char* output = (char*) malloc(pixel_size * dims[0] * dims[1] * rep_vert * rep_hori);
	size_t idx = 0;
	for (int ir = 0; ir < rep_vert; ++ir) {
		for (int i = 0; i < dims[0]; ++i) {
			char* src = ((char*) ptr) + i * rowsize;

			for (int jr = 0; jr < rep_hori; ++jr) {
				memcpy(output + idx, src, rowsize);
				idx += rowsize;
			}
		}
	}

    write_pgm_image(output, maxval, dims[1] * rep_hori, dims[0] * rep_vert, argv[2]);
	printf("output image %s has %d rows and %d columns\n", argv[2], dims[0] * rep_vert, dims[1] * rep_hori);

	free(output);
	free(ptr);

    return 0;
}