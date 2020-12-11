#include <stdio.h>
#include <stdlib.h>

#include "utils_slicing.h"

void print_array2d_int(int* v, int* sizes, int* lower, int* upper) {
	printf("array(%d:%d, %d:%d):\n", lower[0], upper[0], lower[1], upper[1]);
	for (int i = lower[0]; i < upper[0]; ++i) {
		printf("[");
		int pi = i * sizes[1];
		for (int j = lower[1]; j < upper[1]; ++j) {
			printf("%d ", v[pi + j]);		
		}
		printf("]\n");	
	}
}

int main(int argc, char** argv) {
	int zeros[2] = {0, 0};
	int src_dims = 2;
	int src_sizes[2] = {8, 10};
	int dest_dims = 2;
	int dest_sizes[2] = {6, 8};
	
	int cnt = src_sizes[0] * src_sizes[1];

	int* src = (int*) malloc(sizeof(int) * cnt);
	int* dest = (int*) malloc(sizeof(int) * cnt);
	
	for (int i = 0; i < cnt; ++i) {
		src[i] = dest[i] = i;	
	}
	printf("src: \n");
	print_array2d_int(src, src_sizes, zeros, src_sizes);
	printf("dest: \n");
	print_array2d_int(dest, dest_sizes, zeros, dest_sizes);

	int src_lower[2] = {2, 1};
	int src_upper[2] = {src_sizes[0]-1, src_sizes[1]-2};
	int src_cnt = (src_upper[0]-src_lower[0])*(src_upper[1]-src_lower[1]);
	printf("src_cnt: %d\n", src_cnt);
	print_array2d_int(src, src_sizes, src_lower, src_upper);
	
	int dest_lower[2] = {1, 1};
	int dest_upper[2] = {dest_sizes[0], dest_sizes[1]};
	int dest_cnt = (dest_upper[0]-dest_lower[0])*(dest_upper[1]-dest_lower[1]);
	printf("dest_cnt: %d\n", dest_cnt);
	print_array2d_int(dest, dest_sizes, dest_lower, dest_upper);

	slicing_copy(src, dest, sizeof(int)
		, src_dims, src_sizes, src_lower, src_upper
		, dest_dims, dest_sizes, dest_lower, dest_upper
		);

	//printf("final src: \n");
	//print_array2d_int(src, src_sizes, zeros, src_sizes);
	printf("final dest: \n");
	print_array2d_int(dest, dest_sizes, zeros, dest_sizes);
}
