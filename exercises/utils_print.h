#ifndef __utils_print_h__
#define __utils_print_h__

#include <string.h>
#include <assert.h>

//double
void print_array_ptr_noname(const void** array, const int size) {
	printf("[");
	for (int i = 0; i < size; ++i) {
		printf("%p ", array[i]);
	}
	printf("]\n");
}

void print_array_ptr(const char* name, const void** array, const int size) {
	printf("%s(%d):\n", name, size);
	print_array_ptr_noname(array, size);
}

//double
void print_array_double_noname(const double* array, const int size) {
	printf("[");
	for (int i = 0; i < size; ++i) {
		printf("%f ", array[i]);
	}
	printf("]\n");
}

void print_array_double(const char* name, const double* array, const int size) {
	printf("%s(%d):\n", name, size);
	print_array_double_noname(array, size);
}

void print_multiarray_double_noname(const double* array, const int dims, const int* sizes) {
	if (dims == 1) {
		print_array_double_noname(array, *sizes);
	}
	
	int columns = 1;
	for (int i = 1; i < dims; ++i) {
		columns *= sizes[i];
	}
	printf("[");
	for (int i = 0; i < sizes[0]; ++i) {
		print_multiarray_double_noname(array + i * columns, dims-1, sizes + 1); 
		printf(",\n");
	}
	printf("]\n");
}

void print_multiarray_double(const char* name, const double* array, const int dims, const int* sizes) {
	printf("%s(", name);
	for (int i = 0; i < dims; ++i) {
		printf("%d, ", sizes[i]);
	}
	printf("):\n");
	print_multiarray_double_noname(array, dims, sizes);
}

//int
void print_array_int_noname(const int* array, const int size) {
	printf("[");
	for (int i = 0; i < size; ++i) {
		printf("%d ", array[i]);
	}
	printf("]\n");
}

void print_array_int(const char* name, const int* array, const int size) {
	printf("%s(%d):\n", name, size);
	print_array_int_noname(array, size);
}

void print_multiarray_int_noname(const int* array, const int dims, const int* sizes) {
	if (dims == 1) {
		print_array_int_noname(array, *sizes);
	}
	
	int columns = 1;
	for (int i = 1; i < dims; ++i) {
		columns *= sizes[i];
	}
	printf("[");
	for (int i = 0; i < sizes[0]; ++i) {
		print_multiarray_int_noname(array + i * columns, dims-1, sizes + 1); 
		printf(",\n");
	}
	printf("]\n");
}

void print_multiarray_int(const char* name, const int* array, const int dims, const int* sizes) {
	printf("%s(", name);
	for (int i = 0; i < dims; ++i) {
		printf("%d, ", sizes[i]);
	}
	printf("):\n");
	print_multiarray_int_noname(array, dims, sizes);
}

#endif
