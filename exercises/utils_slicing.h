#ifndef __wait_set_h__
#define __wait_set_h__

#include <string.h>
#include <assert.h>

#define loop2d_open(sizes, lower, upper) \
	for (int ii = lower[0]; ii < upper[0]; ++ii) { \
		int pi = ii * sizes[0]; \
		for (int jj = lower[1]; jj < upper[1]; ++jj) { \
			int pj = p0 * sizes[1] + jj; \

#define loop2d_close \
		} \
	}

#define loop3d_open(sizes, lower, upper) \
	loop2d_open(sizes, lower, upper) \
			for (int kk = lower[2]; kk < upper[2]; ++kk) { \
				int pk = pj * sizes[2] + kk; \

#define loop3d_close \
			} \
	loop2d_close

typedef struct {
	int dims;
	int step;
	int count;
	int* sizes;
	int* lower;
	int* upper;
	int* idx;
} idx;

int calc_consecutive_step(idx* idx) {
	int substep = idx->upper[idx->dims-1] - idx->lower[idx->dims-1];
	int step = substep;
	for (int i = idx->dims-2; substep == idx->sizes[i] && i > -1; --i) {
		step *= substep;
		substep = idx->upper[i] - idx->lower[i];
	}
	return step;
}

int idx_count(idx* idx) {
	int ret = 1;
	for (int i = 0; i < idx->dims; ++i) {
		ret *= idx->upper[i] - idx->lower[i];
	}
	return ret;
}

void idx_init(idx* idx, int dims, int* dim_sizes, int* idx_lower, int* idx_upper) {
	idx->dims = dims;

	idx->sizes = (int*) malloc(sizeof(int) * dims * 4);
	idx->lower = idx->sizes + dims;
	idx->upper = idx->lower + dims;
	idx->idx = idx->upper + dims;

	for (int i = 0; i < dims; ++i) {
		idx->sizes[i] = dim_sizes[i];
		idx->lower[i] = idx_lower[i];
		idx->upper[i] = idx_upper[i];
		idx->idx[i] = idx_lower[i];
	}

	idx->step = calc_consecutive_step(idx);
	idx->count = idx_count(idx);
}

int idx_advance(idx* idx) {
	for (int i = idx->dims-1; i > -1; --i) {
		if (idx->idx[i] < idx->upper[i]-1) {
			++(idx->idx[i]);
			return 1;
		} else {
			idx->idx[i] = idx->lower[i];
		}
	}
	return 0;
}

int idx_advance_multi(idx* idx, int steps) {
	int ret = 1;
	/*
	if (steps % idx->sizes[idx->dims-1] == 0) {
		//can advance the lower dim by steps / idx->sizes[idx->dims-1]
	}
	*/
	//trivial...
	for (int i = 0; ret && i < steps; ++i) {
		ret = idx_advance(idx);
	}
	return ret;
}

int idx_linearize(idx* idx) {
	int pos = idx->idx[0];
	for (int i = 1; i < idx->dims; ++i) {
		pos = pos * idx->sizes[i] + idx->idx[i];
	}
	return pos;
}

void slicing_copy(void* src, void* dest, size_t element_size
		, int src_dims, int* src_sizes, int* src_lower, int* src_upper
		, int dest_dims, int* dest_sizes, int* dest_lower, int* dest_upper
		) {
	idx idx_src, idx_dest;
	idx_init(&idx_src, src_dims, src_sizes, src_lower, src_upper); 
	idx_init(&idx_dest, dest_dims, dest_sizes, dest_lower, dest_upper);

	int cnt = idx_count(&idx_src);
	if (!cnt) {
		return;
	}
	assert(cnt == idx_count(&idx_dest));

	int step = idx_src.step > idx_dest.step ? idx_dest.step : idx_src.step;

	char* src2 = (char*) src;
	char* dest2 = (char*) dest;
	while (cnt) {
		memcpy(dest2 + element_size * idx_linearize(&idx_dest), src2 + element_size * idx_linearize(&idx_src), step * element_size);
		idx_advance_multi(&idx_src, step);
		idx_advance_multi(&idx_dest, step);
		cnt -= step;
	}
}

#endif
