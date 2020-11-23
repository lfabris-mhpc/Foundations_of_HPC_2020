#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#include <immintrin.h>

#include <omp.h>

typedef struct mandelbrot_params_t {
	double x_step;
	double y_step;
	
	double x_lower;
	double x_upper;
	
	double y_lower;
	double y_upper;
	
	unsigned int columns;
	unsigned int rows;
	unsigned short int max_iter;
	unsigned short int threads_num;
	unsigned short int simd;
} mandelbrot_params;

void print_params(const mandelbrot_params* params) {
	printf("image size %u x %u centered on (%f, %f)\n"
		   , params->columns, params->rows
		   , (params->x_upper + params->x_lower) / 2
		   , (params->y_upper + params->y_lower) / 2);
	printf("ratio %f extent %f x %f\n"
		   , params->columns / (double) params->rows
		   , params->x_upper - params->x_lower
		   , params->y_upper - params->y_lower);
	printf("x [%f, %f] y [%f, %f]\n"
		   , params->x_lower, params->x_upper
		   , params->y_lower, params->y_upper);
}

void mandelbrot_params_init_from_center(mandelbrot_params* params, const unsigned int columns, const unsigned int rows, const double x, const double y, const double zoom, const unsigned short int max_iter) {
	//round columns to be multiple of 4
	params->columns = columns;
	if (params->columns % 4) {
		params->columns = 4 * (columns / 4 + 1);
	}
	params->rows = rows;
	params->max_iter = max_iter;
	
	//base / height = columns / rows
	//size is 2.0 for height
	double ratio = (params->columns - 1) / (double) (params->rows - 1);
	
	//2s are elided
	double x_hext = ratio * 2.0 / zoom;
	double y_hext = 2.0 / zoom;
	
	params->x_step = 2 * x_hext / (params->columns - 1);
	params->y_step = 2 * y_hext / (params->rows - 1);
	
	params->x_lower = x - x_hext;
	params->x_upper = x + x_hext;
	params->y_lower = y - y_hext;
	params->y_upper = y + y_hext;
}

int cardioid_or_bulb(const double x, const double y) {
	register double y2 = y*y;
	
	register double xq = x - 0.25;
	register double q = xq*xq + y2;
	if (q * (q + xq) <= y2 * 0.25) {
		return 1;
	}
	
	if ((x+1)*(x+1) + y2 <= 1 / 16.0) {
		return 1;
	}

	return 0;
}

unsigned int escapes_at(const double x, const double y, const unsigned int max_iter) {
	if (cardioid_or_bulb(x, y)) {
		return max_iter;
	}
	
	register double zx = 0;
	register double zy = 0;
	register double zx2 = 0;
	register double zy2 = 0;

	//printf("testing point (%f, %f)\n", zx, zy);
	register unsigned short int i = 0;
	while (i < max_iter && (zx2 + zy2 <= 4)) {
		zy = 2*zx*zy + y;
		zx = zx2 - zy2 + x;

		zx2 = zx * zx;
		zy2 = zy * zy;

		++i;
	}

	return i;
};

unsigned short int calc_escapes_nosimd(const mandelbrot_params* params, unsigned short int* iters) {
	unsigned short int max = 0;
	//iters[0, 0] is the top left
	#pragma omp parallel for schedule(dynamic) reduction(max: max)
	for (unsigned int i = 0; i < params->rows; ++i) {
		double y = params->y_upper - (1 + 2*i) * params->y_step / 2;
		for (unsigned int j = 0; j < params->columns; ++j) {
			double x = params->x_lower + (1 + 2*j) * params->x_step / 2;

			iters[i * params->columns + j] = escapes_at(x, y, params->max_iter);
			
			if (iters[i * params->columns + j] < params->max_iter && max < iters[i * params->columns + j]) {
				max = iters[i * params->columns + j];
			}
		}
	}
	
	return max;
}

//SIMD
/*
typedef double v4df __attribute__ ((vector_size (4*sizeof(double))));
typedef union {
	v4df V;
	double v[4];
} v4df_u;
*/
#define simd_df_print(a) printf("(%f, %f, %f, %f)\n", a[0], a[1], a[2], a[3]);

int cardioid_or_bulb_simd(const double x, const double y) {
	register double y2 = y*y;
	
	register double xq = x - 0.25;
	register double q = xq*xq + y2;
	if (q * (q + xq) <= y2 * 0.25) {
		return 1;
	}
	
	if ((x+1)*(x+1) + y2 <= 1 / 16.0) {
		return 1;
	}

	return 0;
}

unsigned short int calc_escapes_simd(const mandelbrot_params* params, unsigned short int* iters) {
	//vector of steps, to be used in increments
	__m256d x_steps = _mm256_broadcast_sd(&(params->x_step));
	//printf("x_steps: "); simd_df_print(x_steps);
	
	double row_advance = 1.0, column_advance = 4.0;
	__m256d ones = _mm256_broadcast_sd(&row_advance);
	__m256d fours = _mm256_broadcast_sd(&column_advance);
	
	unsigned short int max = 0;
	//iters[0, 0] is the top left
	#pragma omp parallel for schedule(dynamic) reduction(max: max)
	for (unsigned int i = 0; i < params->rows; ++i) {
		double y = params->y_upper - (1 + 2*i) * params->y_step / 2;
		//load initial img components - all equal as it's the same row
		__m256d ys = _mm256_broadcast_sd(&y);
		
		//offsets to be multiplied by x_step to get the actual real components
		//will be incremented by 4 at each internal loop
		double column_offsets_[4] = {0.0, 1.0, 2.0, 3.0};
		__m256d column_offsets = _mm256_load_pd(column_offsets_);
		
		double x_base = params->x_lower + params->x_step / 2;
		
		for (unsigned int j = 0; j < params->columns; j += 4) {
			//printf("column_offsets: "); simd_df_print(column_offsets);
			
			//alternative to updating column offsets: double x = params->x_lower + (1 + 2*j) * params->x_step / 2;
			//load base real components
			__m256d xs = _mm256_broadcast_sd(&x_base);
			
			//get current displacements in real component
			__m256d x_offsets = _mm256_mul_pd(column_offsets, x_steps);
			//these are the actual real components for this part of the row
			xs = _mm256_add_pd(xs, x_offsets);
			
			int skip = 0;
			//check if in cardioid or bulb
			{
				double quarter = 0.25;
				__m256d quarters = _mm256_broadcast_sd(&quarter);
				//register double y2 = y*y;
				__m256d y2s = _mm256_mul_pd(ys, ys);
				//printf("y2s: "); simd_df_print(y2s);

				//register double xq = x - 0.25;
				__m256d xqs = _mm256_sub_pd(xs, quarters);
				//register double q = xq*xq + y2;
				__m256d qs  = _mm256_mul_pd(xqs, xqs);
				qs = _mm256_add_pd(qs, y2s);
				
				//if (q * (q + xq) <= y2 * 0.25) {
				//	return 1;
				//}
				//left hand side
				__m256d lhs = _mm256_add_pd(xqs, qs);
				lhs = _mm256_mul_pd(lhs, qs);
				
				//right hand side
				__m256d rhs = _mm256_mul_pd(y2s, quarters);
				
				// comparison
				__m256d cmp = _mm256_cmp_pd(lhs, rhs, _CMP_LT_OQ);
				//test to easily check bits for escaped values
				unsigned int test = 0;
				test = _mm256_movemask_pd(cmp) & 15; //lower 4 bits are comparison
				if (test == 15) {
					//all values are in cardioid
					int top = (i + 3) < params->columns ? 4 : params->columns & 3;
					for (int k = 0; k < top; ++k) {
						iters[i * params->columns + j + k] = params->max_iter;
					}
					skip = 1;
				}
				
				//if ((x+1)*(x+1) + y2 <= 1 / 16.0) {
				//	return 1;
				//}
				//left hand side
				lhs = _mm256_add_pd(xs, ones);
				lhs = _mm256_mul_pd(lhs, lhs);
				lhs = _mm256_add_pd(lhs, y2s);
				
				//right hand side
				rhs = _mm256_mul_pd(quarters, quarters);
				
				// comparison
				cmp = _mm256_cmp_pd(lhs, rhs, _CMP_LT_OQ);
				//test to easily check bits for escaped values
				unsigned int test2 = 0;
				test2 = _mm256_movemask_pd(cmp) & 15; //lower 4 bits are comparison
				test = test | test2;
				if (test == 15) {
					//all values are in cardioid or in bulb
					int top = (i + 3) < params->columns ? 4 : params->columns & 3;
					for (int k = 0; k < top; ++k) {
						iters[i * params->columns + j + k] = params->max_iter;
					}
					skip = 1;
				}
			}
			
			if (!skip) {
				//these are the escape counters of each point - init to 0
				__m256d counters = _mm256_xor_pd(x_steps, x_steps);

				//z are the iteration values
				__m256d zxs = xs, zys = ys;
				//printf("zxs: "); simd_df_print(zxs);
				//printf("zys: "); simd_df_print(zys);

				unsigned int test = 0;
				unsigned short int iter = 0;
				do {
					//compute squares
					__m256d zx2s = _mm256_mul_pd(zxs, zxs);
					//printf("zx2s: "); simd_df_print(zx2s);
					__m256d zy2s = _mm256_mul_pd(zys, zys);
					//printf("zy2s: "); simd_df_print(zy2s);

					//compute norm for escape test in temporary var
					__m256d tmp = _mm256_add_pd(zx2s, zy2s);
					//printf("norms: "); simd_df_print(tmp);
					//check threshold escaped - tmp will have 1s in the positions of unescaped values
					tmp = _mm256_cmp_pd(tmp, fours, _CMP_LT_OQ);
					test = _mm256_movemask_pd(tmp) & 15; //lower 4 bits are comparison
					//printf("norm <= 4: "); simd_df_print(tmp);
					//printf("test: %d\n", test);

					//zeroes the escaped counter increments; others become 1s
					tmp = _mm256_and_pd(tmp, ones);
					//printf("counters delta: "); simd_df_print(tmp);
					//thus, counters are incremented for unescaped values
					counters = _mm256_add_pd(counters, tmp);
					//printf("updated counters: "); simd_df_print(counters);

					//xi*yi
					tmp = _mm256_mul_pd(zxs, zys);
					//xi^2 - yi^2 (overriding current real comps)
					zxs = _mm256_sub_pd(zx2s, zy2s);
					//xi <- xi^2 - yi^2 + x0 (final current real comps)
					zxs = _mm256_add_pd(zxs, xs);
					//2*xi*yi
					zys = _mm256_add_pd(tmp, tmp);
					//yi <- 2*xi*yi + y0
					zys = _mm256_add_pd(zys, ys);

					++iter;
				} while (test && iter < params->max_iter);

				//handles case when columns is not multiple of 4
				int top = (i + 3) < params->columns ? 4 : params->columns & 3;

				//buggy? convert to access members
				//buggy? __m128i int_counters = _mm256_cvtpd_epi32(counters);
				//printf("iters[%d:4]: (", i * params->columns + j);
				for (int k = 0; k < top; ++k) {
					//buggy? iters[i * params->columns + j + k] = int_counters[k];//.m256i_i16[2*k];
					iters[i * params->columns + j + k] = (unsigned short int) counters[k];
					//printf("%d, ", iters[i * params->columns + j + k]);
					
					if (iters[i * params->columns + j] < params->max_iter && max < iters[i * params->columns + j]) {
						max = iters[i * params->columns + j];
					}
				}
				//printf(")\n");
			}
			
			//increment offsets by 4 -> advance to next segment of row
			column_offsets = _mm256_add_pd(column_offsets, fours);
		}
	}
	
	return max;
}

void save_to_file(const unsigned short int* escaped, const unsigned int columns, const unsigned int rows, const unsigned short int max_iter, const char* file_name) {
	FILE* fp = fopen(file_name, "wb");
	
	if (fp == NULL)
	{
		fprintf(stderr, "Could not open file %s\n", file_name);
	}
	else
	{
		size_t check = fwrite(&columns, sizeof columns, 1, fp);
		check = fwrite(&rows, sizeof rows, 1, fp);
		check = fwrite(&max_iter, sizeof max_iter, 1, fp);

		check = fwrite(escaped, sizeof escaped[0], (size_t) columns * rows, fp);
		printf("writing %lu elements; written %lu\n", (size_t) columns * rows, check);
		
		fclose(fp);
	}
}

void render_ascii(const unsigned short int* escaped, const unsigned int columns, const unsigned int rows, const unsigned short int max_iter) {
	for (unsigned int i = 0; i < rows; ++i) {
		printf("[");
		for (unsigned int j = 0; j < columns; ++j) {
			if (escaped[i * columns + j] < max_iter) {
				//printf("@@");
				char c = 'a' + 26 * (int) escaped[i * columns + j] / max_iter;
				printf("%c%c", c, c);
			} else {
				printf("  ");
			}
		}
		printf("]\n");
	}
}

unsigned short int calc_escapes(const mandelbrot_params* params, unsigned short int* iters) {
	unsigned short int max;
	
	omp_set_num_threads(params->threads_num);
	if (params->simd) {
		max = calc_escapes_simd(params, iters);
	} else {
		max = calc_escapes_nosimd(params, iters);
	}
	
	return max;
}

void dive_to_images(unsigned int columns, unsigned int rows, double x, double y, unsigned short max_iter, const double min_zoom, const double max_zoom, unsigned int steps) {
	unsigned short int* escapes = (unsigned short int*) malloc(sizeof(unsigned short int) * columns * rows);
	
	double zoom = min_zoom, zoom_mult = pow(max_zoom / min_zoom, 1.0 / steps);
	for (unsigned int i = 0; i < steps; ++i) {
		mandelbrot_params params;
		mandelbrot_params_init_from_center(&params, columns, rows, x, y, zoom, max_iter);
		print_params(&params);
		
		unsigned short int max = calc_escapes(&params, escapes);
		char filename[100];
		snprintf(filename, sizeof filename, "images/dive_%f_%f_%f.mandelbrot", x, y, zoom);
		save_to_file(escapes, columns, rows, max, filename);
		
		zoom *= zoom_mult;
		printf("\n");
	}
}

int main (int argc, char ** argv) {
	/*
	if (argc < 2)
	{
		printf(" Usage: %s iterations (nthreads)\n",argv[0]);
		return 1;
	}
	
	long long int N = atoll(argv[1]);
	
	if (argc >= 3) {
		int nthreads = atoi(argv[2]);
		omp_set_num_threads(nthreads);
	}
	*/
	
	dive_to_images(1024, 1024, -0.749988253, -0.01259452, 4000, 1, 5000000, 40); 
	
	return 0;
}
