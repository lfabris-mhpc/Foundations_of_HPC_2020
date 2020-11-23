CC = gcc
EXE = omp_mandelbrot.x
OBJS = omp_mandelbrot.o
LCFLAGS = -Wall -Wextra -O3 -march=native -fopenmp $(CFLAGS)

default: $(EXE)
.PHONY: default

clean:
	rm -f $(EXE) $(OBJS)
.PHONY: clean

$(EXE): $(OBJS)
	$(CC) $^ -o $@ $(LCFLAGS)
	
profile: $(EXE)
	perf stat --repeat 10 ./$(EXE) 
	#> /dev/null

%.o: %.c
	$(CC) -c $< -o $@ $(LCFLAGS)

omp_mandelbrot.o: omp_mandelbrot.c
