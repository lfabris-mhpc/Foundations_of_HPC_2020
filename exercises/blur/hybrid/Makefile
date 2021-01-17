CC = mpicc
LCCFLAGS = -Wall -Wextra -fopenmp -O3 -DTIMING $(CCFLAGS) -DNDEBUG
LLDFLAGS = -Wall -Wextra -lm -fopenmp -O3 -DTIMING $(LDFLAGS) -DNDEBUG

EXE_HYBRID = blur.mpi_omp.x
OBJS_HYBRID = blur.mpi_omp.o

default: $(EXE_HYBRID)
.PHONY: default

clean:
	rm -f $(EXE_HYBRID) $(OBJS_HYBRID)
.PHONY: clean

$(EXE_HYBRID): $(OBJS_HYBRID)
	$(CC) $^ -o $@ $(LLDFLAGS)
	
%.o: %.c
	$(CC) -c $< -o $@ $(LCCFLAGS) 

blur.mpi_omp.o: blur.mpi_omp.c
