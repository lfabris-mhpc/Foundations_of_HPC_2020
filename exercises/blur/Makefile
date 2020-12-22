CC = mpicc
INCLUDEDIRS = -I include/
LCCFLAGS = -Wall -Wextra -march=native $(INCLUDEDIRS) -O3 -DTIMING $(CCFLAGS)
LLDFLAGS = -Wall -Wextra -lm $(LDFLAGS)
# -fopenmp
# -O0
# -O3 -DNDEBUG

EXE_HYBRID = blur_hybrid.x
OBJS_HYBRID = blur_hybrid.o
EXE_MPI = blur_mpi.x
OBJS_MPI = blur_mpi.o src/blur_utils.o src/pgm_utils.o
EXE_OMP = blur_omp.x
OBJS_OMP = blur_omp.o src/blur_utils.o src/pgm_utils.o

# $(EXE_MPI) $(EXE_OMP)
default: $(EXE_HYBRID)
.PHONY: default

clean:
	rm -f $(EXE_HYBRID) $(OBJS_HYBRID) $(EXE_MPI) $(OBJS_MPI) $(EXE_OMP) $(OBJS_OMP)
.PHONY: clean

$(EXE_HYBRID): $(OBJS_HYBRID)
	$(CC) $^ -o $@ $(LLDFLAGS)
	
$(EXE_MPI): $(OBJS_MPI)
	$(CC) $^ -o $@ $(LLDFLAGS)

$(EXE_OMP): $(OBJS_OMP)
	$(CC) $^ -o $@ $(LLDFLAGS)
	
%.o: %.c
	$(CC) -c $< -o $@ $(LCCFLAGS) 

blur_mpi.o: blur_mpi.c
blur_omp.o: blur_omp.c
blur_hybrid.o: blur_hybrid.c

src/blur_utils.o: src/blur_utils.c
src/pgm_utils.o: src/pgm_utils.c
