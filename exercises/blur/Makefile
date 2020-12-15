CC = mpicc
INCLUDEDIRS = -I include/
LCFLAGS = -Wall -Wextra -O0 -lm $(INCLUDEDIRS) -DNDEBUG -DTIMING $(CFLAGS)

EXE_MPI = blur_mpi.x
OBJS_MPI = blur_mpi.o src/blur_utils.o src/pgm_utils.o
EXE_OMP = blur_omp.x
OBJS_OMP = blur_omp.o src/blur_utils.o src/pgm_utils.o
EXE_HYBRID = blur_hybrid.x
OBJS_HYBRID = blur_hybri.o src/blur_utils.o src/pgm_utils.o

default: $(EXE_MPI) $(EXE_OMP) $(EXE_HYBRID)
.PHONY: default

clean:
	rm -f $(EXE_MPI) $(OBJS_MPI) $(EXE_OMP) $(OBJS_OMP) $(EXE_HYBRID) $(OBJS_HYBRID)
.PHONY: clean

$(EXE_MPI): $(OBJS_MPI)
	$(CC) $^ -o $@ $(LCFLAGS)

$(EXE_OMP): $(OBJS_OMP)
	$(CC) $^ -o $@ -fopenmp $(LCFLAGS)

$(EXE_HYBRID): $(OBJS_HYBRID)
	$(CC) $^ -o $@ -fopenmp $(LCFLAGS)
	
%.o: %.c
	$(CC) -c $< -o $@ $(LCFLAGS) 

blur_mpi.o: blur_mpi.c
blur_omp.o: blur_omp.c
blur_hybrid.o: blur_hybrid.c

src/blur_utils.o: src/blur_utils.c
src/pgm_utils.o: src/pgm_utils.c
