CC = mpicc
INCLUDEDIRS = -I ../include/
LCCFLAGS = -Wall -Wextra -march=native $(INCLUDEDIRS) -fopenmp -O3 -DTIMING $(CCFLAGS)
LLDFLAGS = -Wall -Wextra -lm -fopenmp $(LDFLAGS)
# -fopenmp
# -O0
# -O3 -DNDEBUG

EXE_HYBRID = blur_hybrid.x
OBJS_HYBRID = blur_hybrid.o

default: $(EXE_HYBRID)
.PHONY: default

clean:
	rm -f $(EXE_HYBRID) $(OBJS_HYBRID)
.PHONY: clean

$(EXE_HYBRID): $(OBJS_HYBRID)
	$(CC) $^ -o $@ $(LLDFLAGS)
	
%.o: %.c
	$(CC) -c $< -o $@ $(LCCFLAGS) 

blur_hybrid.o: blur_hybrid.c
