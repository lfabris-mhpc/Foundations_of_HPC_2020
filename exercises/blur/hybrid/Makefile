CC = mpicc
INCLUDEDIRS = -I ../include/
LCCFLAGS = -Wall -Wextra -fopenmp $(INCLUDEDIRS) -O3 -DNDEBUG -DTIMING $(CCFLAGS)
LLDFLAGS = -Wall -Wextra -lm -fopenmp -O3 -DNDEBUG -DTIMING $(LDFLAGS)
# -fopenmp
# -DNDEBUG
# -O0
# -O3 -march=native

EXE_HYBRID = blur_hybrid.x
OBJS_HYBRID = blur_hybrid.o

EXE_HYBRID_WCOMM = blur_hybrid_wcomm.x
OBJS_HYBRID_WCOMM = blur_hybrid_wcomm.o

default: $(EXE_HYBRID) $(EXE_HYBRID_WCOMM)
.PHONY: default

clean:
	rm -f $(EXE_HYBRID) $(OBJS_HYBRID) $(EXE_HYBRID_WCOMM) $(OBJS_HYBRID_WCOMM)
.PHONY: clean

$(EXE_HYBRID): $(OBJS_HYBRID)
	$(CC) $^ -o $@ $(LLDFLAGS)
	
$(EXE_HYBRID_WCOMM): $(OBJS_HYBRID_WCOMM)
	$(CC) $^ -o $@ $(LLDFLAGS)
	
%.o: %.c
	$(CC) -c $< -o $@ $(LCCFLAGS) 

blur_hybrid.o: blur_hybrid.c
blur_hybrid_wcomm.o: blur_hybrid_wcomm.c
