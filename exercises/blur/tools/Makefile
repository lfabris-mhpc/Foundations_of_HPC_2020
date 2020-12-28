CC = mpicc
INCLUDEDIRS = -I .
LCCFLAGS = -Wall -Wextra $(INCLUDEDIRS) -march=native -O2 -DNDEBUG $(CCFLAGS)
LLDFLAGS = -Wall -Wextra $(LDFLAGS)

EXE_NORMALIZE = img_normalize.x
OBJS_NORMALIZE = img_normalize.o

EXE_TILE = img_tile.x
OBJS_TILE = img_tile.o

EXE_DIFF = img_diff.x
OBJS_DIFF = img_diff.o

EXE_GRADIENT = make_gradient.x
OBJS_GRADIENT = make_gradient.o

default: $(EXE_NORMALIZE) $(EXE_TILE) $(EXE_DIFF) $(EXE_GRADIENT)
.PHONY: default

clean:
	rm -f $(EXE_NORMALIZE) $(OBJS_NORMALIZE) $(EXE_TILE) $(OBJS_TILE) $(EXE_DIFF) $(OBJS_DIFF) $(EXE_GRADIENT) $(OBJS_GRADIENT)
.PHONY: clean

$(EXE_NORMALIZE): $(OBJS_NORMALIZE)
	$(CC) $^ -o $@ $(LLDFLAGS)

$(EXE_TILE): $(OBJS_TILE)
	$(CC) $^ -o $@ $(LLDFLAGS)
	
$(EXE_DIFF): $(OBJS_DIFF)
	$(CC) $^ -o $@ $(LLDFLAGS)
	
$(EXE_GRADIENT): $(OBJS_GRADIENT)
	$(CC) $^ -o $@ $(LLDFLAGS)
	
%.o: %.c
	$(CC) -c $< -o $@ $(LCCFLAGS)
