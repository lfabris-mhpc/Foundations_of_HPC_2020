TILES = 1 2 4 8 16 32 64

IMG_BASE = check_me eevee
IMG_TILED = $(addsuffix .pgm, $(foreach img, $(IMG_BASE), $(addprefix $(img)_, $(TILES)))) 

TILE = ../tools/img_tile.x
NORM = ../tools/img_normalize.x

default: $(IMG_TILED)
.PHONY: default

clean:
	rm -f $(IMG_TILED)
.PHONY: clean

%_1.pgm: %.pgm $(TILE) $(NORM)
	$(TILE) $< $@ 1 1 ; $(NORM) $@
%_2.pgm: %.pgm $(TILE) $(NORM)
	$(TILE) $< $@ 1 2 ; $(NORM) $@
%_4.pgm: %.pgm $(TILE) $(NORM)
	$(TILE) $< $@ 2 2 ; $(NORM) $@
%_8.pgm: %.pgm $(TILE) $(NORM)
	$(TILE) $< $@ 2 4 ; $(NORM) $@
%_16.pgm: %.pgm $(TILE) $(NORM)
	$(TILE) $< $@ 4 4 ; $(NORM) $@
%_32.pgm: %.pgm $(TILE) $(NORM)
	$(TILE) $< $@ 4 8 ; $(NORM) $@
%_64.pgm: %.pgm $(TILE) $(NORM)
	$(TILE) $< $@ 8 8 ; $(NORM) $@
	
$(TILE) $(NORM):
	make -B ../tools clean ; make -B ../tools

#check_me_1.pgm: $(TILE) $(NORM)
#	$(TILE) check_me_1.pgm $@ 1 1 ; $(NORM) $@
