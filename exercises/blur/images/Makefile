TILES = 1 2 4 6 8 10 12 14 16 18 20 22 24

IMG_BASE = gradient
IMG_TILED = $(addsuffix .pgm, $(foreach img, $(IMG_BASE), $(addprefix $(img)_, $(TILES)))) 

TILE = ../tools/img_tile.x

default: $(IMG_TILED)
.PHONY: default

clean:
	rm -f $(IMG_TILED)
.PHONY: clean

%_1.pgm: %.pgm $(TILE)
	$(TILE) $< $@ 1 1
%_2.pgm: %.pgm $(TILE)
	$(TILE) $< $@ 2 1
%_4.pgm: %.pgm $(TILE)
	$(TILE) $< $@ 2 2
%_6.pgm: %.pgm $(TILE)
	$(TILE) $< $@ 3 2
%_8.pgm: %.pgm $(TILE)
	$(TILE) $< $@ 4 2
%_10.pgm: %.pgm $(TILE)
	$(TILE) $< $@ 5 2
%_12.pgm: %.pgm $(TILE)
	$(TILE) $< $@ 4 3
%_14.pgm: %.pgm $(TILE)
	$(TILE) $< $@ 7 2
%_16.pgm: %.pgm $(TILE)
	$(TILE) $< $@ 4 4
%_18.pgm: %.pgm $(TILE)
	$(TILE) $< $@ 6 3
%_20.pgm: %.pgm $(TILE)
	$(TILE) $< $@ 5 4
%_22.pgm: %.pgm $(TILE)
	$(TILE) $< $@ 11 2
%_24.pgm: %.pgm $(TILE)
	$(TILE) $< $@ 6 4
