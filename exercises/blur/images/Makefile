TILES = 1 4 8 12 16 20 24 28 32 36 40 44 48

IMG_BASE = test_picture
IMG_TILED = $(addsuffix .pgm, $(foreach img, $(IMG_BASE), $(addprefix $(img)_, $(TILES)))) 

TILE = ../tools/img_tile.x

default: $(IMG_TILED)
.PHONY: default

clean:
	rm -f $(IMG_TILED)
.PHONY: clean

%_1.pgm: %.pgm $(TILE)
	$(TILE) $< $@ 1 1
%_4.pgm: %.pgm $(TILE)
	$(TILE) $< $@ 2 2
%_8.pgm: %.pgm $(TILE)
	$(TILE) $< $@ 4 2
%_12.pgm: %.pgm $(TILE)
	$(TILE) $< $@ 4 3
%_16.pgm: %.pgm $(TILE)
	$(TILE) $< $@ 4 4
%_20.pgm: %.pgm $(TILE)
	$(TILE) $< $@ 5 4
%_24.pgm: %.pgm $(TILE)
	$(TILE) $< $@ 6 4
%_28.pgm: %.pgm $(TILE)
	$(TILE) $< $@ 7 4
%_32.pgm: %.pgm $(TILE)
	$(TILE) $< $@ 8 4
%_36.pgm: %.pgm $(TILE)
	$(TILE) $< $@ 9 4
%_40.pgm: %.pgm $(TILE)
	$(TILE) $< $@ 8 5
%_44.pgm: %.pgm $(TILE)
	$(TILE) $< $@ 11 4
%_48.pgm: %.pgm $(TILE)
	$(TILE) $< $@ 8 6
