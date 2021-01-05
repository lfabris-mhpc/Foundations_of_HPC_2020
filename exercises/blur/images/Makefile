TILES = 1 2 4 8 16 32 64 128 144

IMG_BASE = test_picture check_me eevee test_picture
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
	$(TILE) $< $@ 1 2
%_4.pgm: %.pgm $(TILE)
	$(TILE) $< $@ 2 2
%_8.pgm: %.pgm $(TILE)
	$(TILE) $< $@ 2 4
%_16.pgm: %.pgm $(TILE)
	$(TILE) $< $@ 4 4
%_32.pgm: %.pgm $(TILE)
	$(TILE) $< $@ 4 8
%_64.pgm: %.pgm $(TILE)
	$(TILE) $< $@ 8 8 
%_128.pgm: %.pgm $(TILE)
	$(TILE) $< $@ 8 16
%_144.pgm: %.pgm $(TILE)
	$(TILE) $< $@ 12 12
