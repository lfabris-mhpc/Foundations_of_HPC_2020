import sys
import os

import glob
import struct
from PIL import Image, ImageDraw

def ratio2color(ratio):
	return tuple(0, int(255 * ratio), 100)

def convert(fname):
	with open(fname, "rb") as f:
		fbuffer = f.read()
		(columns, rows, max_iter) = struct.unpack("IIH", fbuffer[:10])
		print(f"columns {columns} rows {rows} max_iter {max_iter}")
		print(f"calc len: {2 * columns * rows} actual len: {len(fbuffer) - 10}")

		img = Image.new("RGB", (columns, rows), color="black")
		pixels = img.load()

		for i in range(rows):
			for j in range(columns):
				begin = 10 + 2 * (i * columns + j)
				end = begin + 2
				#print(f"fbuffer[{begin}:{end}]: {fbuffer[begin:end]}")
				iters = struct.unpack("H", fbuffer[begin:end])[0]
				if iters < max_iter:
					#pixels[i, j] = (int(255*i/rows), int(255*j/columns), 100)
					#pixels[i, j] = ratio2color(iters / max_iter)

					#why is this transposed?
					pixels[j, i] = (0, int(255 * iters / max_iter), 100)

		#img.show()
		img.save(fname[:-len("mandelbrot")] + "png")

pattern = "*.mandelbrot"
if len(sys.argv) >= 2:
	pattern = os.path.join(sys.argv[1], pattern)

for fname in glob.glob(pattern):
	convert(fname)
