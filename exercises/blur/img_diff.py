import sys
import os
from PIL import Image

if len(sys.argv) < 3:
	print(f"usage: {sys.argv[0]} img_a img_b")

img_a = Image.open(sys.argv[1], mode="r")
print(f"img_a: {img_a}")
ysize_a, xsize_a = img_a.size

img_b = Image.open(sys.argv[2], mode="r")
print(f"img_b: {img_b}")
ysize_b, xsize_b = img_b.size

#print(f"ysize_a {ysize_a} xsize_a {xsize_a}")
#print(f"ysize_b {ysize_b} xsize_b {xsize_b}")

if ysize_a != ysize_b:
	print(f"different widths: {ysize_a} vs {ysize_b}")
	sys.exit(0)
if xsize_a != xsize_b:
	print(f"different heights: {xsize_a} vs {xsize_b}")
	sys.exit(0)

for i in range(xsize_a):
	for j in range(ysize_a):
		try:
			a = img_a.getpixel((j, i))
			b = img_b.getpixel((j, i))
			if a != b:
				print(f"different pixels at {i}, {j}: {a} vs {b}")
		except IndexError:
			raise IndexError(f"error accessing pixels at {i}, {j}")
