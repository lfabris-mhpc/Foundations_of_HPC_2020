import sys
import os
from PIL import Image

if len(sys.argv) < 4:
	print(f"usage: {sys.argv[0]} base_img repeat_rows repeat_columns {output_file}")

base = Image.open(sys.argv[1], mode="r")
print(f"base: {base}")

repeats = tuple(int(v) for v in sys.argv[2:4])
output = Image.new("L", (repeats[1] * base.width, repeats[0] * base.height))
print(f"output: {output}")

for i in range(repeats[0]):
	for j in range(repeats[1]):
		output.paste(base, (j * base.width, i * base.height))

if len(sys.argv) >= 5:
	output_file = sys.argv[4]
else:
	root, ext = os.path.splitext(sys.argv[1])
	output_file = root + f"_{repeats[0]}x{repeats[1]}" + ext
output.save(output_file)
print(f"saved to {output_file}")