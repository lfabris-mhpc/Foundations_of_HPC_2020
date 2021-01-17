#!/usr/bin/python

import sys
import math as m

if len(sys.argv) > 1:
	a = int(sys.argv[1])
	if a < 1:
		exit(1)
	
	c = 1
	diff = a - c
	for b in range(2, a+1):
		#print(f"testing {b}")
		if a % b == 0:
			d = int(abs(b - a // b))
			#print(f"diff {d}")
			if diff > d:
				diff = d
				c = b
	c = max(c, a // c)
	print(f"{c}\n{a // c}\n")