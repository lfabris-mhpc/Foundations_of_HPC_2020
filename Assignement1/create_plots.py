import sys
import os as os
import glob as glob
#import math as math
import pandas as pd
#import mathplotlib as plt

def merge(d, prefix, suffix=".csv", usecols=[0, 4]):
	ret = None
	for fn in sorted(glob.glob(os.path.join(d, prefix + "*[0-9]" + suffix))):
		f = os.path.basename(fn)
		nbase = 10**int(f[len(prefix):-len(suffix)])
		ncol = f"{nbase:.0E}"

		df = pd.read_csv(fn, header=None, names=["P", ncol], index_col=False, usecols=[0, 4], comment="#")

		#print(f"file: {fn}")
		#print(df)
		
		if ret is None:
			ret = df
		else:
			ret[ncol] = df[ncol]
	
	return ret

def process(d, prefix, context, usecols=[0, 4]):
	suffix = ""
	process = merge(d, prefix, suffix+".csv", usecols)
	print(context + "-process:")
	print(process)

	suffix = "-elapsed"
	elapsed = merge(d, prefix, suffix+".csv", usecols)
	print(context + "-elapsed:")
	print(elapsed)

	suffix = "-system"
	system = merge(d, prefix, suffix+".csv", usecols)
	system = system.div(system["P"], axis=0)
	system["P"] = elapsed["P"]
	print(context + "-system (scaled):")
	print(system)

	overhead = elapsed - process
	overhead = overhead.div(elapsed["P"], axis=0)
	overhead["P"] = elapsed["P"]
	print(context + "-overhead (scaled):")
	print(overhead)

d = "csvs"

#serial

#strong scalability

context = "strong-scalability"
prefix = "strong-scalability-10to"
usecols = [0, 4]

process(d, prefix, context, usecols)

#weak scalability
context = "weak-scalability"
prefix = "weak-scalability-10to"
usecols = [0, 4]

process(d, prefix, context, usecols)
