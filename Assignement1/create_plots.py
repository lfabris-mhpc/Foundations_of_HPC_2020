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

		df = pd.read_csv(fn, header=None, names=["P", str(nbase)], index_col=False, usecols=[0, 4], comment="#")

		#print(f"file: {fn}")
		#print(df)
		
		if ret is None:
			ret = df
		else:
			ret[str(nbase)] = df[str(nbase)]
	
	return ret

#serial

#strong scalability
d = "csvs"

prefix = "strong-scalability-10to"
usecols = [0, 4]

suffix = ""
process = merge(d, prefix, suffix+".csv", usecols)
print("weak-scalability-process:")
print(process)

suffix = "-elapsed"
elapsed = merge(d, prefix, suffix+".csv", usecols)
print("weak-scalability-elapsed:")
print(elapsed)

suffix = "-system"
system = merge(d, prefix, suffix+".csv", usecols)
system = system.div(system["P"], axis=0)
system["P"] = elapsed["P"]
print("weak-scalability-system (scaled):")
print(system)

overhead = elapsed - process
overhead = overhead.div(elapsed["P"], axis=0)
overhead["P"] = elapsed["P"]
print("weak-scalability-overhead (scaled):")
print(overhead)

#weak_scalability
