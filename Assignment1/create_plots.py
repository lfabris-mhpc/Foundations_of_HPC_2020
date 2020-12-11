import sys
import os as os
import glob as glob
#import math as math
import pandas as pd
import matplotlib.pyplot as plt

def merge(d, prefix, suffix=".csv", usecols=[0, 1]):
	ret = None
	for fn in sorted(glob.glob(os.path.join(d, prefix + "*[0-9]" + suffix))):
		f = os.path.basename(fn)
		exps = f[len(prefix):-len(suffix)]
		nbase = 10**int(exps)
		ncol = f"{nbase:.0E}"

		df = pd.read_csv(fn, header=None, names=["P", ncol], index_col=[0], usecols=usecols, comment="#")

		#print(f"file: {fn}")
		#print(df)
		
		if ret is None:
			ret = df
		else:
			ret[ncol] = df[ncol]
	
	return ret

def plot(df, title, ylabel, yscale="linear"):
    for c in df.columns:
        plt.plot(df.index, df[c], label=f"{1}{c[2:]}")

    plt.title(title)

    plt.xlabel("P")
    #plt.xscale("log")
    plt.ylabel(ylabel)
    plt.yscale(yscale)
    plt.legend()
 
    plt.show()

def process(d, prefix, context, usecols=[0, 4]):
	suffix = ""
	process = merge(d, prefix, suffix+".csv", usecols)
	title = "process time"
	plot(process, title, "time (s)", "log")

	scalability = pd.DataFrame()
	scalability[process.columns[0]] = process[process.columns[0]]
	for c in process.columns:
		scalability[c] = process[c][1] / process[c]
	
	title = "process time scalability"
	plot(scalability, title, "scalability")

	suffix = "-elapsed"
	elapsed = merge(d, prefix, suffix+".csv", usecols)
	title = "elapsed time"
	plot(elapsed, title, "time (s)", "log")
	
	scalability = pd.DataFrame()
	scalability[elapsed.columns[0]] = elapsed[elapsed.columns[0]]
	for c in elapsed.columns:
		scalability[c] = elapsed[c][1] / elapsed[c]
	
	title = "elapsed time scalability"
	plot(scalability, title, "scalability")
	
	suffix = "-system"
	system = merge(d, prefix, suffix+".csv", usecols)
	title = "system time"
	plot(system, title, "time (s)", "log")

	suffix = "-system"
	system = merge(d, prefix, suffix+".csv", usecols)
	system = system.div(system.index, axis=0)
	title = "system time (scaled)"
	plot(system, title, "time (s)", "log")
	
	overhead = elapsed - process
	title = "overhead time"
	plot(overhead, title, "time (s)", "log")
	
	overhead = elapsed - process
	overhead = overhead.div(overhead.index, axis=0)
	title = "overhead time (scaled)"
	plot(overhead, title, "time (s)", "log")

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
