import sys
import os as os
import glob as glob
#import math as math
import pandas as pd
import matplotlib.pyplot as plt

def merge(d, prefix, suffix = ""):
	ret = None
	g = os.path.join(d, prefix + "*[0-9]" + suffix + ".csv")
	print(f"merging {g}")
	for fn in sorted(glob.glob(g)):
		f = os.path.basename(fn)
		exps = f[len(prefix):-len(suffix)-4]
		nbase = 10**int(exps)
		ncol = str(nbase)
		nerr = ncol + "_error"

		df = pd.read_csv(fn, header=None, names=["P", ncol, nerr], index_col=[0], usecols=[0, 1, 2], comment="#")
		
		if ret is None:
			ret = df
		else:
			ret[ncol] = df[ncol]
			ret[nerr] = df[nerr]
	
	return ret

def plot(df, title, ylabel, yscale="linear", save=""):
    for c in df.columns:
        if not c.endswith("_error"):
            cerr = c + "_error"
            label = f"{float(c):.0E}"
            if cerr in df.columns:
                plt.errorbar(df.index, df[c], yerr=df[cerr], label=label)
            else:
                plt.plot(df.index, df[c], label=label)

    plt.title(title)

    plt.xlabel("P")
    #plt.xscale("log")
    plt.ylabel(ylabel)
    plt.yscale(yscale)
    plt.legend()

    if save:
        plt.savefig(save+".png", transparent=True)
    else:
        plt.show()

def process(d, prefix, context):
	internal = merge(d, prefix)
	title = "process time"
	plot(internal, title, "time (s)", "log")

	elapsed = merge(d, prefix, "-elapsed")
	title = "elapsed time"
	plot(elapsed, title, "time (s)", "log")
	
	system = merge(d, prefix, "-system")
	title = "system time"
	plot(system, title, "time (s)", "log")

    #speedups
	internal_speedup = pd.DataFrame()
	internal_speedup[internal.columns[0]] = internal[internal.columns[0]]
	for c in internal.columns:
		if not c.endswith("_error"):
			internal_speedup[c] = internal[c][1] / internal[c]
		else:
			#internal_speedup[c] = internal[c][1] / internal[c]
			pass
	
	title = "process time speedup"
	plot(internal_speedup, title, "speedup")
	
	elapsed_speedup = pd.DataFrame()
	elapsed_speedup[elapsed.columns[0]] = elapsed[elapsed.columns[0]]
	for c in elapsed.columns:
		if not c.endswith("_error"):
			elapsed_speedup[c] = elapsed[c][1] / elapsed[c]
		else:
			#elapsed_speedup[c] = elapsed[c][1] / elapsed[c]
			pass
	
	title = "elapsed time speedup"
	plot(elapsed_speedup, title, "speedup")

    #composite
	system_scaled = system.copy().div(system.index, axis=0)
	title = "system time (scaled)"
	plot(system_scaled, title, "time (s)", "log")
	
	overhead = elapsed - internal
	title = "overhead time"
	plot(overhead, title, "time (s)", "log")
	
	overhead_scaled = overhead.copy().div(overhead.index, axis=0)
	title = "overhead time (scaled)"
	plot(overhead_scaled, title, "time (s)", "log")

	return internal, elapsed, system, internal_speedup, elapsed_speedup

def getDataframes(d, prefix):
	internal = merge(d, prefix)
	elapsed = merge(d, prefix, "-elapsed")
	system = merge(d, prefix, "-system")

	internal_speedup = pd.DataFrame()
	internal_speedup[internal.columns[0]] = internal[internal.columns[0]]
	for c in internal.columns:
		if not c.endswith("_error"):
			cerr = c + "_error"
			internal_speedup[c] = internal[c][1] / internal[c]
			internal_speedup[cerr] = (internal[c] * internal[cerr][1] + internal[c][1] * internal[cerr]) / (internal[c]**2 - internal[c] * internal[cerr][1])

	
	elapsed_speedup = pd.DataFrame()
	elapsed_speedup[elapsed.columns[0]] = elapsed[elapsed.columns[0]]
	for c in elapsed.columns:
		if not c.endswith("_error"):
			cerr = c + "_error"
			elapsed_speedup[c] = elapsed[c][1] / elapsed[c]
			elapsed_speedup[cerr] = (elapsed[c] * elapsed[cerr][1] + elapsed[c][1] * elapsed[cerr]) / (elapsed[c]**2 - elapsed[c] * elapsed[cerr][1])
		else:
			pass
	
	return internal, elapsed, system, internal_speedup, elapsed_speedup

def getDataframesSerial(d, prefix):
	internal = merge(d, prefix)
	elapsed = merge(d, prefix,  "-elapsed")
	system = merge(d, prefix, "-system")

	return internal, elapsed, system

if len(sys.argv) < 2:
	print("Usage: python create_plots.py csvs_folder [output_folder]")

dOut = "plots"
suffix = ""
if len(sys.argv) >= 3:
	suffix = "_".join(sys.argv[1].split("_")[1:])
	dOut = sys.argv[2]
else:
	suffix = "_".join(sys.argv[1].split("_")[1:])
	if suffix:
		dOut = dOut + "_" + suffix

#serial
prefix = "serial-10to"
(serial_internal, serial_elapsed, serial_system) = getDataframesSerial(sys.argv[1], prefix)

print("serial_internal")
print(serial_internal)
print("serial_elapsed")
print(serial_elapsed)
print("serial_system")
print(serial_system)

#plot(df, title, ylabel, yscale="linear", save=""):
plot(serial_internal, "internal time", "time (s)", "linear")
plot(serial_elapsed, "elapsed time", "time (s)", "log")
#plot(serial_system, "system time", "time (s)", "log")

#strong scalability
prefix = "strong-scalability-10to"

(strong_internal, strong_elapsed, strong_system, strong_internal_speedup, strong_elapsed_speedup) = getDataframes(sys.argv[1], prefix)

#weak scalability
prefix = "weak-scalability-10to"

(weak_internal, weak_elapsed, weak_system, weak_internal_speedup, weak_elapsed_speedup) = getDataframes(sys.argv[1], prefix)

#mpi overhead
print(strong_elapsed)

mpi_overhead_internal = strong_internal.loc[1, :] - serial_internal.loc[1, :]
mpi_overhead_elapsed = strong_elapsed.loc[1, :] - serial_elapsed.loc[1, :]
mpi_overhead_system = strong_system.loc[1, :] - serial_system.loc[1, :]
"""
print(mpi_overhead_internal)
print(mpi_overhead_elapsed)
print(mpi_overhead_system)
"""


