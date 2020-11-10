import sys
import os as os
import glob as glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score

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

def plotComparison(df, df2, title, ylabel, yscale="linear", save="", useErrors=False, doDiag=False, legend=[]):
	prop_cycle = plt.rcParams['axes.prop_cycle']
	colors = prop_cycle.by_key()['color']

	ilabel = 0
	for (i, c) in enumerate(df.columns):
		if not c.endswith("_error"):
			cerr = c + "_error"
			ic = i // 2
			idx = df.index[df[c].notna()]
			if useErrors and cerr in df.columns:
			    plt.errorbar(idx, df.loc[idx, c], yerr=df.loc[idx, cerr], label=legend[ilabel], c=colors[ic])
			    ilabel += 1
			    plt.errorbar(idx, df2.loc[idx, c], yerr=df2.loc[idx, cerr], label=legend[ilabel], c=colors[ic], linestyle="--")
			    ilabel += 1
			else:
			    plt.plot(idx, df.loc[idx, c], label=legend[ilabel], c=colors[ic])
			    ilabel += 1
			    plt.plot(idx, df2.loc[idx, c], label=legend[ilabel], c=colors[ic], linestyle="--")
			    ilabel += 1

	plt.title(title)

	plt.xlabel("P")
	#plt.xscale("log")
	plt.ylabel(ylabel)
	plt.yscale(yscale)
	if doDiag:
		plt.plot(df.index, df.index, label=f"perfect scalability", linestyle="--", c=colors[4])
	
	plt.legend()

	if save:
		plt.savefig(save+".png", transparent=True)
	else:
		plt.show()

def plot(df, title, ylabel, yscale="linear", save="", useErrors=False, labelPrefix="", doDiag=False):
    for c in df.columns:
        if not c.endswith("_error"):
            cerr = c + "_error"
            label = labelPrefix + f"{float(c):.0E}"
            idx = df.index[df[c].notna()]

            if useErrors and cerr in df.columns:
                plt.errorbar(idx, df.loc[idx, c], yerr=df[cerr], label=label)
            else:
                plt.plot(idx, df.loc[idx, c], label=label)

    plt.title(title)

    plt.xlabel("P")
    #plt.xscale("log")
    plt.ylabel(ylabel)
    plt.yscale(yscale)
    if doDiag:
        plt.plot(df.index, df.index, label=f"perfect scalability", linestyle="--")
	
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
"""
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
"""

#strong scalability
prefix = "strong-scalability-10to"

(strong_internal, strong_elapsed, strong_system, strong_internal_speedup, strong_elapsed_speedup) = getDataframes(sys.argv[1], prefix)

labelPrefix="N="
procs = strong_internal.index

#10^08 plots
c = "100000000"
cerr = c + "_error"
strong_internal_08 = strong_internal.loc[:, [c, cerr]]
strong_elapsed_08 = strong_elapsed.loc[:, [c, cerr]]
plotComparison(strong_internal_08, strong_elapsed_08, "", "time (s)", yscale="linear", save="plots/strong_scalability_time_compare", useErrors=False, doDiag=False, legend=["internal", "elapsed"])
plt.close()

strong_internal_speedup_08 = strong_internal_speedup.loc[:, [c, cerr]]
strong_elapsed_speedup_08 = strong_elapsed_speedup.loc[:, [c, cerr]]
plotComparison(strong_internal_speedup_08, strong_elapsed_speedup_08, "", "scalability", yscale="linear", save="plots/strong_scalability_scalability_compare", useErrors=False, doDiag=True, legend=["internal", "elapsed"])
plt.close()

#title="internal time"
plot(strong_internal, "", "time (s)", yscale="log", save="plots/strong_scalability_08_internal", useErrors=False, labelPrefix=labelPrefix)
plt.close()
#title="elapsed time"
plot(strong_elapsed, "", "time (s)", yscale="log", save="plots/strong_scalability_08_elapsed", useErrors=False, labelPrefix=labelPrefix)
plt.close()

#title="internal time scalability"
plot(strong_internal_speedup, "", "scalability", yscale="linear", save="plots/strong_scalability_08_internal_scalability", useErrors=False, labelPrefix=labelPrefix, doDiag=True)
plt.close()
#title="elapsed time scalability"
plot(strong_elapsed_speedup, "", "scalability", yscale="linear", save="plots/strong_scalability_08_elapsed_scalability", useErrors=False, labelPrefix=labelPrefix, doDiag=True)
plt.close()

#full plots
#title="internal time"
plot(strong_internal, "", "time (s)", yscale="log", save="plots/strong_scalability_internal", useErrors=False, labelPrefix=labelPrefix)
plt.close()
#title="elapsed time"
plot(strong_elapsed, "", "time (s)", yscale="log", save="plots/strong_scalability_elapsed", useErrors=False, labelPrefix=labelPrefix)
plt.close()

#title="internal time scalability"
plot(strong_internal_speedup, "", "scalability", yscale="linear", save="plots/strong_scalability_internal_scalability", useErrors=False, labelPrefix=labelPrefix, doDiag=True)
plt.close()
#title="elapsed time scalability"
plot(strong_elapsed_speedup, "", "scalability", yscale="linear", save="plots/strong_scalability_elapsed_scalability", useErrors=False, labelPrefix=labelPrefix, doDiag=True)
plt.close()

#weak scalability
prefix = "weak-scalability-10to"

(weak_internal, weak_elapsed, weak_system, weak_internal_speedup, weak_elapsed_speedup) = getDataframes(sys.argv[1], prefix)

labelPrefix="N=P*"
procs = weak_internal.index
#title="internal time"
plot(weak_internal, "", "time (s)", yscale="log", save="plots/weak_scalability_internal", useErrors=False, labelPrefix=labelPrefix)
plt.close()
#title="elapsed time"
plot(weak_elapsed, "", "time (s)", yscale="log", save="plots/weak_scalability_elapsed", useErrors=False, labelPrefix=labelPrefix)
plt.close()

#title="internal time scalability"
plot(weak_internal_speedup, "", "efficiency", yscale="linear", save="plots/weak_scalability_internal_efficiency", useErrors=False, labelPrefix=labelPrefix, doDiag=False)
plt.close()
#title="elapsed time scalability"
plot(weak_elapsed_speedup, "", "efficiency", yscale="linear", save="plots/weak_scalability_elapsed_efficiency", useErrors=False, labelPrefix=labelPrefix, doDiag=False)
plt.close()

if False:
	#test fits
	pfit_all = np.empty((13*7+4))
	tfit_all = np.empty((13*7+4))
	for exp in range(8, 12):
		c = str(10**exp)
		fit = strong_elapsed.loc[:, [c]] - strong_internal.loc[:, [c]]
		#mini-fit
		pfit = fit.index.to_numpy().reshape((-1, 1))

		a = (exp-8)*26
		pfit_all[a:a+13] = pfit.T
		tfit_all[a:a+13] = fit[c].to_numpy()

		fit = weak_elapsed.loc[:, [c]] - weak_internal.loc[:, [c]]
		#mini-fit
		pfit = fit.index.to_numpy().reshape((-1, 1))

		a +=13
		if exp != 11:
			pfit_all[a:a+13] = pfit.T
			tfit_all[a:a+13] = fit[c].to_numpy()
		else:
			pfit_all[a:a+4] = pfit[[0, 3, 6, 12]].T
			tfit_all[a:a+4] = fit[c].to_numpy()[[0, 3, 6, 12]]

		regr = linear_model.LinearRegression()
		regr.fit(pfit, fit[c])
		pred = regr.predict(pfit)

		print(f"elapsed - internal (10**{exp}):")
		print(f"Coefficients: {regr.coef_}*P + {regr.intercept_}")
		#print(f"Mean squared error: {mean_squared_error(fit[c], pred):.2f}")
		print(f"Mean squared error: {np.mean(np.sum((fit[c] - pred)**2)):.2f}")
		print(f"Coefficient of determination: {r2_score(fit[c], pred):.2f}")

		plt.scatter(pfit.T, fit[c].to_numpy())
		ps = np.array([1] + [4*i for i in range(1, 13)])
		ts = regr.coef_[0] * ps + regr.intercept_

		plt.plot(ps, ts, label=f"predict")
		plt.show()
		
	print(f"global overhead:")
	regr = linear_model.LinearRegression()
	pfit_all = pfit_all.reshape((-1, 1))
	regr.fit(pfit_all, tfit_all)
	pred = regr.predict(pfit_all)

	print(f"Coefficients: {regr.coef_}*P + {regr.intercept_}")
	print(f"Mean squared error: {mean_squared_error(tfit_all, pred):.2f}")
	print(f"Coefficient of determination: {r2_score(tfit_all, pred):.2f}")

	plt.scatter(pfit_all, tfit_all)

	ps = np.array([1] + [4*i for i in range(1, 13)])
	ts = regr.coef_[0] * ps + regr.intercept_

	plt.plot(ps, ts, label=f"predict")
	plt.show()

