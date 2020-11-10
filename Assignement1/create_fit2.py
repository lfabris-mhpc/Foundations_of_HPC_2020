import pandas as pd
import numpy as np
import scipy as sp
from scipy.optimize import curve_fit
from scipy.optimize import differential_evolution
import warnings
import matplotlib.pyplot as plt

def func(ps, a, b):
	return a * (ns / ps) + b * ps

def funcEval(ps, ns, a, b):
	return a * (ns / ps) + b * ps

#####################################
def funcElapsed(ps, a, b, c):
	return a * (ns / ps) + b * ps + c

def funcElapsedEval(ps, ns, a, b, c):
	return a * (ns / ps) + b * ps + c

#####################################
def funcError(params):
    warnings.filterwarnings("ignore")

    val = func(ps, *params)
    return np.sum((ts - val)**2)

def funcErrorRel(params):
    warnings.filterwarnings("ignore")

    val = func(ps, *params)
    return np.sum(((ts - val)/ts)**2)

#####################################
def funcElapsedError(params):
    warnings.filterwarnings("ignore")

    val = funcElapsed(ps, *params)
    return np.sum((ts - val)**2)

def funcElapsedErrorRel(params):
    warnings.filterwarnings("ignore")

    val = funcElapsed(ps, *params)
    return np.sum(((ts - val)/ts)**2)

#####################################
def calcInitialParams(fError, bounds, seed=4321):#, maxiter=10000, popsize=200):
    result = differential_evolution(fError, list(zip(*bounds)), seed=seed)#, maxiter=maxiter)#, popsize=popsize)#, strategy="currenttobest1bin", popsize=1000)
    return result.x

def fitCurve(ps, ts, f, fError, bounds, tscaling=1):
	# generate initial parameter values
	initialParams = calcInitialParams(fError, bounds)
	print(f"initialParams: {initialParams / tscaling}")

	fittedParams, pcov = curve_fit(f, ps, ts, initialParams, bounds=bounds, maxfev=10000, loss="soft_l1")

	predicted = f(ps, *fittedParams) 
	absError = (predicted - ts) / tscaling

	SE = np.square(absError)
	MSE = np.mean(SE)
	RMSE = np.sqrt(MSE)
	Rsquared = 1.0 - (np.var(absError) / np.var(ts / tscaling))

	print(f"fittedParams: {fittedParams / tscaling}")
	print('RMSE:', RMSE)
	print('R-squared:', Rsquared)

	return fittedParams / tscaling

def plot(df, timeCol, fEval, title, ylabel, yscale="linear", save=""):
	prop_cycle = plt.rcParams['axes.prop_cycle']
	colors = prop_cycle.by_key()['color']

	for weak in [False, True]:
		tmp = df.loc[(df["weak"] == weak) | (df["P"] == 1)]

		for (i, e) in enumerate(tmp.groupby(["exp"]).groups.keys()):
			label = f"10^{e}"
			if weak:
				label = "weak " + label
			else:
				label = "strong " + label

			dfe = tmp.loc[tmp["exp"] == e]
			pse = dfe["P"].to_numpy()
			nse = dfe["N"].to_numpy()

			plt.scatter(pse, dfe[timeCol], label=label, facecolor="none", edgecolor=colors[i])
			
			if weak and e == 11 and len(pse) < 6:
				pse = np.array([28, 28, 28, pse[0], pse[1], pse[2]], dtype=np.float64)
				nse2 = np.empty((6), dtype=np.float64)
				nse2[0:3] = nse * 28 / 48
				nse2[3:] = nse
				nse = nse2

			pred = fEval(pse, nse, *fittedParams)
			plt.plot(pse, pred, label="pred " + label, c=colors[i])

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

tscaling = 10**6

#2.68738806e-08 8.81329129e-04 9.99999900e-10 5.28324479e-05
lower = np.array([0, 0, 0, 0], dtype=np.float64)
upper = np.array([1, 1, 1, 1], dtype=np.float64)

lower = np.array([10**-9, 10**-5, 10**-9], dtype=np.float64) * tscaling
upper = np.array([10**-7, 10**-1, 1], dtype=np.float64) * tscaling

###########
lower = np.array([0, 0], dtype=np.float64) * tscaling
upper = np.array([1, 1], dtype=np.float64) * tscaling
fBounds = (lower, upper)

lower = np.array([0, 0, 0], dtype=np.float64) * tscaling
upper = np.array([1, 1, 1], dtype=np.float64) * tscaling
fElapsedBounds = (lower, upper)
print(f"fBounds: {fBounds}")
print(f"fElapsedBounds: {fElapsedBounds}")

df = pd.read_csv("csvs_gpu/fit.csv")
print(df)

df.sort_values(["P", "N", "exp"], axis=0, inplace=True)
##########################################
if True:
	for col in ["internal", "elapsed"]:
		print(f"fitting {col} P <= 24")

		f = func
		fError = funcError
		fErrorRel = funcErrorRel
		fEval = funcEval
		bounds = fBounds
		if col == "elapsed":
			f = funcElapsed
			fError = funcElapsedError
			fErrorRel = funcElapsedErrorRel
			fEval = funcElapsedEval
			bounds = fElapsedBounds

		df2 = df.loc[df["P"] <= 24]

		ps = df2["P"].to_numpy()
		ns = df2["N"].to_numpy()
		ts = df2[col].to_numpy() * tscaling

		print(f"{bounds}")
		fittedParams = fitCurve(ps, ts, f, fErrorRel, bounds, tscaling)
		#fittedParams = calcInitialParams(funcErrorRel, bounds)
		#print(f"fittedParams: {fittedParams}")
		plot(df2, col, fEval, col, "time(s)", yscale="log", save="")
		plt.close()
		print("#"*20)

		print(f"fitting {col} P > 24")
		
		df2 = df.loc[df["P"] > 24]

		ps = df2["P"].to_numpy()
		ns = df2["N"].to_numpy()
		ts = df2[col].to_numpy() * tscaling

		fittedParams = fitCurve(ps, ts, f, fErrorRel, bounds, tscaling)
		#fittedParams = calcInitialParams(funcError, bounds)
		#print(f"fittedParams: {fittedParams}")
		plot(df2, col, fEval, col, "time(s)", yscale="log", save="")
		plt.close()
		print("#"*20)
