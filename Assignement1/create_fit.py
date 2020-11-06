import pandas as pd
import numpy as np
import scipy as sp
from scipy.optimize import curve_fit
from scipy.optimize import differential_evolution
import warnings
import matplotlib.pyplot as plt

def funcLine(ps, a, b, c):
	#a * ps + b
	#a * ps**2 + b * ps + c
	return a * ps + b * np.where(ps > 24, ps - 24, 0) + c

def funcLineEval(ps, ns, a, b, c):
	return funcLine(ps, a, b, c)

def func(ps, a, b, c):#, d):
	#return tscaling * 
	return (a * (ns / ps) + b * ps + c)# + d * np.where(ps > 24, ps - 24, 0)


def funcEval(ps, ns, a, b, c):#, d):
	return a * (ns / ps) + b * ps + c# + d * np.where(ps > 24, ps - 24, 0)

def funcLineError(params):
    warnings.filterwarnings("ignore")

    val = funcLine(ps, *params)
    return np.sum(((ts - val)/ts)**2)

def funcError(params):
    warnings.filterwarnings("ignore")

    val = func(ps, *params)
    return np.sum((1 + (ts - val) / ts)**2)

def calcInitialParams(fError, bounds, seed=4321):
    result = differential_evolution(fError, list(zip(*bounds)), seed=seed)#, strategy="currenttobest1bin", maxiter=1000, popsize=1000)
    return result.x
"""
def calcInitialParams(seed=4321):
    result = differential_evolution(sqErrors, list(zip(*bounds)), seed=seed, strategy="currenttobest1bin", maxiter=1000, popsize=1000)
    return result.x
"""
def fitCurve(ps, ts, f, fError, bounds, tscaling):
	# generate initial parameter values
	initialParams = calcInitialParams(fError, bounds)
	print(f"initialParams: {initialParams / tscaling}")

	# curve fit the test data
	fittedParams, pcov = curve_fit(f, ps, ts, initialParams, bounds=bounds, loss="soft_l1")

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
	#exps = list(dict.fromkeys(df["exp"]))

	for (i, e) in enumerate(df.groupby(["exp"]).groups.keys()):
		dfe = df.loc[df2["exp"] == e]

		pse = dfe["P"].to_numpy()
		nse = dfe["N"].to_numpy()
		pred = fEval(pse, nse, *fittedParams)

		plt.scatter(pse, dfe[timeCol], label=f"1E+{e:02d}", facecolor='none', edgecolor=colors[i])
		plt.plot(pse, pred, label=f"pred 1E+{e:02d}", c=colors[i])

	plt.title("elapsed")

	plt.xlabel("P")
	#plt.xscale("log")
	plt.ylabel("elapsed")
	plt.yscale("log")
	plt.legend()

	if save:
		plt.savefig(save+".png", transparent=True)
	else:
		plt.show()

prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']

tscaling = 10**6

#2.68738806e-08 8.81329129e-04 9.99999900e-10 5.28324479e-05
lower = np.array([0, 0, 0, 0])
upper = np.array([1, 1, 1, 1])
lower = np.array([10**-9, 10**-5, 10**-9]) * tscaling
upper = np.array([10**-7, 10**-1, 1]) * tscaling
bounds = (lower, upper)
print(f"bounds: {bounds}")

df = pd.read_csv("csvs_gpu/fit.csv")
print(df)

df.sort_values(["P", "N", "exp"], axis=0, inplace=True)
##########################################
df2 = df.loc[(df["weak"] == True) | (df["P"] == 1)]
"""
minByExp = df2.groupby("exp")["elapsed"].min()
print(df2.groupby("exp")["elapsed"].min())

for i in df2.index:
	df2.loc[i, "elapsed"] -= minByExp.loc[df2.loc[i, "exp"]]
print(df2)
"""
exps = df2.groupby(["exp"]).groups.keys()

for e in exps:
	df3 = df2.loc[df2["exp"] == e]

	ps = df3["P"].to_numpy()
	ns = df3["N"].to_numpy()
	ts = df3["elapsed"].to_numpy()

	print(f"fitting elapsed with line")
	fittedParams = fitCurve(ps, ts, funcLine, funcLineError, [[0, 0, 0], [10**(e-7), 10**(e-7), 10**(e-7)]], 1)
	plot(df3, "internal", funcLineEval, "internal", "time(s)", yscale="linear", save="")

	print("#"*20)

##########################################
ps = df["P"].to_numpy()
ns = df["N"].to_numpy()
ts = df["internal"].to_numpy() * tscaling

print(f"fitting internal")
fittedParams = fitCurve(ps, ts, func, funcError, bounds, tscaling)

df2 = df.loc[df["weak"] == False]
plot(df2, "internal", funcEval, "internal", "time(s)", yscale="log", save="")

df2 = df.loc[(df["weak"] == True) | (df["P"] == 1)]
plot(df2, "internal", funcEval, "internal", "time(s)", yscale="log", save="")

print("#"*20)
##########################################
ps = df["P"].to_numpy()
ns = df["N"].to_numpy()
ts = df["internal"].to_numpy() * tscaling

print(f"fitting elapsed")
fittedParams = fitCurve(ps, ts, func, funcError, bounds, tscaling)

df2 = df.loc[df["weak"] == False]
plot(df2, "elapsed", funcEval, "elapsed", "time(s)", yscale="log", save="")

df2 = df.loc[(df["weak"] == True) | (df["P"] == 1)]
plot(df2, "elapsed", funcEval, "elapsed", "time(s)", yscale="log", save="")

