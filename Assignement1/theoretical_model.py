import math as m
import numpy as np
import matplotlib.pyplot as plt

tread = 10**-4
tcomm = 10**-6
tcomp = 2 * 10**-9

def serial(n):
	return tread + (n - 1) * tcomp

def mpi_naive(n, p):
	return tread + 2 * (p - 1) * tcomm + (m.ceil(n / p) + p - 2) * tcomp

def mpi_naive_p(n):
	ratio = 1 / (1 + 2 * tcomm / tcomp)
	opt = m.sqrt(n * ratio)
	if opt < 1:
		return 1
	elif mpi_naive(n, m.ceil(opt)) > mpi_naive(n, m.floor(opt)):
		return m.floor(opt)
	return m.ceil(opt)

def mpi_opt(n, p):
	return tread + m.ceil(m.log(p, 2)) * (2 * tcomm + tcomp) + (m.ceil(n / p) - 1) * tcomp

def mpi_opt_p(n):
	opt = (n * m.log(2) * tcomp) / (2 * tcomm + tcomp)
	if opt < 1:
		print(f"mpi_opt_p({n}): relaxed {opt} chosen 1")
		return 1
	else:
		p1 = int(2**m.floor(m.log(opt, 2)))
		p2 = int(2**m.ceil(m.log(opt, 2)))

		if mpi_opt(n, p1) > mpi_opt(n, p2):
			print(f"mpi_opt_p({n}): relaxed {opt} chosen {p2}")
			return p2
		print(f"mpi_opt_p({n}): relaxed {opt} chosen {p1}")
		return p1

ns = [10**(4 + i) for i in range(4)]
procs = [1+i for i in range(100)]

dims = (len(ns), len(procs))

tserial = np.zeros((len(ns)))
tmpi_naive = np.zeros(dims)
smpi_naive = np.ones(dims)
tmpi_opt = np.zeros(dims)
smpi_opt = np.ones(dims)

for (i, n) in enumerate(ns):
	for (j, p) in enumerate(procs):
		tserial[i] = serial(n)

		tmpi_naive[i, j] = mpi_naive(n, p)
		tmpi_opt[i, j] = mpi_opt(n, p)

	smpi_naive[i, :] = tserial[i] / tmpi_naive[i, :]
	smpi_opt[i, :] = tserial[i] / tmpi_opt[i, :]

def plot(values, title, ylabel, yscale="linear"):
	for (i, n) in enumerate(ns):
		plt.plot(procs, values[i, :], label=f"N={n:.0E}")

	plt.title(title)

	plt.xlabel("P")
	#plt.xscale("log")
	plt.ylabel(ylabel)
	plt.yscale(yscale)
	plt.legend()

	#plt.show()

root = "plots/theoretical"
ylabel = "time (s)"
plot(tmpi_naive, "", ylabel, yscale="log")
plt.savefig(root + "_naive_time.png", transparent=True)
plt.close()

plot(tmpi_opt, "", ylabel, yscale="log")
plt.savefig(root + "_enhanced_time.png", transparent=True)
plt.close()

ylabel = "speedup"
plot(smpi_naive, "", ylabel)
plt.savefig(root + "_naive_scalability.png", transparent=True)
plt.close()

plot(smpi_opt, "", ylabel)
plt.savefig(root + "_enhanced_scalability.png", transparent=True)
plt.close()

with open("performance-model.csv", "w") as f:
	f.write("#header: N, best P naive algorithm , best P for enhanced algorithm if any, if not just put XXX\n")
	
	for n in [20000, 100000, 200000, 1000000, 2000000]:#ns:
		n_p = min(100, max(mpi_naive_p(n), 1))
		o_p = min(100, max(mpi_opt_p(n), 1))
		
		#if o_p > 1:
		#	print(f"naive({n}, {o_p-1}): {mpi_opt(n, o_p-1)}")
		#print(f"naive({n}, {o_p}): {mpi_opt(n, o_p)}")
		#print(f"naive({n}, {o_p+1}): {mpi_opt(n, o_p+1)}")
		
		f.write(f"{n},{n_p},{o_p}" + "\n")

