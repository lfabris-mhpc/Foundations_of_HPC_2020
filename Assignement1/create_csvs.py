import sys
import os as os
import glob as glob
import math as m
import pandas as pd

def saveCsv(fn, content, timesCol, timesTop, node="GPU"):
	try:
		with open(fn + ".csv", "w") as f:
			#print(f"writing {fn} with len {len(content.items())}")
			f.write(f"#header line: {node} processors,avg,error_bar,run1,run2,run3,... runN" + "\n")

			def key(v):
				((n, p), times) = v
				return p
			
			ok = True
			for ((n, p), times) in sorted(content.items(), key=key):
				if len(times[timesCol]) < timesTop:
					print(f"Error writing {fn}: found only {len(times[timesCol])} samples for P{p} instead of at least {timesTop}")
					ok = False
					#raise Exception

				#vals = sorted(times[timesCol])[:timesTop]
				vals = times[timesCol]

				f.write(f"{p}")
				f.write(f",{sum(vals) / len(vals):.3f}")
				f.write(f",{(max(vals) - min(vals)) / 2:.3f}")
				f.write("," + ",".join([f"{v:.3f}" for v in vals]))
				f.write("\n")

			if not ok:
				raise Exception
		print(f"Written {fn}.csv")
	except Exception:
		pass

def processExp(n, p):
	#discern weak, strong and serial
	(expFrac, exp) = m.modf(m.log(n/p, 10))
	exp = int(exp)
	weak = False

	#print(f"n: {n} p: {p} exp: {exp} expFrac: {expFrac}")
	if abs(expFrac) < 0.01 or abs(1 - expFrac) < 0.01:
		#this is weak scalability
		weak = p != 1
		exp += int(round(expFrac))
		expFrac = 0
	else:
		exp = round(m.log(n, 10))

	return exp, weak

def processDir(d, dOut="csvs", node="GPU"):
	raw = {}

	for fn in sorted(glob.glob(os.path.join(d, "*.o*"))):
		#print(f"reading {fn}")
		with open(fn, "r") as f:
			n = 0
			p = 1
			wtime = 0
			elapsed = 0
			user = 0
			sys = 0
			walltimes = dict()			

			serial = False

			for l in f.readlines():
				tokens = l[:-1].split(" ")

				
				
				if len(tokens) > 3 and tokens[3] == "trials":
					n = int(tokens[5])
				elif l.startswith(" # walltime : "):
					wtime = max(wtime, float(tokens[4]))
					serial = True
				elif len(tokens) > 2 and tokens[2] == "walltime":
					wtime = max(wtime, float(tokens[7]))

					if tokens[4] == "processor":
						p = max(p, 1 + int(tokens[5]))
						walltimes[0] = float(tokens[7])
					else:						
						walltimes[int(tokens[5])] = float(tokens[7])
				elif tokens[0] == "elapsed:":
					elapsed = float(tokens[1])
				elif tokens[0] == "user:":
					user = float(tokens[1])
				elif tokens[0] == "system:":
					sys = float(tokens[1])

					#this is the last line
					#print(f"attempt to add n: {n} p: {p} wtime: {wtime}")
					if n and p and wtime > 0:
						#valid entry
						#print(f"{fn}: n[{n}] p[{p}] wtime[{wtime}]")
						if len(walltimes) > 2:
							for (proc, t) in sorted(walltimes.items())[2:]:
								print(f"time diff{proc}{proc-1}: {t-walltimes[proc-1]}") 
						
						def append():
							if not fid in raw:
								raw[fid] = {}

							if not (n, p) in raw[fid]:
								#wtimes, etimes, stimes
								raw[fid][(n, p)] = [[], [], []]
							raw[fid][(n, p)][0].append(wtime)
							raw[fid][(n, p)][1].append(elapsed)
							raw[fid][(n, p)][2].append(sys)

						"""
						#discern weak, strong and serial
						(expFrac, exp) = m.modf(m.log(n/p, 10))
						exp = int(exp)
						weak = False

						#print(f"n: {n} p: {p} exp: {exp} expFrac: {expFrac}")
						if abs(expFrac) < 0.01 or abs(1 - expFrac) < 0.01:
							#this is weak scalability
							weak = p != 1
							exp += int(round(expFrac))
							expFrac = 0
						else:
							exp = round(m.log(n, 10))
						"""
						exp, weak = processExp(n, p)
						#print(f"file: {fn} exp: {exp} weak: {weak}")
						
						if serial:
							fid = f"serial-10to{exp:02}"
							append()					
						elif not weak:
							#strong(p) and weak(1) if p == 1
							fid = f"strong-scalability-10to{exp:02}"
							append()

							if p == 1:
								fid = f"weak-scalability-10to{exp:02}"
								append()			
						else:
							#this is surely weak(p)
							fid = f"weak-scalability-10to{exp:02}"
							append()

					#reset
					n = 0
					p = 1
					wtime = 0
					elapsed = 0
					user = 0
					sys = 0
					walltimes = dict()	

					serial = False

	timesTop = 3
	for (fid, content) in sorted(raw.items()):
		saveCsv(os.path.join(dOut, fid), content, 0, timesTop)
		saveCsv(os.path.join(dOut, fid + "-elapsed"), content, 1, timesTop)
		saveCsv(os.path.join(dOut, fid + "-system"), content, 2, timesTop)

	master = {"P":[], "N":[], "internal":[], "elapsed":[], "system":[], "exp":[], "weak":[]}
	for (fid, content) in sorted(raw.items()):
		if not fid.startswith("serial"):
			for ((n, p), times) in content.items():
				exp, weak = processExp(n, p)
				
				if p == 1 and fid.startswith("weak"):
					continue				

				for i in range(len(times[0])):
					master["P"].append(int(p))
					master["N"].append(int(n))
					master["internal"].append(times[0][i])
					master["elapsed"].append(times[1][i])
					master["system"].append(times[2][i])
					master["exp"].append(int(exp))
					master["weak"].append(weak)					

	fit = pd.DataFrame.from_dict(master)
	#print(fit.describe())
	fit.to_csv(os.path.join(dOut, "fit.csv"), index=False)

if len(sys.argv) < 2:
	print("Usage: python create_csvs.py work_folder [output_folder]")

dOut = "csvs"
suffix = ""
if len(sys.argv) >= 3:
	suffix = "_".join(sys.argv[1].split("_")[1:])
	dOut = sys.argv[2]
else:
	suffix = "_".join(sys.argv[1].split("_")[1:])
	if suffix:
		dOut = dOut + "_" + suffix

processDir(sys.argv[1], dOut, suffix.upper())
