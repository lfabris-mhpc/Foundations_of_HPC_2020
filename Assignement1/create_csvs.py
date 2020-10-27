import sys
import os as os
import glob as glob
import math as m

def saveCsv(fn, content, timesCol, timesTop):
	try:
		with open(fn + ".csv", "w") as f:
			f.write("#header line  # processors run1,run2,run3,avg,error_bar  \n")

			def key(v):
				((n, p), times) = v
				return p

			for ((n, p), times) in sorted(content.items(), key=key):
				if len(times[timesCol]) < timesTop:
					print(f"Error writing {fn}: found only {len(times[timesCol])} instead of at least {timesTop}")
					raise Exception

				times = sorted(times[timesCol])[:timesTop]
			
				f.write(str(p))
				f.write("," + ",".join([str(v) for v in times]))
				f.write("," + str(sum(times) / timesTop))
				f.write("," + str((times[-1] - times[0]) / 2))
				f.write("\n")

		print(f"Written {fn}.csv")
	except Exception:
		pass

def processDir(d, dOut="csvs"):
	raw = {}

	for fn in sorted(glob.glob(os.path.join(d, "*.o*"))):
		with open(fn, "r") as f:
			n = 0
			p = 1
			wtime = 0
			elapsed = 0
			user = 0
			sys = 0

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
				elif tokens[0] == "elapsed:":
					elapsed = float(tokens[1])
				elif tokens[0] == "user:":
					user = float(tokens[1])
				elif tokens[0] == "system:":
					sys = float(tokens[1])

					#this is the last line
					if n and p and wtime > 0:
						#valid entry
						#print(f"{fn}: n[{n}] p[{p}] wtime[{wtime}]")
						
						def append():
							if not fid in raw:
								raw[fid] = {}

							if not (n, p) in raw[fid]:
								#wtimes, etimes, stimes
								raw[fid][(n, p)] = [[], [], []]
							raw[fid][(n, p)][0].append(wtime)
							raw[fid][(n, p)][1].append(elapsed)
							raw[fid][(n, p)][2].append(sys)

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

						#print(f"file: {fn} exp: {exp} weak: {weak}")
						
						if serial:
							fid = f"serial"
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

					serial = False

	topwtimes=3
	for (fid, content) in sorted(raw.items()):
		saveCsv(os.path.join(dOut, fid), content, 0, 3)
		saveCsv(os.path.join(dOut, fid + "-elapsed"), content, 1, 3)
		saveCsv(os.path.join(dOut, fid + "-system"), content, 2, 3)

if len(sys.argv) < 2:
	print("Usage: python create_csvs.py work_folder [output_folder]")

dOut = "csvs"
if len(sys.argv) >= 3:
	dOut = sys.argv[2]
else:
	suffix = "_".join(sys.argv[1].split("_")[1:])
	if suffix:
		dOut = dOut + "_" + suffix

processDir(sys.argv[1], dOut)
